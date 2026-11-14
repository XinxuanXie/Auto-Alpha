"""Real LLM, checker, and backtest tools for the V2 smoke workflow."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_workflow.factor_library import FactorKnowledgeBase
from factor_pipeline.compiler import compile_factor_spec
from factor_pipeline.spec import (
    DataField,
    FactorSpec,
    FormulaNode,
    FormulaOperator,
    PortfolioRules,
)

from .contracts import (
    CheckIssue,
    DevelopedHypothesis,
    ExperimentVariantType,
    FactorExperimentSpec,
    FactorFormulaSpecV2,
    HypothesisCheckResult,
    InsightAnalysis,
    InsightDecision,
    MeasurementPlan,
    MeasurementProxy,
    PreflightResult,
    ResearchDirection,
    ResearchIdea,
    SplitMetrics,
    SplitName,
)
from .real_backtest import RealBacktestConfig, RealV2BacktestEngine


SUPPORTED_CONCEPTS = frozenset(
    {
        "intraday_range",
        "liquidity",
        "liquidity_proxy",
        "market_cap",
        "price_momentum",
        "price_reversal",
        "price_volume_confirmation",
        "price_volume_interaction",
        "realized_volatility",
        "recent_return",
        "volatility",
        "volume_abnormality",
        "volume_trend",
    }
)
LEAKAGE_CONCEPTS = frozenset(
    {
        "forward_return",
        "future_price",
        "future_return",
        "future_volume",
        "label",
        "next_day_price",
        "target",
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _InsightReviewDraft(_StrictModel):
    decision: InsightDecision
    supporting_arguments: list[str] = Field(min_length=1, max_length=12)
    objections: list[str] = Field(min_length=1, max_length=12)
    alternative_explanations: list[str] = Field(min_length=1, max_length=12)
    final_research_idea: str = Field(min_length=20, max_length=10_000)
    analysis_text: str = Field(min_length=20, max_length=30_000)


class _HypothesisDraft(_StrictModel):
    title: str = Field(min_length=1, max_length=240)
    claim: str = Field(min_length=10, max_length=4_000)
    economic_rationale: str = Field(min_length=10, max_length=4_000)
    core_mechanism: str = Field(min_length=10, max_length=4_000)
    counterargument: str = Field(min_length=10, max_length=4_000)
    falsification_criteria: list[str] = Field(min_length=1, max_length=12)
    distinction_from_siblings: str = Field(min_length=10, max_length=2_000)
    prediction_horizons_days: list[int] = Field(min_length=1, max_length=4)
    required_concepts: list[str] = Field(min_length=1, max_length=12)
    market_conditions: list[str] = Field(default_factory=list, max_length=12)
    failure_conditions: list[str] = Field(default_factory=list, max_length=12)


class _HypothesisBatch(_StrictModel):
    hypotheses: list[_HypothesisDraft] = Field(min_length=1, max_length=4)


class _MeasurementPlanDraft(_StrictModel):
    mechanism_summary: str = Field(min_length=10, max_length=4_000)
    proxies: list[MeasurementProxy] = Field(min_length=1, max_length=8)
    baseline_logic: str = Field(min_length=10, max_length=2_000)
    full_mechanism_logic: str = Field(min_length=10, max_length=4_000)
    unobservable_aspects: list[str] = Field(default_factory=list, max_length=12)


class _ExperimentDraft(_StrictModel):
    name: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    variation_reason: str = Field(min_length=10, max_length=2_000)
    variant_type: ExperimentVariantType
    represented_proxy_ids: list[str] = Field(min_length=1, max_length=8)
    formula: FormulaNode
    prediction_horizon_days: int = Field(gt=0)


class _ExperimentBatch(_StrictModel):
    experiments: list[_ExperimentDraft] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_required_variants(self):
        variants = {item.variant_type for item in self.experiments}
        required = {
            ExperimentVariantType.BASELINE,
            ExperimentVariantType.FULL_MECHANISM,
        }
        if not required.issubset(variants):
            raise ValueError("experiment batch requires baseline and full_mechanism")
        return self


class _ExperimentRepairDraft(_StrictModel):
    name: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    variation_reason: str = Field(min_length=10, max_length=2_000)
    formula: FormulaNode


class _ExperimentRepairBatch(_StrictModel):
    experiments: list[_ExperimentRepairDraft] = Field(min_length=1, max_length=1)


class RealV2Tools:
    def __init__(
        self,
        panel_path: str | Path,
        membership_path: str | Path,
        *,
        min_hypotheses_per_idea: int = 1,
        max_hypotheses_per_idea: int = 1,
        max_experiments_per_hypothesis: int = 2,
        deepseek_model: str | None = None,
        openai_model: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not 1 <= min_hypotheses_per_idea <= max_hypotheses_per_idea <= 4:
            raise ValueError(
                "hypotheses per idea must satisfy 1 <= minimum <= maximum <= 4"
            )
        if not 2 <= max_experiments_per_hypothesis <= 5:
            raise ValueError("max_experiments_per_hypothesis must be between 2 and 5")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.min_hypotheses_per_idea = min_hypotheses_per_idea
        self.max_hypotheses_per_idea = max_hypotheses_per_idea
        self.max_experiments_per_hypothesis = max_experiments_per_hypothesis
        self.deepseek_model = deepseek_model or os.getenv(
            "DEEPSEEK_RESEARCH_MODEL",
            os.getenv("DEEPSEEK_REFINEMENT_MODEL", "deepseek-chat"),
        )
        self.openai_model = openai_model or os.getenv(
            "OPENAI_MODEL", "gpt-5.6-luna"
        )
        self.deepseek = OpenAI(
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=180.0,
            max_retries=2,
        )
        self.school_openai = OpenAI(
            api_key=openai_key,
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://hkust.azure-api.net/hkust-genai/v1",
            ),
            default_headers={"api-key": openai_key},
            timeout=180.0,
            max_retries=2,
        )
        self.panel_path = Path(panel_path)
        self.membership_path = Path(membership_path)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.factor_library = FactorKnowledgeBase()
        self.backtest_engine = RealV2BacktestEngine(
            RealBacktestConfig(
                panel_path=self.panel_path,
                membership_path=self.membership_path,
            )
        )

    def load_context(self, state: dict[str, Any]) -> dict[str, Any]:
        del state
        entries = [
            entry.model_dump(mode="json") for entry in self.factor_library.entries
        ]
        return {
            "available_fields": [field.value for field in DataField],
            "data_frequency": "daily",
            "universe": "historical point-in-time S&P 500 constituents",
            "factor_library_count": len(entries),
            "factor_library": entries,
            "factor_backtest_metrics_included": False,
            "web_search_enabled": False,
        }

    def generate_research_ideas(
        self, state: dict[str, Any], context: dict[str, Any]
    ) -> list[ResearchIdea]:
        prefix = str(state["run_id"])[:8].upper()
        ideas = []
        for direction in ResearchDirection:
            system_prompt = (
                f"你是专注于 {direction.value} 的量化金融 Research Agent。"
                f"研究范围：{_direction_focus(direction)}。"
                "提出一个抽象但有金融意义的研究洞察。只讨论市场现象、金融或行为机制、"
                "横截面收益差异的形成路径及可能成立或失效的市场条件。"
                "不要生成公式、具体窗口、DSL 算子、FactorSpec 或回测结论。"
                "可以参考完整因子定义库中的机制，但不要直接复制已有公式。"
                "只返回一段聚焦于单一核心洞察的中文文本。"
            )
            user_payload = {
                "task": "提出本方向一个新的抽象研究洞察。",
                "available_data": {
                    "fields": context["available_fields"],
                    "frequency": context["data_frequency"],
                    "universe": context["universe"],
                },
                "factor_library": context["factor_library"],
                "historical_experience": [],
            }
            text = self._deepseek_text(system_prompt, user_payload)
            ideas.append(
                ResearchIdea(
                    idea_id=f"I-{prefix}-{direction.value.upper()}",
                    run_id=str(state["run_id"]),
                    direction=direction,
                    idea_text=text,
                    model=self.deepseek_model,
                    prompt_version="v2.1-critical-development",
                )
            )
        return ideas

    def analyze_insights(
        self, state: dict[str, Any], ideas: list[ResearchIdea]
    ) -> list[InsightAnalysis]:
        del state
        analyses = []
        for idea in ideas:
            review = self._school_json(
                (
                    "你是 Hypothesis Development Agent 的批判性审查与重构阶段。"
                    "不要顺着原始 Idea 扩写；必须同时提出支持理由、反对理由和替代解释，"
                    "检查机制是否混淆相关性与因果、是否只是已有因子的重新表述、是否具有"
                    "清晰的可证伪边界。你可以 ACCEPT、REVISE 或 REPLACE。REPLACE 时必须"
                    "在同一研究方向提出更合理的新 Idea；REVISE 时必须实质性修正机制。"
                    "本阶段保持纯金融分析，不讨论数据字段、DSL、公式或实现。"
                    "最终给出唯一的 final_research_idea，并返回符合 output_schema 的 JSON。"
                ),
                {
                    "research_direction": idea.direction.value,
                    "research_idea": idea.idea_text,
                    "decision_definitions": {
                        "accept": "Idea is already coherent after adversarial review.",
                        "revise": "Keep the core intuition but materially rewrite it.",
                        "replace": "Reject it and create a better idea in the same direction.",
                    },
                    "output_schema": _InsightReviewDraft.model_json_schema(),
                },
                _InsightReviewDraft,
            )
            analyses.append(
                InsightAnalysis(
                    analysis_id=idea.idea_id.replace("I-", "A-", 1),
                    idea_id=idea.idea_id,
                    **review.model_dump(),
                    model=self.openai_model,
                    prompt_version="v2.1-critical-development",
                )
            )
        return analyses

    def expand_hypotheses(
        self,
        state: dict[str, Any],
        ideas: list[ResearchIdea],
        analyses: list[InsightAnalysis],
    ) -> list[DevelopedHypothesis]:
        del state
        analysis_map = {item.idea_id: item for item in analyses}
        hypotheses = []
        batch_model = _HypothesisBatch
        for idea in ideas:
            analysis = analysis_map.get(idea.idea_id)
            if analysis is None:
                continue
            batch = self._school_json(
                (
                    "你是 Hypothesis Development Agent 的 Hypothesis Expansion 阶段。"
                    "只根据批判性审查后的 final_research_idea 生成明确、可证伪的横截面假设。"
                    "每个假设必须代表不同的经济机制或相互竞争的可证伪解释。"
                    "禁止仅通过更换预测周期、市场状态、流动性分组或公司规模分组来凑数量。"
                    "如果只有一个高质量机制，就只输出一个假设。每个假设必须说明核心机制、"
                    "最强反驳、明确的证伪标准，以及它与同批其他假设的实质区别。"
                    "不要生成公式。预测周期可以是任意正整数；不声明预测方向，方向由训练集确定。"
                    "required_concepts 只能从给定 supported_concepts 中选择。"
                    f"必须生成 {self.min_hypotheses_per_idea} 至 "
                    f"{self.max_hypotheses_per_idea} 个假设，具体数量由研究价值决定。"
                    "只返回符合 output_schema 的 JSON。"
                ),
                {
                    "research_direction": idea.direction.value,
                    "research_idea": idea.idea_text,
                    "critical_review": analysis.model_dump(mode="json"),
                    "final_research_idea": analysis.final_research_idea,
                    "supported_concepts": sorted(SUPPORTED_CONCEPTS),
                    "output_schema": batch_model.model_json_schema(),
                },
                batch_model,
            )
            for index, draft in enumerate(
                batch.hypotheses[: self.max_hypotheses_per_idea], start=1
            ):
                hypotheses.append(
                    DevelopedHypothesis(
                        hypothesis_id=f"H-{idea.idea_id[2:]}-{index:02d}",
                        idea_id=idea.idea_id,
                        direction=idea.direction,
                        **draft.model_dump(),
                    )
                )
        return hypotheses

    def check_hypotheses(
        self,
        hypotheses: list[DevelopedHypothesis],
        generation_round: int,
    ) -> list[HypothesisCheckResult]:
        signatures: Counter[str] = Counter(
            _hypothesis_signature(item) for item in hypotheses
        )
        results = []
        for hypothesis in hypotheses:
            issues = []
            unsupported = sorted(
                set(hypothesis.required_concepts) - SUPPORTED_CONCEPTS
            )
            for concept in unsupported:
                issues.append(
                    CheckIssue(
                        code="unsupported_concept",
                        message=f"Unsupported concept: {concept}",
                        field="required_concepts",
                    )
                )
            leaked = sorted(
                set(hypothesis.required_concepts) & LEAKAGE_CONCEPTS
            )
            for concept in leaked:
                issues.append(
                    CheckIssue(
                        code="future_information_leakage",
                        message=f"Leakage concept: {concept}",
                        field="required_concepts",
                    )
                )
            if signatures[_hypothesis_signature(hypothesis)] > 1:
                issues.append(
                    CheckIssue(
                        code="exact_duplicate_hypothesis",
                        message="An identical normalized hypothesis exists in this run.",
                    )
                )
            results.append(
                HypothesisCheckResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    accepted=not issues,
                    issues=issues,
                    generation_round=generation_round,
                )
            )
        return results

    def repair_hypotheses(
        self,
        rejected: list[DevelopedHypothesis],
        issues_by_id: dict[str, list[CheckIssue]],
    ) -> list[DevelopedHypothesis]:
        repaired = []
        for hypothesis in rejected:
            batch = self._school_json(
                (
                    "修正一个未通过确定性检查的假设。保留原经济机制，但 required_concepts "
                    "只能使用 supported_concepts，且不得复制输入中的完整 claim。"
                    "恰好返回一个假设 JSON。"
                ),
                {
                    "rejected_hypothesis": hypothesis.model_dump(mode="json"),
                    "checker_issues": [
                        item.model_dump(mode="json")
                        for item in issues_by_id.get(hypothesis.hypothesis_id, [])
                    ],
                    "supported_concepts": sorted(SUPPORTED_CONCEPTS),
                    "output_schema": _HypothesisBatch.model_json_schema(),
                },
                _HypothesisBatch,
            )
            draft = batch.hypotheses[0]
            repaired.append(
                DevelopedHypothesis(
                    hypothesis_id=f"{hypothesis.hypothesis_id}-R1",
                    idea_id=hypothesis.idea_id,
                    direction=hypothesis.direction,
                    **draft.model_dump(),
                )
            )
        return repaired

    def design_measurement_plans(
        self, hypotheses: list[DevelopedHypothesis]
    ) -> list[MeasurementPlan]:
        plans = []
        for hypothesis in hypotheses:
            draft = self._deepseek_json(
                (
                    "你是 Measurement Planning Agent。先不要生成公式。把纯金融假设拆成"
                    "可由当前日频字段观测的机制组件和代理变量。必须明确哪些机制无法被"
                    "OHLCV/市值忠实观测，并为简单 baseline 与完整机制分别给出测量逻辑。"
                    "代理必须使用给定字段，proxy_id 使用 P1、P2 等稳定编号。"
                    "不要声称代理等于真实机制；必须列出每个代理的局限。"
                    "只返回符合 output_schema 的 JSON。"
                ),
                {
                    "hypothesis": hypothesis.model_dump(mode="json"),
                    "available_fields": [field.value for field in DataField],
                    "output_schema": _MeasurementPlanDraft.model_json_schema(),
                },
                _MeasurementPlanDraft,
            )
            plans.append(
                MeasurementPlan(
                    plan_id=f"MP-{hypothesis.hypothesis_id[2:]}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    **draft.model_dump(),
                )
            )
        return plans

    def generate_experiments(
        self,
        hypotheses: list[DevelopedHypothesis],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]:
        plan_by_hypothesis = {
            plan.hypothesis_id: plan for plan in measurement_plans
        }
        experiments = []
        for hypothesis in hypotheses:
            plan = plan_by_hypothesis.get(hypothesis.hypothesis_id)
            if plan is None:
                continue
            batch = self._read_cached_experiment_batch(hypothesis.hypothesis_id)
            if batch is None:
                batch = self._deepseek_json(
                    (
                        "你是 FactorSpec Agent。把一个结构化假设转换成可执行的结构化因子实验。"
                        "必须严格遵循 measurement_plan，不得用一个方便的技术指标替代整个机制。"
                        "只能使用给定字段和 DSL 算子，不得生成 Python。"
                        "每个实验必须对应一个预测周期，且周期来自 hypothesis。"
                        "同一实验可以同时改变公式结构和窗口。"
                        "参数约束：return 与 lag 只设置 window；mean/std/sum/rolling_max/"
                        "rolling_min/quantile/ts_rank 等滚动算子才可设置 min_periods；"
                        "field 与 constant 不设置 window 或 min_periods。"
                        "所有一元算子（包括 residual）必须在 field 或单个 args 子节点之间二选一，"
                        "绝不能同时填写 field 和 args。"
                        "本 DSL 的 residual、slope、rsquare 都是单序列相对时间趋势的滚动统计，"
                        "只接收一个输入，不是两个变量之间的回归。"
                        "必须生成一个 baseline 和一个 full_mechanism；其余只能是 ablation 或"
                        "robustness。禁止仅更换预测周期来制造新实验。每个实验必须列出实际"
                        "表达的 proxy_id。"
                        f"总数为 2 至 {self.max_experiments_per_hypothesis} 个实验。"
                        "只返回一个符合 output_schema 的 JSON 对象。"
                    ),
                    {
                        "hypothesis": hypothesis.model_dump(mode="json"),
                        "measurement_plan": plan.model_dump(mode="json"),
                        "available_fields": [field.value for field in DataField],
                        "available_operators": [item.value for item in FormulaOperator],
                        "formula_node_schema": FormulaNode.model_json_schema(),
                        "output_schema": _ExperimentBatch.model_json_schema(),
                    },
                    _ExperimentBatch,
                )
                self._write_cached_experiment_batch(
                    hypothesis.hypothesis_id, batch
                )
            accepted_drafts = batch.experiments[
                : self.max_experiments_per_hypothesis
            ]
            for index, draft in enumerate(accepted_drafts, start=1):
                horizon = draft.prediction_horizon_days
                if horizon not in hypothesis.prediction_horizons_days:
                    horizon = hypothesis.prediction_horizons_days[
                        (index - 1) % len(hypothesis.prediction_horizons_days)
                    ]
                formula = draft.formula
                factor_id = f"F-{hypothesis.hypothesis_id[2:]}-{index:02d}"
                factor = FactorFormulaSpecV2(
                    factor_id=factor_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    name=draft.name,
                    description=draft.description,
                    formula=formula,
                    required_fields=sorted(
                        formula.referenced_fields(), key=lambda item: item.value
                    ),
                    lookback_days=formula.required_history(),
                )
                experiments.append(
                    FactorExperimentSpec(
                        experiment_id=f"E-{factor_id[2:]}",
                        hypothesis_id=hypothesis.hypothesis_id,
                        variation_reason=draft.variation_reason,
                        measurement_plan_id=plan.plan_id,
                        variant_type=draft.variant_type,
                        represented_proxy_ids=draft.represented_proxy_ids,
                        factor=factor,
                        prediction_horizon_days=horizon,
                    )
                )
        return experiments

    def _experiment_cache_path(self, hypothesis_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hypothesis_id}.json"

    def _read_cached_experiment_batch(
        self, hypothesis_id: str
    ) -> _ExperimentBatch | None:
        path = self._experiment_cache_path(hypothesis_id)
        if path is None or not path.is_file():
            return None
        return _ExperimentBatch.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_cached_experiment_batch(
        self, hypothesis_id: str, batch: _ExperimentBatch
    ) -> None:
        path = self._experiment_cache_path(hypothesis_id)
        if path is None:
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(batch.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def preflight_experiments(
        self,
        experiments: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
        correction_round: int,
    ) -> list[PreflightResult]:
        signatures: Counter[str] = Counter(
            _experiment_signature(item) for item in experiments
        )
        per_hypothesis = Counter(item.hypothesis_id for item in experiments)
        results = []
        plan_by_id = {plan.plan_id: plan for plan in measurement_plans}
        panel = pl.scan_parquet(self.panel_path)
        sample_start = pl.datetime(2008, 1, 1)
        sample_end = pl.datetime(2010, 12, 31)
        sample_ids = (
            panel.filter(pl.col("date").is_between(sample_start, sample_end))
            .select("permno")
            .unique()
            .sort("permno")
            .limit(64)
            .collect(engine="streaming")
            .get_column("permno")
            .to_list()
        )
        sample_panel = panel.filter(
            pl.col("date").is_between(sample_start, sample_end)
            & pl.col("permno").is_in(sample_ids)
        )
        for experiment in experiments:
            issues = []
            plan = plan_by_id.get(experiment.measurement_plan_id)
            if plan is None or plan.hypothesis_id != experiment.hypothesis_id:
                issues.append(
                    CheckIssue(
                        code="invalid_measurement_plan_lineage",
                        message="Experiment does not reference its hypothesis measurement plan.",
                        field="measurement_plan_id",
                    )
                )
            else:
                plan_proxy_ids = {proxy.proxy_id for proxy in plan.proxies}
                represented = set(experiment.represented_proxy_ids)
                if not represented.issubset(plan_proxy_ids):
                    issues.append(
                        CheckIssue(
                            code="unknown_measurement_proxy",
                            message="Experiment references a proxy outside its measurement plan.",
                            field="represented_proxy_ids",
                        )
                    )
                if (
                    experiment.variant_type == ExperimentVariantType.FULL_MECHANISM
                    and represented != plan_proxy_ids
                ):
                    issues.append(
                        CheckIssue(
                            code="incomplete_full_mechanism",
                            message="Full-mechanism experiment must represent every plan proxy.",
                            field="represented_proxy_ids",
                        )
                    )
                proxy_fields = {
                    proxy.proxy_id: set(proxy.required_fields) for proxy in plan.proxies
                }
                required_fields = set().union(
                    *(
                        proxy_fields[proxy_id]
                        for proxy_id in represented
                        if proxy_id in proxy_fields
                    )
                )
                if not required_fields.issubset(
                    experiment.factor.formula.referenced_fields()
                ):
                    issues.append(
                        CheckIssue(
                            code="measurement_proxy_not_implemented",
                            message=(
                                "Formula does not use every field required by its declared "
                                "measurement proxies."
                            ),
                            field="factor.formula",
                        )
                    )
            if per_hypothesis[experiment.hypothesis_id] > self.max_experiments_per_hypothesis:
                issues.append(
                    CheckIssue(
                        code="experiment_limit_exceeded",
                        message="Hypothesis generated too many experiment configurations.",
                    )
                )
            if signatures[_experiment_signature(experiment)] > 1:
                issues.append(
                    CheckIssue(
                        code="exact_duplicate_experiment",
                        message="An identical formula and horizon exists in this run.",
                    )
                )
            for message in self.backtest_engine.validate_sample_capacity(
                experiment.prediction_horizon_days
            ):
                issues.append(
                    CheckIssue(
                        code="insufficient_sample_capacity",
                        message=message,
                        field="prediction_horizon_days",
                    )
                )
            try:
                compiled = compile_factor_spec(
                    sample_panel, _legacy_factor_spec(experiment)
                )
                schema = compiled.collect_schema()
                if "raw_factor" not in schema:
                    raise ValueError("compiled formula did not produce raw_factor")
                sample = (
                    compiled
                    .select("date", "permno", "raw_factor")
                    .filter(pl.col("raw_factor").is_finite())
                    .limit(1)
                    .collect(engine="streaming")
                )
                if sample.is_empty():
                    raise ValueError("formula produced no finite value in preflight sample")
            except Exception as error:
                issues.append(
                    CheckIssue(
                        code="formula_compile_failed",
                        message=str(error),
                        field="formula",
                    )
                )
            results.append(
                PreflightResult(
                    experiment_id=experiment.experiment_id,
                    accepted=not issues,
                    issues=issues,
                    correction_round=correction_round,
                )
            )
        return results

    def repair_experiments(
        self,
        rejected: list[FactorExperimentSpec],
        issues_by_id: dict[str, list[CheckIssue]],
        accepted: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]:
        repaired = []
        plan_by_id = {plan.plan_id: plan for plan in measurement_plans}
        for experiment in rejected:
            plan = plan_by_id.get(experiment.measurement_plan_id)
            batch = self._deepseek_json(
                (
                    "修正一个未通过 Preflight 的 FactorSpec。只使用允许的 OHLCV/市值字段"
                    "和 DSL 算子，返回一个更简单、可执行且经济含义一致的实验。"
                    "必须针对 checker_issues 中的真实错误修复，不得猜测失败原因。"
                    "若错误是重复，修正后的公式与 existing_experiments 必须具有实质差异。"
                    "只返回符合 output_schema 的完整 JSON 对象。"
                ),
                {
                    "rejected_experiment": experiment.model_dump(mode="json"),
                    "measurement_plan": (
                        plan.model_dump(mode="json") if plan is not None else None
                    ),
                    "checker_issues": [
                        item.model_dump(mode="json")
                        for item in issues_by_id.get(experiment.experiment_id, [])
                    ],
                    "existing_experiments": [
                        {
                            "experiment_id": item.experiment_id,
                            "prediction_horizon_days": item.prediction_horizon_days,
                            "formula": item.factor.formula.model_dump(mode="json"),
                        }
                        for item in accepted
                        if item.hypothesis_id == experiment.hypothesis_id
                    ],
                    "available_fields": [field.value for field in DataField],
                    "available_operators": [item.value for item in FormulaOperator],
                    "formula_node_schema": FormulaNode.model_json_schema(),
                    "output_schema": _ExperimentRepairBatch.model_json_schema(),
                },
                _ExperimentRepairBatch,
            )
            draft = batch.experiments[0]
            formula = draft.formula
            repaired_factor = FactorFormulaSpecV2(
                factor_id=f"{experiment.factor.factor_id}-R1",
                hypothesis_id=experiment.hypothesis_id,
                name=draft.name,
                description=draft.description,
                formula=formula,
                required_fields=sorted(
                    formula.referenced_fields(), key=lambda item: item.value
                ),
                lookback_days=formula.required_history(),
            )
            repaired.append(
                FactorExperimentSpec(
                    experiment_id=f"{experiment.experiment_id}-R1",
                    hypothesis_id=experiment.hypothesis_id,
                    variation_reason=draft.variation_reason,
                    measurement_plan_id=experiment.measurement_plan_id,
                    variant_type=experiment.variant_type,
                    represented_proxy_ids=experiment.represented_proxy_ids,
                    factor=repaired_factor,
                    prediction_horizon_days=experiment.prediction_horizon_days,
                )
            )
        return repaired

    def run_backtests(
        self,
        experiments: list[FactorExperimentSpec],
        split: SplitName,
        train_metrics: dict[str, SplitMetrics] | None = None,
    ) -> list[SplitMetrics]:
        return self.backtest_engine.evaluate(experiments, split, train_metrics)

    def build_memory_summary(
        self,
        train_metrics: list[SplitMetrics],
        validation_metrics: list[SplitMetrics],
        test_metrics: list[SplitMetrics],
    ) -> dict[str, Any]:
        train_pass = [
            item
            for item in train_metrics
            if item.mean_rank_ic > 0.02 and item.net_return > 0
        ]
        validation_pass = [
            item
            for item in validation_metrics
            if item.mean_rank_ic > 0.02 and item.net_return > 0
        ]
        return {
            "summary_version": "v2.1-real-1",
            "train_experiments": len(train_metrics),
            "train_passed": len(train_pass),
            "validation_experiments": len(validation_metrics),
            "validation_passed": len(validation_pass),
            "test_experiments": len(test_metrics),
            "research_experience": [
                (
                    "Some generated mechanisms passed the training threshold."
                    if train_pass
                    else "No generated mechanism passed the training threshold."
                ),
                (
                    "At least one signal remained valid in frozen validation."
                    if validation_pass
                    else "No training signal remained valid in frozen validation."
                ),
            ],
        }

    def _deepseek_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        return self._completion_content(
            self.deepseek,
            self.deepseek_model,
            system_prompt,
            payload,
            max_tokens=2_048,
            provider_name="DeepSeek",
        )

    def _deepseek_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: type[BaseModel],
    ) -> Any:
        return self._validated_json_completion(
            self.deepseek,
            self.deepseek_model,
            system_prompt,
            payload,
            model,
            provider_name="DeepSeek",
        )

    def _school_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        return self._completion_content(
            self.school_openai,
            self.openai_model,
            system_prompt,
            payload,
            max_tokens=8_192,
            provider_name="School OpenAI endpoint",
        )

    def _school_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: type[BaseModel],
    ) -> Any:
        return self._validated_json_completion(
            self.school_openai,
            self.openai_model,
            system_prompt,
            payload,
            model,
            provider_name="School OpenAI endpoint",
        )

    def _validated_json_completion(
        self,
        client: OpenAI,
        model_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        output_model: type[BaseModel],
        *,
        provider_name: str,
    ) -> Any:
        request_payload = dict(payload)
        last_error: ValidationError | json.JSONDecodeError | None = None
        for schema_attempt in range(3):
            content = self._completion_content(
                client,
                model_name,
                system_prompt,
                request_payload,
                max_tokens=8_192,
                provider_name=provider_name,
                response_format={"type": "json_object"},
            )
            try:
                raw_payload = json.loads(content)
                if output_model in {_ExperimentBatch, _ExperimentRepairBatch}:
                    raw_payload = _normalize_experiment_batch(raw_payload)

                return output_model.model_validate(raw_payload)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error

                if isinstance(error, ValidationError):
                    validation_errors = error.errors(
                        include_url=False,
                        include_input=False,
                    )
                else:
                    validation_errors = [
                        {
                            "type": "json_decode_error",
                            "message": str(error),
                            "line": error.lineno,
                            "column": error.colno,
                            "position": error.pos,
                        }
                    ]

                if schema_attempt < 2:
                    request_payload = {
                        **payload,
                        "previous_invalid_output": content,
                        "validation_errors": validation_errors,
                        "correction_instruction": (
                            "Return one complete syntactically valid JSON object. Correct every "
                            "listed JSON or schema error. Return the full required experiment "
                            "batch, including baseline and full_mechanism when the output schema "
                            "requires them. Preserve valid economic content."
                        ),
                    }
        raise ValueError(
            f"{provider_name} returned schema-invalid JSON after three attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _completion_content(
        client: OpenAI,
        model: str,
        system_prompt: str,
        payload: dict[str, Any],
        *,
        max_tokens: int,
        provider_name: str,
        response_format: dict[str, str] | None = None,
        attempts: int = 3,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        finish_reasons: list[str] = []
        for _ in range(attempts):
            options: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                options["response_format"] = response_format
            response = client.chat.completions.create(**options)
            finish_reasons.append(
                str(response.choices[0].finish_reason)
                if response.choices
                else "no_choice"
            )
            content = (
                response.choices[0].message.content if response.choices else None
            )
            if content and content.strip():
                return content.strip()
        raise ValueError(
            f"{provider_name} returned empty content in {attempts} consecutive "
            f"responses; finish_reasons={finish_reasons}"
        )


def _direction_focus(direction: ResearchDirection) -> str:
    return {
        ResearchDirection.MOMENTUM: "信息扩散、趋势延续与拥挤过滤",
        ResearchDirection.REVERSAL: "过度反应、价格冲击与卖压衰竭",
        ResearchDirection.LIQUIDITY_VOLATILITY: "流动性变化、风险补偿与波动状态",
        ResearchDirection.PRICE_VOLUME: "价格与成交量的确认、背离和非线性交互",
    }[direction]


def _hypothesis_signature(hypothesis: DevelopedHypothesis) -> str:
    payload = {
        "title": " ".join(hypothesis.title.split()).casefold(),
        "claim": " ".join(hypothesis.claim.split()).casefold(),
        "economic_rationale": " ".join(
            hypothesis.economic_rationale.split()
        ).casefold(),
        "core_mechanism": " ".join(hypothesis.core_mechanism.split()).casefold(),
        "horizons": hypothesis.prediction_horizons_days,
        "concepts": hypothesis.required_concepts,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _experiment_signature(experiment: FactorExperimentSpec) -> str:
    return json.dumps(
        {
            "formula": experiment.factor.formula.model_dump(mode="json"),
            "prediction_horizon_days": experiment.prediction_horizon_days,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_factor_spec(experiment: FactorExperimentSpec) -> FactorSpec:
    factor = experiment.factor
    return FactorSpec(
        factor_id=factor.factor_id,
        hypothesis_id=factor.hypothesis_id,
        name=factor.name,
        description=factor.description,
        formula=factor.formula,
        required_fields=factor.required_fields,
        lookback_days=factor.lookback_days,
        higher_is_better=True,
        rebalance_frequency="weekly",
        universe_filters=factor.universe_filters,
        portfolio_rules=PortfolioRules(),
    )


_UNARY_OPERATOR_NAMES = {
    item.value
    for item in (
        FormulaOperator.LAG,
        FormulaOperator.RETURN,
        FormulaOperator.MEAN,
        FormulaOperator.STD,
        FormulaOperator.SUM,
        FormulaOperator.ROLLING_MAX,
        FormulaOperator.ROLLING_MIN,
        FormulaOperator.QUANTILE,
        FormulaOperator.TS_RANK,
        FormulaOperator.IDX_MAX,
        FormulaOperator.IDX_MIN,
        FormulaOperator.SLOPE,
        FormulaOperator.RSQUARE,
        FormulaOperator.RESIDUAL,
        FormulaOperator.ZSCORE,
        FormulaOperator.RANK,
        FormulaOperator.ABS,
        FormulaOperator.LOG,
        FormulaOperator.NEGATE,
    )
}


def _normalize_experiment_batch(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        return payload
    normalized = dict(payload)
    normalized["experiments"] = [
        {
            **item,
            "formula": _normalize_formula_node(item.get("formula")),
        }
        if isinstance(item, dict)
        else item
        for item in experiments
    ]
    return normalized


def _normalize_formula_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    normalized = dict(node)
    args = normalized.get("args")
    if isinstance(args, list):
        normalized["args"] = [_normalize_formula_node(item) for item in args]
    if (
        normalized.get("op") in _UNARY_OPERATOR_NAMES
        and normalized.get("field") is not None
        and isinstance(normalized.get("args"), list)
        and len(normalized["args"]) == 1
    ):
        normalized.pop("field", None)
    return normalized


def _valid_experiment_drafts(payload: Any) -> list[_ExperimentDraft]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("experiments"), list
    ):
        return []
    valid = []
    for item in payload["experiments"]:
        try:
            valid.append(_ExperimentDraft.model_validate(item))
        except ValidationError:
            continue
    return valid
