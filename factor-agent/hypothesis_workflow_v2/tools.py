"""Tool contracts and deterministic mocks for the V2 workflow."""

from __future__ import annotations

from typing import Any, Protocol

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


class V2Tools(Protocol):
    def load_context(self, state: dict[str, Any]) -> dict[str, Any]: ...

    def generate_research_ideas(
        self, state: dict[str, Any], context: dict[str, Any]
    ) -> list[ResearchIdea]: ...

    def analyze_insights(
        self, state: dict[str, Any], ideas: list[ResearchIdea]
    ) -> list[InsightAnalysis]: ...

    def expand_hypotheses(
        self,
        state: dict[str, Any],
        ideas: list[ResearchIdea],
        analyses: list[InsightAnalysis],
    ) -> list[DevelopedHypothesis]: ...

    def check_hypotheses(
        self,
        hypotheses: list[DevelopedHypothesis],
        generation_round: int,
    ) -> list[HypothesisCheckResult]: ...

    def repair_hypotheses(
        self,
        rejected: list[DevelopedHypothesis],
        issues_by_id: dict[str, list[CheckIssue]],
    ) -> list[DevelopedHypothesis]: ...

    def design_measurement_plans(
        self, hypotheses: list[DevelopedHypothesis]
    ) -> list[MeasurementPlan]: ...

    def generate_experiments(
        self,
        hypotheses: list[DevelopedHypothesis],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]: ...

    def preflight_experiments(
        self,
        experiments: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
        correction_round: int,
    ) -> list[PreflightResult]: ...

    def repair_experiments(
        self,
        rejected: list[FactorExperimentSpec],
        issues_by_id: dict[str, list[CheckIssue]],
        accepted: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]: ...

    def run_backtests(
        self,
        experiments: list[FactorExperimentSpec],
        split: SplitName,
        train_metrics: dict[str, SplitMetrics] | None = None,
    ) -> list[SplitMetrics]: ...

    def build_memory_summary(
        self,
        train_metrics: list[SplitMetrics],
        validation_metrics: list[SplitMetrics],
        test_metrics: list[SplitMetrics],
    ) -> dict[str, Any]: ...


class MockV2Tools:
    """Deterministic tools that exercise every normal V2 route."""

    allowed_concepts = {
        "recent_return",
        "price_reversal",
        "volume_abnormality",
        "realized_volatility",
        "liquidity_proxy",
        "price_volume_confirmation",
    }

    def load_context(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": state["run_id"],
            "available_fields": ["open", "high", "low", "close", "volume", "mktcap"],
            "factor_library_count": 279,
            "factor_library_source": "mock_factor_library",
            "web_search_enabled": False,
        }

    def generate_research_ideas(
        self, state: dict[str, Any], context: dict[str, Any]
    ) -> list[ResearchIdea]:
        descriptions = {
            ResearchDirection.MOMENTUM: (
                "Information diffusion may create continuation until trading becomes "
                "crowded, after which the same signal can weaken."
            ),
            ResearchDirection.REVERSAL: (
                "Large price shocks may reflect temporary pressure whose reversal depends "
                "on whether selling pressure is exhausted."
            ),
            ResearchDirection.LIQUIDITY_VOLATILITY: (
                "Changes in liquidity and volatility may reveal shifts in risk-bearing "
                "capacity and alter subsequent cross-sectional returns."
            ),
            ResearchDirection.PRICE_VOLUME: (
                "Price moves confirmed or contradicted by trading volume may distinguish "
                "information-driven moves from transient demand."
            ),
        }
        prefix = str(state["run_id"])[:8].upper()
        return [
            ResearchIdea(
                idea_id=f"I-{prefix}-{direction.value.upper()}",
                run_id=str(state["run_id"]),
                direction=direction,
                idea_text=descriptions[direction],
                model="mock-deepseek-flash",
                prompt_version="v2-mock-1",
            )
            for direction in ResearchDirection
        ]

    def analyze_insights(
        self, state: dict[str, Any], ideas: list[ResearchIdea]
    ) -> list[InsightAnalysis]:
        del state
        return [
            InsightAnalysis(
                analysis_id=idea.idea_id.replace("I-", "A-", 1),
                idea_id=idea.idea_id,
                decision=InsightDecision.REVISE,
                supporting_arguments=["The mechanism has a plausible financial channel."],
                objections=["The original idea requires clearer falsification boundaries."],
                alternative_explanations=["A common risk exposure may explain the pattern."],
                final_research_idea=(
                    f"A critically revised {idea.direction.value} idea with a distinct "
                    "mechanism and explicit failure conditions."
                ),
                analysis_text=(
                    f"The proposed {idea.direction.value} mechanism can generate different "
                    "return patterns across horizons and market states. The analysis should "
                    "separate the core signal from confirmation and failure conditions."
                ),
                model="mock-openai",
                prompt_version="v2-mock-1",
            )
            for idea in ideas
        ]

    def expand_hypotheses(
        self,
        state: dict[str, Any],
        ideas: list[ResearchIdea],
        analyses: list[InsightAnalysis],
    ) -> list[DevelopedHypothesis]:
        del state
        analysis_ids = {analysis.idea_id for analysis in analyses}
        hypotheses: list[DevelopedHypothesis] = []
        for idea in ideas:
            if idea.idea_id not in analysis_ids:
                continue
            concepts = self._concepts_for_direction(idea.direction)
            for number in (1, 2):
                required_concepts = list(concepts)
                if idea.direction == ResearchDirection.MOMENTUM and number == 2:
                    required_concepts = ["unsupported_mock_concept"]
                hypotheses.append(
                    DevelopedHypothesis(
                        hypothesis_id=f"H-{idea.idea_id[2:]}-{number:02d}",
                        idea_id=idea.idea_id,
                        direction=idea.direction,
                        title=f"{idea.direction.value} mechanism {number}",
                        claim=(
                            "The proposed mechanism creates a measurable cross-sectional "
                            "difference in subsequent returns across multiple horizons."
                        ),
                        economic_rationale=(
                            "Investor response, risk-bearing capacity, and trading pressure "
                            "can transmit the mechanism into future returns."
                        ),
                        core_mechanism=(
                            "A distinct behavioral or risk-bearing channel transmits the "
                            "signal into subsequent cross-sectional returns."
                        ),
                        counterargument=(
                            "The association may instead reflect a common risk exposure."
                        ),
                        falsification_criteria=[
                            "The proposed mechanism adds no information beyond its baseline."
                        ],
                        distinction_from_siblings=(
                            f"Mock branch {number} represents a separate mechanism test."
                        ),
                        prediction_horizons_days=[5, 20],
                        required_concepts=required_concepts,
                        market_conditions=["liquid US equities"],
                        failure_conditions=["abrupt structural regime change"],
                    )
                )
        return hypotheses

    def check_hypotheses(
        self,
        hypotheses: list[DevelopedHypothesis],
        generation_round: int,
    ) -> list[HypothesisCheckResult]:
        results = []
        for hypothesis in hypotheses:
            unsupported = sorted(
                set(hypothesis.required_concepts) - self.allowed_concepts
            )
            issues = [
                CheckIssue(
                    code="unsupported_concept",
                    message=f"Unsupported research concept: {concept}",
                    field="required_concepts",
                )
                for concept in unsupported
            ]
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
        del issues_by_id
        repaired = []
        for hypothesis in rejected:
            repaired.append(
                hypothesis.model_copy(
                    update={
                        "hypothesis_id": f"{hypothesis.hypothesis_id}-R1",
                        "required_concepts": list(
                            self._concepts_for_direction(hypothesis.direction)
                        ),
                    }
                )
            )
        return repaired

    def design_measurement_plans(
        self, hypotheses: list[DevelopedHypothesis]
    ) -> list[MeasurementPlan]:
        return [
            MeasurementPlan(
                plan_id=f"MP-{hypothesis.hypothesis_id[2:]}",
                hypothesis_id=hypothesis.hypothesis_id,
                mechanism_summary=hypothesis.core_mechanism,
                proxies=[
                    MeasurementProxy(
                        proxy_id="P1",
                        mechanism_component="Recent price adjustment component.",
                        observable_definition="Five-day close-to-close return.",
                        required_fields=["close"],
                        limitations=["The proxy may contain common market movement."],
                    )
                ],
                baseline_logic="Use the simplest recent-return proxy as the baseline.",
                full_mechanism_logic=(
                    "Use the same proxy with the hypothesis-specific interpretation."
                ),
                unobservable_aspects=["Investor identity is not observed."],
            )
            for hypothesis in hypotheses
        ]

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
            plan = plan_by_hypothesis[hypothesis.hypothesis_id]
            horizons = hypothesis.prediction_horizons_days[:2]
            for number, horizon in enumerate(horizons, start=1):
                experiment_id = f"E-{hypothesis.hypothesis_id[2:]}-{number:02d}"
                factor = FactorFormulaSpecV2.model_validate(
                    {
                        "factor_id": f"F-{experiment_id[2:]}",
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "name": f"{hypothesis.title} horizon {horizon}",
                        "description": "Mock executable formula for workflow validation.",
                        "formula": {
                            "op": "return",
                            "field": "close",
                            "window": 5 if number == 1 else 20,
                        },
                        "required_fields": ["close"],
                        "lookback_days": 5 if number == 1 else 20,
                    }
                )
                experiments.append(
                    FactorExperimentSpec(
                        experiment_id=experiment_id,
                        hypothesis_id=hypothesis.hypothesis_id,
                        variation_reason=(
                            "Compare a baseline with the full measurement mechanism."
                        ),
                        measurement_plan_id=plan.plan_id,
                        variant_type=(
                            ExperimentVariantType.BASELINE
                            if number == 1
                            else ExperimentVariantType.FULL_MECHANISM
                        ),
                        represented_proxy_ids=["P1"],
                        factor=factor,
                        prediction_horizon_days=horizon,
                    )
                )
        return experiments

    def preflight_experiments(
        self,
        experiments: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
        correction_round: int,
    ) -> list[PreflightResult]:
        del measurement_plans
        results = []
        for index, experiment in enumerate(experiments):
            force_failure = correction_round == 0 and index == 0
            issues = (
                [
                    CheckIssue(
                        code="mock_preflight_failure",
                        message="Synthetic first-pass failure used to test correction routing.",
                        field="factor",
                    )
                ]
                if force_failure
                else []
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
        del issues_by_id, accepted, measurement_plans
        repaired = []
        for experiment in rejected:
            new_experiment_id = f"{experiment.experiment_id}-R1"
            repaired_factor = experiment.factor.model_copy(
                update={"factor_id": f"{experiment.factor.factor_id}-R1"}
            )
            repaired.append(
                experiment.model_copy(
                    update={
                        "experiment_id": new_experiment_id,
                        "variation_reason": (
                            f"{experiment.variation_reason} Corrected after preflight."
                        ),
                        "factor": repaired_factor,
                    }
                )
            )
        return repaired

    def run_backtests(
        self,
        experiments: list[FactorExperimentSpec],
        split: SplitName,
        train_metrics: dict[str, SplitMetrics] | None = None,
    ) -> list[SplitMetrics]:
        metrics = []
        for index, experiment in enumerate(experiments):
            if split == SplitName.TRAIN:
                passes = index % 2 == 0
                direction = 1 if index % 4 else -1
                direction_source = "train"
            else:
                if train_metrics is None or experiment.experiment_id not in train_metrics:
                    raise ValueError("frozen training metric is required")
                direction = train_metrics[experiment.experiment_id].direction_multiplier
                direction_source = "frozen_train"
                passes = index % 2 == 0 if split == SplitName.VALIDATION else True
            mean_rank_ic = 0.03 if passes else 0.01
            net_return = 0.06 if passes else -0.01
            metrics.append(
                SplitMetrics(
                    split=split,
                    experiment_id=experiment.experiment_id,
                    prediction_horizon_days=experiment.prediction_horizon_days,
                    direction_multiplier=direction,
                    direction_source=direction_source,
                    valid_ic_dates={
                        SplitName.TRAIN: 2_000,
                        SplitName.VALIDATION: 400,
                        SplitName.TEST: 650,
                    }[split],
                    median_stock_coverage=0.95,
                    mean_rank_ic=mean_rank_ic,
                    rank_icir=0.35 if passes else 0.08,
                    newey_west_t_stat=2.1 if passes else 0.7,
                    newey_west_p_value=0.04 if passes else 0.48,
                    fdr_q_value=0.08 if passes else 0.55,
                    gross_return=net_return + 0.02,
                    net_return=net_return,
                    annualized_return=net_return / 2,
                    annualized_volatility=0.15,
                    sharpe=0.45 if passes else -0.05,
                    max_drawdown=-0.12 if passes else -0.25,
                    turnover=0.40,
                )
            )
        return metrics

    def build_memory_summary(
        self,
        train_metrics: list[SplitMetrics],
        validation_metrics: list[SplitMetrics],
        test_metrics: list[SplitMetrics],
    ) -> dict[str, Any]:
        return {
            "summary_version": "v2-mock-1",
            "train_experiments": len(train_metrics),
            "validation_experiments": len(validation_metrics),
            "test_experiments": len(test_metrics),
            "research_experience": [
                "Some mechanisms remain useful only at selected prediction horizons.",
                "Executable formula checks should remain separate from idea generation.",
            ],
        }

    @staticmethod
    def _concepts_for_direction(direction: ResearchDirection) -> tuple[str, ...]:
        return {
            ResearchDirection.MOMENTUM: ("recent_return",),
            ResearchDirection.REVERSAL: ("price_reversal",),
            ResearchDirection.LIQUIDITY_VOLATILITY: (
                "liquidity_proxy",
                "realized_volatility",
            ),
            ResearchDirection.PRICE_VOLUME: (
                "price_volume_confirmation",
                "volume_abnormality",
            ),
        }[direction]
