"""Two-stage specialist pipeline: financial intuition, then hypothesis."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_workflow.factor_library import FactorKnowledgeBase
from agentic_workflow.state import WorkflowState
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.base import ToolResult
from agentic_workflow.tools.research_agent import (
    AVAILABLE_DATA_FIELDS,
    ResearchStrategy,
    SupportedConcept,
    build_research_context,
)
from agentic_workflow.tools.web_research import (
    KeyFreeWebResearchProvider,
    WebResearchProvider,
    build_strategy_search_queries,
)


class IntuitionFactorComponent(BaseModel):
    concept: SupportedConcept
    role: Literal["core_signal", "condition", "confirmation", "risk_control"]
    mechanism: str = Field(min_length=1, max_length=500)


class IntuitionSourceReference(BaseModel):
    source_id: str = Field(pattern=r"^WEB-\d{2}$")
    relevance: str = Field(min_length=1, max_length=500)


class IntuitionFactorReference(BaseModel):
    factor_id: str = Field(min_length=1, max_length=100)
    adaptation: str = Field(min_length=1, max_length=500)


class FinancialIntuitionProposal(BaseModel):
    intuition_title: str = Field(min_length=1, max_length=120)
    core_mechanism: str = Field(min_length=1, max_length=800)
    causal_chain: list[str] = Field(min_length=2, max_length=6)
    market_condition: str = Field(min_length=1, max_length=500)
    expected_pattern: str = Field(min_length=1, max_length=500)
    factor_components: list[IntuitionFactorComponent] = Field(
        min_length=2,
        max_length=4,
    )
    falsification_risks: list[str] = Field(min_length=1, max_length=4)
    source_references: list[IntuitionSourceReference] = Field(
        min_length=1,
        max_length=3,
    )
    factor_references: list[IntuitionFactorReference] = Field(
        min_length=1,
        max_length=3,
    )


class HypothesisFactorComponent(BaseModel):
    concept: SupportedConcept
    role: Literal["core_signal", "condition", "confirmation", "risk_control"]
    expected_relation: Literal["high", "low", "increasing", "decreasing"]


class FormulatedHypothesis(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1, max_length=500)
    economic_rationale: str = Field(min_length=1, max_length=800)
    prediction_direction: Literal["positive", "negative"]
    prediction_horizon_days: Literal[5]
    required_concepts: list[SupportedConcept] = Field(min_length=2, max_length=4)
    factor_structure: list[HypothesisFactorComponent] = Field(
        min_length=2,
        max_length=4,
    )


def _openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for two-stage research")
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://hkust.azure-api.net/hkust-genai/v1",
        ),
        default_headers={"api-key": api_key},
    )


class FinancialIntuitionAgentTool:
    def __init__(
        self,
        repository: SQLiteResearchRepository,
        strategy: ResearchStrategy,
        model: str | None = None,
        history_limit: int = 10,
        max_attempts: int = 2,
        *,
        client: Any | None = None,
        web_research: WebResearchProvider | None = None,
        factor_library: FactorKnowledgeBase | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.name = f"financial_intuition_agent_{strategy.key}"
        self.repository = repository
        self.strategy = strategy
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.history_limit = history_limit
        self.max_attempts = max_attempts
        self.client = client or _openai_client()
        self.web_research = web_research or KeyFreeWebResearchProvider()
        self.factor_library = factor_library or FactorKnowledgeBase()

    def run(self, state: WorkflowState) -> ToolResult:
        try:
            web_sources = self.web_research.search(
                build_strategy_search_queries(self.strategy.key)
            )
            concepts = [concept.value for concept in self.strategy.allowed_concepts]
            factor_examples = self.factor_library.search(
                concepts,
                iteration=state["iteration"],
            )
            retrieval_state = {
                **state,
                "hypothesis": {"required_concepts": concepts},
            }
            relevant_experiments = self.repository.relevant_experiments(
                state=retrieval_state,
                limit=self.history_limit,
            )
            research_context = build_research_context(
                state=state,
                relevant_experiments=relevant_experiments,
                strategy=self.strategy,
                web_sources=web_sources,
                factor_library_examples=factor_examples,
            )
            request = {
                "task": "形成一个可进一步转化为多因子假设的金融直觉。",
                "research_context": research_context,
                "output_schema": FinancialIntuitionProposal.model_json_schema(),
            }
            allowed = {concept.value for concept in self.strategy.allowed_concepts}
            source_map = {source.source_id: source for source in web_sources}
            factor_map = {
                str(factor["factor_id"]): factor for factor in factor_examples
            }
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False),
                },
            ]
            for attempt in range(1, self.max_attempts + 1):
                content = ""
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    content = (
                        response.choices[0].message.content
                        if response.choices
                        else ""
                    )
                    proposal = FinancialIntuitionProposal.model_validate_json(
                        content or ""
                    )
                    selected = {
                        component.concept.value
                        for component in proposal.factor_components
                    }
                    if selected - allowed:
                        raise ValueError(
                            "intuition selected concepts outside strategy scope: "
                            + ", ".join(sorted(selected - allowed))
                            + f"; allowed concepts are {sorted(allowed)}"
                        )
                    unknown_sources = {
                        reference.source_id
                        for reference in proposal.source_references
                        if reference.source_id not in source_map
                    }
                    unknown_factors = {
                        reference.factor_id
                        for reference in proposal.factor_references
                        if reference.factor_id not in factor_map
                    }
                    if unknown_sources or unknown_factors:
                        raise ValueError(
                            "intuition cited unknown references: "
                            + ", ".join(
                                sorted(unknown_sources | unknown_factors)
                            )
                        )
                    break
                except Exception as error:
                    if attempt >= self.max_attempts:
                        raise ValueError(
                            "generation failed after "
                            f"{attempt} attempts: {error}"
                        ) from error
                    if content:
                        messages.append(
                            {"role": "assistant", "content": content}
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "上一份金融直觉未通过确定性校验。"
                                f"校验错误：{error}。"
                                f"factor_components 只能使用：{sorted(allowed)}。"
                                "引用 ID 必须来自输入上下文。请返回完整 JSON。"
                            ),
                        }
                    )
            intuition = proposal.model_dump(
                mode="json",
                exclude={"source_references", "factor_references"},
            )
            intuition["research_strategy"] = self.strategy.key
            intuition["sources"] = [
                {
                    "title": source_map[reference.source_id].title,
                    "url": str(source_map[reference.source_id].url),
                    "relevance": reference.relevance,
                }
                for reference in proposal.source_references
            ]
            intuition["factor_references"] = [
                {
                    "factor_id": reference.factor_id,
                    "name": factor_map[reference.factor_id]["name"],
                    "family": factor_map[reference.factor_id]["family"],
                    "implementation_status": factor_map[reference.factor_id][
                        "implementation_status"
                    ],
                    "adaptation": reference.adaptation,
                }
                for reference in proposal.factor_references
            ]
            return ToolResult.ok(
                {
                    "financial_intuition": intuition,
                    "research_context": research_context,
                    "intuition_generation_attempts": attempt,
                }
            )
        except Exception as error:
            return ToolResult.fail(f"Financial Intuition Agent failed: {error}")

    def _system_prompt(self) -> str:
        concepts = ", ".join(
            concept.value for concept in self.strategy.allowed_concepts
        )
        return (
            f"你是 {self.strategy.name} 中的 Financial Intuition Agent。"
            f"{self.strategy.focus}你的任务不是直接写最终假设或公式，而是形成金融机制。"
            "从行为偏差、风险补偿、流动性冲击、信息扩散或交易拥挤角度建立因果链。"
            "必须组合二至四个因子组件，并明确 core_signal、condition、confirmation "
            "或 risk_control 的角色；复杂性来自有逻辑的条件结构，而不是信号堆叠。"
            "参考 factor_library_examples、web_sources 和历史失败记录。不可实现因子只能"
            "提供机制灵感，最终组件必须属于当前角色概念范围。不得直接输出 FactorSpec。"
            "source_references 与 factor_references 只能引用上下文中的真实 ID。"
            f"允许概念：{concepts}。仅返回符合 output_schema 的 JSON。"
        )


class HypothesisFormulationAgentTool:
    def __init__(
        self,
        strategy: ResearchStrategy,
        model: str | None = None,
        max_attempts: int = 2,
        *,
        client: Any | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.name = f"hypothesis_formulation_agent_{strategy.key}"
        self.strategy = strategy
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.max_attempts = max_attempts
        self.client = client or _openai_client()

    def run(self, state: dict[str, Any]) -> ToolResult:
        intuition = state.get("financial_intuition")
        research_context = state.get("research_context")
        if not isinstance(intuition, dict) or not isinstance(research_context, dict):
            return ToolResult.fail("Financial intuition and research context are required")
        request = {
            "task": "将金融直觉转换为一个可由当前数据证伪的横截面假设。",
            "financial_intuition": intuition,
            "available_data_fields": list(AVAILABLE_DATA_FIELDS),
            "historical_constraints": research_context.get("relevant_experiments", []),
            "output_schema": FormulatedHypothesis.model_json_schema(),
        }
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(request, ensure_ascii=False),
            },
        ]
        for attempt in range(1, self.max_attempts + 1):
            content = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                content = (
                    response.choices[0].message.content if response.choices else ""
                )
                proposal = FormulatedHypothesis.model_validate_json(content or "")
                required = {
                    concept.value for concept in proposal.required_concepts
                }
                structured = {
                    component.concept.value
                    for component in proposal.factor_structure
                }
                intuition_concepts = {
                    component["concept"]
                    for component in intuition["factor_components"]
                }
                if required != structured:
                    raise ValueError(
                        "required_concepts must exactly match factor_structure concepts"
                    )
                if not required <= intuition_concepts:
                    raise ValueError(
                        "hypothesis introduced concepts absent from financial intuition; "
                        f"allowed concepts are {sorted(intuition_concepts)}"
                    )
                break
            except Exception as error:
                if attempt >= self.max_attempts:
                    return ToolResult.fail(
                        "Hypothesis Formulation Agent failed after "
                        f"{attempt} attempts: {error}"
                    )
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一份假设未通过确定性校验。"
                            f"校验错误：{error}。"
                            "请只使用 financial_intuition.factor_components 中已有概念，"
                            "并返回完整 JSON。"
                        ),
                    }
                )
        hypothesis = proposal.model_dump(mode="json")
        hypothesis.update(
            {
                "hypothesis_id": self._build_hypothesis_id(state),
                "research_strategy": self.strategy.key,
                "sources": intuition["sources"],
                "factor_references": intuition["factor_references"],
            }
        )
        return ToolResult.ok(
            {
                "hypothesis": hypothesis,
                "formulation_attempts": attempt,
            }
        )

    def _system_prompt(self) -> str:
        return (
            "你是 Hypothesis Formulation Agent。你只负责把给定 financial_intuition 转换成"
            "明确、可证伪、可由 open/high/low/close/volume/mktcap 实现的五日横截面假设。"
            "保留直觉中的核心因果机制，将多个组件组织成主信号、条件、确认或风险控制。"
            "claim 必须说明各组件的高低或变化关系以及未来收益方向。"
            "required_concepts 必须与 factor_structure 中概念完全一致，不得引入新概念。"
            "不要生成公式、阈值搜索、行业字段、财务字段或动态调仓规则。"
            "仅返回符合 output_schema 的 JSON。"
        )

    def _build_hypothesis_id(self, state: dict[str, Any]) -> str:
        run_prefix = str(state["run_id"])[:8].upper()
        return (
            f"H-{run_prefix}-{int(state['iteration']):04d}-"
            f"{self.strategy.key.upper()}"
        )


class TwoStageResearchSpecialist:
    def __init__(
        self,
        intuition_agent: FinancialIntuitionAgentTool,
        formulation_agent: HypothesisFormulationAgentTool,
    ) -> None:
        self.intuition_agent = intuition_agent
        self.formulation_agent = formulation_agent
        self.name = f"two_stage_research_{intuition_agent.strategy.key}"

    def run(self, state: WorkflowState) -> ToolResult:
        intuition_result = self.intuition_agent.run(state)
        if not intuition_result.success:
            return intuition_result
        formulation_state = {**state, **intuition_result.payload}
        hypothesis_result = self.formulation_agent.run(formulation_state)
        if not hypothesis_result.success:
            return hypothesis_result
        return ToolResult.ok(
            {
                **hypothesis_result.payload,
                "financial_intuition": intuition_result.payload[
                    "financial_intuition"
                ],
            }
        )
