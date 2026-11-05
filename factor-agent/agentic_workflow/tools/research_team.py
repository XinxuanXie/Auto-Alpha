"""Multi-strategy research team with deterministic screening and Manager selection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

from agentic_workflow.state import WorkflowState
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.base import ToolResult, WorkflowTool
from agentic_workflow.tools.research_agent import (
    RESEARCH_STRATEGIES,
)
from agentic_workflow.tools.two_stage_research import (
    FinancialIntuitionAgentTool,
    HypothesisFormulationAgentTool,
    TwoStageResearchSpecialist,
)


class CandidateSelection(BaseModel):
    selected_hypothesis_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1_500)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class CandidateSelectionResult:
    selection: CandidateSelection
    policy: str
    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None

    def to_state_payload(self) -> dict[str, Any]:
        payload = self.selection.model_dump(mode="json")
        payload.update(
            {
                "policy": self.policy,
                "model": self.model,
                "response_id": self.response_id,
                "usage": self.usage,
            }
        )
        return payload


class ResearchCandidateSelector(Protocol):
    name: str

    def select(
        self,
        state: WorkflowState,
        candidates: Sequence[dict[str, Any]],
    ) -> CandidateSelectionResult:
        """Choose one deterministically screened research hypothesis."""


class RuleBasedResearchCandidateSelector:
    name = "rule_based_manager_candidate_selector"

    @staticmethod
    def _metric(candidate: dict[str, Any], name: str) -> float:
        value = candidate.get("backtest_result", {}).get(name)
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    def select(
        self,
        state: WorkflowState,
        candidates: Sequence[dict[str, Any]],
    ) -> CandidateSelectionResult:
        if not candidates:
            raise ValueError("At least one accepted candidate is required")
        selected_candidate = max(
            candidates,
            key=lambda candidate: (
                candidate.get("evaluation", {}).get("status") == "PASS",
                self._metric(candidate, "rank_ic"),
                self._metric(candidate, "net_return"),
            ),
        )
        selected = selected_candidate["hypothesis"]
        return CandidateSelectionResult(
            selection=CandidateSelection(
                selected_hypothesis_id=selected["hypothesis_id"],
                rationale="选择验收状态、Rank IC 与净收益综合表现最优的候选因子。",
                confidence=1.0,
            ),
            policy=self.name,
        )


class OpenAIManagerCandidateSelector:
    name = "openai_manager_candidate_selector"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for candidate selection"
                )
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "OPENAI_BASE_URL",
                    "https://hkust.azure-api.net/hkust-genai/v1",
                ),
                default_headers={"api-key": api_key},
            )
        self.client = client

    def select(
        self,
        state: WorkflowState,
        candidates: Sequence[dict[str, Any]],
    ) -> CandidateSelectionResult:
        if not candidates:
            raise ValueError("At least one accepted candidate is required")
        request = {
            "research_goal": state["research_goal"],
            "iteration": state["iteration"],
            "max_iterations": state["max_iterations"],
            "previous_manager_decision": state["manager_decision"],
            "accepted_candidates": list(candidates),
            "output_schema": CandidateSelection.model_json_schema(),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是量化研究系统的 Manager Agent。"
                        "四个 Specialist Research Agents 的候选已经完成因子实现、"
                        "真实回测和评估。请从中选择一个作为本轮主因子。"
                        "优先考虑验收状态、Rank IC、净收益、换手率、回撤、"
                        "经济逻辑和研究方向多样性。"
                        "selected_hypothesis_id 必须来自 accepted_candidates。"
                        "仅返回符合 output_schema 的 JSON 对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, default=str),
                },
            ],
            response_format={"type": "json_object"},
        )
        if not response.choices:
            raise ValueError("Manager candidate selector returned no choices")
        content = response.choices[0].message.content or ""
        if not content:
            raise ValueError("Manager candidate selector returned empty content")
        selection = CandidateSelection.model_validate_json(content)
        candidate_ids = {
            candidate["hypothesis"]["hypothesis_id"]
            for candidate in candidates
        }
        if selection.selected_hypothesis_id not in candidate_ids:
            raise ValueError(
                "Manager candidate selector chose an unknown hypothesis: "
                f"{selection.selected_hypothesis_id}"
            )
        usage = None
        if getattr(response, "usage", None) is not None:
            usage_object = response.usage
            usage = (
                usage_object.model_dump(mode="json")
                if hasattr(usage_object, "model_dump")
                else asdict(usage_object)
            )
        return CandidateSelectionResult(
            selection=selection,
            policy=self.name,
            model=self.model,
            response_id=getattr(response, "id", None),
            usage=usage,
        )


class MultiStrategyResearchTool:
    name = "multi_strategy_research_team"

    def __init__(
        self,
        specialists: Sequence[WorkflowTool],
        hypothesis_checker: WorkflowTool,
    ) -> None:
        if not specialists:
            raise ValueError("At least one research specialist is required")
        self.specialists = tuple(specialists)
        self.hypothesis_checker = hypothesis_checker

    def run(self, state: WorkflowState) -> ToolResult:
        candidate_records = []
        accepted_candidates = []
        for specialist in self.specialists:
            proposal_result = specialist.run(state)
            if not proposal_result.success:
                candidate_records.append(
                    {
                        "specialist": specialist.name,
                        "accepted": False,
                        "error": proposal_result.error,
                    }
                )
                continue
            hypothesis = proposal_result.payload.get("hypothesis")
            financial_intuition = proposal_result.payload.get(
                "financial_intuition"
            )
            candidate_state = {**state, "hypothesis": hypothesis}
            check_result = self.hypothesis_checker.run(candidate_state)
            if not check_result.success:
                candidate_records.append(
                    {
                        "specialist": specialist.name,
                        "hypothesis": hypothesis,
                        "accepted": False,
                        "error": check_result.error,
                    }
                )
                continue
            hypothesis_check = check_result.payload["hypothesis_check"]
            candidate = {
                "specialist": specialist.name,
                "hypothesis": hypothesis,
                "financial_intuition": financial_intuition,
                "hypothesis_check": hypothesis_check,
                "accepted": hypothesis_check["accepted"],
            }
            candidate_records.append(candidate)
            if hypothesis_check["accepted"]:
                accepted_candidates.append(candidate)

        if not accepted_candidates:
            return ToolResult.fail(
                "No specialist hypothesis passed deterministic screening"
            )
        return ToolResult.ok(
            {
                "hypothesis_candidates": candidate_records,
            }
        )


def build_default_research_team(
    repository: SQLiteResearchRepository,
    hypothesis_checker: WorkflowTool,
    model: str | None = None,
) -> MultiStrategyResearchTool:
    specialists = [
        TwoStageResearchSpecialist(
            intuition_agent=FinancialIntuitionAgentTool(
                repository=repository,
                model=model,
                strategy=strategy,
            ),
            formulation_agent=HypothesisFormulationAgentTool(
                model=model,
                strategy=strategy,
            ),
        )
        for strategy in RESEARCH_STRATEGIES
    ]
    return MultiStrategyResearchTool(
        specialists=specialists,
        hypothesis_checker=hypothesis_checker,
    )
