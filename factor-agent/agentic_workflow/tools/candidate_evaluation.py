"""Implement, backtest, and evaluate every screened research candidate."""

from __future__ import annotations

from typing import Any, Sequence

from agentic_workflow.state import WorkflowState
from agentic_workflow.tools.base import ToolResult, WorkflowTool
from agentic_workflow.tools.research_team import ResearchCandidateSelector


class CandidateBatchEvaluationTool:
    name = "candidate_batch_evaluation"

    def __init__(
        self,
        implementation: WorkflowTool,
        backtest: WorkflowTool,
        evaluation: WorkflowTool,
        selector: ResearchCandidateSelector,
    ) -> None:
        self.implementation = implementation
        self.backtest = backtest
        self.evaluation = evaluation
        self.selector = selector

    def run(self, state: WorkflowState) -> ToolResult:
        candidates = state.get("hypothesis_candidates") or []
        evaluated_candidates: list[dict[str, Any]] = []
        successful_candidates: list[dict[str, Any]] = []

        for candidate in candidates:
            if not candidate.get("accepted"):
                evaluated_candidates.append(candidate)
                continue
            evaluated = self._evaluate_candidate(state, candidate)
            evaluated_candidates.append(evaluated)
            if evaluated.get("pipeline_status") == "completed":
                successful_candidates.append(evaluated)

        if not successful_candidates:
            return ToolResult.fail(
                "No screened candidate completed implementation, backtest, and evaluation"
            )

        try:
            selection_result = self.selector.select(state, successful_candidates)
        except Exception as error:
            return ToolResult.fail(f"Manager candidate selection failed: {error}")

        selection = selection_result.to_state_payload()
        selected_id = selection["selected_hypothesis_id"]
        selected = next(
            candidate
            for candidate in successful_candidates
            if candidate["hypothesis"]["hypothesis_id"] == selected_id
        )
        return ToolResult.ok(
            {
                "hypothesis_candidates": evaluated_candidates,
                "hypothesis_selection": selection,
                "hypothesis": selected["hypothesis"],
                "financial_intuition": selected.get("financial_intuition"),
                "hypothesis_check": selected.get("hypothesis_check"),
                "attempt": selected["attempt"],
                "factor_spec": selected["factor_spec"],
                "implementation": selected["implementation"],
                "backtest_result": selected["backtest_result"],
                "evaluation": selected["evaluation"],
                "next_action": selected["next_action"],
            }
        )

    def _evaluate_candidate(
        self,
        state: WorkflowState,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_state = {
            **state,
            "hypothesis": candidate["hypothesis"],
            "financial_intuition": candidate.get("financial_intuition"),
            "hypothesis_check": candidate.get("hypothesis_check"),
            "attempt": 1,
            "factor_spec": None,
            "implementation": None,
            "backtest_result": None,
            "evaluation": None,
            "previous_attempt": None,
            "next_action": None,
            "error": None,
        }
        stages: Sequence[tuple[str, WorkflowTool]] = (
            ("implementation", self.implementation),
            ("backtest", self.backtest),
            ("evaluation", self.evaluation),
        )
        for stage_name, tool in stages:
            result = tool.run(candidate_state)
            if not result.success:
                return {
                    **candidate,
                    "pipeline_status": "failed",
                    "failed_stage": stage_name,
                    "error": result.error,
                }
            candidate_state.update(result.payload)

        return {
            **candidate,
            "pipeline_status": "completed",
            "attempt": candidate_state["attempt"],
            "factor_spec": candidate_state["factor_spec"],
            "implementation": candidate_state["implementation"],
            "backtest_result": candidate_state["backtest_result"],
            "evaluation": candidate_state["evaluation"],
            "next_action": candidate_state["next_action"],
        }
