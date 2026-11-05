"""Shared state contract for the research workflow."""

from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4


class WorkflowStage(StrEnum):
    INIT = "init"
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    EVALUATE_CANDIDATES = "evaluate_candidates"
    CHECK_HYPOTHESIS = "check_hypothesis"
    IMPLEMENT_FACTOR = "implement_factor"
    VALIDATE_IMPLEMENTATION = "validate_implementation"
    RUN_BACKTEST = "run_backtest"
    EVALUATE_RESULT = "evaluate_result"
    RECORD_EXPERIMENT = "record_experiment"
    DECIDE_NEXT_ACTION = "decide_next_action"
    APPLY_MANAGER_ACTION = "apply_manager_action"
    COMPLETED = "completed"
    FAILED = "failed"


class ManagerAction(StrEnum):
    NEW_HYPOTHESIS = "new_hypothesis"
    REVISE_HYPOTHESIS = "revise_hypothesis"
    RETRY_IMPLEMENTATION = "retry_implementation"
    ADJUST_FACTOR = "adjust_factor"
    ACCEPT_FACTOR = "accept_factor"
    REJECT_FACTOR = "reject_factor"
    STOP_RESEARCH = "stop_research"


class WorkflowState(TypedDict):
    run_id: str
    research_goal: str
    stage: WorkflowStage
    iteration: int
    attempt: int
    max_iterations: int
    max_attempts_per_hypothesis: int
    financial_intuition: dict[str, Any] | None
    hypothesis: dict[str, Any] | None
    hypothesis_candidates: list[dict[str, Any]] | None
    hypothesis_selection: dict[str, Any] | None
    hypothesis_check: dict[str, Any] | None
    factor_spec: dict[str, Any] | None
    implementation: dict[str, Any] | None
    backtest_result: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    previous_attempt: dict[str, Any] | None
    record: dict[str, Any] | None
    manager_decision: dict[str, Any] | None
    next_action: ManagerAction | None
    retry_counts: dict[str, int]
    events: list[dict[str, Any]]
    error: str | None


def create_initial_state(
    max_iterations: int = 3,
    max_attempts_per_hypothesis: int = 3,
    research_goal: str = "发现具有稳定横截面预测能力的美股因子",
) -> WorkflowState:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if max_attempts_per_hypothesis < 1:
        raise ValueError("max_attempts_per_hypothesis must be at least 1")
    if not research_goal.strip():
        raise ValueError("research_goal cannot be empty")

    return WorkflowState(
        run_id=str(uuid4()),
        research_goal=research_goal.strip(),
        stage=WorkflowStage.INIT,
        iteration=0,
        attempt=0,
        max_iterations=max_iterations,
        max_attempts_per_hypothesis=max_attempts_per_hypothesis,
        financial_intuition=None,
        hypothesis=None,
        hypothesis_candidates=None,
        hypothesis_selection=None,
        hypothesis_check=None,
        factor_spec=None,
        implementation=None,
        backtest_result=None,
        evaluation=None,
        previous_attempt=None,
        record=None,
        manager_decision=None,
        next_action=None,
        retry_counts={},
        events=[],
        error=None,
    )
