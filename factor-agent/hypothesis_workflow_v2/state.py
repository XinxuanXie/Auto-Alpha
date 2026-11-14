"""Small checkpoint state for the hypothesis-first workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

from .contracts import ResearchDirection


class V2Stage(StrEnum):
    INIT = "init"
    LOAD_CONTEXT = "load_context"
    GENERATE_RESEARCH_IDEAS = "generate_research_ideas"
    ANALYZE_INSIGHTS = "analyze_insights"
    EXPAND_HYPOTHESES = "expand_hypotheses"
    CHECK_HYPOTHESES = "check_hypotheses"
    REPAIR_HYPOTHESES = "repair_hypotheses"
    DESIGN_MEASUREMENT_PLANS = "design_measurement_plans"
    GENERATE_EXPERIMENTS = "generate_experiments"
    PREFLIGHT_EXPERIMENTS = "preflight_experiments"
    REPAIR_EXPERIMENTS = "repair_experiments"
    RUN_TRAIN = "run_train"
    FILTER_TRAIN = "filter_train"
    RUN_VALIDATION = "run_validation"
    FILTER_VALIDATION = "filter_validation"
    RUN_TEST = "run_test"
    BUILD_MEMORY = "build_memory"
    RECORD_RUN = "record_run"
    COMPLETED = "completed"
    FAILED = "failed"


class V2WorkflowState(TypedDict):
    run_id: str
    stage: V2Stage
    config_snapshot: dict[str, Any]
    branch_status: dict[str, str]
    artifact_refs: dict[str, dict[str, Any]]
    hypothesis_repair_used: bool
    factor_repair_used: bool
    train_candidate_ids: list[str]
    validation_candidate_ids: list[str]
    test_candidate_ids: list[str]
    retry_counts: dict[str, int]
    events: list[dict[str, Any]]
    error: str | None


def create_initial_state(
    config_snapshot: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
) -> V2WorkflowState:
    resolved_run_id = (run_id or str(uuid4())).strip()
    if not resolved_run_id:
        raise ValueError("run_id cannot be empty")
    directions = [direction.value for direction in ResearchDirection]
    return V2WorkflowState(
        run_id=resolved_run_id,
        stage=V2Stage.INIT,
        config_snapshot=dict(config_snapshot or {}),
        branch_status={direction: "pending" for direction in directions},
        artifact_refs={},
        hypothesis_repair_used=False,
        factor_repair_used=False,
        train_candidate_ids=[],
        validation_candidate_ids=[],
        test_candidate_ids=[],
        retry_counts={},
        events=[],
        error=None,
    )
