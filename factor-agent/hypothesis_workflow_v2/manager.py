"""Deterministic LangGraph manager for the V2 workflow."""

from __future__ import annotations

from typing import Any, TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .artifacts import ArtifactRepository
from .contracts import (
    ArtifactRef,
    DevelopedHypothesis,
    FactorExperimentSpec,
    HypothesisCheckResult,
    InsightAnalysis,
    MeasurementPlan,
    PreflightResult,
    ResearchIdea,
    SplitMetrics,
    SplitName,
)
from .state import V2Stage, V2WorkflowState
from .tools import V2Tools


ModelT = TypeVar("ModelT", bound=BaseModel)


def _event(
    state: V2WorkflowState,
    event_type: str,
    stage: V2Stage,
    **details: Any,
) -> list[dict[str, Any]]:
    return [
        *state["events"],
        {
            "sequence": len(state["events"]) + 1,
            "type": event_type,
            "stage": stage.value,
            **details,
        },
    ]


def _reference(state: V2WorkflowState, key: str) -> ArtifactRef:
    payload = state["artifact_refs"].get(key)
    if payload is None:
        raise ValueError(f"missing artifact reference: {key}")
    return ArtifactRef.model_validate(payload)


def _read_models(
    repository: ArtifactRepository,
    state: V2WorkflowState,
    key: str,
    model: type[ModelT],
) -> list[ModelT]:
    payload = repository.read_json(_reference(state, key))
    if not isinstance(payload, list):
        raise ValueError(f"artifact {key} must contain a list")
    return [model.model_validate(item) for item in payload]


def _save(
    repository: ArtifactRepository,
    state: V2WorkflowState,
    key: str,
    payload: Any,
) -> dict[str, dict[str, Any]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], BaseModel):
        serializable = [item.model_dump(mode="json") for item in payload]
    elif isinstance(payload, BaseModel):
        serializable = payload.model_dump(mode="json")
    else:
        serializable = payload
    reference = repository.write_json(state["run_id"], key, serializable)
    return {**state["artifact_refs"], key: reference.model_dump(mode="json")}


def build_manager(
    tools: V2Tools,
    repository: ArtifactRepository,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_after: list[str] | None = None,
):
    workflow = StateGraph(V2WorkflowState)

    def initialize(state: V2WorkflowState) -> dict[str, Any]:
        return {
            "stage": V2Stage.LOAD_CONTEXT,
            "events": _event(state, "run_started", V2Stage.INIT),
            "error": None,
        }

    def load_context(state: V2WorkflowState) -> dict[str, Any]:
        context = tools.load_context(state)
        return {
            "stage": V2Stage.GENERATE_RESEARCH_IDEAS,
            "artifact_refs": _save(repository, state, "research_context", context),
            "events": _event(state, "context_loaded", V2Stage.LOAD_CONTEXT),
        }

    def generate_research_ideas(state: V2WorkflowState) -> dict[str, Any]:
        context = repository.read_json(_reference(state, "research_context"))
        ideas = tools.generate_research_ideas(state, context)
        if not ideas:
            return _failure(state, "all research branches failed")
        branch_status = {key: "failed" for key in state["branch_status"]}
        for idea in ideas:
            branch_status[idea.direction.value] = "passed"
        return {
            "stage": V2Stage.ANALYZE_INSIGHTS,
            "branch_status": branch_status,
            "artifact_refs": _save(repository, state, "research_ideas", ideas),
            "events": _event(
                state,
                "research_ideas_generated",
                V2Stage.GENERATE_RESEARCH_IDEAS,
                count=len(ideas),
            ),
        }

    def analyze_insights(state: V2WorkflowState) -> dict[str, Any]:
        ideas = _read_models(repository, state, "research_ideas", ResearchIdea)
        analyses = tools.analyze_insights(state, ideas)
        if not analyses:
            return _failure(state, "all insight analyses failed")
        return {
            "stage": V2Stage.EXPAND_HYPOTHESES,
            "artifact_refs": _save(repository, state, "insight_analyses", analyses),
            "events": _event(
                state,
                "insights_analyzed",
                V2Stage.ANALYZE_INSIGHTS,
                count=len(analyses),
            ),
        }

    def expand_hypotheses(state: V2WorkflowState) -> dict[str, Any]:
        ideas = _read_models(repository, state, "research_ideas", ResearchIdea)
        analyses = _read_models(
            repository, state, "insight_analyses", InsightAnalysis
        )
        hypotheses = tools.expand_hypotheses(state, ideas, analyses)
        if not hypotheses:
            return _failure(state, "hypothesis expansion produced no candidates")
        return {
            "stage": V2Stage.CHECK_HYPOTHESES,
            "artifact_refs": _save(repository, state, "hypotheses", hypotheses),
            "events": _event(
                state,
                "hypotheses_expanded",
                V2Stage.EXPAND_HYPOTHESES,
                count=len(hypotheses),
            ),
        }

    def check_hypotheses(state: V2WorkflowState) -> dict[str, Any]:
        hypotheses = _read_models(
            repository, state, "hypotheses", DevelopedHypothesis
        )
        round_number = 1 if state["hypothesis_repair_used"] else 0
        checks = tools.check_hypotheses(hypotheses, round_number)
        accepted = [check for check in checks if check.accepted]
        rejected = [check for check in checks if not check.accepted]
        refs = _save(repository, state, "hypothesis_checks", checks)
        if rejected and not state["hypothesis_repair_used"]:
            next_stage = V2Stage.REPAIR_HYPOTHESES
        elif accepted:
            next_stage = V2Stage.DESIGN_MEASUREMENT_PLANS
        else:
            return _failure(state, "all hypotheses failed deterministic checks", refs)
        return {
            "stage": next_stage,
            "artifact_refs": refs,
            "events": _event(
                state,
                "hypotheses_checked",
                V2Stage.CHECK_HYPOTHESES,
                accepted=len(accepted),
                rejected=len(rejected),
                generation_round=round_number,
            ),
        }

    def repair_hypotheses(state: V2WorkflowState) -> dict[str, Any]:
        hypotheses = _read_models(
            repository, state, "hypotheses", DevelopedHypothesis
        )
        checks = _read_models(
            repository, state, "hypothesis_checks", HypothesisCheckResult
        )
        accepted_ids = {check.hypothesis_id for check in checks if check.accepted}
        accepted = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.hypothesis_id in accepted_ids
        ]
        rejected = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.hypothesis_id not in accepted_ids
        ]
        issues_by_id = {
            check.hypothesis_id: check.issues for check in checks if not check.accepted
        }
        repaired = tools.repair_hypotheses(rejected, issues_by_id)
        combined = [*accepted, *repaired]
        return {
            "stage": V2Stage.CHECK_HYPOTHESES,
            "hypothesis_repair_used": True,
            "artifact_refs": _save(repository, state, "hypotheses", combined),
            "events": _event(
                state,
                "hypotheses_repaired",
                V2Stage.REPAIR_HYPOTHESES,
                repaired=len(repaired),
            ),
        }

    def design_measurement_plans(state: V2WorkflowState) -> dict[str, Any]:
        hypotheses = _accepted_hypotheses(repository, state)
        plans = tools.design_measurement_plans(hypotheses)
        if not plans:
            return _failure(state, "measurement planning produced no candidates")
        return {
            "stage": V2Stage.GENERATE_EXPERIMENTS,
            "artifact_refs": _save(repository, state, "measurement_plans", plans),
            "events": _event(
                state,
                "measurement_plans_designed",
                V2Stage.DESIGN_MEASUREMENT_PLANS,
                count=len(plans),
            ),
        }

    def generate_experiments(state: V2WorkflowState) -> dict[str, Any]:
        hypotheses = _accepted_hypotheses(repository, state)
        plans = _read_models(
            repository, state, "measurement_plans", MeasurementPlan
        )
        experiments = tools.generate_experiments(hypotheses, plans)
        if not experiments:
            return _failure(state, "factor generation produced no experiments")
        return {
            "stage": V2Stage.PREFLIGHT_EXPERIMENTS,
            "artifact_refs": _save(repository, state, "experiment_specs", experiments),
            "events": _event(
                state,
                "experiments_generated",
                V2Stage.GENERATE_EXPERIMENTS,
                count=len(experiments),
            ),
        }

    def preflight_experiments(state: V2WorkflowState) -> dict[str, Any]:
        experiments = _read_models(
            repository, state, "experiment_specs", FactorExperimentSpec
        )
        plans = _read_models(
            repository, state, "measurement_plans", MeasurementPlan
        )
        round_number = 1 if state["factor_repair_used"] else 0
        checks = tools.preflight_experiments(experiments, plans, round_number)
        accepted = [check for check in checks if check.accepted]
        rejected = [check for check in checks if not check.accepted]
        refs = _save(repository, state, "preflight_results", checks)
        if rejected and not state["factor_repair_used"]:
            next_stage = V2Stage.REPAIR_EXPERIMENTS
        elif accepted:
            next_stage = V2Stage.RUN_TRAIN
        else:
            return _failure(state, "all experiments failed preflight", refs)
        return {
            "stage": next_stage,
            "artifact_refs": refs,
            "events": _event(
                state,
                "experiments_preflighted",
                V2Stage.PREFLIGHT_EXPERIMENTS,
                accepted=len(accepted),
                rejected=len(rejected),
                correction_round=round_number,
            ),
        }

    def repair_experiments(state: V2WorkflowState) -> dict[str, Any]:
        experiments = _read_models(
            repository, state, "experiment_specs", FactorExperimentSpec
        )
        checks = _read_models(
            repository, state, "preflight_results", PreflightResult
        )
        accepted_ids = {check.experiment_id for check in checks if check.accepted}
        accepted = [
            experiment
            for experiment in experiments
            if experiment.experiment_id in accepted_ids
        ]
        rejected = [
            experiment
            for experiment in experiments
            if experiment.experiment_id not in accepted_ids
        ]
        issues_by_id = {
            check.experiment_id: check.issues for check in checks if not check.accepted
        }
        plans = _read_models(
            repository, state, "measurement_plans", MeasurementPlan
        )
        repaired = tools.repair_experiments(
            rejected, issues_by_id, accepted, plans
        )
        return {
            "stage": V2Stage.PREFLIGHT_EXPERIMENTS,
            "factor_repair_used": True,
            "artifact_refs": _save(
                repository, state, "experiment_specs", [*accepted, *repaired]
            ),
            "events": _event(
                state,
                "experiments_repaired",
                V2Stage.REPAIR_EXPERIMENTS,
                repaired=len(repaired),
            ),
        }

    def run_train(state: V2WorkflowState) -> dict[str, Any]:
        experiments = _accepted_experiments(repository, state)
        metrics = tools.run_backtests(experiments, SplitName.TRAIN)
        return {
            "stage": V2Stage.FILTER_TRAIN,
            "artifact_refs": _save(repository, state, "train_metrics", metrics),
            "events": _event(
                state,
                "train_backtests_completed",
                V2Stage.RUN_TRAIN,
                count=len(metrics),
            ),
        }

    def filter_train(state: V2WorkflowState) -> dict[str, Any]:
        metrics = _read_models(repository, state, "train_metrics", SplitMetrics)
        candidate_ids = [
            metric.experiment_id
            for metric in metrics
            if metric.mean_rank_ic > 0.02 and metric.net_return > 0
        ]
        return {
            "stage": (
                V2Stage.RUN_VALIDATION if candidate_ids else V2Stage.BUILD_MEMORY
            ),
            "train_candidate_ids": candidate_ids,
            "events": _event(
                state,
                "train_candidates_filtered",
                V2Stage.FILTER_TRAIN,
                passed=len(candidate_ids),
            ),
        }

    def run_validation(state: V2WorkflowState) -> dict[str, Any]:
        experiments = _experiments_by_ids(
            repository, state, state["train_candidate_ids"]
        )
        train = _read_models(repository, state, "train_metrics", SplitMetrics)
        train_by_id = {metric.experiment_id: metric for metric in train}
        metrics = tools.run_backtests(
            experiments, SplitName.VALIDATION, train_by_id
        )
        return {
            "stage": V2Stage.FILTER_VALIDATION,
            "artifact_refs": _save(
                repository, state, "validation_metrics", metrics
            ),
            "events": _event(
                state,
                "validation_backtests_completed",
                V2Stage.RUN_VALIDATION,
                count=len(metrics),
            ),
        }

    def filter_validation(state: V2WorkflowState) -> dict[str, Any]:
        metrics = _read_models(
            repository, state, "validation_metrics", SplitMetrics
        )
        candidate_ids = [
            metric.experiment_id
            for metric in metrics
            if metric.mean_rank_ic > 0.02 and metric.net_return > 0
        ]
        return {
            "stage": V2Stage.RUN_TEST if candidate_ids else V2Stage.BUILD_MEMORY,
            "validation_candidate_ids": candidate_ids,
            "test_candidate_ids": candidate_ids,
            "events": _event(
                state,
                "validation_candidates_filtered",
                V2Stage.FILTER_VALIDATION,
                passed=len(candidate_ids),
            ),
        }

    def run_test(state: V2WorkflowState) -> dict[str, Any]:
        experiments = _experiments_by_ids(
            repository, state, state["test_candidate_ids"]
        )
        train = _read_models(repository, state, "train_metrics", SplitMetrics)
        train_by_id = {metric.experiment_id: metric for metric in train}
        metrics = tools.run_backtests(experiments, SplitName.TEST, train_by_id)
        return {
            "stage": V2Stage.BUILD_MEMORY,
            "artifact_refs": _save(repository, state, "test_metrics", metrics),
            "events": _event(
                state,
                "test_backtests_completed",
                V2Stage.RUN_TEST,
                count=len(metrics),
            ),
        }

    def build_memory(state: V2WorkflowState) -> dict[str, Any]:
        train = _optional_models(repository, state, "train_metrics", SplitMetrics)
        validation = _optional_models(
            repository, state, "validation_metrics", SplitMetrics
        )
        test = _optional_models(repository, state, "test_metrics", SplitMetrics)
        summary = tools.build_memory_summary(train, validation, test)
        return {
            "stage": V2Stage.RECORD_RUN,
            "artifact_refs": _save(repository, state, "memory_summary", summary),
            "events": _event(state, "memory_built", V2Stage.BUILD_MEMORY),
        }

    def record_run(state: V2WorkflowState) -> dict[str, Any]:
        summary = {
            "run_id": state["run_id"],
            "status": "completed",
            "branch_status": state["branch_status"],
            "hypothesis_repair_used": state["hypothesis_repair_used"],
            "factor_repair_used": state["factor_repair_used"],
            "train_candidates": state["train_candidate_ids"],
            "validation_candidates": state["validation_candidate_ids"],
            "test_candidates": state["test_candidate_ids"],
            "artifact_keys": sorted(state["artifact_refs"]),
        }
        return {
            "stage": V2Stage.COMPLETED,
            "artifact_refs": _save(repository, state, "run_summary", summary),
            "events": _event(state, "run_recorded", V2Stage.RECORD_RUN),
        }

    workflow.add_node("initialize", initialize)
    workflow.add_node("load_context", load_context)
    workflow.add_node("generate_research_ideas", generate_research_ideas)
    workflow.add_node("analyze_insights", analyze_insights)
    workflow.add_node("expand_hypotheses", expand_hypotheses)
    workflow.add_node("check_hypotheses", check_hypotheses)
    workflow.add_node("repair_hypotheses", repair_hypotheses)
    workflow.add_node("design_measurement_plans", design_measurement_plans)
    workflow.add_node("generate_experiments", generate_experiments)
    workflow.add_node("preflight_experiments", preflight_experiments)
    workflow.add_node("repair_experiments", repair_experiments)
    workflow.add_node("run_train", run_train)
    workflow.add_node("filter_train", filter_train)
    workflow.add_node("run_validation", run_validation)
    workflow.add_node("filter_validation", filter_validation)
    workflow.add_node("run_test", run_test)
    workflow.add_node("build_memory", build_memory)
    workflow.add_node("record_run", record_run)
    workflow.add_node("completed", lambda state: {})
    workflow.add_node("failed", lambda state: {})

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_context")
    workflow.add_edge("load_context", "generate_research_ideas")
    _route_failure(workflow, "generate_research_ideas", "analyze_insights")
    _route_failure(workflow, "analyze_insights", "expand_hypotheses")
    _route_failure(workflow, "expand_hypotheses", "check_hypotheses")
    workflow.add_conditional_edges(
        "check_hypotheses",
        lambda state: _stage_route(
            state,
            {
                V2Stage.REPAIR_HYPOTHESES: "repair_hypotheses",
                V2Stage.DESIGN_MEASUREMENT_PLANS: "design_measurement_plans",
            },
        ),
    )
    workflow.add_edge("repair_hypotheses", "check_hypotheses")
    _route_failure(workflow, "design_measurement_plans", "generate_experiments")
    _route_failure(workflow, "generate_experiments", "preflight_experiments")
    workflow.add_conditional_edges(
        "preflight_experiments",
        lambda state: _stage_route(
            state,
            {
                V2Stage.REPAIR_EXPERIMENTS: "repair_experiments",
                V2Stage.RUN_TRAIN: "run_train",
            },
        ),
    )
    workflow.add_edge("repair_experiments", "preflight_experiments")
    workflow.add_edge("run_train", "filter_train")
    workflow.add_conditional_edges(
        "filter_train",
        lambda state: {
            V2Stage.RUN_VALIDATION: "run_validation",
            V2Stage.BUILD_MEMORY: "build_memory",
        }[state["stage"]],
    )
    workflow.add_edge("run_validation", "filter_validation")
    workflow.add_conditional_edges(
        "filter_validation",
        lambda state: {
            V2Stage.RUN_TEST: "run_test",
            V2Stage.BUILD_MEMORY: "build_memory",
        }[state["stage"]],
    )
    workflow.add_edge("run_test", "build_memory")
    workflow.add_edge("build_memory", "record_run")
    workflow.add_edge("record_run", "completed")
    workflow.add_edge("completed", END)
    workflow.add_edge("failed", END)

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _failure(
    state: V2WorkflowState,
    message: str,
    artifact_refs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "stage": V2Stage.FAILED,
        "error": message,
        "artifact_refs": artifact_refs or state["artifact_refs"],
        "events": _event(
            state,
            "workflow_failed",
            state["stage"],
            error=message,
        ),
    }


def _stage_route(
    state: V2WorkflowState,
    routes: dict[V2Stage, str],
) -> str:
    if state["stage"] == V2Stage.FAILED:
        return "failed"
    return routes[state["stage"]]


def _route_failure(workflow: StateGraph, source: str, success: str) -> None:
    workflow.add_conditional_edges(
        source,
        lambda state: "failed" if state["stage"] == V2Stage.FAILED else success,
    )


def _accepted_hypotheses(
    repository: ArtifactRepository,
    state: V2WorkflowState,
) -> list[DevelopedHypothesis]:
    hypotheses = _read_models(repository, state, "hypotheses", DevelopedHypothesis)
    checks = _read_models(
        repository, state, "hypothesis_checks", HypothesisCheckResult
    )
    accepted_ids = {check.hypothesis_id for check in checks if check.accepted}
    return [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.hypothesis_id in accepted_ids
    ]


def _accepted_experiments(
    repository: ArtifactRepository,
    state: V2WorkflowState,
) -> list[FactorExperimentSpec]:
    experiments = _read_models(
        repository, state, "experiment_specs", FactorExperimentSpec
    )
    checks = _read_models(repository, state, "preflight_results", PreflightResult)
    accepted_ids = {check.experiment_id for check in checks if check.accepted}
    return [
        experiment
        for experiment in experiments
        if experiment.experiment_id in accepted_ids
    ]


def _experiments_by_ids(
    repository: ArtifactRepository,
    state: V2WorkflowState,
    experiment_ids: list[str],
) -> list[FactorExperimentSpec]:
    requested = set(experiment_ids)
    return [
        experiment
        for experiment in _accepted_experiments(repository, state)
        if experiment.experiment_id in requested
    ]


def _optional_models(
    repository: ArtifactRepository,
    state: V2WorkflowState,
    key: str,
    model: type[ModelT],
) -> list[ModelT]:
    if key not in state["artifact_refs"]:
        return []
    return _read_models(repository, state, key, model)
