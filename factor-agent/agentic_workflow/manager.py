"""Deterministic LangGraph manager for the factor research workflow."""

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agentic_workflow.manager_policy import (
    ManagerPolicy,
    RuleBasedManagerPolicy,
    allowed_manager_actions,
)
from agentic_workflow.state import ManagerAction, WorkflowStage, WorkflowState
from agentic_workflow.tools import (
    MockBacktestTool,
    MockEvaluationTool,
    MockHypothesisCheckerTool,
    MockHypothesisTool,
    MockImplementationTool,
    MockRecordTool,
    WorkflowTool,
)


@dataclass(frozen=True)
class ManagerTools:
    hypothesis: WorkflowTool
    hypothesis_checker: WorkflowTool
    implementation: WorkflowTool
    backtest: WorkflowTool
    evaluation: WorkflowTool
    record: WorkflowTool
    candidate_evaluation: WorkflowTool | None = None

    @classmethod
    def mock( #创建整套mock工具，方便测试工作流
        cls,
        hypothesis: WorkflowTool | None = None,
        hypothesis_checker: WorkflowTool | None = None,
        implementation: WorkflowTool | None = None,
        backtest: WorkflowTool | None = None,
        evaluation: WorkflowTool | None = None,
        record: WorkflowTool | None = None,
        candidate_evaluation: WorkflowTool | None = None,
    ) -> "ManagerTools":
        return cls(
            hypothesis=hypothesis or MockHypothesisTool(),
            hypothesis_checker=(
                hypothesis_checker or MockHypothesisCheckerTool()
            ),
            implementation=implementation or MockImplementationTool(),
            backtest=backtest or MockBacktestTool(),
            evaluation=evaluation or MockEvaluationTool(),
            record=record or MockRecordTool(),
            candidate_evaluation=candidate_evaluation,
        )


def _event( #给工作流添加执行日志
    state: WorkflowState,
    event_type: str,
    stage: WorkflowStage,
    **details: Any,
) -> list[dict[str, Any]]:
    entry = {
        "sequence": len(state["events"]) + 1,
        "type": event_type,
        "stage": stage.value,
        **details,
    }
    return [*state["events"], entry]

#对于每一个节点共用的执行模板：
# 调用tool，检查success，更新state，记录事件，处理error
def _tool_node( 
    state: WorkflowState,
    tool: WorkflowTool,
    success_stage: WorkflowStage,
) -> dict[str, Any]:
    source_stage = state["stage"]
    result = tool.run(state)
    if not result.success:
        return {
            "stage": WorkflowStage.FAILED,
            "error": result.error,
            "events": _event(
                state,
                "tool_failed",
                source_stage,
                tool=tool.name,
                error=result.error,
            ),
        }

    return {
        **result.payload,
        "stage": success_stage,
        "error": None,
        "events": _event(
            state,
            "tool_succeeded",
            source_stage,
            tool=tool.name,
            next_stage=success_stage.value,
        ),
    }

# 创建并编译完整 LangGraph 工作流
def build_manager(
    tools: ManagerTools,
    *,
    policy: ManagerPolicy | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_after: list[str] | None = None,
):
    manager_policy = policy or RuleBasedManagerPolicy()
    workflow = StateGraph(WorkflowState)

    def initialize(state: WorkflowState) -> dict[str, Any]:
        """
        检查迭代次数
        → 决定是否还能开始下一轮
        → 更新 iteration
        → 进入提出假设阶段
        """
        if state["iteration"] >= state["max_iterations"]:
            return {
                "stage": WorkflowStage.COMPLETED,
                "next_action": ManagerAction.STOP_RESEARCH,
                "events": _event(
                    state,
                    "iteration_limit_reached",
                    WorkflowStage.INIT,
                ),
            }

        next_iteration = state["iteration"] + 1
        return { #返回state更新
            "stage": WorkflowStage.PROPOSE_HYPOTHESIS,
            "iteration": next_iteration,
            "attempt": 0,
            "error": None,
            "events": _event(
                state,
                "iteration_started",
                WorkflowStage.INIT,
                iteration=next_iteration,
            ),
        }

    def propose_hypothesis(state: WorkflowState) -> dict[str, Any]:
        return _tool_node(
            state,
            tools.hypothesis,
            (
                WorkflowStage.EVALUATE_CANDIDATES
                if tools.candidate_evaluation is not None
                else WorkflowStage.CHECK_HYPOTHESIS
            ),
        )

    def evaluate_candidates(state: WorkflowState) -> dict[str, Any]:
        if tools.candidate_evaluation is None:
            return {
                "stage": WorkflowStage.FAILED,
                "error": "Candidate evaluation tool is not configured",
                "events": _event(
                    state,
                    "tool_failed",
                    WorkflowStage.EVALUATE_CANDIDATES,
                    error="Candidate evaluation tool is not configured",
                ),
            }
        return _tool_node(
            state,
            tools.candidate_evaluation,
            WorkflowStage.DECIDE_NEXT_ACTION,
        )

    def check_hypothesis(state: WorkflowState) -> dict[str, Any]:
        result = tools.hypothesis_checker.run(state)
        if not result.success: #checker工具执行失败，直接返回失败状态
            return {
                "stage": WorkflowStage.FAILED,
                "error": result.error,
                "events": _event(
                    state,
                    "tool_failed",
                    WorkflowStage.CHECK_HYPOTHESIS,
                    tool=tools.hypothesis_checker.name,
                    error=result.error,
                ),
            }
        hypothesis_check = result.payload.get("hypothesis_check")
        if not hypothesis_check or not hypothesis_check["accepted"]: #假设不合格
            issues = hypothesis_check["issues"] if hypothesis_check else []
            return {
                **result.payload,
                "stage": WorkflowStage.DECIDE_NEXT_ACTION,
                "evaluation": {
                    "status": "REJECTED_HYPOTHESIS",
                    "reasons": [issue["message"] for issue in issues],
                },
                "next_action": ManagerAction.NEW_HYPOTHESIS,
                "error": None,
                "events": _event(
                    state,
                    "hypothesis_rejected",
                    WorkflowStage.CHECK_HYPOTHESIS,
                    tool=tools.hypothesis_checker.name,
                    next_stage=WorkflowStage.DECIDE_NEXT_ACTION.value,
                ),
            }
        return {
            **result.payload,
            "stage": WorkflowStage.IMPLEMENT_FACTOR,
            "error": None,
            "events": _event(
                state,
                "tool_succeeded",
                WorkflowStage.CHECK_HYPOTHESIS,
                tool=tools.hypothesis_checker.name,
                next_stage=WorkflowStage.IMPLEMENT_FACTOR.value,
            ),
        }

    def implement_factor(state: WorkflowState) -> dict[str, Any]:
        next_attempt = state["attempt"] + 1
        attempt_state = {**state, "attempt": next_attempt}
        result = _tool_node(
            attempt_state,
            tools.implementation,
            WorkflowStage.VALIDATE_IMPLEMENTATION,
        )
        return {**result, "attempt": next_attempt}

    def validate_implementation(state: WorkflowState) -> dict[str, Any]:
        implementation = state["implementation"]
        if implementation is None or not implementation.get("unit_tests_passed"):
            return {
                "stage": WorkflowStage.FAILED,
                "error": "Factor implementation validation failed",
                "events": _event(
                    state,
                    "validation_failed",
                    WorkflowStage.VALIDATE_IMPLEMENTATION,
                ),
            }
        return {
            "stage": WorkflowStage.RUN_BACKTEST,
            "events": _event(
                state,
                "validation_succeeded",
                WorkflowStage.VALIDATE_IMPLEMENTATION,
                next_stage=WorkflowStage.RUN_BACKTEST.value,
            ),
        }

    def run_backtest(state: WorkflowState) -> dict[str, Any]:
        return _tool_node(
            state,
            tools.backtest,
            WorkflowStage.EVALUATE_RESULT,
        )

    def evaluate_result(state: WorkflowState) -> dict[str, Any]:
        return _tool_node(
            state,
            tools.evaluation,
            WorkflowStage.DECIDE_NEXT_ACTION,
        )

    def record_experiment(state: WorkflowState) -> dict[str, Any]:
        return _tool_node(
            state,
            tools.record,
            WorkflowStage.APPLY_MANAGER_ACTION,
        )

    def decide_next_action(state: WorkflowState) -> dict[str, Any]:
        allowed_actions = allowed_manager_actions(state)
        try:
            policy_result = manager_policy.decide(state, allowed_actions)
        except Exception as error:
            return {
                "stage": WorkflowStage.FAILED,
                "error": f"Manager Agent decision failed: {error}",
                "events": _event(
                    state,
                    "manager_decision_failed",
                    WorkflowStage.DECIDE_NEXT_ACTION,
                    policy=manager_policy.name,
                    error=str(error),
                ),
            }
        action = policy_result.decision.action
        if action not in allowed_actions: #检查policy选择是否合法
            return {
                "stage": WorkflowStage.FAILED,
                "error": (
                    "Manager Agent selected an action outside the allowed set: "
                    f"{action.value}"
                ),
                "events": _event(
                    state,
                    "manager_decision_failed",
                    WorkflowStage.DECIDE_NEXT_ACTION,
                    policy=manager_policy.name,
                    action=action.value,
                ),
            }
        manager_decision = policy_result.to_state_payload()
        return { #返回state更新
            "stage": WorkflowStage.RECORD_EXPERIMENT,
            "manager_decision": manager_decision,
            "next_action": action,
            "events": _event(
                state,
                "manager_decided",
                WorkflowStage.DECIDE_NEXT_ACTION,
                action=action.value,
                policy=manager_policy.name,
                next_stage=WorkflowStage.RECORD_EXPERIMENT.value,
            ),
        }
    
    # 执行manager policy的动作，更新state
    def apply_manager_action(state: WorkflowState) -> dict[str, Any]:
        action = state["next_action"]
        if action in { # 终止研究的动作，直接结束工作流
            ManagerAction.ACCEPT_FACTOR,
            ManagerAction.REJECT_FACTOR,
            ManagerAction.STOP_RESEARCH,
        }:
            return {
                "stage": WorkflowStage.COMPLETED,
                "events": _event(
                    state,
                    "workflow_completed",
                    WorkflowStage.APPLY_MANAGER_ACTION,
                    action=action.value,
                    policy=manager_policy.name,
                ),
            }

        if action in { # 开始新一轮研究的动作，重置state，进入下一轮
            ManagerAction.NEW_HYPOTHESIS,
            ManagerAction.REVISE_HYPOTHESIS,
        }:
            if state["iteration"] >= state["max_iterations"]:
                return {
                    "stage": WorkflowStage.COMPLETED,
                    "next_action": ManagerAction.STOP_RESEARCH,
                    "events": _event(
                        state,
                        "iteration_limit_reached",
                        WorkflowStage.APPLY_MANAGER_ACTION,
                    ),
                }
            return {
                "stage": WorkflowStage.INIT,
                "hypothesis": None,
                "financial_intuition": None,
                "hypothesis_candidates": None,
                "hypothesis_selection": None,
                "hypothesis_check": None,
                "attempt": 0,
                "factor_spec": None,
                "implementation": None,
                "backtest_result": None,
                "evaluation": None,
                "previous_attempt": None,
                "record": None,
                "next_action": None,
                "events": _event(
                    state,
                    "new_iteration_requested",
                    WorkflowStage.APPLY_MANAGER_ACTION,
                    action=action.value,
                ),
            }

        if action in { # 重新尝试实现因子的动作，重置相关state，进入实现阶段
            ManagerAction.RETRY_IMPLEMENTATION,
            ManagerAction.ADJUST_FACTOR,
        }:
            return {
                "stage": WorkflowStage.IMPLEMENT_FACTOR,
                "previous_attempt": {
                    "attempt": state["attempt"],
                    "factor_spec": state["factor_spec"],
                    "implementation": state["implementation"],
                    "backtest_result": state["backtest_result"],
                    "evaluation": state["evaluation"],
                    "manager_decision": state["manager_decision"],
                },
                "implementation": None,
                "backtest_result": None,
                "evaluation": None,
                "record": None,
                "next_action": None,
                "events": _event(
                    state,
                    "implementation_retry_requested",
                    WorkflowStage.APPLY_MANAGER_ACTION,
                    action=action.value,
                ),
            }

        return {
            "stage": WorkflowStage.FAILED,
            "error": "Manager received no valid next action",
            "events": _event(
                state,
                "decision_failed",
                WorkflowStage.APPLY_MANAGER_ACTION,
            ),
        }

    #组装LangGraph workflow
    workflow.add_node("initialize", initialize)
    workflow.add_node("propose_hypothesis", propose_hypothesis)
    workflow.add_node("evaluate_candidates", evaluate_candidates)
    workflow.add_node("check_hypothesis", check_hypothesis)
    workflow.add_node("implement_factor", implement_factor)
    workflow.add_node("validate_implementation", validate_implementation)
    workflow.add_node("run_backtest", run_backtest)
    workflow.add_node("evaluate_result", evaluate_result)
    workflow.add_node("record_experiment", record_experiment)
    workflow.add_node("decide_next_action", decide_next_action)
    workflow.add_node("apply_manager_action", apply_manager_action)
    workflow.add_node("completed", lambda state: {})
    workflow.add_node("failed", lambda state: {})

    # 添加工作流边和条件边
    workflow.add_edge(START, "initialize")
    workflow.add_conditional_edges(
        "initialize",
        lambda state: "completed"
        if state["stage"] == WorkflowStage.COMPLETED
        else "propose_hypothesis",
    )
    workflow.add_conditional_edges(
        "propose_hypothesis",
        lambda state: {
            WorkflowStage.FAILED: "failed",
            WorkflowStage.CHECK_HYPOTHESIS: "check_hypothesis",
            WorkflowStage.EVALUATE_CANDIDATES: "evaluate_candidates",
        }[state["stage"]],
    )
    workflow.add_conditional_edges(
        "evaluate_candidates",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "decide_next_action",
    )
    workflow.add_conditional_edges(
        "check_hypothesis",
        lambda state: {
            WorkflowStage.FAILED: "failed",
            WorkflowStage.IMPLEMENT_FACTOR: "implement_factor",
            WorkflowStage.DECIDE_NEXT_ACTION: "decide_next_action",
        }[state["stage"]],
    )
    workflow.add_conditional_edges(
        "implement_factor",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "validate_implementation",
    )
    workflow.add_conditional_edges(
        "validate_implementation",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "run_backtest",
    )
    workflow.add_conditional_edges(
        "run_backtest",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "evaluate_result",
    )
    workflow.add_conditional_edges(
        "evaluate_result",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "decide_next_action",
    )
    workflow.add_conditional_edges(
        "decide_next_action",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "record_experiment",
    )
    workflow.add_conditional_edges(
        "record_experiment",
        lambda state: "failed"
        if state["stage"] == WorkflowStage.FAILED
        else "apply_manager_action",
    )
    workflow.add_conditional_edges(
        "apply_manager_action",
        lambda state: {
            WorkflowStage.COMPLETED: "completed",
            WorkflowStage.FAILED: "failed",
            WorkflowStage.INIT: "initialize",
            WorkflowStage.IMPLEMENT_FACTOR: "implement_factor",
        }[state["stage"]],
    )
    workflow.add_edge("completed", END)
    workflow.add_edge("failed", END)

    # 编译工作流
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )
