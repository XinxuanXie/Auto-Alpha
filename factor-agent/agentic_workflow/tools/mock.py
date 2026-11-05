"""Deterministic tools used to validate the workflow before real integrations."""

from agentic_workflow.state import ManagerAction, WorkflowState
from agentic_workflow.tools.base import ToolResult
from factor_pipeline.spec import FactorSpec, abnormal_volume_reversal_spec


class MockHypothesisTool:
    name = "hypothesis"

    def run(self, state: WorkflowState) -> ToolResult:
        hypothesis = {
            "hypothesis_id": f"H-{state['iteration']:04d}",
            "title": "异常放量后的短期价格反转",
            "claim": "近期放量上涨的股票未来五个交易日收益较低。",
            "economic_rationale": "异常放量上涨可能反映短期情绪过度反应。",
            "prediction_direction": "negative",
            "prediction_horizon_days": 5,
            "required_concepts": ["recent_return", "volume_abnormality"],
        }
        return ToolResult.ok({"hypothesis": hypothesis})


class MockHypothesisCheckerTool:
    name = "hypothesis_checker"

    def run(self, state: WorkflowState) -> ToolResult:
        hypothesis = state["hypothesis"]
        if hypothesis is None:
            return ToolResult.fail("A hypothesis is required before checking")

        required_fields = {
            "hypothesis_id",
            "claim",
            "prediction_direction",
            "prediction_horizon_days",
            "required_concepts",
        }
        missing_fields = sorted(required_fields - hypothesis.keys())
        hypothesis_check = {
            "accepted": not missing_fields,
            "issues": [f"Missing field: {field}" for field in missing_fields],
        }
        return ToolResult.ok({"hypothesis_check": hypothesis_check})


class MockImplementationTool:
    name = "implementation"

    def run(self, state: WorkflowState) -> ToolResult:
        hypothesis = state["hypothesis"]
        if hypothesis is None:
            return ToolResult.fail("A hypothesis is required before implementation")

        factor_payload = abnormal_volume_reversal_spec().model_dump(
            mode="json", exclude_none=True
        )
        factor_payload.update(
            {
                "factor_id": hypothesis["hypothesis_id"].replace("H-", "F-"),
                "hypothesis_id": hypothesis["hypothesis_id"],
            }
        )
        factor_spec = FactorSpec.model_validate(factor_payload).model_dump(
            mode="json", exclude_none=True
        )
        implementation = {
            "factor_spec_schema": "factor_spec_v1",
            "compiled_artifact": "factor_pipeline.compile_factor_spec",
            "unit_tests_passed": True,
        }
        return ToolResult.ok(
            {
                "factor_spec": factor_spec,
                "implementation": implementation,
            }
        )


class MockBacktestTool:
    name = "backtest"

    def run(self, state: WorkflowState) -> ToolResult:
        if state["implementation"] is None:
            return ToolResult.fail("An implementation is required before backtesting")

        backtest_result = {
            "rank_ic": 0.025,
            "icir": 0.42,
            "net_excess_return": 0.031,
            "turnover": 0.48,
            "max_drawdown": -0.09,
            "transaction_cost_bps": 10,
        }
        return ToolResult.ok({"backtest_result": backtest_result})


class MockEvaluationTool:
    name = "evaluation"

    def run(self, state: WorkflowState) -> ToolResult:
        backtest_result = state["backtest_result"]
        if backtest_result is None:
            return ToolResult.fail("A backtest result is required before evaluation")

        rank_ic = backtest_result.get(
            "rank_ic", backtest_result.get("mean_rank_ic")
        )
        net_return = backtest_result.get(
            "net_return", backtest_result.get("net_excess_return")
        )
        if rank_ic is None or net_return is None:
            return ToolResult.fail(
                "Backtest result must contain rank_ic and net_return"
            )
        passed = rank_ic > 0.02 and net_return > 0
        evaluation = {
            "status": "PASS" if passed else "FAIL_HYPOTHESIS",
            "reasons": [
                "Rank IC exceeds the rule threshold",
                "Net return is positive after costs",
            ]
            if passed
            else ["The rule-based acceptance thresholds were not satisfied"],
        }
        next_action = (
            ManagerAction.ACCEPT_FACTOR
            if passed
            else ManagerAction.NEW_HYPOTHESIS
        )
        return ToolResult.ok(
            {"evaluation": evaluation, "next_action": next_action}
        )


class MockRecordTool:
    name = "record"

    def run(self, state: WorkflowState) -> ToolResult:
        record = {
            "record_id": (
                f"{state['run_id']}:{state['iteration']}:{state['attempt']}"
            ),
            "run_id": state["run_id"],
            "iteration": state["iteration"],
            "attempt": state["attempt"],
            "hypothesis_id": (
                state["hypothesis"]["hypothesis_id"]
                if state["hypothesis"] is not None
                else None
            ),
            "evaluation_status": (
                state["evaluation"]["status"]
                if state["evaluation"] is not None
                else None
            ),
        }
        return ToolResult.ok({"record": record})
