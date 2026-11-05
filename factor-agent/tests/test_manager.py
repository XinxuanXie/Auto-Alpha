import unittest
from copy import deepcopy

from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.manager_policy import ManagerDecision, ManagerPolicyResult
from agentic_workflow.state import ManagerAction, WorkflowStage, create_initial_state
from agentic_workflow.tools import (
    MockBacktestTool,
    MockEvaluationTool,
    MockHypothesisCheckerTool,
    MockImplementationTool,
    MockRecordTool,
    ToolResult,
)


class FailingHypothesisTool:
    name = "failing_hypothesis"

    def run(self, state):
        return ToolResult.fail("Mock hypothesis failure")


class IteratingManagerPolicy:
    name = "test_iterating_manager"

    def decide(self, state, allowed_actions):
        action = (
            ManagerAction.NEW_HYPOTHESIS
            if state["iteration"] == 1
            else ManagerAction.ACCEPT_FACTOR
        )
        return ManagerPolicyResult(
            decision=ManagerDecision(
                action=action,
                rationale=f"Test decision for iteration {state['iteration']}",
                instructions="Continue the controlled test workflow.",
                confidence=0.9,
            ),
            policy=self.name,
            model="fake-model",
            response_id=f"response-{state['iteration']}",
        )


class AdjustOnceManagerPolicy:
    name = "adjust_once_manager"

    def decide(self, state, allowed_actions):
        action = (
            ManagerAction.ADJUST_FACTOR
            if state["attempt"] == 1
            else ManagerAction.ACCEPT_FACTOR
        )
        return ManagerPolicyResult(
            decision=ManagerDecision(
                action=action,
                rationale="Use the first result to revise once.",
                instructions="Revise the previous factor using its evaluation.",
                confidence=1.0,
            ),
            policy=self.name,
        )


class TrackingImplementationTool:
    name = "tracking_implementation"

    def __init__(self):
        self.delegate = MockImplementationTool()
        self.received_states = []

    def run(self, state):
        self.received_states.append(deepcopy(state))
        return self.delegate.run(state)


class ManagerWorkflowTest(unittest.TestCase):
    def test_completes_one_mock_research_iteration(self) -> None:
        manager = build_manager(ManagerTools.mock())

        result = manager.invoke(create_initial_state())

        self.assertEqual(result["stage"], WorkflowStage.COMPLETED)
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(result["attempt"], 1)
        self.assertEqual(result["next_action"], ManagerAction.ACCEPT_FACTOR)
        self.assertEqual(result["evaluation"]["status"], "PASS")
        self.assertIsNotNone(result["record"])
        self.assertGreaterEqual(len(result["events"]), 9)

    def test_stops_when_a_tool_fails(self) -> None:
        tools = ManagerTools(
            hypothesis=FailingHypothesisTool(),
            hypothesis_checker=MockHypothesisCheckerTool(),
            implementation=MockImplementationTool(),
            backtest=MockBacktestTool(),
            evaluation=MockEvaluationTool(),
            record=MockRecordTool(),
        )
        manager = build_manager(tools)

        result = manager.invoke(create_initial_state())

        self.assertEqual(result["stage"], WorkflowStage.FAILED)
        self.assertEqual(result["error"], "Mock hypothesis failure")
        self.assertEqual(result["events"][-1]["type"], "tool_failed")

    def test_outer_manager_policy_controls_research_iterations(self) -> None:
        manager = build_manager(
            ManagerTools.mock(),
            policy=IteratingManagerPolicy(),
        )

        result = manager.invoke(
            create_initial_state(
                max_iterations=3,
                research_goal="验证外层 Manager Agent 的迭代控制",
            )
        )

        self.assertEqual(result["stage"], WorkflowStage.COMPLETED)
        self.assertEqual(result["iteration"], 2)
        self.assertEqual(result["next_action"], ManagerAction.ACCEPT_FACTOR)
        self.assertEqual(
            result["manager_decision"]["policy"],
            "test_iterating_manager",
        )
        self.assertEqual(result["manager_decision"]["model"], "fake-model")

    def test_factor_adjustment_passes_previous_attempt_to_implementation(self):
        implementation = TrackingImplementationTool()
        manager = build_manager(
            ManagerTools.mock(implementation=implementation),
            policy=AdjustOnceManagerPolicy(),
        )

        result = manager.invoke(create_initial_state())

        self.assertEqual(result["attempt"], 2)
        self.assertEqual(len(implementation.received_states), 2)
        self.assertIsNone(
            implementation.received_states[0]["previous_attempt"]
        )
        previous_attempt = implementation.received_states[1][
            "previous_attempt"
        ]
        self.assertEqual(previous_attempt["attempt"], 1)
        self.assertEqual(
            previous_attempt["factor_spec"]["factor_id"],
            "F-0001",
        )
        self.assertEqual(
            previous_attempt["backtest_result"]["rank_ic"],
            0.025,
        )
        self.assertEqual(previous_attempt["evaluation"]["status"], "PASS")
        self.assertEqual(
            previous_attempt["manager_decision"]["action"],
            "adjust_factor",
        )


if __name__ == "__main__":
    unittest.main()
