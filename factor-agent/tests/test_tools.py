import unittest

from agentic_workflow.state import ManagerAction, create_initial_state
from agentic_workflow.tools import (
    MockBacktestTool,
    MockEvaluationTool,
    MockHypothesisTool,
    MockImplementationTool,
    MockRecordTool,
)
from factor_pipeline.spec import FactorSpec


class MockToolPipelineTest(unittest.TestCase):
    def test_mock_tools_produce_compatible_outputs(self) -> None:
        state = create_initial_state()

        hypothesis_result = MockHypothesisTool().run(state)
        self.assertTrue(hypothesis_result.success)
        state.update(hypothesis_result.payload)

        implementation_result = MockImplementationTool().run(state)
        self.assertTrue(implementation_result.success)
        state.update(implementation_result.payload)
        validated_spec = FactorSpec.model_validate(state["factor_spec"])
        self.assertEqual(validated_spec.hypothesis_id, "H-0000")
        self.assertEqual(
            state["implementation"]["compiled_artifact"],
            "factor_pipeline.compile_factor_spec",
        )

        backtest_result = MockBacktestTool().run(state)
        self.assertTrue(backtest_result.success)
        state.update(backtest_result.payload)

        evaluation_result = MockEvaluationTool().run(state)
        self.assertTrue(evaluation_result.success)
        state.update(evaluation_result.payload)

        record_result = MockRecordTool().run(state)

        self.assertTrue(record_result.success)
        self.assertEqual(state["evaluation"]["status"], "PASS")
        self.assertEqual(state["next_action"], ManagerAction.ACCEPT_FACTOR)
        self.assertEqual(
            record_result.payload["record"]["hypothesis_id"],
            state["hypothesis"]["hypothesis_id"],
        )

    def test_implementation_requires_hypothesis(self) -> None:
        result = MockImplementationTool().run(create_initial_state())

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "A hypothesis is required before implementation",
        )


if __name__ == "__main__":
    unittest.main()
