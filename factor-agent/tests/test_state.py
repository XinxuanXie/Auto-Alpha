import unittest

from agentic_workflow.state import WorkflowStage, create_initial_state


class CreateInitialStateTest(unittest.TestCase):
    def test_creates_clean_initial_state(self) -> None:
        state = create_initial_state(max_iterations=5)

        self.assertTrue(state["run_id"])
        self.assertEqual(state["stage"], WorkflowStage.INIT)
        self.assertEqual(state["iteration"], 0)
        self.assertEqual(state["attempt"], 0)
        self.assertEqual(state["max_attempts_per_hypothesis"], 3)
        self.assertEqual(
            state["research_goal"],
            "发现具有稳定横截面预测能力的美股因子",
        )
        self.assertEqual(state["max_iterations"], 5)
        self.assertIsNone(state["financial_intuition"])
        self.assertIsNone(state["hypothesis"])
        self.assertIsNone(state["hypothesis_candidates"])
        self.assertIsNone(state["hypothesis_selection"])
        self.assertIsNone(state["previous_attempt"])
        self.assertEqual(state["retry_counts"], {})
        self.assertEqual(state["events"], [])

    def test_rejects_non_positive_iteration_limit(self) -> None:
        with self.assertRaises(ValueError):
            create_initial_state(max_iterations=0)

    def test_rejects_non_positive_attempt_limit(self) -> None:
        with self.assertRaises(ValueError):
            create_initial_state(max_attempts_per_hypothesis=0)

    def test_rejects_empty_research_goal(self) -> None:
        with self.assertRaises(ValueError):
            create_initial_state(research_goal=" ")


if __name__ == "__main__":
    unittest.main()
