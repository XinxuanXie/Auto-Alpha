import tempfile
import unittest
from pathlib import Path

from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.state import ManagerAction, create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools import (
    RuleBasedHypothesisCheckerTool,
    SQLiteRecordTool,
)


def valid_hypothesis() -> dict:
    return {
        "hypothesis_id": "H-TEST-001",
        "title": "异常放量后的短期价格反转",
        "claim": "近期放量上涨的股票未来五个交易日收益较低。",
        "economic_rationale": "异常放量上涨可能反映短期情绪过度反应。",
        "prediction_direction": "negative",
        "prediction_horizon_days": 5,
        "required_concepts": ["recent_return", "volume_abnormality"],
    }


class RuleBasedHypothesisCheckerTest(unittest.TestCase):
    def test_accepts_supported_new_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            state = create_initial_state()
            state["hypothesis"] = valid_hypothesis()

            result = RuleBasedHypothesisCheckerTool(repository).run(state)

            self.assertTrue(result.success)
            self.assertTrue(result.payload["hypothesis_check"]["accepted"])
            self.assertFalse(
                result.payload["hypothesis_check"]["checks"]["exact_duplicate"]
            )

    def test_rejects_exact_historical_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            historical_state = create_initial_state()
            historical_state["iteration"] = 1
            historical_state["hypothesis"] = valid_hypothesis()
            repository.save(historical_state)

            candidate_state = create_initial_state()
            candidate_state["hypothesis"] = {
                **valid_hypothesis(),
                "hypothesis_id": "H-TEST-999",
            }
            result = RuleBasedHypothesisCheckerTool(repository).run(
                candidate_state
            )

            check = result.payload["hypothesis_check"]
            self.assertFalse(check["accepted"])
            self.assertTrue(check["checks"]["exact_duplicate"])
            self.assertIn(
                "exact_duplicate_hypothesis",
                {issue["code"] for issue in check["issues"]},
            )

    def test_rejects_leakage_and_unsupported_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            state = create_initial_state()
            state["hypothesis"] = {
                **valid_hypothesis(),
                "required_concepts": ["future_return", "earnings_growth"],
            }

            result = RuleBasedHypothesisCheckerTool(repository).run(state)

            issue_codes = {
                issue["code"]
                for issue in result.payload["hypothesis_check"]["issues"]
            }
            self.assertIn("future_information_leakage", issue_codes)
            self.assertIn("unsupported_data_concept", issue_codes)

    def test_manager_records_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            historical_state = create_initial_state()
            historical_state["iteration"] = 1
            historical_state["hypothesis"] = valid_hypothesis()
            repository.save(historical_state)
            checker = RuleBasedHypothesisCheckerTool(repository)
            manager = build_manager(
                ManagerTools.mock(
                    hypothesis_checker=checker,
                    record=SQLiteRecordTool(repository),
                )
            )

            result = manager.invoke(create_initial_state(max_iterations=1))

            self.assertEqual(
                result["evaluation"]["status"],
                "REJECTED_HYPOTHESIS",
            )
            self.assertEqual(result["next_action"], ManagerAction.STOP_RESEARCH)
            self.assertEqual(repository.count(), 2)

if __name__ == "__main__":
    unittest.main()
