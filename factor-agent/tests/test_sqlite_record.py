import json
import tempfile
import unittest
from pathlib import Path

from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.manager_policy import ManagerDecision, ManagerPolicyResult
from agentic_workflow.state import (
    ManagerAction,
    WorkflowStage,
    create_initial_state,
)
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools import SQLiteRecordTool


class AdjustThenAcceptPolicy:
    name = "adjust_then_accept"

    def decide(self, state, allowed_actions):
        action = (
            ManagerAction.ADJUST_FACTOR
            if state["attempt"] == 1
            else ManagerAction.ACCEPT_FACTOR
        )
        return ManagerPolicyResult(
            decision=ManagerDecision(
                action=action,
                rationale="Exercise attempt-level memory.",
                instructions="Run the next controlled attempt.",
                confidence=1.0,
            ),
            policy=self.name,
        )


class SQLiteRecordToolTest(unittest.TestCase):
    def test_persists_completed_mock_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "research.db"
            repository = SQLiteResearchRepository(database_path)
            manager = build_manager(
                ManagerTools.mock(record=SQLiteRecordTool(repository))
            )

            result = manager.invoke(create_initial_state())
            stored = repository.fetch(result["record"]["record_id"])

            self.assertEqual(result["stage"], WorkflowStage.COMPLETED)
            self.assertEqual(repository.count(), 1)
            self.assertIsNotNone(stored)
            self.assertEqual(stored["hypothesis_id"], "H-0001")
            self.assertEqual(stored["attempt"], 1)
            self.assertEqual(stored["factor_id"], "F-0001")
            self.assertEqual(stored["evaluation_status"], "PASS")
            self.assertEqual(stored["next_action"], "accept_factor")
            state_snapshot = json.loads(stored["state_json"])
            self.assertEqual(state_snapshot["backtest_result"]["rank_ic"], 0.025)
            self.assertEqual(
                state_snapshot["manager_decision"]["action"],
                "accept_factor",
            )
            self.assertEqual(
                state_snapshot["manager_decision"]["policy"],
                "rule_based_manager",
            )

    def test_preserves_distinct_attempts_and_updates_same_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            state = create_initial_state()
            state["iteration"] = 1
            state["attempt"] = 1

            repository.save(state)
            repository.save(state)
            state["attempt"] = 2
            repository.save(state)

            self.assertEqual(repository.count(), 2)
            attempts = repository.fetch_run(state["run_id"])
            self.assertEqual([row["attempt"] for row in attempts], [1, 2])

    def test_workflow_records_every_factor_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            manager = build_manager(
                ManagerTools.mock(record=SQLiteRecordTool(repository)),
                policy=AdjustThenAcceptPolicy(),
            )

            result = manager.invoke(create_initial_state())
            records = repository.fetch_run(result["run_id"])

            self.assertEqual(result["iteration"], 1)
            self.assertEqual(result["attempt"], 2)
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [record["attempt"] for record in records],
                [1, 2],
            )
            self.assertEqual(records[0]["next_action"], "adjust_factor")
            self.assertEqual(records[1]["next_action"], "accept_factor")


if __name__ == "__main__":
    unittest.main()
