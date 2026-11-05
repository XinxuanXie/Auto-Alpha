import tempfile
import unittest
from pathlib import Path

from agentic_workflow.checkpoint import checkpoint_config, open_sqlite_checkpointer
from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.state import WorkflowStage, create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools import SQLiteRecordTool


class SQLiteCheckpointTest(unittest.TestCase):
    def test_resumes_after_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime_path = Path(temporary_directory)
            checkpoint_path = runtime_path / "checkpoints.db"
            repository = SQLiteResearchRepository(runtime_path / "research.db")
            tools = ManagerTools.mock(record=SQLiteRecordTool(repository))
            initial_state = create_initial_state()
            config = checkpoint_config(initial_state["run_id"])

            with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
                interrupted_manager = build_manager(
                    tools,
                    checkpointer=checkpointer,
                    interrupt_after=["implement_factor"],
                )
                paused = interrupted_manager.invoke(initial_state, config=config)

            self.assertEqual(
                paused["stage"],
                WorkflowStage.VALIDATE_IMPLEMENTATION,
            )
            self.assertIsNotNone(paused["implementation"])
            self.assertEqual(repository.count(), 0)

            with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
                resumed_manager = build_manager(tools, checkpointer=checkpointer)
                completed = resumed_manager.invoke(None, config=config)

            self.assertEqual(completed["stage"], WorkflowStage.COMPLETED)
            self.assertEqual(completed["run_id"], initial_state["run_id"])
            self.assertEqual(completed["evaluation"]["status"], "PASS")
            self.assertEqual(repository.count(), 1)

    def test_rejects_empty_thread_id(self) -> None:
        with self.assertRaises(ValueError):
            checkpoint_config(" ")


if __name__ == "__main__":
    unittest.main()
