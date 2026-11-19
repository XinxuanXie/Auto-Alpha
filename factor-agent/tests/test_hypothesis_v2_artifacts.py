import tempfile
import unittest
from pathlib import Path

import polars as pl

from hypothesis_workflow_v2 import (
    ArtifactIntegrityError,
    ArtifactRepository,
    V2Stage,
    create_initial_state,
)
from hypothesis_workflow_v2.contracts import ResearchDirection


class ArtifactRepositoryTest(unittest.TestCase):
    def test_round_trips_json_with_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            payload = [{"idea_id": "I-1", "text": "market mechanism"}]

            reference = repository.write_json("run-1", "research_ideas", payload)

            self.assertEqual(reference.item_count, 1)
            self.assertEqual(repository.read_json(reference), payload)
            self.assertTrue(repository.verify(reference))

    def test_detects_modified_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            reference = repository.write_json("run-1", "ideas", [1, 2])
            artifact_path = Path(temporary_directory) / reference.path
            artifact_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(ArtifactIntegrityError):
                repository.read_json(reference)

    def test_round_trips_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            frame = pl.DataFrame(
                {"experiment_id": ["E-1", "E-2"], "rank_ic": [0.03, 0.01]}
            )

            reference = repository.write_parquet("run-1", "train_metrics", frame)
            restored = repository.read_parquet(reference)

            self.assertEqual(reference.item_count, 2)
            self.assertTrue(restored.equals(frame))

    def test_rejects_unsafe_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            with self.assertRaises(ValueError):
                repository.write_json("../escape", "ideas", [])

    def test_initial_state_is_small_and_has_one_branch_per_direction(self) -> None:
        state = create_initial_state(
            {"llm_concurrency": 4, "backtest_concurrency": 1},
            run_id="run-1",
        )

        self.assertEqual(state["stage"], V2Stage.INIT)
        self.assertEqual(len(state["branch_status"]), len(ResearchDirection))
        self.assertEqual(set(state["branch_status"].values()), {"pending"})
        self.assertEqual(state["artifact_refs"], {})
        self.assertFalse(state["hypothesis_repair_used"])
        self.assertFalse(state["factor_repair_used"])


if __name__ == "__main__":
    unittest.main()
