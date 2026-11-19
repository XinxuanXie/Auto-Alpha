import tempfile
import unittest

from hypothesis_workflow_v2 import (
    ArtifactRef,
    ArtifactRepository,
    MockV2Tools,
    V2Stage,
    build_manager,
    create_initial_state,
)
from hypothesis_workflow_v2.contracts import ResearchDirection, SplitName


class AllButOneDirectionTools(MockV2Tools):
    def generate_research_ideas(self, state, context):
        # One direction produces no idea, so exactly one branch fails.
        return super().generate_research_ideas(state, context)[:-1]


class NoTrainPassTools(MockV2Tools):
    def run_backtests(self, experiments, split, train_metrics=None):
        metrics = super().run_backtests(experiments, split, train_metrics)
        if split != SplitName.TRAIN:
            return metrics
        return [
            metric.model_copy(
                update={"mean_rank_ic": 0.01, "net_return": -0.01}
            )
            for metric in metrics
        ]


class SparseTrainPassTools(MockV2Tools):
    def run_backtests(self, experiments, split, train_metrics=None):
        metrics = super().run_backtests(experiments, split, train_metrics)
        if split != SplitName.TRAIN:
            return metrics
        # Mock passers have IC 0.03 and net 0.06, but their signal covers
        # almost no stocks: the coverage leg of the gate must block them.
        return [
            metric.model_copy(update={"median_stock_coverage": 0.0})
            for metric in metrics
        ]


class HypothesisV2ManagerTest(unittest.TestCase):
    def test_complete_mock_workflow_runs_all_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            initial = create_initial_state({"mode": "mock"}, run_id="run-v2-test")
            manager = build_manager(MockV2Tools(), repository)

            result = manager.invoke(initial)

            self.assertEqual(result["stage"], V2Stage.COMPLETED)
            self.assertIsNone(result["error"])
            self.assertTrue(result["hypothesis_repair_used"])
            self.assertTrue(result["factor_repair_used"])
            # Mock batch: half the generated experiments reach train, half of
            # those reach validation, and all validation passers reach test.
            # Counts scale with the direction count.
            n = len(ResearchDirection)
            self.assertEqual(len(result["train_candidate_ids"]), 2 * n)
            self.assertEqual(len(result["validation_candidate_ids"]), n)
            self.assertEqual(len(result["test_candidate_ids"]), n)
            self.assertEqual(set(result["branch_status"].values()), {"passed"})

            summary_ref = ArtifactRef.model_validate(
                result["artifact_refs"]["run_summary"]
            )
            summary = repository.read_json(summary_ref)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(summary["test_candidates"]), n)

            event_types = [event["type"] for event in result["events"]]
            self.assertEqual(event_types.count("hypotheses_checked"), 2)
            self.assertEqual(event_types.count("experiments_preflighted"), 2)
            self.assertIn("hypotheses_repaired", event_types)
            self.assertIn("measurement_plans_designed", event_types)
            self.assertIn("experiments_repaired", event_types)
            self.assertIn("test_backtests_completed", event_types)
            self.assertEqual(event_types[-1], "run_recorded")

    def test_artifact_counts_match_mock_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            initial = create_initial_state(run_id="run-v2-counts")
            result = build_manager(MockV2Tools(), repository).invoke(initial)

            # Mock batch: 2 hypotheses per idea, 2 experiments per
            # hypothesis; half the train metrics pass to validation, half of
            # those pass to test. Counts scale with the direction count.
            n = len(ResearchDirection)
            expected_counts = {
                "research_ideas": n,
                "insight_analyses": n,
                "hypotheses": 2 * n,
                "measurement_plans": 2 * n,
                "experiment_specs": 4 * n,
                "train_metrics": 4 * n,
                "validation_metrics": 2 * n,
                "test_metrics": n,
            }
            for key, expected in expected_counts.items():
                reference = ArtifactRef.model_validate(result["artifact_refs"][key])
                self.assertEqual(reference.item_count, expected, key)
                self.assertTrue(repository.verify(reference), key)

    def test_one_failed_research_branch_does_not_stop_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            initial = create_initial_state(run_id="run-v2-partial")

            result = build_manager(AllButOneDirectionTools(), repository).invoke(
                initial
            )

            self.assertEqual(result["stage"], V2Stage.COMPLETED)
            self.assertEqual(
                list(result["branch_status"].values()).count("failed"), 1
            )
            self.assertGreater(len(result["test_candidate_ids"]), 0)

    def test_no_training_pass_is_a_completed_research_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            initial = create_initial_state(run_id="run-v2-no-pass")

            result = build_manager(NoTrainPassTools(), repository).invoke(initial)

            self.assertEqual(result["stage"], V2Stage.COMPLETED)
            self.assertEqual(result["train_candidate_ids"], [])
            self.assertNotIn("validation_metrics", result["artifact_refs"])
            self.assertNotIn("test_metrics", result["artifact_refs"])
            self.assertIn("memory_summary", result["artifact_refs"])
            self.assertIn("run_summary", result["artifact_refs"])

    def test_sparse_train_passers_do_not_reach_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = ArtifactRepository(temporary_directory)
            initial = create_initial_state(run_id="run-v2-sparse")

            result = build_manager(
                SparseTrainPassTools(), repository
            ).invoke(initial)

            self.assertEqual(result["stage"], V2Stage.COMPLETED)
            self.assertEqual(result["train_candidate_ids"], [])
            self.assertNotIn("validation_metrics", result["artifact_refs"])
            self.assertNotIn("test_metrics", result["artifact_refs"])


if __name__ == "__main__":
    unittest.main()
