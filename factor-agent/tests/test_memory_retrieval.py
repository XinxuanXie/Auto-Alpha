import tempfile
import unittest
from pathlib import Path

from agentic_workflow.state import ManagerAction, create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository


def experiment_state(
    *,
    run_id: str,
    iteration: int,
    concepts: list[str],
    direction: str,
    rank_ic: float,
    net_return: float,
):
    state = create_initial_state()
    state["run_id"] = run_id
    state["iteration"] = iteration
    state["attempt"] = 1
    state["hypothesis"] = {
        "hypothesis_id": f"H-{run_id}-{iteration}",
        "title": "测试假设",
        "claim": "测试历史记录的相关性检索。",
        "economic_rationale": "用于单元测试。",
        "prediction_direction": direction,
        "prediction_horizon_days": 5,
        "required_concepts": concepts,
    }
    state["factor_spec"] = {
        "factor_id": f"F-{run_id}-{iteration}",
        "name": "测试因子",
        "description": "用于检索测试。",
    }
    state["backtest_result"] = {
        "rank_ic": rank_ic,
        "net_return": net_return,
    }
    state["evaluation"] = {
        "status": "FAIL_HYPOTHESIS",
        "reasons": ["Rank IC threshold failed"],
        "rule_checks": {
            "rank_ic": {"passed": rank_ic > 0.02},
            "net_return": {"passed": net_return > 0},
        },
        "diagnosis": "测试诊断",
        "revision_guidance": ["测试建议"],
    }
    state["next_action"] = ManagerAction.REVISE_HYPOTHESIS
    return state


class MemoryRetrievalTests(unittest.TestCase):
    def test_prioritizes_concepts_and_failure_profile_over_recency(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            repository.save(
                experiment_state(
                    run_id="HISTORY-RELATED",
                    iteration=1,
                    concepts=["price_reversal", "volume_abnormality"],
                    direction="positive",
                    rank_ic=0.01,
                    net_return=0.1,
                )
            )
            repository.save(
                experiment_state(
                    run_id="CURRENT-RUN",
                    iteration=1,
                    concepts=["price_reversal", "volume_abnormality"],
                    direction="positive",
                    rank_ic=0.0,
                    net_return=0.1,
                )
            )
            repository.save(
                experiment_state(
                    run_id="HISTORY-UNRELATED",
                    iteration=1,
                    concepts=["market_cap", "liquidity"],
                    direction="negative",
                    rank_ic=0.03,
                    net_return=0.2,
                )
            )
            query_state = create_initial_state()
            query_state["run_id"] = "CURRENT-RUN"
            query_state["iteration"] = 2

            results = repository.relevant_experiments(
                query_state,
                limit=3,
            )

        ordered_runs = [result["run_id"] for result in results]
        self.assertLess(
            ordered_runs.index("HISTORY-RELATED"),
            ordered_runs.index("HISTORY-UNRELATED"),
        )
        related = next(
            result
            for result in results
            if result["run_id"] == "HISTORY-RELATED"
        )
        self.assertTrue(
            any(
                reason.startswith("concept_overlap:")
                for reason in related["retrieval"]["match_reasons"]
            )
        )
        self.assertIn("rank_ic", related["failure_profile"])

    def test_validates_candidate_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            with self.assertRaises(ValueError):
                repository.relevant_experiments(
                    create_initial_state(),
                    limit=10,
                    candidate_limit=5,
                )


if __name__ == "__main__":
    unittest.main()
