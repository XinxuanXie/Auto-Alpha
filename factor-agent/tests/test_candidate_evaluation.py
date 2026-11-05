import unittest

from agentic_workflow.state import create_initial_state
from agentic_workflow.tools.base import ToolResult
from agentic_workflow.tools.candidate_evaluation import (
    CandidateBatchEvaluationTool,
)
from agentic_workflow.tools.research_team import (
    RuleBasedResearchCandidateSelector,
)


class CandidateImplementation:
    name = "candidate_implementation"

    def run(self, state):
        hypothesis_id = state["hypothesis"]["hypothesis_id"]
        return ToolResult.ok(
            {
                "factor_spec": {
                    "factor_id": hypothesis_id.replace("H-", "F-"),
                    "hypothesis_id": hypothesis_id,
                },
                "implementation": {"unit_tests_passed": True},
            }
        )


class CandidateBacktest:
    name = "candidate_backtest"

    def __init__(self):
        self.hypothesis_ids = []

    def run(self, state):
        hypothesis_id = state["hypothesis"]["hypothesis_id"]
        self.hypothesis_ids.append(hypothesis_id)
        score = float(hypothesis_id.rsplit("-", 1)[-1]) / 100
        return ToolResult.ok(
            {
                "backtest_result": {
                    "rank_ic": score,
                    "net_return": score,
                }
            }
        )


class CandidateEvaluation:
    name = "candidate_evaluation"

    def run(self, state):
        passed = state["backtest_result"]["rank_ic"] > 0.02
        return ToolResult.ok(
            {
                "evaluation": {
                    "status": "PASS" if passed else "FAIL_HYPOTHESIS",
                    "recommended_action": (
                        "accept_factor" if passed else "new_hypothesis"
                    ),
                }
            }
        )


class CandidateBatchEvaluationTests(unittest.TestCase):
    def test_backtests_every_accepted_candidate_then_selects_best(self):
        state = create_initial_state()
        state["iteration"] = 1
        state["hypothesis_candidates"] = [
            {
                "accepted": True,
                "hypothesis": {
                    "hypothesis_id": f"H-0{index}",
                    "title": f"candidate {index}",
                },
                "financial_intuition": {"intuition_title": f"idea {index}"},
                "hypothesis_check": {"accepted": True, "issues": []},
            }
            for index in range(1, 5)
        ]
        backtest = CandidateBacktest()
        tool = CandidateBatchEvaluationTool(
            implementation=CandidateImplementation(),
            backtest=backtest,
            evaluation=CandidateEvaluation(),
            selector=RuleBasedResearchCandidateSelector(),
        )

        result = tool.run(state)

        self.assertTrue(result.success, result.error)
        self.assertEqual(backtest.hypothesis_ids, ["H-01", "H-02", "H-03", "H-04"])
        self.assertEqual(len(result.payload["hypothesis_candidates"]), 4)
        self.assertTrue(
            all(
                candidate["pipeline_status"] == "completed"
                for candidate in result.payload["hypothesis_candidates"]
            )
        )
        self.assertEqual(result.payload["hypothesis"]["hypothesis_id"], "H-04")


if __name__ == "__main__":
    unittest.main()
