import json
import unittest
from types import SimpleNamespace

from agentic_workflow.state import ManagerAction, create_initial_state
from agentic_workflow.tools.evaluation_agent import EvaluationAgentTool


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def evaluation_state(rank_ic: float, net_return: float):
    state = create_initial_state()
    state["iteration"] = 1
    state["hypothesis"] = {
        "hypothesis_id": "H-TEST-0001",
        "claim": "测试假设",
    }
    state["factor_spec"] = {"factor_id": "F-TEST-0001"}
    state["implementation"] = {"unit_tests_passed": True}
    state["backtest_result"] = {
        "rank_ic": rank_ic,
        "net_return": net_return,
        "turnover": 0.5,
        "max_drawdown": -0.1,
    }
    return state


class EvaluationAgentToolTests(unittest.TestCase):
    def test_rule_failure_cannot_be_overridden_by_diagnosis(self):
        diagnosis = {
            "diagnosis": "预测能力未达到验收标准。",
            "evidence": ["Rank IC 低于硬性门槛。"],
            "recommended_action": "new_hypothesis",
            "revision_guidance": ["更换经济机制并提出新假设。"],
            "confidence": 0.9,
        }
        client = FakeClient(json.dumps(diagnosis, ensure_ascii=False))

        result = EvaluationAgentTool(
            model="test-model", client=client
        ).run(evaluation_state(rank_ic=-0.005, net_return=0.39))

        self.assertTrue(result.success, result.error)
        evaluation = result.payload["evaluation"]
        self.assertEqual(evaluation["status"], "FAIL_HYPOTHESIS")
        self.assertFalse(evaluation["rule_checks"]["rank_ic"]["passed"])
        self.assertTrue(evaluation["rule_checks"]["net_return"]["passed"])
        self.assertEqual(
            result.payload["next_action"], ManagerAction.NEW_HYPOTHESIS
        )
        request = json.loads(client.chat.completions.request["messages"][1]["content"])
        allowed = request["evaluation_context"]["allowed_recommendations"]
        self.assertNotIn("accept_factor", allowed)

    def test_pass_preserves_deterministic_status(self):
        diagnosis = {
            "diagnosis": "因子达到基础验收标准。",
            "evidence": ["Rank IC 与净收益均通过硬性门槛。"],
            "recommended_action": "accept_factor",
            "revision_guidance": ["进入后续稳健性检验。"],
            "confidence": 0.8,
        }
        result = EvaluationAgentTool(
            client=FakeClient(json.dumps(diagnosis, ensure_ascii=False))
        ).run(evaluation_state(rank_ic=0.03, net_return=0.05))

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.payload["evaluation"]["status"], "PASS")
        self.assertEqual(
            result.payload["next_action"], ManagerAction.ACCEPT_FACTOR
        )

    def test_rejects_disallowed_recommendation(self):
        diagnosis = {
            "diagnosis": "预测能力不足。",
            "evidence": ["Rank IC 未通过。"],
            "recommended_action": "accept_factor",
            "revision_guidance": ["检查因子方向。"],
            "confidence": 0.5,
        }
        result = EvaluationAgentTool(
            client=FakeClient(json.dumps(diagnosis, ensure_ascii=False))
        ).run(evaluation_state(rank_ic=0.0, net_return=0.01))

        self.assertFalse(result.success)
        self.assertIn("outside the allowed set", result.error)

    def test_requires_backtest_result(self):
        result = EvaluationAgentTool(client=FakeClient("{}")).run(
            create_initial_state()
        )

        self.assertFalse(result.success)
        self.assertIn("backtest result", result.error)

    def test_attempt_limit_removes_adjustment_recommendation(self):
        diagnosis = {
            "diagnosis": "当前尝试未达到验收标准。",
            "evidence": ["Rank IC 未通过硬性门槛。"],
            "recommended_action": "new_hypothesis",
            "revision_guidance": ["转向新的研究假设。"],
            "confidence": 0.9,
        }
        client = FakeClient(json.dumps(diagnosis, ensure_ascii=False))
        state = evaluation_state(rank_ic=0.0, net_return=0.01)
        state["attempt"] = state["max_attempts_per_hypothesis"]

        result = EvaluationAgentTool(client=client).run(state)

        self.assertTrue(result.success, result.error)
        request = json.loads(
            client.chat.completions.request["messages"][1]["content"]
        )
        allowed = request["evaluation_context"]["allowed_recommendations"]
        self.assertNotIn("adjust_factor", allowed)


if __name__ == "__main__":
    unittest.main()
