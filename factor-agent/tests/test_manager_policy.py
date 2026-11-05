import json
import unittest
from types import SimpleNamespace

from agentic_workflow.manager_policy import (
    ManagerDecision,
    OpenAIManagerPolicy,
    allowed_manager_actions,
)
from agentic_workflow.state import ManagerAction, create_initial_state


class FakeUsage:
    def model_dump(self, mode="python"):
        return {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="chatcmpl-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "action": "accept_factor",
                                "rationale": "Mock evaluation passed.",
                                "instructions": (
                                    "Accept the factor for this interface test."
                                ),
                                "confidence": 0.8,
                            }
                        )
                    )
                )
            ],
            usage=FakeUsage(),
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


class OpenAIManagerPolicyTests(unittest.TestCase):
    def test_attempt_limit_removes_factor_retry_actions(self):
        state = create_initial_state(max_attempts_per_hypothesis=3)
        state["iteration"] = 1
        state["attempt"] = 3
        state["evaluation"] = {"status": "FAIL_HYPOTHESIS"}

        actions = allowed_manager_actions(state)

        self.assertNotIn(ManagerAction.ADJUST_FACTOR, actions)
        self.assertNotIn(ManagerAction.RETRY_IMPLEMENTATION, actions)
        self.assertIn(ManagerAction.REVISE_HYPOTHESIS, actions)
        self.assertIn(ManagerAction.NEW_HYPOTHESIS, actions)

    def test_returns_structured_manager_decision(self):
        client = FakeOpenAIClient()
        policy = OpenAIManagerPolicy(model="test-model", client=client)
        state = create_initial_state(research_goal="测试外层 Manager")
        state["iteration"] = 1
        state["evaluation"] = {"status": "PASS", "reasons": ["test"]}
        state["next_action"] = ManagerAction.ACCEPT_FACTOR
        allowed = allowed_manager_actions(state)

        result = policy.decide(state, allowed)

        self.assertEqual(result.decision.action, ManagerAction.ACCEPT_FACTOR)
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.response_id, "chatcmpl-test")
        self.assertEqual(result.usage["total_tokens"], 120)
        self.assertEqual(
            client.chat.completions.request["response_format"],
            {"type": "json_object"},
        )
        prompt = json.loads(
            client.chat.completions.request["messages"][1]["content"]
        )
        self.assertEqual(
            prompt["evidence"]["research_goal"],
            "测试外层 Manager",
        )
        self.assertIn(
            "accept_factor",
            prompt["evidence"]["allowed_actions"],
        )
        self.assertIn("properties", prompt["output_schema"])


if __name__ == "__main__":
    unittest.main()
