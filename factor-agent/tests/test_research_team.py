import json
import unittest
from types import SimpleNamespace

from agentic_workflow.state import create_initial_state
from agentic_workflow.tools.base import ToolResult
from agentic_workflow.tools.research_team import (
    MultiStrategyResearchTool,
    OpenAIManagerCandidateSelector,
    RuleBasedResearchCandidateSelector,
)
from agentic_workflow.tools.research_agent import (
    RESEARCH_STRATEGIES,
    build_research_context,
    build_system_prompt,
)


class StaticSpecialist:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.name = f"research_agent_{strategy}"

    def run(self, state):
        return ToolResult.ok(
            {
                "hypothesis": {
                    "hypothesis_id": (
                        f"H-TEST-{state['iteration']:04d}-{self.strategy.upper()}"
                    ),
                    "title": f"{self.strategy} hypothesis",
                    "claim": "测试候选假设。",
                    "economic_rationale": "测试经济机制。",
                    "prediction_direction": "positive",
                    "prediction_horizon_days": 5,
                    "required_concepts": ["recent_return"],
                    "research_strategy": self.strategy,
                },
                "financial_intuition": {
                    "intuition_title": f"{self.strategy} intuition"
                },
            }
        )


class ScreeningChecker:
    name = "screening_checker"

    def run(self, state):
        strategy = state["hypothesis"]["research_strategy"]
        accepted = strategy != "reversal"
        return ToolResult.ok(
            {
                "hypothesis_check": {
                    "accepted": accepted,
                    "issues": [] if accepted else [{"message": "duplicate"}],
                }
            }
        )


class FakeCompletions:
    def __init__(self, selected_id: str):
        self.selected_id = selected_id
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="selection-response",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "selected_hypothesis_id": self.selected_id,
                                "rationale": "选择具有差异化经济机制的候选。",
                                "confidence": 0.9,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ],
            usage=None,
        )


class FakeClient:
    def __init__(self, selected_id: str):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(selected_id)
        )


class ResearchTeamTests(unittest.TestCase):
    def setUp(self):
        self.specialists = [
            StaticSpecialist("momentum"),
            StaticSpecialist("reversal"),
            StaticSpecialist("liquidity_volatility"),
            StaticSpecialist("price_volume"),
        ]
        self.state = create_initial_state()
        self.state["iteration"] = 1

    def test_screens_four_candidates_for_downstream_backtesting(self):
        tool = MultiStrategyResearchTool(
            specialists=self.specialists,
            hypothesis_checker=ScreeningChecker(),
        )

        result = tool.run(self.state)

        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.payload["hypothesis_candidates"]), 4)
        self.assertEqual(
            sum(
                candidate["accepted"]
                for candidate in result.payload["hypothesis_candidates"]
            ),
            3,
        )
        self.assertNotIn("hypothesis", result.payload)
        self.assertNotIn("hypothesis_selection", result.payload)

    def test_openai_manager_selects_known_candidate(self):
        selected_id = "H-TEST-0001-PRICE_VOLUME"
        client = FakeClient(selected_id)
        selector = OpenAIManagerCandidateSelector(
            model="test-model",
            client=client,
        )
        candidates = [
            specialist.run(self.state).payload
            for specialist in self.specialists
            if specialist.strategy != "reversal"
        ]

        result = selector.select(self.state, candidates)

        self.assertEqual(
            result.selection.selected_hypothesis_id,
            selected_id,
        )
        request = json.loads(
            client.chat.completions.request["messages"][1]["content"]
        )
        self.assertEqual(len(request["accepted_candidates"]), 3)

    def test_research_prompt_uses_failed_experiments_as_constraints(self):
        prompt = build_system_prompt(RESEARCH_STRATEGIES[0])

        self.assertIn("必须把 relevant_experiments 当作经验约束", prompt)
        self.assertIn("Rank IC 较低而失败", prompt)
        self.assertIn("一个核心经济机制", prompt)
        self.assertIn("不得只改标题、措辞、阈值", prompt)

    def test_research_context_exposes_quality_and_learning_policy(self):
        context = build_research_context(
            state=self.state,
            relevant_experiments=[],
            strategy=RESEARCH_STRATEGIES[0],
        )

        self.assertIn("research_quality_criteria", context)
        self.assertIn("historical_learning_policy", context)
        self.assertIn(
            "rank_ic_failure",
            context["historical_learning_policy"],
        )


if __name__ == "__main__":
    unittest.main()
