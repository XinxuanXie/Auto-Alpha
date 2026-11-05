import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentic_workflow.state import create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.research_agent import RESEARCH_STRATEGIES
from agentic_workflow.tools.two_stage_research import (
    FinancialIntuitionAgentTool,
    HypothesisFormulationAgentTool,
    TwoStageResearchSpecialist,
)
from agentic_workflow.tools.web_research import WebResearchSource


class StaticWebResearch:
    def search(self, query):
        return [
            WebResearchSource(
                source_id="WEB-01",
                title="Momentum research",
                url="https://example.com/momentum",
                snippet="Research on momentum conditioned by volume trends.",
            )
        ]


class StaticFactorLibrary:
    def search(self, concepts, *, iteration, limit=12):
        return [
            {
                "factor_id": "ALPHA158-ROC20",
                "name": "Qlib Alpha158 ROC20",
                "family": "alpha158",
                "definition": "Ref($close,20)/$close",
                "economic_rationale": "Twenty-day price trend.",
                "required_data": ["close"],
                "concepts": ["price_momentum", "recent_return"],
                "implementation_status": "available_now",
                "source_url": "https://github.com/microsoft/qlib",
            }
        ]


class StaticCompletions:
    def __init__(self, payload):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload, ensure_ascii=False)
                    )
                )
            ]
        )


class StaticClient:
    def __init__(self, payload):
        self.chat = SimpleNamespace(completions=StaticCompletions(payload))


class TwoStageResearchTests(unittest.TestCase):
    def test_generates_intuition_then_formulates_hypothesis(self):
        intuition_payload = {
            "intuition_title": "趋势中的成交量确认",
            "core_mechanism": "价格趋势需要持续交易参与才能延续。",
            "causal_chain": ["信息逐步扩散", "成交量确认", "趋势延续"],
            "market_condition": "中期趋势为正且成交量趋势增强",
            "expected_pattern": "确认充分的股票未来收益更高",
            "factor_components": [
                {
                    "concept": "price_momentum",
                    "role": "core_signal",
                    "mechanism": "识别中期价格趋势",
                },
                {
                    "concept": "volume_trend",
                    "role": "confirmation",
                    "mechanism": "确认交易参与持续增强",
                },
            ],
            "falsification_risks": ["趋势可能已被价格充分反映"],
            "source_references": [
                {"source_id": "WEB-01", "relevance": "支持量价确认机制"}
            ],
            "factor_references": [
                {
                    "factor_id": "ALPHA158-ROC20",
                    "adaptation": "作为中期趋势主信号",
                }
            ],
        }
        hypothesis_payload = {
            "title": "成交量确认的中期价格趋势",
            "claim": "中期价格趋势较高且成交量趋势增强的股票未来五日收益更高。",
            "economic_rationale": "持续交易参与支持信息逐步扩散。",
            "prediction_direction": "positive",
            "prediction_horizon_days": 5,
            "required_concepts": ["price_momentum", "volume_trend"],
            "factor_structure": [
                {
                    "concept": "price_momentum",
                    "role": "core_signal",
                    "expected_relation": "high",
                },
                {
                    "concept": "volume_trend",
                    "role": "confirmation",
                    "expected_relation": "increasing",
                },
            ],
        }
        invalid_intuition_payload = {
            **intuition_payload,
            "factor_components": [
                *intuition_payload["factor_components"],
                {
                    "concept": "price_reversal",
                    "role": "condition",
                    "mechanism": "该概念超出 momentum 角色范围",
                },
            ],
        }
        intuition_client = StaticClient(
            [invalid_intuition_payload, intuition_payload]
        )
        formulation_client = StaticClient(hypothesis_payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            strategy = RESEARCH_STRATEGIES[0]
            specialist = TwoStageResearchSpecialist(
                FinancialIntuitionAgentTool(
                    repository=repository,
                    strategy=strategy,
                    client=intuition_client,
                    web_research=StaticWebResearch(),
                    factor_library=StaticFactorLibrary(),
                ),
                HypothesisFormulationAgentTool(
                    strategy=strategy,
                    client=formulation_client,
                ),
            )
            state = create_initial_state()
            state["iteration"] = 1

            result = specialist.run(state)

        self.assertTrue(result.success, result.error)
        self.assertEqual(intuition_client.chat.completions.calls, 2)
        self.assertEqual(formulation_client.chat.completions.calls, 1)
        self.assertEqual(
            result.payload["financial_intuition"]["intuition_title"],
            "趋势中的成交量确认",
        )
        hypothesis = result.payload["hypothesis"]
        self.assertEqual(len(hypothesis["factor_structure"]), 2)
        self.assertEqual(
            hypothesis["factor_references"][0]["factor_id"],
            "ALPHA158-ROC20",
        )

    def test_formulation_retries_after_introducing_new_concept(self):
        invalid = {
            "title": "无效假设",
            "claim": "引入直觉中不存在的成交量概念。",
            "economic_rationale": "测试自动重试。",
            "prediction_direction": "positive",
            "prediction_horizon_days": 5,
            "required_concepts": ["price_momentum", "volume_trend"],
            "factor_structure": [
                {
                    "concept": "price_momentum",
                    "role": "core_signal",
                    "expected_relation": "high",
                },
                {
                    "concept": "volume_trend",
                    "role": "confirmation",
                    "expected_relation": "increasing",
                },
            ],
        }
        valid = {
            **invalid,
            "title": "有效假设",
            "claim": "中期价格趋势较高且近期收益温和的股票未来收益更高。",
            "required_concepts": ["price_momentum", "recent_return"],
            "factor_structure": [
                {
                    "concept": "price_momentum",
                    "role": "core_signal",
                    "expected_relation": "high",
                },
                {
                    "concept": "recent_return",
                    "role": "condition",
                    "expected_relation": "high",
                },
            ],
        }
        client = StaticClient([invalid, valid])
        agent = HypothesisFormulationAgentTool(
            strategy=RESEARCH_STRATEGIES[0],
            client=client,
        )
        state = create_initial_state()
        state["iteration"] = 1
        state["financial_intuition"] = {
            "factor_components": [
                {"concept": "price_momentum"},
                {"concept": "recent_return"},
            ],
            "sources": [],
            "factor_references": [],
        }
        state["research_context"] = {"relevant_experiments": []}

        result = agent.run(state)

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.payload["formulation_attempts"], 2)
        self.assertEqual(client.chat.completions.calls, 2)


if __name__ == "__main__":
    unittest.main()
