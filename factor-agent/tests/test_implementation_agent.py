import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from agentic_workflow.state import create_initial_state
from agentic_workflow.tools.implementation_agent import ImplementationAgentTool
from factor_pipeline.spec import FactorSpec


class FakeCompletions:
    def __init__(self, content: str | list[str]):
        self.contents = content if isinstance(content, list) else [content]
        self.request = None
        self.requests = []

    def create(self, **kwargs):
        self.request = kwargs
        self.requests.append(kwargs)
        content = self.contents[min(len(self.requests) - 1, len(self.contents) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, content: str | list[str]):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


class ImplementationAgentToolTests(unittest.TestCase):
    def test_generates_and_compiles_factor_spec(self):
        proposal = {
            "name": "二十日低波动率因子",
            "description": "使用过去二十日收益率标准差刻画股票波动率。",
            "formula": {
                "op": "std",
                "window": 20,
                "args": [{"op": "return", "field": "close", "window": 1}],
            },
            "required_fields": ["close"],
            "lookback_days": 20,
            "higher_is_better": False,
            "rebalance_frequency": "weekly",
        }
        client = FakeClient(json.dumps(proposal, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary_directory:
            panel_path = Path(temporary_directory) / "panel.parquet"
            pl.DataFrame(
                {
                    "date": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                    "permno": [1, 1],
                    "close": [10.0, 10.5],
                }
            ).write_parquet(panel_path)
            state = create_initial_state(research_goal="测试 Implementation Agent")
            state["iteration"] = 1
            state["hypothesis"] = {
                "hypothesis_id": "H-RUN-0001",
                "title": "低波动率股票的短期收益优势",
                "claim": "低波动率股票未来五日收益较高。",
                "economic_rationale": "投资者可能高估高波动股票。",
                "prediction_direction": "negative",
                "prediction_horizon_days": 5,
                "required_concepts": ["volatility"],
            }

            result = ImplementationAgentTool(
                panel_path=panel_path,
                model="test-model",
                client=client,
            ).run(state)

        self.assertTrue(result.success, result.error)
        spec = FactorSpec.model_validate(result.payload["factor_spec"])
        self.assertEqual(spec.factor_id, "F-RUN-0001")
        self.assertEqual(spec.hypothesis_id, "H-RUN-0001")
        self.assertFalse(spec.higher_is_better)
        self.assertEqual(spec.portfolio_rules.top_k, 20)
        self.assertEqual(spec.portfolio_rules.exit_rank, 30)
        self.assertTrue(result.payload["implementation"]["unit_tests_passed"])
        self.assertEqual(
            client.chat.completions.request["response_format"],
            {"type": "json_object"},
        )
        request = json.loads(
            client.chat.completions.request["messages"][1]["content"]
        )
        self.assertEqual(
            request["implementation_context"]["hypothesis"]["hypothesis_id"],
            "H-RUN-0001",
        )
        self.assertEqual(
            request["implementation_context"]["task_mode"],
            "initial",
        )
        self.assertEqual(
            request["implementation_context"]["fixed_portfolio_rules"],
            {"top_k": 20, "exit_rank": 30},
        )

    def test_requires_hypothesis(self):
        result = ImplementationAgentTool(
            panel_path="unused.parquet",
            client=FakeClient("{}"),
        ).run(create_initial_state())

        self.assertFalse(result.success)
        self.assertIn("hypothesis", result.error)

    def test_retries_with_validation_feedback(self):
        invalid = {
            "name": "错误窗口因子",
            "description": "第一次输出故意把 window 放在 zscore 上。",
            "formula": {
                "op": "zscore",
                "window": 20,
                "field": "close",
            },
            "required_fields": ["close"],
            "lookback_days": 20,
            "higher_is_better": True,
            "rebalance_frequency": "weekly",
        }
        valid = {
            "name": "五日收益横截面标准分",
            "description": "先计算五日收益，再进行当日横截面标准化。",
            "formula": {
                "op": "zscore",
                "args": [{"op": "return", "field": "close", "window": 5}],
            },
            "required_fields": ["close"],
            "lookback_days": 5,
            "higher_is_better": True,
            "rebalance_frequency": "weekly",
        }
        client = FakeClient(
            [json.dumps(invalid), json.dumps(valid)]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            panel_path = Path(temporary_directory) / "panel.parquet"
            pl.DataFrame(
                {
                    "date": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                    "permno": [1, 1],
                    "close": [10.0, 10.5],
                }
            ).write_parquet(panel_path)
            state = create_initial_state()
            state["iteration"] = 1
            state["hypothesis"] = {
                "hypothesis_id": "H-RETRY-0001",
                "title": "收益延续",
                "claim": "近期收益较高的股票未来收益较高。",
                "economic_rationale": "信息逐步扩散。",
                "prediction_direction": "positive",
                "prediction_horizon_days": 5,
                "required_concepts": ["price_momentum"],
            }
            result = ImplementationAgentTool(
                panel_path=panel_path,
                client=client,
            ).run(state)

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.payload["implementation"]["generation_attempts"], 2)
        self.assertEqual(len(client.chat.completions.requests), 2)
        retry_message = client.chat.completions.requests[1]["messages"][-1][
            "content"
        ]
        self.assertIn("zscore does not accept window", retry_message)


if __name__ == "__main__":
    unittest.main()
