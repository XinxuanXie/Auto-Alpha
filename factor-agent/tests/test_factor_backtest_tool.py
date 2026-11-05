import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.state import WorkflowStage, create_initial_state
from agentic_workflow.tools import FactorBacktestTool
from factor_pipeline.spec import FactorSpec


def write_test_data(root: Path) -> tuple[Path, Path]:
    panel_rows = []
    start = datetime(2014, 11, 1)
    for day_index in range(180):
        timestamp = start + timedelta(days=day_index)
        for stock_index in range(60):
            base = 20.0 + stock_index
            panel_rows.append(
                {
                    "date": timestamp,
                    "permno": stock_index,
                    "open": base + day_index * (stock_index + 1) * 0.001,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base
                    + day_index * (stock_index + 1) * 0.001
                    + ((day_index + stock_index) % 7) * 0.01,
                    "volume": 1_000.0
                    + stock_index * 10.0
                    + (day_index % 13) * 5.0,
                    "mktcap": 100_000.0 + stock_index * 1_000.0,
                }
            )
    panel_path = root / "panel.parquet"
    pl.DataFrame(panel_rows).with_columns(
        pl.col("permno").cast(pl.Int32)
    ).write_parquet(panel_path)
    membership_path = root / "membership.parquet"
    pl.DataFrame(
        {
            "permno": pl.Series(range(60), dtype=pl.Int32),
            "start_date": [datetime(2014, 1, 1)] * 60,
            "end_date": [datetime(2016, 1, 1)] * 60,
        }
    ).write_parquet(membership_path)
    return panel_path, membership_path


def test_factor_spec() -> dict:
    return FactorSpec.model_validate(
        {
            "factor_id": "F-TOOL-TEST",
            "name": "工具测试因子",
            "description": "使用两日价格反转验证真实回测工具。",
            "hypothesis_id": "H-TOOL-TEST",
            "formula": {
                "op": "negate",
                "args": [
                    {"op": "return", "field": "close", "window": 2}
                ],
            },
            "required_fields": ["close"],
            "lookback_days": 2,
            "higher_is_better": True,
            "rebalance_frequency": "weekly",
            "universe_filters": {
                "min_price": 5.0,
                "min_market_cap_percentile": 0.1,
            },
        }
    ).model_dump(mode="json", exclude_none=True)


class FactorBacktestToolTests(unittest.TestCase):
    def test_requires_factor_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = FactorBacktestTool(
                root / "panel.parquet",
                root / "membership.parquet",
                root / "artifacts",
            )

            result = tool.run(create_initial_state())

            self.assertFalse(result.success)
            self.assertEqual(
                result.error,
                "A FactorSpec is required before backtesting",
            )

    def test_runs_real_pipeline_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path, membership_path = write_test_data(root)
            tool = FactorBacktestTool(
                panel_path,
                membership_path,
                root / "artifacts",
                start_date=date(2015, 1, 1),
                end_date=date(2015, 3, 31),
                top_k=20,
                transaction_cost_bps=10.0,
            )
            state = create_initial_state()
            state["iteration"] = 1
            state["attempt"] = 2
            state["factor_spec"] = test_factor_spec()

            result = tool.run(state)

            self.assertTrue(result.success, result.error)
            backtest = result.payload["backtest_result"]
            self.assertEqual(backtest["factor_id"], "F-TOOL-TEST")
            self.assertGreater(backtest["periods"], 0)
            self.assertEqual(backtest["top_k"], 20)
            self.assertEqual(backtest["exit_rank"], 30)
            self.assertEqual(backtest["universe_filters"]["min_price"], 5.0)
            self.assertEqual(backtest["transaction_cost_bps"], 10.0)
            self.assertEqual(backtest["rank_ic"], backtest["mean_rank_ic"])
            self.assertEqual(backtest["net_return"], backtest["cumulative_net_return"])
            periods_path = Path(backtest["periods_artifact"])
            summary_path = Path(backtest["summary_artifact"])
            self.assertTrue(periods_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(periods_path.parent.name, "attempt_0002")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["factor_spec"]["factor_id"], "F-TOOL-TEST")

    def test_manager_runs_real_backtest_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path, membership_path = write_test_data(root)
            tool = FactorBacktestTool(
                panel_path,
                membership_path,
                root / "artifacts",
                start_date=date(2015, 1, 1),
                end_date=date(2015, 3, 31),
            )
            manager = build_manager(ManagerTools.mock(backtest=tool))

            result = manager.invoke(create_initial_state(max_iterations=1))

            self.assertEqual(result["stage"], WorkflowStage.COMPLETED)
            self.assertEqual(
                result["backtest_result"]["factor_id"],
                "F-0001",
            )
            self.assertGreater(result["backtest_result"]["periods"], 0)
            self.assertIsNotNone(result["manager_decision"])


if __name__ == "__main__":
    unittest.main()
