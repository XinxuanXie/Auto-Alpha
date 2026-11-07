import unittest
from datetime import date, datetime

import polars as pl

from evoalpha_workflow.alpha158_benchmark import (
    BenchmarkWindow,
    aggregate_backtest_window,
    aggregate_prediction_window,
    build_benchmark_windows,
)


class Alpha158BenchmarkTests(unittest.TestCase):
    def test_builds_annual_and_rolling_windows_with_partial_last_year(self):
        windows = build_benchmark_windows(
            date(2000, 1, 3),
            date(2026, 9, 30),
        )

        counts = {
            kind: sum(item.window_type == kind for item in windows)
            for kind in ("annual", "rolling_5y", "rolling_10y", "rolling_20y")
        }
        self.assertEqual(
            counts,
            {"annual": 27, "rolling_5y": 23, "rolling_10y": 18, "rolling_20y": 8},
        )
        self.assertEqual(len(windows), 76)
        self.assertTrue(
            next(
                item
                for item in windows
                if item.window_type == "rolling_20y" and item.end_year == 2026
            ).is_partial
        )

    def test_aggregates_prediction_direction_and_coverage(self):
        daily = pl.DataFrame(
            {
                "date": [datetime(2020, 1, day) for day in range(2, 6)],
                "eligible_stocks_5d": [100] * 4,
                "valid_stocks_5d": [100] * 4,
                "target_date_5d": [datetime(2020, 1, day + 5) for day in range(2, 6)],
                "rank_ic_5d": [-0.1, -0.2, -0.1, -0.2],
                "eligible_stocks_10d": [100] * 4,
                "valid_stocks_10d": [100] * 4,
                "target_date_10d": [datetime(2020, 1, day + 10) for day in range(2, 6)],
                "rank_ic_10d": [-0.2, -0.3, -0.2, -0.3],
                "eligible_stocks_20d": [100] * 4,
                "valid_stocks_20d": [90] * 4,
                "target_date_20d": [datetime(2020, 2, day) for day in range(2, 6)],
                "rank_ic_20d": [-0.3, -0.1, -0.2, -0.4],
            }
        )
        window = BenchmarkWindow(
            "annual", 1, 2020, date(2020, 1, 1), date(2020, 12, 31), False
        )

        metrics = aggregate_prediction_window(daily, window)

        self.assertEqual(metrics["direction"], -1)
        self.assertAlmostEqual(metrics["mean_rank_ic_20d"], 0.25)
        self.assertAlmostEqual(metrics["median_stock_coverage"], 0.9)
        self.assertEqual(metrics["cross_horizon_consistency"], 1.0)

    def test_aggregates_backtest_for_selected_direction(self):
        positive = pl.DataFrame(
            {
                "signal_date": [datetime(2020, 1, 3), datetime(2020, 1, 10)],
                "positions": [20, 20],
                "turnover": [1.0, 0.5],
                "gross_return": [0.01, 0.02],
                "net_return": [0.009, 0.0195],
            }
        )
        negative = positive.with_columns(
            (-pl.col("gross_return")).alias("gross_return"),
            (-pl.col("net_return")).alias("net_return"),
        )
        window = BenchmarkWindow(
            "annual", 1, 2020, date(2020, 1, 1), date(2020, 12, 31), False
        )

        result = aggregate_backtest_window(positive, negative, window, -1)

        self.assertEqual(result["backtest_periods"], 2)
        self.assertLess(result["cumulative_net_return"], 0)
        self.assertAlmostEqual(result["average_turnover"], 0.75)


if __name__ == "__main__":
    unittest.main()
