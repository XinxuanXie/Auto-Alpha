import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from factor_pipeline.backtest import (
    _one_way_turnover,
    build_weekly_periods,
    run_weekly_topk_backtest,
)
from factor_pipeline.config import BacktestConfig, DataConfig, FactorConfig
from factor_pipeline.data import DataValidationError, PanelDataSource
from factor_pipeline.factor import (
    build_abnormal_volume_reversal,
    build_factor_from_spec,
)
from factor_pipeline.spec import FactorSpec


class PanelDataSourceTests(unittest.TestCase):
    def test_validate_and_point_in_time_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            membership_path = root / "membership.parquet"
            dates = [datetime(2020, 1, day) for day in range(1, 6)]
            rows = []
            for permno in [1, 2]:
                for timestamp in dates:
                    rows.append(
                        {
                            "date": timestamp,
                            "permno": permno,
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.0,
                            "volume": 100.0,
                            "mktcap": 1_000.0,
                        }
                    )
            pl.DataFrame(rows).with_columns(
                pl.col("permno").cast(pl.Int32)
            ).write_parquet(panel_path)
            pl.DataFrame(
                {
                    "permno": pl.Series([1, 2], dtype=pl.Int32),
                    "start_date": [datetime(2020, 1, 2), datetime(2020, 1, 4)],
                    "end_date": [datetime(2020, 1, 3), datetime(2020, 1, 5)],
                }
            ).write_parquet(membership_path)
            source = PanelDataSource(
                DataConfig(
                    panel_path=panel_path,
                    membership_path=membership_path,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 1, 5),
                    warmup_calendar_days=0,
                    exit_buffer_calendar_days=0,
                )
            )

            summary = source.validate()
            history = source.panel_history().collect()
            frame = source.point_in_time_panel().collect()

            self.assertEqual(summary["rows"], 10)
            self.assertEqual(history.height, 10)
            self.assertEqual(frame.height, 4)
            self.assertEqual(
                set(frame.select("date", "permno").iter_rows()),
                {
                    (datetime(2020, 1, 2), 1),
                    (datetime(2020, 1, 3), 1),
                    (datetime(2020, 1, 4), 2),
                    (datetime(2020, 1, 5), 2),
                },
            )

    def test_validate_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            membership_path = root / "membership.parquet"
            pl.DataFrame({"date": [datetime(2020, 1, 1)]}).write_parquet(
                panel_path
            )
            pl.DataFrame(
                {
                    "permno": pl.Series([1], dtype=pl.Int32),
                    "start_date": [datetime(2020, 1, 1)],
                    "end_date": [datetime(2020, 1, 2)],
                }
            ).write_parquet(membership_path)
            source = PanelDataSource(
                DataConfig(
                    panel_path=panel_path,
                    membership_path=membership_path,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 1, 2),
                )
            )

            with self.assertRaises(DataValidationError):
                source.validate()


class FactorProcessingTests(unittest.TestCase):
    def test_factor_is_standardized_and_size_neutral(self):
        rows = []
        start = datetime(2020, 1, 1)
        for day_index in range(35):
            timestamp = start + timedelta(days=day_index)
            for stock_index in range(60):
                rows.append(
                    {
                        "date": timestamp,
                        "permno": stock_index,
                        "open": 100.0 + stock_index,
                        "close": 100.0
                        + stock_index
                        + day_index * (0.05 + stock_index * 0.0007)
                        + np.sin((day_index + stock_index) / 4.0),
                        "volume": 1_000.0
                        + stock_index * 13.0
                        + day_index * 7.0
                        + 100.0 * np.cos((day_index * stock_index + 1) / 11.0),
                        "mktcap": float(np.exp(8.0 + stock_index / 20.0)),
                    }
                )
        panel = pl.DataFrame(rows).lazy()
        factor = build_abnormal_volume_reversal(
            panel,
            FactorConfig(
                minimum_cross_section=50,
            ),
        ).collect()
        latest = factor.filter(pl.col("date") == factor.get_column("date").max())
        factor_values = latest.get_column("factor").to_numpy()
        log_size = np.log(latest.get_column("mktcap").to_numpy())

        self.assertEqual(latest.height, 60)
        self.assertAlmostEqual(float(np.mean(factor_values)), 0.0, places=10)
        self.assertAlmostEqual(float(np.std(factor_values)), 1.0, places=10)
        self.assertAlmostEqual(float(np.cov(factor_values, log_size)[0, 1]), 0.0, places=8)

    def test_universe_filters_apply_before_factor_cleaning(self):
        rows = []
        start = datetime(2020, 1, 1)
        for day_index in range(25):
            for stock_index in range(60):
                rows.append(
                    {
                        "date": start + timedelta(days=day_index),
                        "permno": stock_index,
                        "open": 10.0 + stock_index,
                        "close": 10.0
                        + stock_index
                        + np.sin((day_index + stock_index) / 5.0),
                        "volume": 1_000.0 + stock_index * 100.0,
                        "mktcap": 10_000.0 + stock_index * 1_000.0,
                    }
                )
        spec = FactorSpec.model_validate(
            {
                "factor_id": "F-FILTER-TEST",
                "name": "股票池筛选测试",
                "description": "验证价格、流动性和市值筛选。",
                "formula": {"op": "field", "field": "close"},
                "required_fields": ["close"],
                "lookback_days": 19,
                "universe_filters": {
                    "min_price": 30.0,
                    "min_liquidity_percentile": 0.5,
                    "liquidity_lookback_days": 20,
                    "min_market_cap_percentile": 0.5,
                },
            }
        )

        result = build_factor_from_spec(
            pl.DataFrame(rows).lazy(),
            spec,
            FactorConfig(minimum_cross_section=20),
        ).collect()
        latest = result.filter(pl.col("date") == result["date"].max())

        self.assertGreater(latest.height, 0)
        self.assertLess(latest.height, 60)
        self.assertGreaterEqual(float(latest["close"].min()), 30.0)


class BacktestTests(unittest.TestCase):
    def test_weekly_period_uses_next_trading_day_open(self):
        calendar = [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 6),
            datetime(2020, 1, 10),
            datetime(2020, 1, 13),
            datetime(2020, 1, 17),
            datetime(2020, 1, 21),
        ]

        periods = build_weekly_periods(calendar, date(2020, 1, 1), date(2020, 1, 10))

        self.assertEqual(periods.height, 2)
        self.assertEqual(periods[0, "signal_date"], datetime(2020, 1, 3))
        self.assertEqual(periods[0, "entry_date"], datetime(2020, 1, 6))
        self.assertEqual(periods[0, "exit_date"], datetime(2020, 1, 13))

    def test_one_way_turnover_for_equal_weight_portfolios(self):
        self.assertEqual(_one_way_turnover({}, {1: 0.5, 2: 0.5}), 1.0)
        self.assertEqual(
            _one_way_turnover({1: 0.5, 2: 0.5}, {2: 0.5, 3: 0.5}),
            0.5,
        )

    def test_backtest_selects_top_k_and_applies_cost(self):
        calendar = [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 6),
            datetime(2020, 1, 10),
            datetime(2020, 1, 13),
            datetime(2020, 1, 17),
            datetime(2020, 1, 21),
            datetime(2020, 1, 24),
            datetime(2020, 1, 27),
        ]
        score_rows = []
        for signal_date, ranking in [
            (datetime(2020, 1, 3), {1: 3.0, 2: 2.0, 3: 1.0}),
            (datetime(2020, 1, 10), {1: 1.0, 2: 3.0, 3: 2.0}),
        ]:
            for permno, factor in ranking.items():
                score_rows.append(
                    {
                        "date": signal_date,
                        "permno": permno,
                        "factor": factor,
                        "raw_factor": factor,
                    }
                )
        price_rows = []
        for timestamp, prices in [
            (datetime(2020, 1, 6), {1: 100.0, 2: 100.0, 3: 100.0}),
            (datetime(2020, 1, 13), {1: 110.0, 2: 100.0, 3: 90.0}),
            (datetime(2020, 1, 21), {1: 110.0, 2: 105.0, 3: 99.0}),
        ]:
            for permno, opening_price in prices.items():
                price_rows.append(
                    {"date": timestamp, "permno": permno, "open": opening_price}
                )

        result, periods = run_weekly_topk_backtest(
            factor_frame=pl.DataFrame(score_rows).lazy(),
            price_panel=pl.DataFrame(price_rows).lazy(),
            calendar=calendar,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 10),
            config=BacktestConfig(top_k=2, transaction_cost_bps=10.0),
        )

        self.assertEqual(result.periods, 2)
        self.assertEqual(result.selected_positions, 4)
        self.assertEqual(result.missing_price_positions, 0)
        self.assertAlmostEqual(periods[0, "turnover"], 1.0)
        self.assertAlmostEqual(periods[1, "turnover"], 0.5)
        self.assertAlmostEqual(periods[0, "gross_return"], 0.05)
        self.assertAlmostEqual(periods[0, "net_return"], 0.049)

    def test_exit_buffer_retains_positions_inside_exit_rank(self):
        calendar = [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 6),
            datetime(2020, 1, 10),
            datetime(2020, 1, 13),
            datetime(2020, 1, 17),
            datetime(2020, 1, 21),
        ]
        scores = pl.DataFrame(
            [
                {"date": datetime(2020, 1, 3), "permno": 1, "factor": 3.0, "raw_factor": 3.0},
                {"date": datetime(2020, 1, 3), "permno": 2, "factor": 2.0, "raw_factor": 2.0},
                {"date": datetime(2020, 1, 3), "permno": 3, "factor": 1.0, "raw_factor": 1.0},
                {"date": datetime(2020, 1, 10), "permno": 1, "factor": 1.0, "raw_factor": 1.0},
                {"date": datetime(2020, 1, 10), "permno": 2, "factor": 3.0, "raw_factor": 3.0},
                {"date": datetime(2020, 1, 10), "permno": 3, "factor": 2.0, "raw_factor": 2.0},
            ]
        ).lazy()
        prices = pl.DataFrame(
            [
                {"date": date_value, "permno": permno, "open": 100.0}
                for date_value in [
                    datetime(2020, 1, 6),
                    datetime(2020, 1, 13),
                    datetime(2020, 1, 21),
                ]
                for permno in [1, 2, 3]
            ]
        ).lazy()

        _, periods = run_weekly_topk_backtest(
            factor_frame=scores,
            price_panel=prices,
            calendar=calendar,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 10),
            config=BacktestConfig(
                top_k=2,
                exit_rank=3,
                transaction_cost_bps=10.0,
            ),
        )

        self.assertAlmostEqual(periods[0, "turnover"], 1.0)
        self.assertAlmostEqual(periods[1, "turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
