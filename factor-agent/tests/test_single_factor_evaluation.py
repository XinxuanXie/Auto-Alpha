import unittest
from datetime import date, datetime, timedelta

import polars as pl

from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    build_score_reference,
    build_forward_return_labels,
    evaluate_single_factor,
    score_factor_metrics,
)


class SingleFactorEvaluationTests(unittest.TestCase):
    def setUp(self):
        start = datetime(2020, 1, 1)
        rows = []
        for day_index in range(45):
            for stock_index in range(6):
                rows.append(
                    {
                        "date": start + timedelta(days=day_index),
                        "permno": stock_index + 1,
                        "close": 100.0
                        + stock_index * (day_index + 1)
                        + day_index,
                    }
                )
        self.panel = pl.DataFrame(rows)
        self.period = EvaluationPeriod(date(2020, 1, 1), date(2020, 2, 14))
        self.labels = build_forward_return_labels(
            self.panel,
            self.period,
            horizons=(5, 10, 20),
        )
        self.config = FactorEvaluationConfig(
            minimum_cross_section=3,
            minimum_valid_day_ratio=0.8,
            minimum_median_stock_coverage=0.8,
        )

    def factor_signal(self, reverse_every: int, negate: bool = False) -> pl.DataFrame:
        instrument = pl.col("permno").cast(pl.Float64)
        signal = (
            pl.when((pl.col("date").dt.day() % reverse_every) == 0)
            .then(-instrument)
            .otherwise(instrument)
        )
        if negate:
            signal = -signal
        return self.labels.select("date", "permno").with_columns(
            signal.alias("factor")
        )

    def test_labels_do_not_cross_period_end(self):
        last = self.labels.sort("date").tail(1)

        self.assertIsNone(last[0, "target_date_20d"])
        self.assertIsNone(last[0, "forward_return_20d"])

    def test_negative_factor_direction_is_flipped_from_primary_horizon(self):
        values = self.factor_signal(reverse_every=5, negate=True)

        result = evaluate_single_factor(
            values,
            self.labels,
            "NEGATIVE-SIGNAL",
            self.period,
            self.config,
        )

        self.assertEqual(result.direction, -1)
        self.assertGreater(result.mean_rank_ic_20d, 0)
        self.assertTrue(result.passed_hard_filters)

    def test_constant_factor_fails_hard_filters(self):
        values = self.labels.select("date", "permno").with_columns(
            pl.lit(1.0).alias("factor")
        )

        result = evaluate_single_factor(
            values,
            self.labels,
            "CONSTANT",
            self.period,
            self.config,
        )

        self.assertFalse(result.passed_hard_filters)
        self.assertIn("invalid_primary_rank_ic", result.rejection_reasons)

    def test_scores_valid_factors_against_fixed_reference_population(self):
        positive = self.factor_signal(reverse_every=5)
        weaker = self.factor_signal(reverse_every=3)
        metrics = [
            evaluate_single_factor(
                positive,
                self.labels,
                "STRONG",
                self.period,
                self.config,
            ),
            evaluate_single_factor(
                weaker,
                self.labels,
                "WEAKER",
                self.period,
                self.config,
            ),
        ]

        scored = score_factor_metrics(metrics)

        self.assertTrue(all(item.composite_score is not None for item in scored))
        self.assertGreaterEqual(
            scored[0].composite_score,
            scored[1].composite_score,
        )

        reference = build_score_reference(metrics)
        rescored = score_factor_metrics([metrics[0]], reference=reference)
        self.assertEqual(
            rescored[0].composite_score,
            scored[0].composite_score,
        )
        self.assertLessEqual(rescored[0].rank_ic_percentile, 1.0)


if __name__ == "__main__":
    unittest.main()
