import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from evoalpha_workflow.seed_evaluation import (
    Alpha158SeedEvaluationConfig,
    Alpha158SeedEvaluator,
)
from factor_pipeline.alpha158 import alpha158_factor_spec, alpha158_formula_catalog
from factor_pipeline.config import FactorConfig
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
)


class Alpha158SeedEvaluatorTests(unittest.TestCase):
    def test_evaluates_specs_and_writes_reusable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            membership_path = root / "membership.parquet"
            output_dir = root / "results"
            start = datetime(2020, 1, 1)
            rows = []
            for day_index in range(100):
                for stock_index in range(30):
                    cycle = np.sin((day_index + stock_index * 1.7) / 8.0)
                    trend = day_index * (0.03 + stock_index * 0.0009)
                    close = 20.0 + stock_index * 0.4 + trend + cycle
                    rows.append(
                        {
                            "date": start + timedelta(days=day_index),
                            "permno": stock_index + 1,
                            "open": close * (0.995 + 0.0002 * (stock_index % 4)),
                            "high": close * (1.01 + 0.0001 * (day_index % 3)),
                            "low": close * (0.99 - 0.0001 * (stock_index % 3)),
                            "close": close,
                            "volume": 10_000.0
                            + stock_index * 100.0
                            + day_index * (20.0 + stock_index),
                            "mktcap": 1_000_000.0 + stock_index * 50_000.0,
                        }
                    )
            pl.DataFrame(rows).with_columns(
                pl.col("permno").cast(pl.Int32)
            ).write_parquet(panel_path)
            pl.DataFrame(
                {
                    "permno": pl.Series(range(1, 31), dtype=pl.Int32),
                    "start_date": [start] * 30,
                    "end_date": [start + timedelta(days=99)] * 30,
                }
            ).write_parquet(membership_path)
            formulas = dict(alpha158_formula_catalog())
            specs = [
                alpha158_factor_spec(name, formulas[name])
                for name in ("ROC5", "MA5", "STD5")
            ]
            evaluator = Alpha158SeedEvaluator(
                Alpha158SeedEvaluationConfig(
                    panel_path=panel_path,
                    membership_path=membership_path,
                    training_period=EvaluationPeriod(
                        start_date=date(2020, 2, 1),
                        end_date=date(2020, 4, 9),
                    ),
                    warmup_calendar_days=31,
                    factor_config=FactorConfig(minimum_cross_section=10),
                    evaluation_config=FactorEvaluationConfig(
                        minimum_cross_section=10,
                    ),
                    output_dir=output_dir,
                ),
                specs=specs,
            )

            candidates = evaluator.evaluate_seeds()

            self.assertEqual(len(candidates), 3)
            self.assertTrue(any(item.passed_hard_filters for item in candidates))
            self.assertIsNotNone(evaluator.last_score_reference)
            self.assertTrue((output_dir / "alpha158_seed_metrics.parquet").is_file())
            self.assertTrue((output_dir / "alpha158_seed_candidates.json").is_file())
            self.assertTrue((output_dir / "alpha158_score_reference.json").is_file())


if __name__ == "__main__":
    unittest.main()
