import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from evoalpha_workflow.similarity import (
    SimilarityConfig,
    TrainingFactorSimilarityAnalyzer,
)
from factor_pipeline.config import FactorConfig
from factor_pipeline.single_factor_evaluation import EvaluationPeriod
from factor_pipeline.spec import FactorSpec


def make_spec(factor_id: str, formula: dict, fields: list[str]) -> FactorSpec:
    return FactorSpec.model_validate(
        {
            "factor_id": factor_id,
            "name": factor_id,
            "description": "Similarity test factor.",
            "formula": formula,
            "required_fields": fields,
            "lookback_days": 0,
        }
    )


class FactorSimilarityTests(unittest.TestCase):
    def test_computes_daily_cross_sectional_median_spearman(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            membership_path = root / "membership.parquet"
            output_dir = root / "results"
            start = datetime(2020, 1, 1)
            rows = []
            for day_index in range(50):
                for stock_index in range(30):
                    close = (
                        20.0
                        + stock_index * 0.5
                        + day_index * 0.05
                        + np.sin((stock_index + day_index) / 5.0)
                    )
                    rows.append(
                        {
                            "date": start + timedelta(days=day_index),
                            "permno": stock_index + 1,
                            "open": close * 0.99,
                            "high": close * 1.01,
                            "low": close * 0.98,
                            "close": close,
                            "volume": 1_000.0
                            + ((stock_index * 17 + day_index * 13) % 29) * 100.0,
                            "mktcap": 1_000_000.0 + stock_index * 70_000.0,
                        }
                    )
            pl.DataFrame(rows).with_columns(
                pl.col("permno").cast(pl.Int32)
            ).write_parquet(panel_path)
            pl.DataFrame(
                {
                    "permno": pl.Series(range(1, 31), dtype=pl.Int32),
                    "start_date": [start] * 30,
                    "end_date": [start + timedelta(days=49)] * 30,
                }
            ).write_parquet(membership_path)
            close = {"op": "field", "field": "close"}
            specs = [
                make_spec("A", close, ["close"]),
                make_spec(
                    "B",
                    {
                        "op": "multiply",
                        "args": [close, {"op": "constant", "value": 2.0}],
                    },
                    ["close"],
                ),
                make_spec("C", {"op": "field", "field": "volume"}, ["volume"]),
            ]
            result = TrainingFactorSimilarityAnalyzer(
                SimilarityConfig(
                    panel_path=panel_path,
                    membership_path=membership_path,
                    training_period=EvaluationPeriod(
                        start_date=date(2020, 1, 10),
                        end_date=date(2020, 2, 19),
                    ),
                    warmup_calendar_days=9,
                    minimum_cross_section=10,
                    factor_config=FactorConfig(minimum_cross_section=10),
                    output_dir=output_dir,
                )
            ).analyze_specs(specs)

            self.assertEqual(len(result.pairs), 3)
            pair_ab = next(
                pair
                for pair in result.pairs
                if {pair.factor_a, pair.factor_b} == {"A", "B"}
            )
            self.assertAlmostEqual(pair_ab.median_absolute_spearman, 1.0)
            self.assertFalse(pair_ab.eligible_for_parent_pairing)
            self.assertTrue((output_dir / "top30_pair_similarity.parquet").is_file())
            self.assertTrue((output_dir / "top30_similarity_matrix.csv").is_file())


if __name__ == "__main__":
    unittest.main()
