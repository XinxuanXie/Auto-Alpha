import unittest
from datetime import datetime, timedelta

import polars as pl

from factor_pipeline.alpha158 import (
    alpha158_factor_spec,
    alpha158_formula_catalog,
    alpha158_support_manifest,
    executable_alpha158_specs,
)
from factor_pipeline.compiler import compile_factor_spec


class Alpha158CatalogTests(unittest.TestCase):
    def test_catalog_contains_158_and_executable_set_excludes_only_vwap(self):
        catalog = alpha158_formula_catalog()
        specs = executable_alpha158_specs()

        self.assertEqual(len(catalog), 158)
        self.assertEqual(len(specs), 157)
        self.assertNotIn("ALPHA158-VWAP0", {spec.factor_id for spec in specs})
        self.assertEqual(len({spec.factor_id for spec in specs}), 157)
        manifest = alpha158_support_manifest()
        excluded = [record for record in manifest if not record.supported]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].factor_id, "ALPHA158-VWAP0")
        self.assertIn("true VWAP", excluded[0].reason)

    def test_vwap_formula_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "true VWAP"):
            alpha158_factor_spec("VWAP0", "$vwap/$close")


class Alpha158CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = datetime(2020, 1, 1)
        rows = []
        for day_index in range(70):
            for stock_index, permno in enumerate((1, 2, 3)):
                close = 20.0 + stock_index * 5.0 + day_index * (0.1 + 0.03 * stock_index)
                rows.append(
                    {
                        "date": start + timedelta(days=day_index),
                        "permno": permno,
                        "open": close - 0.2 + 0.03 * ((day_index + stock_index) % 3),
                        "high": close + 0.8 + 0.01 * stock_index,
                        "low": close - 0.9 - 0.01 * stock_index,
                        "close": close + 0.05 * ((day_index + stock_index) % 5),
                        "volume": 1_000.0 + day_index * (11 + stock_index) + stock_index * 100,
                        "mktcap": 1_000_000.0 + stock_index * 100_000.0,
                    }
                )
        cls.panel = pl.DataFrame(rows).lazy()

    def test_all_157_specs_compile_and_produce_values(self):
        for spec in executable_alpha158_specs():
            with self.subTest(factor_id=spec.factor_id):
                result = compile_factor_spec(self.panel, spec).select("raw_factor").collect()
                self.assertGreater(result.get_column("raw_factor").drop_nulls().len(), 0)

    def test_elementwise_max_uses_qlib_greater_semantics(self):
        spec = alpha158_factor_spec("KUP", "($high-Greater($open,$close))/$open")
        result = compile_factor_spec(self.panel, spec).collect()
        first = result.sort("permno", "date").row(0, named=True)
        expected = (first["high"] - max(first["open"], first["close"])) / first["open"]

        self.assertAlmostEqual(first["raw_factor"], expected)

    def test_time_series_rank_and_index_are_one_based(self):
        rank_spec = alpha158_factor_spec("RANK5", "Rank($close,5)")
        index_spec = alpha158_factor_spec("IMAX5", "IdxMax($high,5)/5")

        rank = compile_factor_spec(self.panel, rank_spec).collect().filter(pl.col("permno") == 1)
        index = compile_factor_spec(self.panel, index_spec).collect().filter(pl.col("permno") == 1)

        self.assertEqual(rank[4, "raw_factor"], 1.0)
        self.assertEqual(index[4, "raw_factor"], 1.0)

    def test_rolling_regression_statistics_match_linear_series(self):
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=day) for day in range(5)],
                "permno": [1] * 5,
                "close": [1.0, 3.0, 5.0, 7.0, 9.0],
            }
        ).lazy()
        slope = compile_factor_spec(
            panel,
            alpha158_factor_spec("TEST-SLOPE", "Slope($close,5)"),
        ).collect()
        r_square = compile_factor_spec(
            panel,
            alpha158_factor_spec("TEST-RSQUARE", "Rsquare($close,5)"),
        ).collect()
        residual = compile_factor_spec(
            panel,
            alpha158_factor_spec("TEST-RESIDUAL", "Resi($close,5)"),
        ).collect()

        self.assertAlmostEqual(slope[-1, "raw_factor"], 2.0)
        self.assertAlmostEqual(r_square[-1, "raw_factor"], 1.0)
        self.assertAlmostEqual(residual[-1, "raw_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
