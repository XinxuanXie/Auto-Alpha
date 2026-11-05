import unittest
from datetime import datetime, timedelta

import polars as pl
from pydantic import ValidationError

from factor_pipeline.compiler import FactorCompilationError, compile_factor_spec
from factor_pipeline.spec import FactorSpec, FormulaNode


def make_spec(**overrides) -> FactorSpec:
    payload = {
        "factor_id": "F-TEST",
        "name": "测试因子",
        "description": "用于验证结构化公式编译。",
        "formula": {"op": "return", "field": "close", "window": 2},
        "required_fields": ["close"],
        "lookback_days": 2,
        "higher_is_better": True,
        "rebalance_frequency": "weekly",
    }
    payload.update(overrides)
    return FactorSpec.model_validate(payload)


class FactorSpecTests(unittest.TestCase):
    def test_accepts_structured_return_formula(self):
        spec = make_spec()

        self.assertEqual(spec.formula.op.value, "return")
        self.assertEqual(spec.formula.field.value, "close")
        self.assertEqual(spec.formula.required_history(), 2)
        self.assertEqual(spec.portfolio_rules.top_k, 20)
        self.assertEqual(spec.portfolio_rules.exit_rank, 30)

    def test_rejects_unknown_operator(self):
        with self.assertRaises(ValidationError):
            FormulaNode.model_validate(
                {"op": "future_return", "field": "close", "window": 5}
            )

    def test_rejects_incorrect_required_fields(self):
        with self.assertRaises(ValidationError):
            make_spec(required_fields=["close", "volume"])

    def test_rejects_insufficient_lookback(self):
        with self.assertRaises(ValidationError):
            make_spec(lookback_days=1)

    def test_rejects_extra_node_properties(self):
        with self.assertRaises(ValidationError):
            FormulaNode.model_validate(
                {
                    "op": "return",
                    "field": "close",
                    "window": 5,
                    "python_code": "import os",
                }
            )

    def test_accepts_universe_filters_and_portfolio_rules(self):
        spec = make_spec(
            lookback_days=19,
            universe_filters={
                "min_price": 5.0,
                "min_liquidity_percentile": 0.2,
                "liquidity_lookback_days": 20,
                "min_market_cap_percentile": 0.1,
            },
            portfolio_rules={"top_k": 20, "exit_rank": 30},
        )

        self.assertEqual(spec.universe_filters.min_price, 5.0)
        self.assertEqual(spec.portfolio_rules.exit_rank, 30)

    def test_rejects_filter_lookback_mismatch(self):
        with self.assertRaises(ValidationError):
            make_spec(
                lookback_days=10,
                universe_filters={
                    "min_liquidity_percentile": 0.2,
                    "liquidity_lookback_days": 20,
                },
            )


class FactorCompilerTests(unittest.TestCase):
    def setUp(self):
        rows = []
        start = datetime(2020, 1, 1)
        for day_index in range(5):
            for stock_index, permno in enumerate([1, 2, 3]):
                rows.append(
                    {
                        "date": start + timedelta(days=day_index),
                        "permno": permno,
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.0 + day_index * (stock_index + 1),
                        "volume": 100.0 + stock_index,
                        "mktcap": 1_000.0 + stock_index,
                    }
                )
        self.panel = pl.DataFrame(rows).lazy()

    def test_compiles_return_formula(self):
        result = compile_factor_spec(self.panel, make_spec()).collect()
        stock = result.filter(pl.col("permno") == 2).sort("date")

        self.assertIsNone(stock[0, "raw_factor"])
        self.assertIsNone(stock[1, "raw_factor"])
        self.assertAlmostEqual(stock[2, "raw_factor"], 14.0 / 10.0 - 1.0)

    def test_applies_higher_is_better_direction(self):
        result = compile_factor_spec(
            self.panel,
            make_spec(higher_is_better=False),
        ).collect()
        stock = result.filter(pl.col("permno") == 2).sort("date")

        self.assertAlmostEqual(stock[2, "raw_factor"], -(14.0 / 10.0 - 1.0))

    def test_compiles_cross_sectional_rank(self):
        spec = make_spec(
            formula={
                "op": "rank",
                "args": [
                    {"op": "return", "field": "close", "window": 1}
                ],
            },
            lookback_days=1,
        )
        result = compile_factor_spec(self.panel, spec).collect()
        latest = result.filter(pl.col("date") == datetime(2020, 1, 5)).sort(
            "permno"
        )

        self.assertEqual(
            latest.get_column("raw_factor").to_list(),
            [1 / 3, 2 / 3, 1.0],
        )

    def test_compiles_remaining_supported_operators(self):
        formulas = [
            ({"op": "lag", "field": "close", "window": 1}, 1),
            ({"op": "mean", "field": "close", "window": 2}, 1),
            ({"op": "std", "field": "close", "window": 2}, 1),
            ({"op": "zscore", "field": "close"}, 0),
            (
                {
                    "op": "add",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 1.0},
                    ],
                },
                0,
            ),
            (
                {
                    "op": "subtract",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 1.0},
                    ],
                },
                0,
            ),
            (
                {
                    "op": "multiply",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 2.0},
                    ],
                },
                0,
            ),
            (
                {
                    "op": "divide",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 2.0},
                    ],
                },
                0,
            ),
            (
                {
                    "op": "less_than",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 15.0},
                    ],
                },
                0,
            ),
            (
                {
                    "op": "negate",
                    "args": [{"op": "field", "field": "close"}],
                },
                0,
            ),
        ]
        for formula, lookback in formulas:
            with self.subTest(operator=formula["op"]):
                result = compile_factor_spec(
                    self.panel,
                    make_spec(formula=formula, lookback_days=lookback),
                ).collect()
                self.assertGreater(
                    result.get_column("raw_factor").drop_nulls().len(),
                    0,
                )

    def test_compiles_conditional_operators(self):
        spec = make_spec(
            formula={
                "op": "where",
                "args": [
                    {
                        "op": "greater_than",
                        "args": [
                            {"op": "field", "field": "close"},
                            {"op": "constant", "value": 15.0},
                        ],
                    },
                    {"op": "field", "field": "close"},
                    {"op": "constant", "value": 0.0},
                ],
            },
            lookback_days=0,
        )

        result = compile_factor_spec(self.panel, spec).collect()
        latest = result.filter(pl.col("date") == datetime(2020, 1, 5)).sort(
            "permno"
        )

        self.assertEqual(
            latest.get_column("raw_factor").to_list(),
            [0.0, 18.0, 22.0],
        )

    def test_rejects_missing_panel_field(self):
        panel = self.panel.select(
            "date", "permno", "open", "high", "low", "volume", "mktcap"
        )

        with self.assertRaises(FactorCompilationError):
            compile_factor_spec(panel, make_spec())


if __name__ == "__main__":
    unittest.main()
