import unittest

from pydantic import ValidationError

from hypothesis_workflow_v2.contracts import (
    BacktestProtocol,
    DevelopedHypothesis,
    ExperimentDecision,
    FactorExperimentSpec,
    FactorFormulaSpecV2,
    HypothesisCheckResult,
    MeasurementPlan,
    SplitMetrics,
)


def _factor() -> FactorFormulaSpecV2:
    return FactorFormulaSpecV2.model_validate(
        {
            "factor_id": "F-MOM-001-E01",
            "hypothesis_id": "H-MOM-001",
            "name": "five-day return",
            "description": "Recent price movement as a raw signal.",
            "formula": {"op": "return", "field": "close", "window": 5},
            "required_fields": ["close"],
            "lookback_days": 5,
        }
    )


class HypothesisV2ContractTest(unittest.TestCase):
    def test_canonicalizes_horizons_and_concepts(self) -> None:
        hypothesis = DevelopedHypothesis(
            hypothesis_id="H-MOM-001",
            idea_id="I-MOM-001",
            direction="momentum",
            title="Conditional continuation",
            claim="Price continuation may vary across prediction horizons.",
            economic_rationale="Information may diffuse at different speeds.",
            core_mechanism="Delayed information diffusion creates continuation.",
            counterargument="The pattern may only reflect common risk exposure.",
            falsification_criteria=["No incremental effect beyond recent return."],
            distinction_from_siblings="This hypothesis tests delayed diffusion directly.",
            prediction_horizons_days=[20, 5, 20],
            required_concepts=[" Recent_Return ", "recent_return"],
            market_conditions=[],
            failure_conditions=[],
        )

        self.assertEqual(hypothesis.prediction_horizons_days, [5, 20])
        self.assertEqual(hypothesis.required_concepts, ["recent_return"])

    def test_rejects_non_positive_horizon(self) -> None:
        with self.assertRaises(ValidationError):
            DevelopedHypothesis(
                hypothesis_id="H-1",
                idea_id="I-1",
                direction="momentum",
                title="invalid",
                claim="This claim contains enough text.",
                economic_rationale="This rationale contains enough text.",
                core_mechanism="This mechanism contains enough text.",
                counterargument="This counterargument contains enough text.",
                falsification_criteria=["A clear failure criterion."],
                distinction_from_siblings="A distinct mechanism description.",
                prediction_horizons_days=[0],
                required_concepts=["return"],
            )

    def test_formula_metadata_must_match_structure(self) -> None:
        with self.assertRaises(ValidationError):
            FactorFormulaSpecV2.model_validate(
                {
                    **_factor().model_dump(mode="json"),
                    "lookback_days": 20,
                }
            )

    def test_experiment_requires_matching_hypothesis(self) -> None:
        with self.assertRaises(ValidationError):
            FactorExperimentSpec(
                experiment_id="E-1",
                hypothesis_id="H-OTHER",
                variation_reason="Test an alternative prediction horizon.",
                measurement_plan_id="MP-MOM-001",
                variant_type="baseline",
                represented_proxy_ids=["P1"],
                factor=_factor(),
                prediction_horizon_days=20,
            )

    def test_measurement_plan_requires_unique_proxy_ids(self) -> None:
        proxy = {
            "proxy_id": "P1",
            "mechanism_component": "Delayed price adjustment after information arrival.",
            "observable_definition": "Recent close-to-close return measures adjustment.",
            "required_fields": ["close"],
            "limitations": ["Recent return cannot identify the information source."],
        }
        with self.assertRaises(ValidationError):
            MeasurementPlan(
                plan_id="MP-MOM-001",
                hypothesis_id="H-MOM-001",
                mechanism_summary="Delayed information diffusion creates continuation.",
                proxies=[proxy, proxy],
                baseline_logic="Use recent return as the simplest mechanism proxy.",
                full_mechanism_logic=(
                    "Combine all declared proxies to represent the complete mechanism."
                ),
            )

    def test_experiment_requires_unique_represented_proxy_ids(self) -> None:
        with self.assertRaises(ValidationError):
            FactorExperimentSpec(
                experiment_id="E-1",
                hypothesis_id="H-MOM-001",
                variation_reason="Test the declared baseline measurement of the mechanism.",
                measurement_plan_id="MP-MOM-001",
                variant_type="baseline",
                represented_proxy_ids=["P1", "P1"],
                factor=_factor(),
                prediction_horizon_days=20,
            )

    def test_rejects_inconsistent_check_result(self) -> None:
        with self.assertRaises(ValidationError):
            HypothesisCheckResult(
                hypothesis_id="H-1",
                accepted=True,
                issues=[{"code": "bad", "message": "unexpected issue"}],
                generation_round=0,
            )

    def test_cohort_weight_must_match_horizon(self) -> None:
        protocol = BacktestProtocol(
            holding_period_days=20,
            cohort_capital_weight=0.05,
        )
        self.assertEqual(protocol.top_k, 20)
        with self.assertRaises(ValidationError):
            BacktestProtocol(
                holding_period_days=20,
                cohort_capital_weight=1.0,
            )

    def test_validation_metrics_require_frozen_direction(self) -> None:
        payload = {
            "split": "validation",
            "experiment_id": "E-1",
            "prediction_horizon_days": 20,
            "direction_multiplier": 1,
            "direction_source": "train",
            "valid_ic_dates": 400,
            "median_stock_coverage": 0.95,
            "mean_rank_ic": 0.03,
            "rank_icir": 0.4,
            "gross_return": 0.1,
            "net_return": 0.08,
            "annualized_return": 0.04,
            "annualized_volatility": 0.12,
            "sharpe": 0.33,
            "max_drawdown": -0.15,
            "turnover": 0.5,
        }
        with self.assertRaises(ValidationError):
            SplitMetrics.model_validate(payload)
        metrics = SplitMetrics.model_validate(
            {**payload, "direction_source": "frozen_train"}
        )
        self.assertEqual(metrics.direction_multiplier, 1)

    def test_test_report_requires_validation_pass(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentDecision(
                experiment_id="E-1",
                train_passed=True,
                validation_passed=False,
                test_reported=True,
            )


if __name__ == "__main__":
    unittest.main()
