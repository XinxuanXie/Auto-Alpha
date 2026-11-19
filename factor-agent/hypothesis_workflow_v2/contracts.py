"""Typed contracts for the hypothesis-first workflow."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from factor_pipeline.spec import DataField, FormulaNode, UniverseFilterSpec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ResearchDirection(StrEnum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    LIQUIDITY_VOLATILITY = "liquidity_volatility"
    PRICE_VOLUME = "price_volume"
    OVERNIGHT_INTRADAY = "overnight_intraday"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


class InsightDecision(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REPLACE = "replace"


class ExperimentVariantType(StrEnum):
    BASELINE = "baseline"
    FULL_MECHANISM = "full_mechanism"
    ABLATION = "ablation"
    ROBUSTNESS = "robustness"


class ResearchIdea(StrictModel):
    idea_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    direction: ResearchDirection
    idea_text: str = Field(min_length=20, max_length=20_000)
    model: str = Field(min_length=1, max_length=160)
    prompt_version: str = Field(min_length=1, max_length=80)


class InsightAnalysis(StrictModel):
    analysis_id: str = Field(min_length=1, max_length=160)
    idea_id: str = Field(min_length=1, max_length=160)
    decision: InsightDecision
    supporting_arguments: list[str] = Field(min_length=1, max_length=12)
    objections: list[str] = Field(min_length=1, max_length=12)
    alternative_explanations: list[str] = Field(min_length=1, max_length=12)
    final_research_idea: str = Field(min_length=20, max_length=10_000)
    analysis_text: str = Field(min_length=20, max_length=30_000)
    model: str = Field(min_length=1, max_length=160)
    prompt_version: str = Field(min_length=1, max_length=80)


class DevelopedHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=160)
    idea_id: str = Field(min_length=1, max_length=160)
    direction: ResearchDirection
    title: str = Field(min_length=1, max_length=240)
    claim: str = Field(min_length=10, max_length=4_000)
    economic_rationale: str = Field(min_length=10, max_length=4_000)
    core_mechanism: str = Field(min_length=10, max_length=4_000)
    counterargument: str = Field(min_length=10, max_length=4_000)
    falsification_criteria: list[str] = Field(min_length=1, max_length=12)
    distinction_from_siblings: str = Field(min_length=10, max_length=2_000)
    prediction_horizons_days: list[int] = Field(min_length=1, max_length=10)
    required_concepts: list[str] = Field(min_length=1, max_length=30)
    market_conditions: list[str] = Field(default_factory=list, max_length=20)
    failure_conditions: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("prediction_horizons_days")
    @classmethod
    def canonicalize_horizons(cls, value: list[int]) -> list[int]:
        if any(isinstance(item, bool) or item <= 0 for item in value):
            raise ValueError("prediction horizons must be positive integers")
        return sorted(set(value))

    @field_validator("required_concepts")
    @classmethod
    def canonicalize_concepts(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.split()).casefold() for item in value]
        if any(not item for item in normalized):
            raise ValueError("required concepts cannot be empty")
        return sorted(set(normalized))


class CheckIssue(StrictModel):
    code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2_000)
    field: str | None = Field(default=None, max_length=160)


class HypothesisCheckResult(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=160)
    accepted: bool
    issues: list[CheckIssue] = Field(default_factory=list)
    generation_round: Literal[0, 1]

    @model_validator(mode="after")
    def validate_accepted_issues(self) -> Self:
        if self.accepted and self.issues:
            raise ValueError("accepted hypothesis cannot contain issues")
        if not self.accepted and not self.issues:
            raise ValueError("rejected hypothesis must contain at least one issue")
        return self


class FactorFormulaSpecV2(StrictModel):
    factor_id: str = Field(min_length=1, max_length=160)
    hypothesis_id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    formula: FormulaNode
    required_fields: list[DataField] = Field(min_length=1)
    lookback_days: int = Field(ge=0, le=504)
    universe_filters: UniverseFilterSpec | None = None

    @model_validator(mode="after")
    def validate_formula_metadata(self) -> Self:
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError("required_fields cannot contain duplicates")
        referenced = self.formula.referenced_fields()
        if set(self.required_fields) != referenced:
            expected = sorted(field.value for field in referenced)
            raise ValueError(
                f"required_fields must exactly match formula fields: {expected}"
            )
        required_history = max(
            self.formula.required_history(),
            (
                self.universe_filters.required_history()
                if self.universe_filters is not None
                else 0
            ),
        )
        if self.lookback_days != required_history:
            raise ValueError(
                "lookback_days must equal the history derived from formula and filters: "
                f"{required_history}"
            )
        return self


class MeasurementProxy(StrictModel):
    proxy_id: str = Field(pattern=r"^P[1-9][0-9]*$", max_length=20)
    mechanism_component: str = Field(min_length=10, max_length=2_000)
    observable_definition: str = Field(min_length=10, max_length=2_000)
    required_fields: list[DataField] = Field(min_length=1, max_length=6)
    limitations: list[str] = Field(min_length=1, max_length=8)

    @field_validator("required_fields")
    @classmethod
    def canonicalize_fields(cls, value: list[DataField]) -> list[DataField]:
        return sorted(set(value), key=lambda item: item.value)


class MeasurementPlan(StrictModel):
    plan_id: str = Field(min_length=1, max_length=180)
    hypothesis_id: str = Field(min_length=1, max_length=160)
    mechanism_summary: str = Field(min_length=10, max_length=4_000)
    proxies: list[MeasurementProxy] = Field(min_length=1, max_length=8)
    baseline_logic: str = Field(min_length=10, max_length=2_000)
    full_mechanism_logic: str = Field(min_length=10, max_length=4_000)
    unobservable_aspects: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_proxy_ids(self) -> Self:
        identifiers = [proxy.proxy_id for proxy in self.proxies]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("measurement proxy IDs must be unique")
        return self


class FactorExperimentSpec(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=180)
    hypothesis_id: str = Field(min_length=1, max_length=160)
    variation_reason: str = Field(min_length=10, max_length=2_000)
    measurement_plan_id: str = Field(min_length=1, max_length=180)
    variant_type: ExperimentVariantType
    represented_proxy_ids: list[str] = Field(min_length=1, max_length=8)
    factor: FactorFormulaSpecV2
    prediction_horizon_days: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.factor.hypothesis_id != self.hypothesis_id:
            raise ValueError("factor and experiment hypothesis_id must match")
        if len(self.represented_proxy_ids) != len(set(self.represented_proxy_ids)):
            raise ValueError("represented_proxy_ids cannot contain duplicates")
        return self


class PreflightResult(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=180)
    accepted: bool
    issues: list[CheckIssue] = Field(default_factory=list)
    correction_round: Literal[0, 1]

    @model_validator(mode="after")
    def validate_accepted_issues(self) -> Self:
        if self.accepted and self.issues:
            raise ValueError("accepted experiment cannot contain issues")
        if not self.accepted and not self.issues:
            raise ValueError("rejected experiment must contain at least one issue")
        return self


class BacktestProtocol(StrictModel):
    signal_frequency: Literal["daily"] = "daily"
    top_k: Literal[20] = 20
    holding_period_days: int = Field(gt=0)
    overlapping_cohorts: Literal[True] = True
    cohort_capital_weight: float = Field(gt=0.0, le=1.0)
    transaction_cost_bps: float = Field(default=10.0, ge=0.0)

    @model_validator(mode="after")
    def validate_cohort_weight(self) -> Self:
        expected = 1.0 / self.holding_period_days
        if not math.isclose(self.cohort_capital_weight, expected, abs_tol=1e-12):
            raise ValueError(
                "cohort_capital_weight must equal 1 / holding_period_days"
            )
        return self


class SplitMetrics(StrictModel):
    split: SplitName
    experiment_id: str = Field(min_length=1, max_length=180)
    prediction_horizon_days: int = Field(gt=0)
    direction_multiplier: Literal[-1, 1]
    direction_source: Literal["train", "frozen_train"]
    valid_ic_dates: int = Field(ge=0)
    median_stock_coverage: float = Field(ge=0.0, le=1.0)
    mean_rank_ic: float
    rank_icir: float
    newey_west_t_stat: float | None = None
    newey_west_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    fdr_q_value: float | None = Field(default=None, ge=0.0, le=1.0)
    gross_return: float
    net_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe: float
    max_drawdown: float = Field(le=0.0)
    turnover: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_direction_source(self) -> Self:
        expected = "train" if self.split == SplitName.TRAIN else "frozen_train"
        if self.direction_source != expected:
            raise ValueError(
                f"direction_source for {self.split.value} must be {expected}"
            )
        return self


# Selection gate (Patch 10): a factor passes a split only if it clears the
# IC bar, makes money after costs, AND ranks a broad cross-section. The
# coverage leg is new: 0d53bd4b's PRICE_VOLUME-01-04 passed train
# (IC 0.0266) with median_stock_coverage 0.00 — its signal exists on almost
# no stocks, so the Top-20 portfolio and the IC are computed on a sliver.
# Observed coverages are 0.00 (sparse junk) vs 0.99-1.00 (real factors),
# so 0.5 sits in the gap. The IC bar (0.02) and the data splits are
# unchanged.
MIN_MEDIAN_STOCK_COVERAGE = 0.5


def passes_selection_gate(metric: SplitMetrics) -> bool:
    return (
        metric.mean_rank_ic > 0.02
        and metric.net_return > 0
        and metric.median_stock_coverage >= MIN_MEDIAN_STOCK_COVERAGE
    )


class ExperimentDecision(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=180)
    train_passed: bool
    validation_passed: bool | None = None
    test_reported: bool = False
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_consistency(self) -> Self:
        if not self.train_passed and self.validation_passed is not None:
            raise ValueError("failed training experiment cannot have validation result")
        if self.test_reported and self.validation_passed is not True:
            raise ValueError("test can only be reported after validation passes")
        return self


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1, max_length=240)
    kind: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1, max_length=1_000)
    checksum: str
    item_count: int = Field(ge=0)

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        normalized = value.casefold()
        if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
            raise ValueError("checksum must be a SHA-256 hex digest")
        return normalized
