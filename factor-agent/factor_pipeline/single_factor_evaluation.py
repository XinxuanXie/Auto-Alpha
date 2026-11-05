from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from math import isfinite
from statistics import median

import numpy as np
import polars as pl


@dataclass(frozen=True)
class EvaluationPeriod:
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")


@dataclass(frozen=True)
class FactorEvaluationConfig:
    primary_horizon_days: int = 20
    diagnostic_horizons_days: tuple[int, ...] = (5, 10)
    minimum_valid_day_ratio: float = 0.80
    minimum_median_stock_coverage: float = 0.80
    minimum_cross_section: int = 50

    def __post_init__(self) -> None:
        horizons = (self.primary_horizon_days, *self.diagnostic_horizons_days)
        if any(horizon < 1 for horizon in horizons):
            raise ValueError("all horizons must be positive")
        if len(horizons) != len(set(horizons)):
            raise ValueError("primary and diagnostic horizons must be unique")
        if not 0 <= self.minimum_valid_day_ratio <= 1:
            raise ValueError("minimum_valid_day_ratio must be between 0 and 1")
        if not 0 <= self.minimum_median_stock_coverage <= 1:
            raise ValueError(
                "minimum_median_stock_coverage must be between 0 and 1"
            )
        if self.minimum_cross_section < 3:
            raise ValueError("minimum_cross_section must be at least 3")

    @property
    def horizons(self) -> tuple[int, ...]:
        return (self.primary_horizon_days, *self.diagnostic_horizons_days)


@dataclass(frozen=True)
class SingleFactorMetrics:
    factor_id: str
    direction: int
    mean_rank_ic_20d: float
    rank_icir_20d: float
    monthly_stability: float
    cross_horizon_consistency: float
    mean_rank_ic_5d: float
    mean_rank_ic_10d: float
    valid_rank_ic_day_ratio: float
    median_stock_coverage: float
    valid_rank_ic_days: int
    eligible_days: int
    passed_hard_filters: bool
    rejection_reasons: tuple[str, ...]
    composite_score: float | None = None
    rank_ic_percentile: float | None = None
    rank_icir_percentile: float | None = None
    monthly_stability_percentile: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FactorScoreReference:
    rank_ic_values: tuple[float, ...]
    rank_icir_values: tuple[float, ...]
    monthly_stability_values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _HorizonMetrics:
    mean_rank_ic: float
    rank_icir: float
    rank_ics: pl.DataFrame
    valid_days: int
    eligible_days: int
    valid_day_ratio: float
    median_stock_coverage: float


def build_forward_return_labels(
    panel: pl.DataFrame,
    period: EvaluationPeriod,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pl.DataFrame:
    required = {"date", "permno", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing label columns: {sorted(missing)}")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("all horizons must be positive")
    if len(horizons) != len(set(horizons)):
        raise ValueError("horizons must be unique")

    prices = (
        panel.select("date", "permno", "close")
        .filter(
            pl.col("close").is_finite()
            & (pl.col("close") > 0)
            & pl.col("date").dt.date().is_between(
                period.start_date,
                period.end_date,
            )
        )
        .unique(subset=["date", "permno"], keep="first")
    )
    calendar = prices.select("date").unique().sort("date")
    mappings = calendar
    for horizon in horizons:
        mappings = mappings.with_columns(
            pl.col("date").shift(-horizon).alias(f"target_date_{horizon}d")
        )

    labels = prices.join(mappings, on="date", how="inner")
    for horizon in horizons:
        target_date = f"target_date_{horizon}d"
        future_close = f"future_close_{horizon}d"
        target_prices = prices.rename(
            {"date": target_date, "close": future_close}
        )
        labels = labels.join(
            target_prices,
            on=[target_date, "permno"],
            how="left",
        ).with_columns(
            (
                pl.col(future_close) / pl.col("close") - 1.0
            ).alias(f"forward_return_{horizon}d")
        ).drop(future_close)
    return labels


def evaluate_single_factor(
    factor_values: pl.DataFrame,
    labels: pl.DataFrame,
    factor_id: str,
    period: EvaluationPeriod,
    config: FactorEvaluationConfig = FactorEvaluationConfig(),
    direction: int | None = None,
) -> SingleFactorMetrics:
    required_factor_columns = {"date", "permno", "factor"}
    missing = required_factor_columns - set(factor_values.columns)
    if missing:
        raise ValueError(f"factor values are missing columns: {sorted(missing)}")
    label_columns = {
        "date",
        "permno",
        *(f"forward_return_{horizon}d" for horizon in config.horizons),
    }
    missing_labels = label_columns - set(labels.columns)
    if missing_labels:
        raise ValueError(f"labels are missing columns: {sorted(missing_labels)}")
    if direction not in {None, -1, 1}:
        raise ValueError("direction must be -1, 1, or None")

    frame = (
        labels.select(*sorted(label_columns))
        .join(
            factor_values.select("date", "permno", "factor").unique(
                subset=["date", "permno"], keep="first"
            ),
            on=["date", "permno"],
            how="left",
        )
        .filter(
            pl.col("date").dt.date().is_between(
                period.start_date,
                period.end_date,
            )
        )
    )

    raw_primary = _evaluate_horizon(
        frame,
        config.primary_horizon_days,
        config.minimum_cross_section,
    )
    resolved_direction = direction
    if resolved_direction is None:
        resolved_direction = (
            -1
            if isfinite(raw_primary.mean_rank_ic)
            and raw_primary.mean_rank_ic < 0
            else 1
        )

    horizon_metrics = {
        config.primary_horizon_days: _apply_direction(
            raw_primary, resolved_direction
        )
    }
    for horizon in config.diagnostic_horizons_days:
        horizon_metrics[horizon] = _apply_direction(
            _evaluate_horizon(frame, horizon, config.minimum_cross_section),
            resolved_direction,
        )

    primary = horizon_metrics[config.primary_horizon_days]
    monthly_stability = _monthly_stability(primary.rank_ics)
    diagnostic_signs = [
        horizon_metrics[horizon].mean_rank_ic > 0
        for horizon in config.diagnostic_horizons_days
        if isfinite(horizon_metrics[horizon].mean_rank_ic)
    ]
    cross_horizon_consistency = (
        sum(diagnostic_signs) / len(config.diagnostic_horizons_days)
        if len(diagnostic_signs) == len(config.diagnostic_horizons_days)
        else 0.0
    )

    reasons = []
    if primary.valid_day_ratio < config.minimum_valid_day_ratio:
        reasons.append("insufficient_valid_rank_ic_days")
    if primary.median_stock_coverage < config.minimum_median_stock_coverage:
        reasons.append("insufficient_stock_coverage")
    if not isfinite(primary.mean_rank_ic):
        reasons.append("invalid_primary_rank_ic")
    if not isfinite(primary.rank_icir):
        reasons.append("invalid_primary_rank_icir")

    return SingleFactorMetrics(
        factor_id=factor_id,
        direction=resolved_direction,
        mean_rank_ic_20d=primary.mean_rank_ic,
        rank_icir_20d=primary.rank_icir,
        monthly_stability=monthly_stability,
        cross_horizon_consistency=cross_horizon_consistency,
        mean_rank_ic_5d=horizon_metrics[5].mean_rank_ic,
        mean_rank_ic_10d=horizon_metrics[10].mean_rank_ic,
        valid_rank_ic_day_ratio=primary.valid_day_ratio,
        median_stock_coverage=primary.median_stock_coverage,
        valid_rank_ic_days=primary.valid_days,
        eligible_days=primary.eligible_days,
        passed_hard_filters=not reasons,
        rejection_reasons=tuple(reasons),
    )


def score_factor_metrics(
    metrics: list[SingleFactorMetrics],
    reference: FactorScoreReference | None = None,
) -> list[SingleFactorMetrics]:
    score_reference = reference or build_score_reference(metrics)

    scored = []
    for item in metrics:
        if not item.passed_hard_filters:
            scored.append(item)
            continue
        rank_ic_percentile = _percentile(
            item.mean_rank_ic_20d,
            score_reference.rank_ic_values,
        )
        rank_icir_percentile = _percentile(
            item.rank_icir_20d,
            score_reference.rank_icir_values,
        )
        stability_percentile = _percentile(
            item.monthly_stability,
            score_reference.monthly_stability_values,
        )
        composite = (
            0.45 * rank_ic_percentile
            + 0.35 * rank_icir_percentile
            + 0.15 * stability_percentile
            + 0.05 * item.cross_horizon_consistency
        )
        scored.append(
            replace(
                item,
                composite_score=composite,
                rank_ic_percentile=rank_ic_percentile,
                rank_icir_percentile=rank_icir_percentile,
                monthly_stability_percentile=stability_percentile,
            )
        )
    return scored


def build_score_reference(
    metrics: list[SingleFactorMetrics],
) -> FactorScoreReference:
    valid = [item for item in metrics if item.passed_hard_filters]
    if not valid:
        raise ValueError("cannot build a score reference without valid factors")
    return FactorScoreReference(
        rank_ic_values=tuple(sorted(item.mean_rank_ic_20d for item in valid)),
        rank_icir_values=tuple(sorted(item.rank_icir_20d for item in valid)),
        monthly_stability_values=tuple(
            sorted(item.monthly_stability for item in valid)
        ),
    )


def _evaluate_horizon(
    frame: pl.DataFrame,
    horizon: int,
    minimum_cross_section: int,
) -> _HorizonMetrics:
    label = f"forward_return_{horizon}d"
    eligible = frame.filter(pl.col(label).is_finite())
    eligible_counts = eligible.group_by("date").len().rename(
        {"len": "eligible_stocks"}
    )
    eligible_days = eligible_counts.height
    valid = eligible.filter(pl.col("factor").is_finite())
    valid_counts = valid.group_by("date").len().rename({"len": "valid_stocks"})
    coverage = eligible_counts.join(valid_counts, on="date", how="left").with_columns(
        pl.col("valid_stocks").fill_null(0),
        (
            pl.col("valid_stocks") / pl.col("eligible_stocks")
        ).alias("stock_coverage"),
    )
    median_stock_coverage = (
        float(coverage.get_column("stock_coverage").median())
        if coverage.height
        else 0.0
    )

    valid = valid.join(
        valid_counts.filter(pl.col("valid_stocks") >= minimum_cross_section),
        on="date",
        how="inner",
    ).with_columns(
        pl.col("factor").rank(method="average").over("date").alias("factor_rank"),
        pl.col(label).rank(method="average").over("date").alias("return_rank"),
    )
    rank_ics = (
        valid.group_by("date")
        .agg(
            pl.corr("factor_rank", "return_rank").alias("rank_ic"),
            pl.col("factor_rank").n_unique().alias("factor_unique"),
            pl.col("return_rank").n_unique().alias("return_unique"),
        )
        .filter(
            (pl.col("factor_unique") > 1)
            & (pl.col("return_unique") > 1)
            & pl.col("rank_ic").is_finite()
        )
        .select("date", "rank_ic")
        .sort("date")
    )
    values = rank_ics.get_column("rank_ic").to_numpy()
    mean_rank_ic = float(np.mean(values)) if len(values) else float("nan")
    standard_deviation = (
        float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    )
    rank_icir = (
        mean_rank_ic / standard_deviation
        if isfinite(standard_deviation) and standard_deviation > 0
        else float("nan")
    )
    valid_days = rank_ics.height
    valid_day_ratio = valid_days / eligible_days if eligible_days else 0.0
    return _HorizonMetrics(
        mean_rank_ic=mean_rank_ic,
        rank_icir=rank_icir,
        rank_ics=rank_ics,
        valid_days=valid_days,
        eligible_days=eligible_days,
        valid_day_ratio=valid_day_ratio,
        median_stock_coverage=median_stock_coverage,
    )


def _apply_direction(metrics: _HorizonMetrics, direction: int) -> _HorizonMetrics:
    return replace(
        metrics,
        mean_rank_ic=metrics.mean_rank_ic * direction,
        rank_icir=metrics.rank_icir * direction,
        rank_ics=metrics.rank_ics.with_columns(
            (pl.col("rank_ic") * direction).alias("rank_ic")
        ),
    )


def _monthly_stability(rank_ics: pl.DataFrame) -> float:
    if rank_ics.is_empty():
        return 0.0
    monthly = (
        rank_ics.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("rank_ic").mean().alias("monthly_rank_ic"))
    )
    if monthly.is_empty():
        return 0.0
    return float(
        monthly.select((pl.col("monthly_rank_ic") > 0).mean()).item()
    )


def _percentile(value: float, reference: tuple[float, ...]) -> float:
    if not reference:
        raise ValueError("cannot score factors without a valid reference population")
    left = bisect_left(reference, value)
    right = bisect_right(reference, value)
    average_rank = (left + right - 1) / 2.0
    if len(reference) == 1:
        return 1.0
    return min(1.0, max(0.0, average_rank / (len(reference) - 1)))
