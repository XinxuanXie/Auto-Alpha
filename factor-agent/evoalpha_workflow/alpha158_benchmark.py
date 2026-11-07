"""Efficient annual and rolling-window diagnostics for the Alpha158 seed library."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite, sqrt
from typing import Sequence

import numpy as np
import polars as pl

from factor_pipeline.single_factor_evaluation import FactorEvaluationConfig


@dataclass(frozen=True)
class BenchmarkWindow:
    window_type: str
    window_years: int
    end_year: int
    start_date: date
    end_date: date
    is_partial: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


def build_benchmark_windows(
    first_date: date,
    last_date: date,
    rolling_years: Sequence[int] = (5, 10, 20),
) -> tuple[BenchmarkWindow, ...]:
    if last_date < first_date:
        raise ValueError("last_date must not precede first_date")
    if any(years < 2 for years in rolling_years):
        raise ValueError("rolling windows must span at least two years")
    if len(set(rolling_years)) != len(tuple(rolling_years)):
        raise ValueError("rolling windows must be unique")

    windows = []
    for year in range(first_date.year, last_date.year + 1):
        start = max(first_date, date(year, 1, 1))
        end = min(last_date, date(year, 12, 31))
        windows.append(
            BenchmarkWindow(
                window_type="annual",
                window_years=1,
                end_year=year,
                start_date=start,
                end_date=end,
                is_partial=start != date(year, 1, 1) or end != date(year, 12, 31),
            )
        )

    for years in rolling_years:
        first_end_year = first_date.year + years - 1
        for end_year in range(first_end_year, last_date.year + 1):
            start_year = end_year - years + 1
            start = max(first_date, date(start_year, 1, 1))
            end = min(last_date, date(end_year, 12, 31))
            windows.append(
                BenchmarkWindow(
                    window_type=f"rolling_{years}y",
                    window_years=years,
                    end_year=end_year,
                    start_date=start,
                    end_date=end,
                    is_partial=(
                        start != date(start_year, 1, 1)
                        or end != date(end_year, 12, 31)
                    ),
                )
            )
    return tuple(windows)


def build_forward_return_labels_lazy(
    panel: pl.LazyFrame,
    calendar: Sequence[object],
    horizons: Sequence[int] = (5, 10, 20),
) -> pl.LazyFrame:
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("all horizons must be positive")
    if len(set(horizons)) != len(tuple(horizons)):
        raise ValueError("horizons must be unique")
    dates = pl.DataFrame({"date": sorted(set(calendar))})
    mappings = dates
    for horizon in horizons:
        mappings = mappings.with_columns(
            pl.col("date").shift(-horizon).alias(f"target_date_{horizon}d")
        )
    prices = (
        panel.select("date", "permno", "close")
        .filter(pl.col("close").is_finite() & (pl.col("close") > 0))
        .unique(subset=["date", "permno"], keep="first")
    )
    labels = prices.join(mappings.lazy(), on="date", how="inner")
    for horizon in horizons:
        target_date = f"target_date_{horizon}d"
        future_close = f"future_close_{horizon}d"
        target_prices = prices.rename(
            {"date": target_date, "close": future_close}
        )
        labels = (
            labels.join(
                target_prices,
                on=[target_date, "permno"],
                how="left",
            )
            .with_columns(
                (pl.col(future_close) / pl.col("close") - 1.0).alias(
                    f"forward_return_{horizon}d"
                )
            )
            .drop(future_close)
        )
    return labels.select(
        "date",
        "permno",
        *(f"target_date_{horizon}d" for horizon in horizons),
        *(f"forward_return_{horizon}d" for horizon in horizons),
    )


def build_daily_factor_statistics(
    factor_values: pl.LazyFrame,
    labels: pl.LazyFrame,
    config: FactorEvaluationConfig = FactorEvaluationConfig(),
) -> pl.DataFrame:
    frame = labels.join(
        factor_values.select("date", "permno", "factor"),
        on=["date", "permno"],
        how="left",
    )
    horizon_frames = []
    for horizon in config.horizons:
        label = f"forward_return_{horizon}d"
        target_date = f"target_date_{horizon}d"
        eligible = frame.filter(pl.col(label).is_finite())
        counts = eligible.group_by("date").agg(
            pl.col(target_date).first().alias(target_date),
            pl.len().alias(f"eligible_stocks_{horizon}d"),
            pl.col("factor")
            .is_finite()
            .sum()
            .alias(f"valid_stocks_{horizon}d"),
        )
        valid = (
            eligible.filter(pl.col("factor").is_finite())
            .with_columns(
                pl.len().over("date").alias("_valid_count"),
                pl.col("factor").rank(method="average").over("date").alias("_factor_rank"),
                pl.col(label).rank(method="average").over("date").alias("_return_rank"),
            )
            .filter(pl.col("_valid_count") >= config.minimum_cross_section)
        )
        rank_ics = (
            valid.group_by("date")
            .agg(
                pl.corr("_factor_rank", "_return_rank").alias(
                    f"rank_ic_{horizon}d"
                ),
                pl.col("_factor_rank").n_unique().alias("_factor_unique"),
                pl.col("_return_rank").n_unique().alias("_return_unique"),
            )
            .with_columns(
                pl.when(
                    (pl.col("_factor_unique") > 1)
                    & (pl.col("_return_unique") > 1)
                    & pl.col(f"rank_ic_{horizon}d").is_finite()
                )
                .then(pl.col(f"rank_ic_{horizon}d"))
                .otherwise(None)
                .alias(f"rank_ic_{horizon}d")
            )
            .select("date", f"rank_ic_{horizon}d")
        )
        horizon_frames.append(counts.join(rank_ics, on="date", how="left"))

    result = horizon_frames[0]
    for horizon_frame in horizon_frames[1:]:
        result = result.join(horizon_frame, on="date", how="full", coalesce=True)
    return result.sort("date").collect(engine="streaming")


def aggregate_prediction_window(
    daily_statistics: pl.DataFrame,
    window: BenchmarkWindow,
    config: FactorEvaluationConfig = FactorEvaluationConfig(),
) -> dict[str, object]:
    horizon_data = {
        horizon: daily_statistics.filter(
            pl.col("date").dt.date().is_between(window.start_date, window.end_date)
            & (pl.col(f"target_date_{horizon}d").dt.date() <= window.end_date)
        )
        for horizon in config.horizons
    }
    primary_column = f"rank_ic_{config.primary_horizon_days}d"
    primary_data = horizon_data[config.primary_horizon_days]
    primary_raw = _finite_values(primary_data, primary_column)
    raw_primary_mean = (
        float(np.mean(primary_raw)) if len(primary_raw) else float("nan")
    )
    direction = -1 if isfinite(raw_primary_mean) and raw_primary_mean < 0 else 1

    means = {}
    for horizon in config.horizons:
        values = _finite_values(horizon_data[horizon], f"rank_ic_{horizon}d")
        means[horizon] = (
            float(np.mean(values)) * direction if len(values) else float("nan")
        )
    primary_std = (
        float(np.std(primary_raw, ddof=1))
        if len(primary_raw) > 1
        else float("nan")
    )
    rank_icir = (
        means[config.primary_horizon_days] / primary_std
        if isfinite(primary_std) and primary_std > 0
        else float("nan")
    )
    eligible_column = f"eligible_stocks_{config.primary_horizon_days}d"
    valid_column = f"valid_stocks_{config.primary_horizon_days}d"
    eligible_days = primary_data.filter(pl.col(eligible_column) > 0).height
    valid_days = len(primary_raw)
    valid_day_ratio = valid_days / eligible_days if eligible_days else 0.0
    coverage = (
        primary_data.filter(pl.col(eligible_column) > 0)
        .select((pl.col(valid_column) / pl.col(eligible_column)).alias("coverage"))
        .get_column("coverage")
    )
    median_coverage = float(coverage.median()) if len(coverage) else 0.0
    monthly_stability = _monthly_stability(primary_data, primary_column, direction)
    diagnostic_means = [means[horizon] for horizon in config.diagnostic_horizons_days]
    cross_horizon_consistency = (
        sum(value > 0 for value in diagnostic_means) / len(diagnostic_means)
        if all(isfinite(value) for value in diagnostic_means)
        else 0.0
    )
    reasons = []
    if valid_day_ratio < config.minimum_valid_day_ratio:
        reasons.append("insufficient_valid_rank_ic_days")
    if median_coverage < config.minimum_median_stock_coverage:
        reasons.append("insufficient_stock_coverage")
    if not isfinite(means[config.primary_horizon_days]):
        reasons.append("invalid_primary_rank_ic")
    if not isfinite(rank_icir):
        reasons.append("invalid_primary_rank_icir")
    return {
        "direction": direction,
        "mean_rank_ic_5d": means[5],
        "mean_rank_ic_10d": means[10],
        "mean_rank_ic_20d": means[20],
        "rank_icir_20d": rank_icir,
        "monthly_stability": monthly_stability,
        "cross_horizon_consistency": cross_horizon_consistency,
        "valid_rank_ic_day_ratio": valid_day_ratio,
        "median_stock_coverage": median_coverage,
        "valid_rank_ic_days": valid_days,
        "eligible_days": eligible_days,
        "passed_hard_filters": not reasons,
        "rejection_reasons": reasons,
    }


def aggregate_backtest_window(
    positive_periods: pl.DataFrame,
    negative_periods: pl.DataFrame,
    window: BenchmarkWindow,
    direction: int,
) -> dict[str, object]:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    source = positive_periods if direction == 1 else negative_periods
    periods = source.filter(
        pl.col("signal_date").dt.date().is_between(window.start_date, window.end_date)
    )
    if periods.is_empty():
        return {
            "backtest_periods": 0,
            "average_positions": None,
            "average_turnover": None,
            "cumulative_gross_return": None,
            "cumulative_net_return": None,
            "annualized_net_return": None,
            "annualized_net_volatility": None,
            "annualized_sharpe": None,
            "max_drawdown": None,
        }
    gross = periods.get_column("gross_return").to_numpy()
    net = periods.get_column("net_return").to_numpy()
    wealth = np.cumprod(1.0 + net)
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    period_count = len(net)
    cumulative_net = float(wealth[-1] - 1.0)
    weekly_std = float(np.std(net, ddof=1)) if period_count > 1 else float("nan")
    annualized_volatility = weekly_std * sqrt(52.0)
    return {
        "backtest_periods": period_count,
        "average_positions": float(periods.get_column("positions").mean()),
        "average_turnover": float(periods.get_column("turnover").mean()),
        "cumulative_gross_return": float(np.prod(1.0 + gross) - 1.0),
        "cumulative_net_return": cumulative_net,
        "annualized_net_return": float(
            (1.0 + cumulative_net) ** (52.0 / period_count) - 1.0
        ),
        "annualized_net_volatility": annualized_volatility,
        "annualized_sharpe": (
            float(np.mean(net) / weekly_std * sqrt(52.0))
            if isfinite(weekly_std) and weekly_std > 0
            else float("nan")
        ),
        "max_drawdown": float(np.min(drawdowns)),
    }


def _finite_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    if frame.is_empty():
        return np.asarray([], dtype=float)
    return frame.filter(pl.col(column).is_finite()).get_column(column).to_numpy()


def _monthly_stability(
    frame: pl.DataFrame,
    column: str,
    direction: int,
) -> float:
    valid = frame.filter(pl.col(column).is_finite())
    if valid.is_empty():
        return 0.0
    monthly = (
        valid.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg((pl.col(column) * direction).mean().alias("monthly_rank_ic"))
    )
    return float(monthly.select((pl.col("monthly_rank_ic") > 0).mean()).item())
