"""Real single-factor evaluation and overlapping-cohort backtest for V2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from factor_pipeline.config import DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    build_forward_return_labels,
)
from factor_pipeline.spec import FactorSpec, PortfolioRules

from .contracts import (
    FactorExperimentSpec,
    SplitMetrics,
    SplitName,
)


@dataclass(frozen=True)
class RealBacktestConfig:
    panel_path: Path
    membership_path: Path
    train_period: EvaluationPeriod = EvaluationPeriod(
        start_date=date(2001, 1, 1), end_date=date(2010, 12, 31)
    )
    validation_period: EvaluationPeriod = EvaluationPeriod(
        start_date=date(2011, 1, 1), end_date=date(2012, 12, 31)
    )
    test_period: EvaluationPeriod = EvaluationPeriod(
        start_date=date(2013, 1, 1), end_date=date(2015, 12, 31)
    )
    warmup_calendar_days: int = 730
    minimum_cross_section: int = 50
    minimum_stock_coverage: float = 0.80
    transaction_cost_bps: float = 10.0
    top_k: int = 20

    def period(self, split: SplitName) -> EvaluationPeriod:
        return {
            SplitName.TRAIN: self.train_period,
            SplitName.VALIDATION: self.validation_period,
            SplitName.TEST: self.test_period,
        }[split]

    def minimum_valid_dates(self, split: SplitName) -> int:
        return 504 if split == SplitName.TRAIN else 252


@dataclass
class _SplitData:
    factor_panel: pl.DataFrame
    evaluation_panel: pl.DataFrame
    price_panel: pl.DataFrame
    calendar: list[object]


class RealV2BacktestEngine:
    def __init__(self, config: RealBacktestConfig) -> None:
        self.config = config
        self._split_cache: dict[SplitName, _SplitData] = {}
        # (experiment_id, error) for experiments whose backtest failed in
        # the most recent evaluate() call. Reset at the start of each call;
        # a failed experiment is dropped from the returned metrics, never
        # silently swallowed: the caller can read this list (and the
        # warning printed below) to see what was lost.
        self.failed_experiments: list[tuple[str, str]] = []

    def evaluate(
        self,
        experiments: list[FactorExperimentSpec],
        split: SplitName,
        train_metrics: dict[str, SplitMetrics] | None = None,
    ) -> list[SplitMetrics]:
        if not experiments:
            return []
        self.failed_experiments = []
        data = self._load_split(split)
        horizons = tuple(sorted({item.prediction_horizon_days for item in experiments}))
        labels = build_forward_return_labels(
            data.evaluation_panel,
            self.config.period(split),
            horizons,
        )
        factor_cache: dict[str, pl.DataFrame] = {}
        raw_rows = []
        for experiment in experiments:
            # Per-experiment isolation (Patch 8, 0f45e744): one bad factor
            # used to abort the whole split — REVERSAL-03-02 raised
            # "factor produced no valid Rank IC dates" in _rank_ic_metrics
            # and the 10 experiments already evaluated in train were lost
            # with it, because metrics are only saved after evaluate()
            # returns. Now a failure drops only that experiment; the rest
            # still return their metrics and the run completes.
            try:
                signature = json.dumps(
                    experiment.factor.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature not in factor_cache:
                    factor_cache[signature] = self._factor_values(
                        data.factor_panel,
                        experiment,
                        self.config.period(split),
                    )
                factor_values = factor_cache[signature]
                raw_ic = self._rank_ic_metrics(
                    factor_values,
                    labels,
                    experiment.prediction_horizon_days,
                )
                if split == SplitName.TRAIN:
                    direction = -1 if raw_ic["mean_rank_ic"] < 0 else 1
                    direction_source = "train"
                else:
                    if train_metrics is None or experiment.experiment_id not in train_metrics:
                        raise ValueError(
                            f"missing frozen training direction for {experiment.experiment_id}"
                        )
                    direction = train_metrics[experiment.experiment_id].direction_multiplier
                    direction_source = "frozen_train"
                oriented_ics = raw_ic["rank_ics"] * direction
                mean_rank_ic = float(np.mean(oriented_ics))
                standard_deviation = (
                    float(np.std(oriented_ics, ddof=1))
                    if len(oriented_ics) > 1
                    else float("nan")
                )
                rank_icir = (
                    mean_rank_ic / standard_deviation
                    if math.isfinite(standard_deviation) and standard_deviation > 0
                    else 0.0
                )
                nw_t, nw_p = _newey_west_mean_test(
                    oriented_ics,
                    max_lag=max(1, experiment.prediction_horizon_days - 1),
                )
                portfolio = self._overlapping_cohort_metrics(
                    factor_values,
                    data.price_panel,
                    data.calendar,
                    experiment.prediction_horizon_days,
                    direction,
                )
                raw_rows.append(
                    {
                        "split": split,
                        "experiment_id": experiment.experiment_id,
                        "prediction_horizon_days": experiment.prediction_horizon_days,
                        "direction_multiplier": direction,
                        "direction_source": direction_source,
                        "valid_ic_dates": raw_ic["valid_ic_dates"],
                        "median_stock_coverage": raw_ic["median_stock_coverage"],
                        "mean_rank_ic": mean_rank_ic,
                        "rank_icir": rank_icir,
                        "newey_west_t_stat": nw_t,
                        "newey_west_p_value": nw_p,
                        "fdr_q_value": None,
                        **portfolio,
                    }
                )
            except Exception as error:
                self.failed_experiments.append(
                    (experiment.experiment_id, str(error))
                )
                print(
                    f"warning: backtest failed for {experiment.experiment_id}: "
                    f"{error}",
                    flush=True,
                )
                continue
        q_values = _benjamini_hochberg(
            [float(row["newey_west_p_value"]) for row in raw_rows]
        )
        return [
            SplitMetrics.model_validate({**row, "fdr_q_value": q_value})
            for row, q_value in zip(raw_rows, q_values, strict=True)
        ]

    def validate_sample_capacity(
        self,
        horizon: int,
    ) -> list[str]:
        issues = []
        for split in SplitName:
            data = self._load_split(split)
            complete_dates = max(0, len(data.calendar) - horizon)
            minimum = self.config.minimum_valid_dates(split)
            if complete_dates < minimum:
                issues.append(
                    f"{split.value} has {complete_dates} complete label dates; "
                    f"requires at least {minimum}"
                )
        return issues

    def _load_split(self, split: SplitName) -> _SplitData:
        cached = self._split_cache.get(split)
        if cached is not None:
            return cached
        period = self.config.period(split)
        source = PanelDataSource(
            DataConfig(
                panel_path=self.config.panel_path,
                membership_path=self.config.membership_path,
                start_date=period.start_date,
                end_date=period.end_date,
                warmup_calendar_days=self.config.warmup_calendar_days,
                exit_buffer_calendar_days=0,
            )
        )
        history = source.panel_history().collect(engine="streaming")
        factor_panel = source.apply_membership(history.lazy()).collect(
            engine="streaming"
        )
        evaluation_panel = factor_panel.filter(
            pl.col("date").dt.date().is_between(period.start_date, period.end_date)
        )
        price_panel = history.filter(
            pl.col("date").dt.date().is_between(period.start_date, period.end_date)
        )
        calendar = (
            evaluation_panel.select("date")
            .unique()
            .sort("date")
            .get_column("date")
            .to_list()
        )
        result = _SplitData(
            factor_panel=factor_panel,
            evaluation_panel=evaluation_panel,
            price_panel=price_panel,
            calendar=calendar,
        )
        self._split_cache[split] = result
        return result

    def _factor_values(
        self,
        panel: pl.DataFrame,
        experiment: FactorExperimentSpec,
        period: EvaluationPeriod,
    ) -> pl.DataFrame:
        spec = _legacy_factor_spec(experiment)
        return (
            build_factor_from_spec(panel.lazy(), spec, FactorConfig())
            .filter(
                pl.col("date").dt.date().is_between(
                    period.start_date, period.end_date
                )
            )
            .select("date", "permno", "factor")
            .collect(engine="streaming")
        )

    def _rank_ic_metrics(
        self,
        factor_values: pl.DataFrame,
        labels: pl.DataFrame,
        horizon: int,
    ) -> dict[str, object]:
        label = f"forward_return_{horizon}d"
        eligible = labels.filter(pl.col(label).is_finite())
        eligible_counts = eligible.group_by("date").len().rename(
            {"len": "eligible_stocks"}
        )
        joined = eligible.join(
            factor_values,
            on=["date", "permno"],
            how="left",
        )
        valid = joined.filter(pl.col("factor").is_finite())
        valid_counts = valid.group_by("date").len().rename(
            {"len": "valid_stocks"}
        )
        coverage = (
            eligible_counts.join(valid_counts, on="date", how="left")
            .with_columns(pl.col("valid_stocks").fill_null(0))
            .with_columns(
                (pl.col("valid_stocks") / pl.col("eligible_stocks")).alias(
                    "coverage"
                )
            )
        )
        median_coverage = (
            float(coverage.get_column("coverage").median())
            if coverage.height
            else 0.0
        )
        valid = (
            valid.join(
                valid_counts.filter(
                    pl.col("valid_stocks") >= self.config.minimum_cross_section
                ),
                on="date",
                how="inner",
            )
            .with_columns(
                pl.col("factor").rank(method="average").over("date").alias("x"),
                pl.col(label).rank(method="average").over("date").alias("y"),
            )
        )
        rank_ics = (
            valid.group_by("date")
            .agg(
                pl.corr("x", "y").alias("rank_ic"),
                pl.col("x").n_unique().alias("x_unique"),
                pl.col("y").n_unique().alias("y_unique"),
            )
            .filter(
                (pl.col("x_unique") > 1)
                & (pl.col("y_unique") > 1)
                & pl.col("rank_ic").is_finite()
            )
            .sort("date")
        )
        values = rank_ics.get_column("rank_ic").to_numpy()
        if len(values) == 0:
            raise ValueError("factor produced no valid Rank IC dates")
        return {
            "mean_rank_ic": float(np.mean(values)),
            "rank_ics": values,
            "valid_ic_dates": len(values),
            "median_stock_coverage": median_coverage,
        }

    def _overlapping_cohort_metrics(
        self,
        factor_values: pl.DataFrame,
        price_panel: pl.DataFrame,
        calendar: list[object],
        horizon: int,
        direction: int,
    ) -> dict[str, float]:
        if len(calendar) <= horizon + 2:
            raise ValueError("not enough calendar dates for cohort backtest")
        period_rows = [
            {
                "signal_date": calendar[index],
                "entry_index": index + 1,
                "exit_index": index + 1 + horizon,
            }
            for index in range(len(calendar) - horizon - 1)
        ]
        periods = pl.DataFrame(period_rows)
        signal_dates = periods.get_column("signal_date").to_list()
        selected = (
            factor_values.filter(pl.col("date").is_in(signal_dates))
            .with_columns((pl.col("factor") * direction).alias("oriented_factor"))
            .sort(
                ["date", "oriented_factor", "permno"],
                descending=[False, True, False],
            )
            .group_by("date", maintain_order=True)
            .head(self.config.top_k)
            .rename({"date": "signal_date"})
            .join(periods, on="signal_date", how="inner")
        )
        if selected.is_empty():
            raise ValueError("factor produced no Top-K cohort positions")
        selected = selected.with_columns(
            pl.len().over("signal_date").alias("positions"),
            pl.int_ranges(
                pl.col("entry_index") + 1,
                pl.col("exit_index") + 1,
            ).alias("return_index"),
        ).explode("return_index")

        calendar_frame = pl.DataFrame(
            {"date": calendar, "date_index": list(range(len(calendar)))}
        )
        daily_prices = (
            price_panel.select("date", "permno", "open")
            .unique(subset=["date", "permno"], keep="first")
            .join(calendar_frame, on="date", how="inner")
            .sort("permno", "date_index")
            .with_columns(
                pl.col("open").shift(1).over("permno").alias("previous_open"),
                pl.col("date_index")
                .shift(1)
                .over("permno")
                .alias("previous_index"),
            )
            .with_columns(
                (pl.col("open") / pl.col("previous_open") - 1.0).alias(
                    "open_return"
                )
            )
            .filter(
                (pl.col("date_index") - pl.col("previous_index") == 1)
                & pl.col("open_return").is_finite()
            )
            .select(
                pl.col("date_index").alias("return_index"),
                "permno",
                "open_return",
            )
        )
        cohort_weight = 1.0 / horizon
        daily_gross = (
            selected.join(
                daily_prices,
                on=["return_index", "permno"],
                how="inner",
            )
            .with_columns(
                (
                    pl.col("open_return") * cohort_weight / pl.col("positions")
                ).alias("weighted_return")
            )
            .group_by("return_index")
            .agg(pl.col("weighted_return").sum().alias("gross_return"))
        )
        entries = periods.group_by("entry_index").len().select(
            pl.col("entry_index").alias("date_index"),
            (pl.col("len") * cohort_weight).alias("entry_weight"),
        )
        exits = periods.group_by("exit_index").len().select(
            pl.col("exit_index").alias("date_index"),
            (pl.col("len") * cohort_weight).alias("exit_weight"),
        )
        daily = (
            calendar_frame.join(
                daily_gross,
                left_on="date_index",
                right_on="return_index",
                how="left",
            )
            .join(entries, on="date_index", how="left")
            .join(exits, on="date_index", how="left")
            .with_columns(
                pl.col("gross_return").fill_null(0.0),
                pl.col("entry_weight").fill_null(0.0),
                pl.col("exit_weight").fill_null(0.0),
            )
            .with_columns(
                pl.max_horizontal("entry_weight", "exit_weight").alias("turnover")
            )
            .with_columns(
                (
                    pl.col("gross_return")
                    - pl.col("turnover")
                    * self.config.transaction_cost_bps
                    / 10_000.0
                ).alias("net_return")
            )
        )
        gross = daily.get_column("gross_return").to_numpy()
        net = daily.get_column("net_return").to_numpy()
        turnover = daily.get_column("turnover").to_numpy()
        cumulative_gross = float(np.prod(1.0 + gross) - 1.0)
        cumulative_net = float(np.prod(1.0 + net) - 1.0)
        periods_per_year = 252.0
        annualized_return = (
            float((1.0 + cumulative_net) ** (periods_per_year / len(net)) - 1.0)
            if cumulative_net > -1.0 and len(net)
            else -1.0
        )
        volatility = (
            float(np.std(net, ddof=1) * math.sqrt(periods_per_year))
            if len(net) > 1
            else 0.0
        )
        sharpe = (
            float(np.mean(net) / np.std(net, ddof=1) * math.sqrt(periods_per_year))
            if len(net) > 1 and np.std(net, ddof=1) > 0
            else 0.0
        )
        equity = np.cumprod(1.0 + net)
        running_peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.min(equity / running_peak - 1.0))
        return {
            "gross_return": cumulative_gross,
            "net_return": cumulative_net,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "turnover": float(np.mean(turnover)),
        }


def _legacy_factor_spec(experiment: FactorExperimentSpec) -> FactorSpec:
    factor = experiment.factor
    return FactorSpec(
        factor_id=factor.factor_id,
        hypothesis_id=factor.hypothesis_id,
        name=factor.name,
        description=factor.description,
        formula=factor.formula,
        required_fields=factor.required_fields,
        lookback_days=factor.lookback_days,
        higher_is_better=True,
        rebalance_frequency="weekly",
        universe_filters=factor.universe_filters,
        portfolio_rules=PortfolioRules(),
    )


def _newey_west_mean_test(
    values: np.ndarray,
    max_lag: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = len(array)
    if count < 3:
        return 0.0, 1.0
    centered = array - np.mean(array)
    lag_limit = min(max_lag, count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_limit + 1):
        weight = 1.0 - lag / (lag_limit + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    if standard_error == 0:
        return 0.0, 1.0
    statistic = float(np.mean(array) / standard_error)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for reverse_rank in range(total - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * total / rank)
        adjusted[index] = min(1.0, running)
    return adjusted
