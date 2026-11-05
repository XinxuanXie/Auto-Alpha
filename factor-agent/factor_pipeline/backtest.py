from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import sqrt

import numpy as np
import polars as pl

from .config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    start_date: str
    end_date: str
    periods: int
    selected_positions: int
    missing_price_positions: int
    average_positions: float
    average_turnover: float
    cumulative_gross_return: float
    cumulative_net_return: float
    annualized_net_return: float
    annualized_net_volatility: float
    annualized_sharpe: float
    max_drawdown: float
    mean_rank_ic: float
    rank_icir: float
    annualized_rank_icir: float
    transaction_cost_bps: float
    top_k: int
    exit_rank: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_weekly_periods(
    calendar: list[datetime], start_date: date, end_date: date
) -> pl.DataFrame:
    ordered = sorted(calendar)
    weekly_last: dict[tuple[int, int], datetime] = {}
    for timestamp in ordered:
        iso = timestamp.isocalendar()
        weekly_last[(iso.year, iso.week)] = timestamp
    signal_dates = sorted(weekly_last.values())
    rows: list[dict[str, datetime]] = []
    for index, signal_date in enumerate(signal_dates[:-1]):
        if not (start_date <= signal_date.date() <= end_date):
            continue
        entry_index = bisect_right(ordered, signal_date)
        next_signal = signal_dates[index + 1]
        exit_index = bisect_right(ordered, next_signal)
        if entry_index >= len(ordered) or exit_index >= len(ordered):
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": ordered[entry_index],
                "exit_date": ordered[exit_index],
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "signal_date": pl.Datetime,
            "entry_date": pl.Datetime,
            "exit_date": pl.Datetime,
        },
    )


def run_weekly_topk_backtest(
    factor_frame: pl.LazyFrame,
    price_panel: pl.LazyFrame,
    calendar: list[datetime],
    start_date: date,
    end_date: date,
    config: BacktestConfig,
) -> tuple[BacktestResult, pl.DataFrame]:
    periods = build_weekly_periods(calendar, start_date, end_date)
    if periods.is_empty():
        raise ValueError("no complete weekly holding periods in requested date range")

    signal_dates = periods.get_column("signal_date").to_list()
    scores = (
        factor_frame.filter(pl.col("date").is_in(signal_dates))
        .select(
            pl.col("date").alias("signal_date"),
            "permno",
            "factor",
            "raw_factor",
        )
        .collect(engine="streaming")
    )
    if scores.is_empty():
        raise ValueError("factor pipeline produced no scores on rebalance dates")

    prices = _load_execution_prices(price_panel, periods)
    forward = (
        scores.join(periods, on="signal_date", how="inner")
        .join(
            prices.rename({"date": "entry_date", "open": "entry_open"}),
            on=["entry_date", "permno"],
            how="left",
        )
        .join(
            prices.rename({"date": "exit_date", "open": "exit_open"}),
            on=["exit_date", "permno"],
            how="left",
        )
        .with_columns(
            (pl.col("exit_open") / pl.col("entry_open") - 1.0).alias(
                "forward_return"
            )
        )
    )
    selected = _select_positions(forward, config)
    missing_price_positions = selected.filter(
        pl.col("forward_return").is_null()
        | ~pl.col("forward_return").is_finite()
    ).height
    valid_selected = selected.filter(pl.col("forward_return").is_finite())
    period_rows = _portfolio_periods(valid_selected, periods, config)
    rank_ics = _daily_rank_ics(forward)
    result = _summarize(
        period_rows=period_rows,
        rank_ics=rank_ics,
        start_date=start_date,
        end_date=end_date,
        selected_positions=selected.height,
        missing_price_positions=missing_price_positions,
        config=config,
    )
    return result, period_rows


def _select_positions(
    forward: pl.DataFrame,
    config: BacktestConfig,
) -> pl.DataFrame:
    ordered = forward.sort(
        ["signal_date", "factor", "permno"],
        descending=[False, True, False],
    )
    if config.exit_rank is None:
        return ordered.group_by(
            "signal_date", maintain_order=True
        ).head(config.top_k)

    selected_frames = []
    previous_instruments: set[int] = set()
    for group in ordered.partition_by("signal_date", maintain_order=True):
        ranked_instruments = group.get_column("permno").to_list()
        eligible_to_hold = set(ranked_instruments[: config.exit_rank])
        selected_instruments = [
            instrument
            for instrument in ranked_instruments
            if instrument in previous_instruments
            and instrument in eligible_to_hold
        ]
        selected_set = set(selected_instruments)
        for instrument in ranked_instruments:
            if len(selected_instruments) >= config.top_k:
                break
            if instrument not in selected_set:
                selected_instruments.append(instrument)
                selected_set.add(instrument)
        selected_frames.append(
            group.filter(pl.col("permno").is_in(selected_instruments))
        )
        previous_instruments = selected_set
    return pl.concat(selected_frames) if selected_frames else ordered.head(0)


def _load_execution_prices(
    price_panel: pl.LazyFrame, periods: pl.DataFrame
) -> pl.DataFrame:
    execution_dates = sorted(
        set(periods.get_column("entry_date").to_list())
        | set(periods.get_column("exit_date").to_list())
    )
    return (
        price_panel.filter(pl.col("date").is_in(execution_dates))
        .select("date", "permno", "open")
        .filter(pl.col("open").is_finite() & (pl.col("open") > 0))
        .collect(engine="streaming")
        .unique(subset=["date", "permno"], keep="first")
    )


def _portfolio_periods(
    selected: pl.DataFrame,
    periods: pl.DataFrame,
    config: BacktestConfig,
) -> pl.DataFrame:
    selected_by_date: dict[datetime, set[int]] = {}
    returns_by_date: dict[datetime, list[float]] = {}
    for row in selected.iter_rows(named=True):
        signal_date = row["signal_date"]
        selected_by_date.setdefault(signal_date, set()).add(row["permno"])
        returns_by_date.setdefault(signal_date, []).append(row["forward_return"])

    rows: list[dict[str, object]] = []
    previous_weights: dict[int, float] = {}
    cost_rate = config.transaction_cost_bps / 10_000.0
    for period in periods.iter_rows(named=True):
        signal_date = period["signal_date"]
        instruments = selected_by_date.get(signal_date, set())
        returns = returns_by_date.get(signal_date, [])
        if not instruments or not returns:
            continue
        current_weight = 1.0 / len(instruments)
        current_weights = {instrument: current_weight for instrument in instruments}
        turnover = _one_way_turnover(previous_weights, current_weights)
        gross_return = float(np.mean(returns))
        transaction_cost = turnover * cost_rate
        rows.append(
            {
                **period,
                "positions": len(instruments),
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": gross_return - transaction_cost,
            }
        )
        previous_weights = current_weights
    return pl.DataFrame(rows)


def _one_way_turnover(
    previous_weights: dict[int, float], current_weights: dict[int, float]
) -> float:
    if not previous_weights:
        return float(sum(current_weights.values()))
    instruments = set(previous_weights) | set(current_weights)
    return 0.5 * sum(
        abs(current_weights.get(item, 0.0) - previous_weights.get(item, 0.0))
        for item in instruments
    )


def _daily_rank_ics(frame: pl.DataFrame) -> list[float]:
    values: list[float] = []
    valid = frame.filter(
        pl.col("factor").is_finite() & pl.col("forward_return").is_finite()
    )
    for group in valid.partition_by("signal_date", maintain_order=True):
        if group.height < 3:
            continue
        factor = group.get_column("factor").to_numpy()
        returns = group.get_column("forward_return").to_numpy()
        factor_ranks = _average_ranks(factor)
        return_ranks = _average_ranks(returns)
        correlation = float(np.corrcoef(factor_ranks, return_ranks)[0, 1])
        if np.isfinite(correlation):
            values.append(correlation)
    return values


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0
        ranks[order[position:end]] = average
        position = end
    return ranks


def _summarize(
    period_rows: pl.DataFrame,
    rank_ics: list[float],
    start_date: date,
    end_date: date,
    selected_positions: int,
    missing_price_positions: int,
    config: BacktestConfig,
) -> BacktestResult:
    if period_rows.is_empty():
        raise ValueError("no tradable positions remained after price validation")
    gross = period_rows.get_column("gross_return").to_numpy()
    net = period_rows.get_column("net_return").to_numpy()
    wealth = np.cumprod(1.0 + net)
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    period_count = len(net)
    cumulative_gross = float(np.prod(1.0 + gross) - 1.0)
    cumulative_net = float(wealth[-1] - 1.0)
    annualized_net = float((1.0 + cumulative_net) ** (52.0 / period_count) - 1.0)
    weekly_std = float(np.std(net, ddof=1)) if period_count > 1 else float("nan")
    annualized_volatility = weekly_std * sqrt(52.0)
    sharpe = (
        float(np.mean(net) / weekly_std * sqrt(52.0))
        if weekly_std > 0
        else float("nan")
    )
    rank_ic_array = np.asarray(rank_ics, dtype=float)
    rank_ic_mean = float(np.mean(rank_ic_array)) if len(rank_ic_array) else float("nan")
    rank_ic_std = (
        float(np.std(rank_ic_array, ddof=1))
        if len(rank_ic_array) > 1
        else float("nan")
    )
    rank_icir = (
        rank_ic_mean / rank_ic_std if rank_ic_std > 0 else float("nan")
    )
    return BacktestResult(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        periods=period_count,
        selected_positions=selected_positions,
        missing_price_positions=missing_price_positions,
        average_positions=float(period_rows.get_column("positions").mean()),
        average_turnover=float(period_rows.get_column("turnover").mean()),
        cumulative_gross_return=cumulative_gross,
        cumulative_net_return=cumulative_net,
        annualized_net_return=annualized_net,
        annualized_net_volatility=annualized_volatility,
        annualized_sharpe=sharpe,
        max_drawdown=float(np.min(drawdowns)),
        mean_rank_ic=rank_ic_mean,
        rank_icir=rank_icir,
        annualized_rank_icir=rank_icir * sqrt(52.0),
        transaction_cost_bps=config.transaction_cost_bps,
        top_k=config.top_k,
        exit_rank=config.exit_rank,
    )

