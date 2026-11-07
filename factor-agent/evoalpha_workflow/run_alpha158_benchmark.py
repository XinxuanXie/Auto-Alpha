"""Run resumable full-history annual and rolling Alpha158 diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import polars as pl

from factor_pipeline.alpha158 import executable_alpha158_specs
from factor_pipeline.backtest import run_weekly_topk_backtest
from factor_pipeline.config import BacktestConfig, DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import FactorEvaluationConfig

from .alpha158_benchmark import (
    aggregate_backtest_window,
    aggregate_prediction_window,
    build_benchmark_windows,
    build_daily_factor_statistics,
    build_forward_return_labels_lazy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark all executable Alpha158 factors by year and rolling window."
    )
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/alpha158_full_history_benchmark"),
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("factor-agent/.runtime/alpha158_full_history_benchmark"),
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--factor-id",
        action="append",
        help="Optional repeatable factor ID filter for smoke tests; default runs all 157.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    first_date, last_date = _panel_bounds(args.panel)
    windows = build_benchmark_windows(first_date, last_date)
    all_specs = executable_alpha158_specs()
    requested_ids = set(args.factor_id or ())
    specs = tuple(
        spec for spec in all_specs if not requested_ids or spec.factor_id in requested_ids
    )
    if requested_ids - {spec.factor_id for spec in specs}:
        raise ValueError(
            f"unknown factor IDs: {sorted(requested_ids - {spec.factor_id for spec in specs})}"
        )
    protocol = _protocol(args, first_date, last_date, windows, specs)
    _validate_or_write_protocol(args.output_dir / "protocol.json", protocol)

    source = PanelDataSource(
        DataConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            start_date=first_date,
            end_date=last_date,
            warmup_calendar_days=0,
            exit_buffer_calendar_days=0,
        )
    )
    source.validate()
    panel = source.point_in_time_panel()
    panel_history = source.panel_history()
    calendar = source.market_calendar()
    evaluation_config = FactorEvaluationConfig()
    labels_path = args.runtime_dir / "labels.parquet"
    labels_meta_path = args.runtime_dir / "labels_meta.json"
    labels_meta = {
        "schema_version": 2,
        "panel": _file_identity(args.panel),
        "membership": _file_identity(args.membership),
        "first_date": first_date.isoformat(),
        "last_date": last_date.isoformat(),
        "horizons": list(evaluation_config.horizons),
    }
    if not labels_path.is_file() or _read_json(labels_meta_path) != labels_meta:
        print("[Setup] materialize full-history forward-return labels", flush=True)
        build_forward_return_labels_lazy(
            panel,
            calendar,
            evaluation_config.horizons,
        ).sink_parquet(labels_path)
        _write_json(labels_meta_path, labels_meta)
    labels = pl.scan_parquet(labels_path)

    failures = _read_json(args.output_dir / "failures.json") or {}
    factor_temp_path = args.runtime_dir / "current_factor.parquet"
    for index, spec in enumerate(specs, start=1):
        checkpoint_path = checkpoint_dir / f"{spec.factor_id}.parquet"
        if checkpoint_path.is_file():
            print(f"[{index}/{len(specs)}] resume {spec.factor_id}", flush=True)
            continue
        print(f"[{index}/{len(specs)}] evaluate {spec.factor_id}", flush=True)
        try:
            build_factor_from_spec(panel, spec, FactorConfig()).select(
                "date", "permno", "factor", "raw_factor"
            ).sink_parquet(factor_temp_path)
            factor_values = pl.scan_parquet(factor_temp_path)
            daily = build_daily_factor_statistics(
                factor_values,
                labels,
                evaluation_config,
            )
            backtest_config = BacktestConfig(
                top_k=spec.portfolio_rules.top_k,
                exit_rank=spec.portfolio_rules.exit_rank,
                transaction_cost_bps=args.cost_bps,
            )
            _, positive_periods = run_weekly_topk_backtest(
                factor_values,
                panel_history,
                calendar,
                first_date,
                last_date,
                backtest_config,
            )
            negative_values = factor_values.with_columns(
                (-pl.col("factor")).alias("factor"),
                (-pl.col("raw_factor")).alias("raw_factor"),
            )
            _, negative_periods = run_weekly_topk_backtest(
                negative_values,
                panel_history,
                calendar,
                first_date,
                last_date,
                backtest_config,
            )
            rows = []
            for window in windows:
                prediction = aggregate_prediction_window(
                    daily,
                    window,
                    evaluation_config,
                )
                backtest = aggregate_backtest_window(
                    positive_periods,
                    negative_periods,
                    window,
                    prediction["direction"],
                )
                rows.append(
                    {
                        "factor_id": spec.factor_id,
                        **window.to_dict(),
                        **prediction,
                        **backtest,
                        "transaction_cost_bps": args.cost_bps,
                        "top_k": spec.portfolio_rules.top_k,
                        "exit_rank": spec.portfolio_rules.exit_rank,
                    }
                )
            pl.DataFrame(rows).write_parquet(checkpoint_path)
            failures.pop(spec.factor_id, None)
        except Exception as error:
            failures[spec.factor_id] = f"{type(error).__name__}: {error}"
            _write_json(args.output_dir / "failures.json", failures)
            raise
        finally:
            factor_temp_path.unlink(missing_ok=True)
        _write_json(args.output_dir / "failures.json", failures)

    checkpoint_paths = sorted(checkpoint_dir.glob("ALPHA158-*.parquet"))
    if len(checkpoint_paths) != len(specs):
        raise ValueError(
            f"expected {len(specs)} factor checkpoints, found {len(checkpoint_paths)}"
        )
    results = pl.concat(
        [pl.read_parquet(path) for path in checkpoint_paths],
        how="diagonal_relaxed",
    )
    expected_rows = len(specs) * len(windows)
    if results.height != expected_rows:
        raise ValueError(f"expected {expected_rows} result rows, found {results.height}")
    results = results.sort("window_type", "end_year", "factor_id")
    results.write_parquet(args.output_dir / "alpha158_benchmark_metrics.parquet")
    csv_rows = results.to_dicts()
    for row in csv_rows:
        row["rejection_reasons"] = "|".join(row["rejection_reasons"] or ())
    pl.DataFrame(csv_rows).write_csv(
        args.output_dir / "alpha158_benchmark_metrics.csv"
    )
    _write_analysis_outputs(results, args.output_dir)
    summary = {
        "factor_count": len(specs),
        "window_count_per_factor": len(windows),
        "result_rows": results.height,
        "annual_windows": sum(item.window_type == "annual" for item in windows),
        "rolling_5y_windows": sum(item.window_type == "rolling_5y" for item in windows),
        "rolling_10y_windows": sum(item.window_type == "rolling_10y" for item in windows),
        "rolling_20y_windows": sum(item.window_type == "rolling_20y" for item in windows),
        "data_start": first_date.isoformat(),
        "data_end": last_date.isoformat(),
        "failures": failures,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _panel_bounds(path: Path) -> tuple[date, date]:
    bounds = (
        pl.scan_parquet(path)
        .select(pl.col("date").min().alias("min"), pl.col("date").max().alias("max"))
        .collect()
        .row(0, named=True)
    )
    return bounds["min"].date(), bounds["max"].date()


def _write_analysis_outputs(results: pl.DataFrame, output_dir: Path) -> None:
    for window_type in ("annual", "rolling_5y", "rolling_10y", "rolling_20y"):
        subset = results.filter(pl.col("window_type") == window_type)
        subset.write_parquet(output_dir / f"{window_type}_metrics.parquet")
        csv_rows = subset.to_dicts()
        for row in csv_rows:
            row["rejection_reasons"] = "|".join(row["rejection_reasons"] or ())
        pl.DataFrame(csv_rows).write_csv(output_dir / f"{window_type}_metrics.csv")

    window_summary = (
        results.group_by(
            "window_type",
            "window_years",
            "end_year",
            "start_date",
            "end_date",
            "is_partial",
        )
        .agg(
            pl.len().alias("factor_count"),
            pl.col("passed_hard_filters").sum().alias("passed_hard_filters"),
            pl.col("mean_rank_ic_20d").median().alias("median_mean_rank_ic_20d"),
            pl.col("rank_icir_20d").median().alias("median_rank_icir_20d"),
            pl.col("monthly_stability").median().alias("median_monthly_stability"),
            pl.col("annualized_net_return").median().alias(
                "median_annualized_net_return"
            ),
            pl.col("average_turnover").median().alias("median_average_turnover"),
            pl.col("max_drawdown").median().alias("median_max_drawdown"),
        )
        .sort("window_years", "end_year")
    )
    window_summary.write_parquet(output_dir / "window_summary.parquet")
    window_summary.write_csv(output_dir / "window_summary.csv")

    annual = results.filter(pl.col("window_type") == "annual")
    factor_summary = annual.group_by("factor_id").agg(
        pl.len().alias("annual_windows"),
        pl.col("passed_hard_filters").sum().alias("annual_windows_passed"),
        (pl.col("direction") == 1).mean().alias("positive_direction_share"),
        pl.col("mean_rank_ic_5d").median().alias("median_annual_rank_ic_5d"),
        pl.col("mean_rank_ic_10d").median().alias("median_annual_rank_ic_10d"),
        pl.col("mean_rank_ic_20d").median().alias("median_annual_rank_ic_20d"),
        pl.col("rank_icir_20d").median().alias("median_annual_rank_icir_20d"),
        pl.col("monthly_stability").median().alias(
            "median_annual_monthly_stability"
        ),
        (pl.col("annualized_net_return") > 0).mean().alias(
            "positive_annual_return_share"
        ),
        pl.col("annualized_net_return").median().alias(
            "median_annualized_net_return"
        ),
        pl.col("average_turnover").median().alias("median_annual_turnover"),
        pl.col("max_drawdown").median().alias("median_annual_max_drawdown"),
    )
    latest_20y_end = results.filter(
        pl.col("window_type") == "rolling_20y"
    ).get_column("end_year").max()
    latest_20y = (
        results.filter(
            (pl.col("window_type") == "rolling_20y")
            & (pl.col("end_year") == latest_20y_end)
        )
        .select(
            "factor_id",
            pl.col("direction").alias("latest_20y_direction"),
            pl.col("mean_rank_ic_20d").alias("latest_20y_mean_rank_ic_20d"),
            pl.col("rank_icir_20d").alias("latest_20y_rank_icir_20d"),
            pl.col("monthly_stability").alias("latest_20y_monthly_stability"),
            pl.col("annualized_net_return").alias(
                "latest_20y_annualized_net_return"
            ),
            pl.col("average_turnover").alias("latest_20y_average_turnover"),
            pl.col("max_drawdown").alias("latest_20y_max_drawdown"),
        )
    )
    factor_summary = factor_summary.join(latest_20y, on="factor_id", how="left").sort(
        "median_annual_rank_ic_20d",
        descending=True,
    )
    factor_summary.write_parquet(output_dir / "factor_stability_summary.parquet")
    factor_summary.write_csv(output_dir / "factor_stability_summary.csv")


def _protocol(args, first_date, last_date, windows, specs) -> dict[str, object]:
    return {
        "panel": _file_identity(args.panel),
        "membership": _file_identity(args.membership),
        "data_start": first_date.isoformat(),
        "data_end": last_date.isoformat(),
        "factors": len(specs),
        "factor_ids": [spec.factor_id for spec in specs],
        "rolling_years": [5, 10, 20],
        "rolling_step": "one calendar year",
        "window_count_per_factor": len(windows),
        "direction_policy": "re-estimate sign from 20d Rank IC within each diagnostic window",
        "prediction_horizons_days": [5, 10, 20],
        "portfolio": {"top_k": 20, "exit_rank": 30, "equal_weight": True},
        "transaction_cost_bps": args.cost_bps,
        "uses_lightgbm": False,
    }


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validate_or_write_protocol(path: Path, protocol: dict[str, object]) -> None:
    existing = _read_json(path)
    if existing is not None and existing != protocol:
        raise ValueError("existing benchmark protocol does not match current inputs")
    _write_json(path, protocol)


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
