"""Run the one-time frozen final test for the selected evolved factor."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import polars as pl

from factor_pipeline.backtest import run_weekly_topk_backtest
from factor_pipeline.config import BacktestConfig, DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    build_forward_return_labels,
    evaluate_single_factor,
)
from factor_pipeline.spec import FactorSpec

from .state import EvaluatedCandidate


DEFAULT_CHILD_ID = "G02-CROSS-006-INTERMEDIATE"
DEFAULT_PARENT_ID = "ALPHA158-MIN5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen final test for one evolved factor and its parent."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("factor-agent/results/evoalpha_deepseek_2001_2012_3gen"),
    )
    parser.add_argument("--child-id", default=DEFAULT_CHILD_ID)
    parser.add_argument("--parent-id", default=DEFAULT_PARENT_ID)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2013, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_final_test_2013_2025"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"final test already exists at {summary_path}; do not rerun or tune on it"
        )
    if args.end < args.start:
        raise ValueError("test end date must not precede start date")

    candidates = _load_candidates(args.experiment_root)
    selected = {
        factor_id: candidates[factor_id]
        for factor_id in (args.parent_id, args.child_id)
    }
    period = EvaluationPeriod(args.start, args.end)
    source = PanelDataSource(
        DataConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            start_date=args.start,
            end_date=args.end,
            warmup_calendar_days=366,
            exit_buffer_calendar_days=45,
        )
    )
    data_validation = source.validate()
    panel = source.point_in_time_panel()
    panel_history = source.panel_history()
    calendar = source.market_calendar()
    evaluation_panel = panel.filter(
        pl.col("date").dt.date() <= args.end
    ).collect(engine="streaming")
    evaluation_config = FactorEvaluationConfig()
    labels = build_forward_return_labels(
        evaluation_panel,
        period,
        evaluation_config.horizons,
    )

    results = {}
    for factor_id, candidate in selected.items():
        print(f"[Final Test] {factor_id}", flush=True)
        spec = FactorSpec.model_validate(candidate.artifact["factor_spec"])
        factor_values = build_factor_from_spec(panel, spec, FactorConfig())
        if candidate.direction == -1:
            factor_values = factor_values.with_columns(
                (-pl.col("factor")).alias("factor"),
                (-pl.col("raw_factor")).alias("raw_factor"),
            )
        prediction = evaluate_single_factor(
            factor_values.select("date", "permno", "factor").collect(
                engine="streaming"
            ),
            labels,
            factor_id,
            period,
            evaluation_config,
            direction=1,
        )
        backtest, _ = run_weekly_topk_backtest(
            factor_values,
            panel_history,
            calendar,
            args.start,
            args.end,
            BacktestConfig(
                top_k=spec.portfolio_rules.top_k,
                exit_rank=spec.portfolio_rules.exit_rank,
                transaction_cost_bps=args.cost_bps,
            ),
        )
        results[factor_id] = {
            "factor_id": factor_id,
            "training_direction_frozen": candidate.direction,
            "factor_spec": spec.model_dump(mode="json"),
            "prediction_metrics": prediction.to_dict(),
            "backtest_metrics": backtest.to_dict(),
        }

    parent = results[args.parent_id]
    child = results[args.child_id]
    prediction_keys = (
        "mean_rank_ic_5d",
        "mean_rank_ic_10d",
        "mean_rank_ic_20d",
        "rank_icir_20d",
        "monthly_stability",
    )
    backtest_keys = (
        "annualized_net_return",
        "annualized_sharpe",
        "average_turnover",
        "max_drawdown",
    )
    deltas = {
        **{
            key: child["prediction_metrics"][key]
            - parent["prediction_metrics"][key]
            for key in prediction_keys
        },
        **{
            key: child["backtest_metrics"][key]
            - parent["backtest_metrics"][key]
            for key in backtest_keys
        },
    }
    payload = {
        "protocol": {
            "period": {
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
            },
            "training_period": "2001-01-01/2010-12-31",
            "validation_period": "2011-01-01/2012-12-31",
            "formula_frozen": True,
            "direction_frozen_from_training": True,
            "parameter_tuning_performed": False,
            "selection_performed": False,
            "one_time_final_test": True,
            "transaction_cost_bps": args.cost_bps,
        },
        "data_validation": data_validation,
        "parent_factor_id": args.parent_id,
        "child_factor_id": args.child_id,
        "results": results,
        "child_minus_parent": deltas,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "final_test_results.json").write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "period": payload["protocol"]["period"],
        "parent_factor_id": args.parent_id,
        "child_factor_id": args.child_id,
        "child_minus_parent": deltas,
    }
    summary_path.write_text(
        json.dumps(
            _json_safe(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    return 0


def _load_candidates(root: Path) -> dict[str, EvaluatedCandidate]:
    paths = [
        root / "seed" / "alpha158_seed_candidates.json",
        *(
            root
            / f"generation_{generation:02d}"
            / f"generation_{generation:02d}_candidate_evaluations.json"
            for generation in range(1, 4)
        ),
    ]
    candidates = {}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")):
            candidate = EvaluatedCandidate.model_validate(item)
            candidates[candidate.factor_id] = candidate
    return candidates


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
