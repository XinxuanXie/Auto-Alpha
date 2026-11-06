"""Run frozen out-of-sample validation for one evolved factor."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
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
    score_factor_metrics,
)
from factor_pipeline.spec import FactorSpec

from .candidate_evaluation import load_score_reference
from .state import EvaluatedCandidate


DEFAULT_FACTOR_ID = "G01-CROSS-004-MUTATION"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one frozen evolved factor out of sample."
    )
    generation_root = Path("factor-agent/results/evoalpha_generation_01_real")
    seed_root = Path("factor-agent/results/evoalpha_seed_2001")
    parser.add_argument("--factor-id", default=DEFAULT_FACTOR_ID)
    parser.add_argument(
        "--evaluations",
        type=Path,
        default=generation_root / "generation_01_candidate_evaluations.json",
    )
    parser.add_argument(
        "--score-reference",
        type=Path,
        default=seed_root / "alpha158_score_reference.json",
    )
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2002, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2002, 12, 31))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_validation_2002"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate = _load_candidate(args.evaluations, args.factor_id)
    spec = FactorSpec.model_validate(candidate.artifact["factor_spec"])
    period = EvaluationPeriod(start_date=args.start, end_date=args.end)
    warmup_days = max(90, math.ceil(spec.lookback_days * 7 / 5) + 30)
    data_config = DataConfig(
        panel_path=args.panel,
        membership_path=args.membership,
        start_date=args.start,
        end_date=args.end,
        warmup_calendar_days=warmup_days,
        exit_buffer_calendar_days=45,
    )
    factor_config = FactorConfig()
    source = PanelDataSource(data_config)
    data_validation = source.validate()
    panel_history = source.panel_history()
    point_in_time_panel = source.apply_membership(panel_history)
    factor_values = build_factor_from_spec(
        point_in_time_panel,
        spec,
        factor_config,
    )
    oriented_factor_values = _apply_frozen_direction(
        factor_values,
        candidate.direction,
    )
    evaluation_panel = point_in_time_panel.collect(engine="streaming")
    labels = build_forward_return_labels(
        evaluation_panel,
        period,
        FactorEvaluationConfig().horizons,
    )
    prediction_metrics = evaluate_single_factor(
        factor_values=oriented_factor_values.select(
            "date", "permno", "factor"
        ).collect(engine="streaming"),
        labels=labels,
        factor_id=candidate.factor_id,
        period=period,
        direction=1,
    )
    scored_metrics = score_factor_metrics(
        [prediction_metrics],
        reference=load_score_reference(args.score_reference),
    )[0]
    backtest_config = BacktestConfig(
        top_k=spec.portfolio_rules.top_k,
        exit_rank=spec.portfolio_rules.exit_rank,
        transaction_cost_bps=args.cost_bps,
    )
    backtest, weekly_periods = run_weekly_topk_backtest(
        factor_frame=oriented_factor_values,
        price_panel=panel_history,
        calendar=source.market_calendar(),
        start_date=args.start,
        end_date=args.end,
        config=backtest_config,
    )
    training_metrics = {
        key: candidate.metrics.get(key)
        for key in (
            "mean_rank_ic_5d",
            "mean_rank_ic_10d",
            "mean_rank_ic_20d",
            "rank_icir_20d",
            "monthly_stability",
            "cross_horizon_consistency",
            "composite_score",
        )
    }
    validation_metrics = scored_metrics.to_dict()
    payload = {
        "protocol": {
            "period": {"start": args.start.isoformat(), "end": args.end.isoformat()},
            "formula_frozen": True,
            "direction_frozen": candidate.direction,
            "parameter_tuning_performed": False,
            "selection_performed": False,
            "score_reference": str(args.score_reference),
        },
        "factor_spec": spec.model_dump(mode="json"),
        "data_validation": _json_safe(data_validation),
        "data_config": {
            **asdict(data_config),
            "panel_path": str(data_config.panel_path),
            "membership_path": str(data_config.membership_path),
            "start_date": data_config.start_date.isoformat(),
            "end_date": data_config.end_date.isoformat(),
        },
        "training_metrics_2001": training_metrics,
        "validation_metrics_2002": validation_metrics,
        "validation_minus_training": {
            key: validation_metrics[key] - training_metrics[key]
            for key in training_metrics
            if training_metrics[key] is not None and key in validation_metrics
        },
        "top20_weekly_backtest_2002": backtest.to_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{candidate.factor_id}.json"
    output_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    weekly_periods.write_parquet(
        args.output_dir / f"{candidate.factor_id}.periods.parquet"
    )
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")
    return 0


def _load_candidate(path: Path, factor_id: str) -> EvaluatedCandidate:
    candidates = [
        EvaluatedCandidate.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    matches = [candidate for candidate in candidates if candidate.factor_id == factor_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one candidate for factor_id={factor_id}")
    return matches[0]


def _apply_frozen_direction(
    factor_values: pl.LazyFrame,
    direction: int,
) -> pl.LazyFrame:
    if direction not in {-1, 1}:
        raise ValueError("frozen direction must be -1 or 1")
    if direction == 1:
        return factor_values
    return factor_values.with_columns(
        (pl.col("factor") * direction).alias("factor"),
        (pl.col("raw_factor") * direction).alias("raw_factor"),
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
