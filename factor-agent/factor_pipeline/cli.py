from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .backtest import run_weekly_topk_backtest
from .config import BacktestConfig, DataConfig, FactorConfig
from .data import PanelDataSource
from .factor import build_factor_from_spec
from .spec import FactorSpec, abnormal_volume_reversal_spec


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the deterministic abnormal-volume reversal factor backtest."
    )
    parser.add_argument("--panel", type=Path, default=repository_root / "factor-data/panel.parquet")
    parser.add_argument(
        "--membership",
        type=Path,
        default=repository_root / "factor-data/membership_intervals.parquet",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2019, 12, 31))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--factor-spec",
        type=Path,
        help="Path to a validated JSON FactorSpec. Uses the example factor by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "factor-agent/.runtime/backtests/abnormal_volume_reversal.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = (
        FactorSpec.model_validate_json(args.factor_spec.read_text(encoding="utf-8"))
        if args.factor_spec
        else abnormal_volume_reversal_spec()
    )
    warmup_days = max(90, math.ceil(spec.lookback_days * 7 / 5) + 30)
    data_config = DataConfig(
        panel_path=args.panel,
        membership_path=args.membership,
        start_date=args.start,
        end_date=args.end,
        warmup_calendar_days=warmup_days,
    )
    factor_config = FactorConfig()
    backtest_config = BacktestConfig(
        top_k=spec.portfolio_rules.top_k,
        exit_rank=spec.portfolio_rules.exit_rank,
        transaction_cost_bps=args.cost_bps,
    )
    source = PanelDataSource(data_config)
    validation = source.validate()
    panel_history = source.panel_history()
    factor = build_factor_from_spec(
        source.apply_membership(panel_history), spec, factor_config
    )
    result, periods = run_weekly_topk_backtest(
        factor_frame=factor,
        price_panel=panel_history,
        calendar=source.market_calendar(),
        start_date=args.start,
        end_date=args.end,
        config=backtest_config,
    )
    payload = {
        "factor_spec": spec.model_dump(mode="json", exclude_none=True),
        "data_validation": validation,
        "data_config": {
            **asdict(data_config),
            "panel_path": str(data_config.panel_path),
            "membership_path": str(data_config.membership_path),
            "start_date": data_config.start_date.isoformat(),
            "end_date": data_config.end_date.isoformat(),
        },
        "factor_config": asdict(factor_config),
        "backtest": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    periods.write_parquet(args.output.with_suffix(".periods.parquet"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
