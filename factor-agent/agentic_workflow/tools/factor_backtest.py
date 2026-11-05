"""Real FactorSpec backtest tool backed by the Polars factor pipeline."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
from factor_pipeline.backtest import run_weekly_topk_backtest
from factor_pipeline.config import BacktestConfig, DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.spec import FactorSpec

from agentic_workflow.state import WorkflowState
from agentic_workflow.tools.base import ToolResult


class FactorBacktestTool:
    name = "factor_backtest"

    def __init__(
        self,
        panel_path: str | Path,
        membership_path: str | Path,
        artifact_root: str | Path,
        *,
        start_date: date = date(2015, 1, 1),
        end_date: date = date(2019, 12, 31),
        top_k: int = 20,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        if top_k != 20:
            raise ValueError("FactorBacktestTool requires the fixed top_k=20 policy")
        self.panel_path = Path(panel_path)
        self.membership_path = Path(membership_path)
        self.artifact_root = Path(artifact_root)
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_config = BacktestConfig(
            top_k=20,
            exit_rank=30,
            transaction_cost_bps=transaction_cost_bps,
        )
        self.factor_config = FactorConfig()

    def run(self, state: WorkflowState) -> ToolResult:
        if state["factor_spec"] is None:
            return ToolResult.fail("A FactorSpec is required before backtesting")
        try:
            spec = FactorSpec.model_validate(state["factor_spec"])
            warmup_days = max(
                90,
                math.ceil(spec.lookback_days * 7 / 5) + 30,
            )
            data_config = DataConfig(
                panel_path=self.panel_path,
                membership_path=self.membership_path,
                start_date=self.start_date,
                end_date=self.end_date,
                warmup_calendar_days=warmup_days,
            )
            source = PanelDataSource(data_config)
            validation = source.validate()
            panel_history = source.panel_history()
            eligible_history = source.apply_membership(panel_history)
            factor = build_factor_from_spec(
                eligible_history,
                spec,
                self.factor_config,
            )
            result, periods = run_weekly_topk_backtest(
                factor_frame=factor,
                price_panel=panel_history,
                calendar=source.market_calendar(),
                start_date=self.start_date,
                end_date=self.end_date,
                config=self.backtest_config,
            )
            artifact_directory = (
                self.artifact_root
                / state["run_id"]
                / f"iteration_{state['iteration']:04d}"
                / spec.factor_id
                / f"attempt_{state['attempt']:04d}"
            )
            artifact_directory.mkdir(parents=True, exist_ok=True)
            periods_path = artifact_directory / "periods.parquet"
            summary_path = artifact_directory / "summary.json"
            periods.write_parquet(periods_path)
            backtest_result = {
                **result.to_dict(),
                "factor_id": spec.factor_id,
                "universe_filters": (
                    spec.universe_filters.model_dump(mode="json")
                    if spec.universe_filters is not None
                    else None
                ),
                "portfolio_rules": spec.portfolio_rules.model_dump(mode="json"),
                "rank_ic": result.mean_rank_ic,
                "icir": result.rank_icir,
                "net_return": result.cumulative_net_return,
                "turnover": result.average_turnover,
                "data_validation": validation,
                "periods_artifact": str(periods_path),
                "summary_artifact": str(summary_path),
            }
            summary_path.write_text(
                json.dumps(
                    {
                        "factor_spec": spec.model_dump(
                            mode="json", exclude_none=True
                        ),
                        "backtest_result": backtest_result,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            return ToolResult.fail(f"Factor backtest failed: {error}")
        return ToolResult.ok({"backtest_result": backtest_result})
