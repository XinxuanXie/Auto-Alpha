from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from .config import DataConfig


PANEL_SCHEMA = {
    "date": pl.Datetime,
    "permno": pl.Int32,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "mktcap": pl.Float64,
}
MEMBERSHIP_COLUMNS = {"permno", "start_date", "end_date"}


class DataValidationError(ValueError):
    pass


class PanelDataSource:
    def __init__(self, config: DataConfig):
        self.config = config

    def validate(self) -> dict[str, object]:
        self._require_file(self.config.panel_path)
        self._require_file(self.config.membership_path)
        panel = pl.scan_parquet(self.config.panel_path)
        membership = pl.scan_parquet(self.config.membership_path)
        panel_schema = panel.collect_schema()
        membership_schema = membership.collect_schema()

        missing_panel = set(PANEL_SCHEMA) - set(panel_schema.names())
        if missing_panel:
            raise DataValidationError(
                f"panel is missing required columns: {sorted(missing_panel)}"
            )
        missing_membership = MEMBERSHIP_COLUMNS - set(membership_schema.names())
        if missing_membership:
            raise DataValidationError(
                "membership is missing required columns: "
                f"{sorted(missing_membership)}"
            )
        if not panel_schema["date"].is_temporal():
            raise DataValidationError("panel.date must be a temporal type")
        if not membership_schema["start_date"].is_temporal():
            raise DataValidationError("membership.start_date must be a temporal type")
        if not membership_schema["end_date"].is_temporal():
            raise DataValidationError("membership.end_date must be a temporal type")

        summary = panel.select(
            pl.len().alias("rows"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("permno").n_unique().alias("stocks"),
        ).collect(engine="streaming").row(0, named=True)
        return {
            "panel_path": str(self.config.panel_path),
            "membership_path": str(self.config.membership_path),
            **summary,
        }

    def point_in_time_panel(self) -> pl.LazyFrame:
        self.validate()
        return self.apply_membership(self.panel_history())

    def panel_history(self) -> pl.LazyFrame:
        self.validate()
        scan_start = self.config.start_date - timedelta(
            days=self.config.warmup_calendar_days
        )
        scan_end = self.config.end_date + timedelta(
            days=self.config.exit_buffer_calendar_days
        )
        return (
            pl.scan_parquet(self.config.panel_path)
            .select(*PANEL_SCHEMA)
            .filter(pl.col("date").dt.date().is_between(scan_start, scan_end))
            .filter(
                pl.col("open").is_finite()
                & pl.col("close").is_finite()
                & pl.col("volume").is_finite()
                & pl.col("mktcap").is_finite()
                & (pl.col("open") > 0)
                & (pl.col("close") > 0)
                & (pl.col("volume") >= 0)
                & (pl.col("mktcap") > 0)
            )
            .unique(subset=["date", "permno"], keep="first")
            .sort("permno", "date")
        )

    def apply_membership(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        scan_start = self.config.start_date - timedelta(
            days=self.config.warmup_calendar_days
        )
        scan_end = self.config.end_date + timedelta(
            days=self.config.exit_buffer_calendar_days
        )
        membership = (
            pl.scan_parquet(self.config.membership_path)
            .select("permno", "start_date", "end_date")
            .filter(
                (pl.col("end_date").dt.date() >= scan_start)
                & (pl.col("start_date").dt.date() <= scan_end)
            )
        )
        return (
            frame.join(membership, on="permno", how="inner")
            .filter(
                (pl.col("date") >= pl.col("start_date"))
                & (pl.col("date") <= pl.col("end_date"))
            )
            .drop("start_date", "end_date")
            .unique(subset=["date", "permno"], keep="first")
        )

    def market_calendar(self) -> list[object]:
        scan_start = self.config.start_date - timedelta(days=7)
        scan_end = self.config.end_date + timedelta(
            days=self.config.exit_buffer_calendar_days
        )
        dates = (
            pl.scan_parquet(self.config.panel_path)
            .filter(pl.col("date").dt.date().is_between(scan_start, scan_end))
            .select("date")
            .unique()
            .sort("date")
            .collect(engine="streaming")
        )
        return dates.get_column("date").to_list()

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.is_file():
            raise DataValidationError(f"required data file does not exist: {path}")

