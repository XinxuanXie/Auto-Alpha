from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    panel_path: Path
    membership_path: Path
    start_date: date
    end_date: date
    warmup_calendar_days: int = 90
    exit_buffer_calendar_days: int = 21


@dataclass(frozen=True)
class FactorConfig:
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    minimum_cross_section: int = 50


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int = 20
    exit_rank: int | None = None
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.exit_rank is not None and self.exit_rank < self.top_k:
            raise ValueError("exit_rank must be at least top_k")

