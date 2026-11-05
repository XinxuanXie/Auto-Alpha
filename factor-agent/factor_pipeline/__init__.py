"""Deterministic data and backtest pipeline for factor research."""

from .backtest import BacktestResult, run_weekly_topk_backtest
from .compiler import FactorCompilationError, compile_factor_spec
from .config import BacktestConfig, DataConfig, FactorConfig
from .data import DataValidationError, PanelDataSource
from .factor import build_abnormal_volume_reversal, build_factor_from_spec
from .spec import (
    FactorSpec,
    FormulaNode,
    PortfolioRules,
    UniverseFilterSpec,
    abnormal_volume_reversal_spec,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DataConfig",
    "DataValidationError",
    "FactorCompilationError",
    "FactorConfig",
    "FactorSpec",
    "FormulaNode",
    "PortfolioRules",
    "UniverseFilterSpec",
    "PanelDataSource",
    "abnormal_volume_reversal_spec",
    "build_abnormal_volume_reversal",
    "build_factor_from_spec",
    "compile_factor_spec",
    "run_weekly_topk_backtest",
]
