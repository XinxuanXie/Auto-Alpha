"""Hypothesis-first factor research workflow V2."""

from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactIntegrityError, ArtifactRepository
from .contracts import (
    ArtifactRef,
    BacktestProtocol,
    CheckIssue,
    DevelopedHypothesis,
    ExperimentVariantType,
    ExperimentDecision,
    FactorExperimentSpec,
    FactorFormulaSpecV2,
    HypothesisCheckResult,
    InsightAnalysis,
    InsightDecision,
    ItemStatus,
    MeasurementPlan,
    MeasurementProxy,
    PreflightResult,
    ResearchDirection,
    ResearchIdea,
    SplitMetrics,
    SplitName,
)
from .state import V2Stage, V2WorkflowState, create_initial_state
from .manager import build_manager
from .tools import MockV2Tools, V2Tools

if TYPE_CHECKING:
    from .real_tools import RealV2Tools


def __getattr__(name: str) -> Any:
    if name == "RealV2Tools":
        from .real_tools import RealV2Tools

        return RealV2Tools
    raise AttributeError(name)

__all__ = [
    "ArtifactIntegrityError",
    "ArtifactRef",
    "ArtifactRepository",
    "BacktestProtocol",
    "CheckIssue",
    "DevelopedHypothesis",
    "ExperimentVariantType",
    "ExperimentDecision",
    "FactorExperimentSpec",
    "FactorFormulaSpecV2",
    "HypothesisCheckResult",
    "InsightAnalysis",
    "InsightDecision",
    "ItemStatus",
    "MeasurementPlan",
    "MeasurementProxy",
    "PreflightResult",
    "ResearchDirection",
    "ResearchIdea",
    "SplitMetrics",
    "SplitName",
    "V2Stage",
    "V2Tools",
    "V2WorkflowState",
    "MockV2Tools",
    "RealV2Tools",
    "build_manager",
    "create_initial_state",
]
