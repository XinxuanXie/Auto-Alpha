"""Workflow tool contracts and test doubles."""

from agentic_workflow.tools.base import ToolResult, WorkflowTool
from agentic_workflow.tools.candidate_evaluation import CandidateBatchEvaluationTool
from agentic_workflow.tools.evaluation_agent import (
    EvaluationAgentTool,
    EvaluationThresholds,
)
from agentic_workflow.tools.factor_backtest import FactorBacktestTool
from agentic_workflow.tools.hypothesis_checker import (
    HypothesisRules,
    RuleBasedHypothesisCheckerTool,
)
from agentic_workflow.tools.implementation_agent import ImplementationAgentTool
from agentic_workflow.tools.research_agent import ResearchAgentTool
from agentic_workflow.tools.web_research import (
    DDGSWebResearchProvider,
    CrossrefWebResearchProvider,
    KeyFreeWebResearchProvider,
    WebResearchProvider,
    WebResearchSource,
)
from agentic_workflow.tools.two_stage_research import (
    FinancialIntuitionAgentTool,
    HypothesisFormulationAgentTool,
    TwoStageResearchSpecialist,
)
from agentic_workflow.tools.research_team import (
    MultiStrategyResearchTool,
    OpenAIManagerCandidateSelector,
    RuleBasedResearchCandidateSelector,
    build_default_research_team,
)
from agentic_workflow.tools.sqlite_record import SQLiteRecordTool
from agentic_workflow.tools.mock import (
    MockBacktestTool,
    MockEvaluationTool,
    MockHypothesisCheckerTool,
    MockHypothesisTool,
    MockImplementationTool,
    MockRecordTool,
)

__all__ = [
    "MockBacktestTool",
    "CandidateBatchEvaluationTool",
    "EvaluationAgentTool",
    "EvaluationThresholds",
    "FactorBacktestTool",
    "MockEvaluationTool",
    "MockHypothesisCheckerTool",
    "MockHypothesisTool",
    "MockImplementationTool",
    "MockRecordTool",
    "ToolResult",
    "SQLiteRecordTool",
    "HypothesisRules",
    "ImplementationAgentTool",
    "RuleBasedHypothesisCheckerTool",
    "ResearchAgentTool",
    "DDGSWebResearchProvider",
    "CrossrefWebResearchProvider",
    "KeyFreeWebResearchProvider",
    "WebResearchProvider",
    "WebResearchSource",
    "FinancialIntuitionAgentTool",
    "HypothesisFormulationAgentTool",
    "TwoStageResearchSpecialist",
    "MultiStrategyResearchTool",
    "OpenAIManagerCandidateSelector",
    "RuleBasedResearchCandidateSelector",
    "build_default_research_team",
    "WorkflowTool",
]
