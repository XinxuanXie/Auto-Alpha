"""Run or resume the factor research workflow."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agentic_workflow.checkpoint import checkpoint_config, open_sqlite_checkpointer
from agentic_workflow.manager import ManagerTools, build_manager
from agentic_workflow.manager_policy import (
    OpenAIManagerPolicy,
    RuleBasedManagerPolicy,
)
from agentic_workflow.state import create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools import (
    CandidateBatchEvaluationTool,
    EvaluationAgentTool,
    FactorBacktestTool,
    ImplementationAgentTool,
    OpenAIManagerCandidateSelector,
    RuleBasedResearchCandidateSelector,
    RuleBasedHypothesisCheckerTool,
    SQLiteRecordTool,
    build_default_research_team,
)


# 读取用户在终端输入的运行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="Resume an existing workflow from its latest checkpoint.",
    )
    parser.add_argument(
        "--manager",
        choices=["openai", "rule"],
        default="openai",
        help="Outer Manager policy. Defaults to the OpenAI Manager Agent.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model override. Defaults to OPENAI_MODEL or gpt-5.6-luna.",
    )
    parser.add_argument(
        "--research-goal",
        default="发现具有稳定横截面预测能力的美股因子",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--max-attempts-per-hypothesis",
        type=int,
        default=3,
        help="Maximum implementation/backtest attempts for each hypothesis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_agent_root = Path(__file__).resolve().parents[1]
    repository_root = factor_agent_root.parent
    load_dotenv(repository_root / ".env")
    policy = (
        OpenAIManagerPolicy(model=args.model or os.getenv("OPENAI_MODEL"))
        if args.manager == "openai"
        else RuleBasedManagerPolicy()
    )
    repository = SQLiteResearchRepository(
        factor_agent_root / ".runtime" / "research.db"
    )
    hypothesis_checker = RuleBasedHypothesisCheckerTool(repository)
    candidate_selector = (
        OpenAIManagerCandidateSelector(
            model=args.model or os.getenv("OPENAI_MODEL")
        )
        if args.manager == "openai"
        else RuleBasedResearchCandidateSelector()
    )
    research_team = build_default_research_team(
        repository=repository,
        hypothesis_checker=hypothesis_checker,
        model=args.model or os.getenv("OPENAI_MODEL"),
    )
    backtest_tool = FactorBacktestTool(
        panel_path=repository_root / "factor-data" / "panel.parquet",
        membership_path=(
            repository_root / "factor-data" / "membership_intervals.parquet"
        ),
        artifact_root=factor_agent_root / ".runtime" / "backtests",
    )
    initial_state = (
        None
        if args.resume
        else create_initial_state(
            max_iterations=args.max_iterations,
            max_attempts_per_hypothesis=args.max_attempts_per_hypothesis,
            research_goal=args.research_goal,
        )
    )
    thread_id = args.resume or initial_state["run_id"]
    config = checkpoint_config(thread_id)

    with open_sqlite_checkpointer(
        factor_agent_root / ".runtime" / "checkpoints.db"
    ) as checkpointer:
        implementation_tool = ImplementationAgentTool(
            panel_path=repository_root / "factor-data" / "panel.parquet"
        )
        evaluation_tool = EvaluationAgentTool(
            model=args.model or os.getenv("OPENAI_MODEL")
        )
        manager = build_manager(
            ManagerTools(
                hypothesis=research_team,
                hypothesis_checker=hypothesis_checker,
                implementation=implementation_tool,
                backtest=backtest_tool,
                evaluation=evaluation_tool,
                record=SQLiteRecordTool(repository),
                candidate_evaluation=CandidateBatchEvaluationTool(
                    implementation=implementation_tool,
                    backtest=backtest_tool,
                    evaluation=evaluation_tool,
                    selector=candidate_selector,
                ),
            ),
            policy=policy,
            checkpointer=checkpointer,
        )
        result = manager.invoke(initial_state, config=config)

    if result is None:
        raise SystemExit(f"No checkpoint found for thread_id={thread_id}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
