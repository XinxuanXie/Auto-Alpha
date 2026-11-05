"""Generate and validate generation-one refinement proposals with a mock client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .parent_selection import ParentSelectionPlan
from .refinement import RefinementAgent, RuleBasedMockRefinementClient
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mock EvoAlpha refinements")
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001/alpha158_top30.json"),
    )
    parser.add_argument(
        "--parent-plan",
        type=Path,
        default=Path(
            "factor-agent/results/evoalpha_seed_2001/generation_01_parent_plan.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "factor-agent/results/evoalpha_seed_2001/generation_01_refinement_mock.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    population = {
        candidate.factor_id: candidate
        for candidate in (
            EvaluatedCandidate.model_validate(item)
            for item in json.loads(args.population.read_text(encoding="utf-8"))
        )
    }
    plan = ParentSelectionPlan.model_validate_json(
        args.parent_plan.read_text(encoding="utf-8")
    )
    agent = RefinementAgent(RuleBasedMockRefinementClient())
    proposals = [
        agent.propose(task, population[task.parent_factor_id])
        for task in plan.refinement_tasks
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [proposal.model_dump(mode="json") for proposal in proposals],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generated": len(proposals),
                "window_changes": sum(
                    proposal.change_type.value == "window" for proposal in proposals
                ),
                "normalization_changes": sum(
                    proposal.change_type.value == "normalization"
                    for proposal in proposals
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
