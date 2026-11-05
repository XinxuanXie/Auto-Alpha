"""Generate generation-one crossover intermediates and mutations with mock clients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .crossover import (
    CrossoverAgent,
    MutationAgent,
    RuleBasedMockCrossoverClient,
    RuleBasedMockMutationClient,
)
from .parent_selection import ParentSelectionPlan
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mock EvoAlpha crossover and mutation")
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
            "factor-agent/results/evoalpha_seed_2001/generation_01_crossover_mock.json"
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
    crossover_agent = CrossoverAgent(RuleBasedMockCrossoverClient())
    mutation_agent = MutationAgent(RuleBasedMockMutationClient())
    intermediates = []
    mutations = []
    for task in plan.crossover_tasks:
        parent_a, parent_b = (
            population[task.parent_factor_ids[0]],
            population[task.parent_factor_ids[1]],
        )
        intermediate = crossover_agent.propose(task, parent_a, parent_b)
        intermediates.append(intermediate)
        mutations.append(mutation_agent.propose(intermediate))
    payload = {
        "intermediates": [item.model_dump(mode="json") for item in intermediates],
        "mutations": [item.model_dump(mode="json") for item in mutations],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "intermediates": len(intermediates),
                "mutations": len(mutations),
                "window_mutations": sum(
                    item.mutation_type.value == "window" for item in mutations
                ),
                "normalization_mutations": sum(
                    item.mutation_type.value == "normalization" for item in mutations
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
