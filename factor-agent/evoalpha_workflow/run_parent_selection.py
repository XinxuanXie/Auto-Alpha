"""CLI for deterministic generation-one parent selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .parent_selection import ParentSelector
from .similarity import FactorPairSimilarity
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select EvoAlpha refinement and crossover parents")
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001/alpha158_top30.json"),
    )
    parser.add_argument(
        "--similarities",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001/top30_pair_similarity.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001/generation_01_parent_plan.json"),
    )
    parser.add_argument("--generation", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    population = [
        EvaluatedCandidate.model_validate(item)
        for item in json.loads(args.population.read_text(encoding="utf-8"))
    ]
    allowed = {field.name for field in fields(FactorPairSimilarity)}
    similarities = [
        FactorPairSimilarity(**{key: value for key, value in item.items() if key in allowed})
        for item in json.loads(args.similarities.read_text(encoding="utf-8"))
    ]
    plan = ParentSelector().select(population, similarities, args.generation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
