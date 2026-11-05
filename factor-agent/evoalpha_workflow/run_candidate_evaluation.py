"""CLI for evaluating all generation-one mock candidates on 2001 training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_evaluation import (
    CandidateEvaluationConfig,
    TrainingCandidateEvaluator,
    load_score_reference,
)
from .crossover import CrossoverProposal, MutationProposal
from .refinement import RefinementProposal
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generation-one candidates")
    root = Path("factor-agent/results/evoalpha_seed_2001")
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--population", type=Path, default=root / "alpha158_top30.json")
    parser.add_argument(
        "--score-reference",
        type=Path,
        default=root / "alpha158_score_reference.json",
    )
    parser.add_argument(
        "--refinements",
        type=Path,
        default=root / "generation_01_refinement_mock.json",
    )
    parser.add_argument(
        "--crossovers",
        type=Path,
        default=root / "generation_01_crossover_mock.json",
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    known_candidates = [
        EvaluatedCandidate.model_validate(item)
        for item in json.loads(args.population.read_text(encoding="utf-8"))
    ]
    refinements = [
        RefinementProposal.model_validate(item).to_candidate_proposal()
        for item in json.loads(args.refinements.read_text(encoding="utf-8"))
    ]
    crossover_payload = json.loads(args.crossovers.read_text(encoding="utf-8"))
    crossovers = [
        CrossoverProposal.model_validate(item).to_candidate_proposal()
        for item in crossover_payload["intermediates"]
    ]
    mutations = [
        MutationProposal.model_validate(item).to_candidate_proposal()
        for item in crossover_payload["mutations"]
    ]
    evaluator = TrainingCandidateEvaluator(
        CandidateEvaluationConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            output_dir=args.output_dir,
        ),
        score_reference=load_score_reference(args.score_reference),
        known_candidates=known_candidates,
    )
    evaluated = evaluator.evaluate_candidates(
        [*refinements, *crossovers, *mutations],
        generation=1,
    )
    valid = [item for item in evaluated if item.passed_hard_filters]
    improved = []
    for candidate in valid:
        direct = [
            item
            for item in candidate.metrics.get("parent_comparisons", [])
            if item["relationship"] == "direct_parent"
        ]
        if any(
            comparison["deltas"].get("composite_score", 0.0) > 0
            for comparison in direct
        ):
            improved.append(candidate)
    ranked = sorted(
        valid,
        key=lambda item: (-item.composite_score, item.factor_id),
    )
    print(
        json.dumps(
            {
                "evaluated": len(evaluated),
                "passed_hard_filters": len(valid),
                "improved_over_at_least_one_direct_parent": len(improved),
                "best_candidate": ranked[0].factor_id if ranked else None,
                "best_composite_score": ranked[0].composite_score if ranked else None,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
