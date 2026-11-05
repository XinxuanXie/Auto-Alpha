"""Run one complete real-LLM EvoAlpha generation with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from factor_pipeline.spec import FactorSpec

from .candidate_evaluation import (
    CandidateEvaluationConfig,
    TrainingCandidateEvaluator,
    load_score_reference,
)
from .crossover import CrossoverAgent, CrossoverProposal, MutationAgent, MutationProposal
from .llm_client import OpenAIJSONClient
from .parent_selection import ParentSelectionPlan
from .population import PopulationUpdater
from .refinement import RefinementAgent, RefinementProposal
from .similarity import (
    SimilarityConfig,
    TrainingFactorSimilarityAnalyzer,
)
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one real EvoAlpha generation")
    seed_root = Path("factor-agent/results/evoalpha_seed_2001")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--population", type=Path, default=seed_root / "alpha158_top30.json")
    parser.add_argument(
        "--parent-plan",
        type=Path,
        default=seed_root / "generation_01_parent_plan.json",
    )
    parser.add_argument(
        "--score-reference",
        type=Path,
        default=seed_root / "alpha158_score_reference.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_generation_01_real"),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "generation_01_agent_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    population = {
        candidate.factor_id: candidate
        for candidate in (
            EvaluatedCandidate.model_validate(item)
            for item in json.loads(args.population.read_text(encoding="utf-8"))
        )
    }
    parent_plan = ParentSelectionPlan.model_validate_json(
        args.parent_plan.read_text(encoding="utf-8")
    )
    client = OpenAIJSONClient()
    refinement_agent = RefinementAgent(client, max_attempts=args.max_attempts)
    crossover_agent = CrossoverAgent(client, max_attempts=args.max_attempts)
    mutation_agent = MutationAgent(client, max_attempts=args.max_attempts)

    refinements = {
        item["task_id"]: RefinementProposal.model_validate(item)
        for item in checkpoint["refinements"]
    }
    for index, task in enumerate(parent_plan.refinement_tasks, start=1):
        if task.task_id in refinements:
            print(f"[Refinement {index}/12] resume {task.task_id}", flush=True)
            continue
        print(f"[Refinement {index}/12] request {task.task_id}", flush=True)
        proposal = refinement_agent.propose(task, population[task.parent_factor_id])
        refinements[task.task_id] = proposal
        checkpoint["refinements"] = [
            item.model_dump(mode="json") for item in refinements.values()
        ]
        _write_json(checkpoint_path, checkpoint)

    intermediates = {
        item["task_id"]: CrossoverProposal.model_validate(item)
        for item in checkpoint["intermediates"]
    }
    mutations = {
        item["task_id"]: MutationProposal.model_validate(item)
        for item in checkpoint["mutations"]
    }
    for index, task in enumerate(parent_plan.crossover_tasks, start=1):
        parent_a = population[task.parent_factor_ids[0]]
        parent_b = population[task.parent_factor_ids[1]]
        if task.task_id not in intermediates:
            print(f"[Crossover {index}/8] request {task.task_id}", flush=True)
            intermediates[task.task_id] = crossover_agent.propose(
                task,
                parent_a,
                parent_b,
            )
            checkpoint["intermediates"] = [
                item.model_dump(mode="json") for item in intermediates.values()
            ]
            _write_json(checkpoint_path, checkpoint)
        else:
            print(f"[Crossover {index}/8] resume {task.task_id}", flush=True)
        if task.task_id not in mutations:
            print(f"[Mutation {index}/8] request {task.task_id}", flush=True)
            mutations[task.task_id] = mutation_agent.propose(intermediates[task.task_id])
            checkpoint["mutations"] = [
                item.model_dump(mode="json") for item in mutations.values()
            ]
            _write_json(checkpoint_path, checkpoint)
        else:
            print(f"[Mutation {index}/8] resume {task.task_id}", flush=True)

    proposal_payload = {
        "refinements": [item.model_dump(mode="json") for item in refinements.values()],
        "intermediates": [item.model_dump(mode="json") for item in intermediates.values()],
        "mutations": [item.model_dump(mode="json") for item in mutations.values()],
    }
    _write_json(args.output_dir / "generation_01_real_agent_proposals.json", proposal_payload)
    candidate_proposals = [
        *[item.to_candidate_proposal() for item in refinements.values()],
        *[item.to_candidate_proposal() for item in intermediates.values()],
        *[item.to_candidate_proposal() for item in mutations.values()],
    ]
    print("[Evaluation] evaluate 28 real-agent candidates", flush=True)
    evaluator = TrainingCandidateEvaluator(
        CandidateEvaluationConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            output_dir=args.output_dir,
        ),
        score_reference=load_score_reference(args.score_reference),
        known_candidates=list(population.values()),
    )
    evaluated = evaluator.evaluate_candidates(candidate_proposals, generation=1)

    print("[Similarity] analyze 30 parents + 28 candidates", flush=True)
    all_candidates = [*population.values(), *evaluated]
    specs = [
        FactorSpec.model_validate(candidate.artifact["factor_spec"])
        for candidate in all_candidates
    ]

    def similarity_progress(index: int, total: int, factor_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[Similarity {index}/{total}] {factor_id}", flush=True)

    similarity_result = TrainingFactorSimilarityAnalyzer(
        SimilarityConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            output_dir=args.output_dir,
            output_prefix="generation_01_pool",
        ),
        progress_callback=similarity_progress,
    ).analyze_specs(specs)
    print("[Population] apply diversity filters and select Top 30", flush=True)
    update = PopulationUpdater().update(
        list(population.values()),
        evaluated,
        similarity_result.pairs,
        generation=1,
    )
    _write_json(
        args.output_dir / "generation_01_population_update.json",
        update.model_dump(mode="json"),
    )
    _write_json(
        args.output_dir / "generation_01_population.json",
        [item.model_dump(mode="json") for item in update.population],
    )
    admitted = len(update.admitted_candidate_ids)
    surviving_children = sum(item.generation == 1 for item in update.population)
    summary = {
        "refinements": len(refinements),
        "crossover_intermediates": len(intermediates),
        "mutations": len(mutations),
        "evaluated_candidates": len(evaluated),
        "admitted_after_similarity": admitted,
        "children_in_next_population": surviving_children,
        "next_population_size": len(update.population),
        "best_factor_id": update.population[0].factor_id,
        "best_composite_score": update.population[0].composite_score,
        "output_dir": str(args.output_dir),
    }
    _write_json(args.output_dir / "generation_01_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _load_checkpoint(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        return {"refinements": [], "intermediates": [], "mutations": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "refinements": list(payload.get("refinements", [])),
        "intermediates": list(payload.get("intermediates", [])),
        "mutations": list(payload.get("mutations", [])),
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
