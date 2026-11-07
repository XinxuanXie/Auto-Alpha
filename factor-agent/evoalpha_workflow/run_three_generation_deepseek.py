"""Run the complete three-generation EvoAlpha protocol with DeepSeek."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl
from dotenv import load_dotenv

from factor_pipeline.config import DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    FactorScoreReference,
    build_forward_return_labels,
    evaluate_single_factor,
    score_factor_metrics,
)
from factor_pipeline.spec import FactorSpec

from .candidate_evaluation import (
    CandidateEvaluationConfig,
    TrainingCandidateEvaluator,
    load_score_reference,
)
from .config import EvoAlphaConfig
from .crossover import CrossoverAgent, CrossoverProposal, MutationAgent, MutationProposal
from .llm_client import OpenAIJSONClient
from .parent_selection import ParentSelectionPlan, ParentSelector
from .population import PopulationUpdater
from .refinement import RefinementAgent, RefinementProposal
from .seed_evaluation import Alpha158SeedEvaluationConfig, Alpha158SeedEvaluator
from .similarity import (
    FactorPairSimilarity,
    SimilarityConfig,
    TrainingFactorSimilarityAnalyzer,
)
from .state import EvaluatedCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run three real DeepSeek EvoAlpha generations and frozen validation."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--training-start", type=date.fromisoformat, default=date(2001, 1, 1))
    parser.add_argument("--training-end", type=date.fromisoformat, default=date(2010, 12, 31))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2011, 1, 1))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2012, 12, 31))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--refinement-model", default="deepseek-flash")
    parser.add_argument("--crossover-model", default="deepseek-v4-pro")
    parser.add_argument("--mutation-model", default="deepseek-flash")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_deepseek_2001_2012_3gen"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_periods(args)
    load_dotenv(args.env_file)
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is required")
    protocol = EvoAlphaConfig(generations=args.generations)
    training_period = EvaluationPeriod(args.training_start, args.training_end)
    validation_period = EvaluationPeriod(args.validation_start, args.validation_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "protocol.json",
        {
            "training_period": asdict(training_period),
            "validation_period": asdict(validation_period),
            "warmup_period": "2000 calendar year available before training",
            "generations": args.generations,
            "models": {
                "refinement": args.refinement_model,
                "crossover": args.crossover_model,
                "mutation": args.mutation_model,
            },
            "reasoning_effort": {
                "refinement": "none",
                "crossover": "high",
                "mutation": "none",
            },
            "offspring_per_generation": protocol.offspring_per_generation,
        },
    )

    population, score_reference = _prepare_seed_population(
        args,
        training_period,
        protocol,
    )
    refinement_client = OpenAIJSONClient.deepseek(
        args.refinement_model,
        reasoning_effort="none",
    )
    crossover_client = OpenAIJSONClient.deepseek(
        args.crossover_model,
        reasoning_effort="high",
    )
    mutation_client = OpenAIJSONClient.deepseek(
        args.mutation_model,
        reasoning_effort="none",
    )
    agents = (
        RefinementAgent(refinement_client, max_attempts=args.max_attempts),
        CrossoverAgent(crossover_client, max_attempts=args.max_attempts),
        MutationAgent(mutation_client, max_attempts=args.max_attempts),
    )

    generation_summaries = []
    for generation in range(1, args.generations + 1):
        generation_dir = args.output_dir / f"generation_{generation:02d}"
        completed_population = generation_dir / "population.json"
        if completed_population.is_file():
            population = _load_candidates(completed_population)
            summary_path = generation_dir / "summary.json"
            if summary_path.is_file():
                generation_summaries.append(_read_json(summary_path))
            print(f"[Generation {generation}] resume completed population", flush=True)
            continue
        population, summary = _run_generation(
            args=args,
            protocol=protocol,
            training_period=training_period,
            score_reference=score_reference,
            population=population,
            generation=generation,
            generation_dir=generation_dir,
            agents=agents,
        )
        generation_summaries.append(summary)

    print("[Validation] evaluate final Top 30 on frozen 2011-2012 period", flush=True)
    validation_results, validation_top10 = _evaluate_frozen_validation(
        panel_path=args.panel,
        membership_path=args.membership,
        population=population,
        score_reference=score_reference,
        period=validation_period,
        output_dir=args.output_dir / "validation_2011_2012",
        selection_size=protocol.validation_selection_size,
    )
    request_counts = {
        "refinement": refinement_client.request_count,
        "crossover": crossover_client.request_count,
        "mutation": mutation_client.request_count,
    }
    final_summary = {
        "training_period": asdict(training_period),
        "validation_period": asdict(validation_period),
        "generations_completed": args.generations,
        "generation_summaries": generation_summaries,
        "final_population_size": len(population),
        "validation_valid_candidates": sum(
            item.passed_hard_filters for item in validation_results
        ),
        "validation_top10": [item.factor_id for item in validation_top10],
        "best_validation_factor_id": validation_top10[0].factor_id,
        "best_validation_score": validation_top10[0].composite_score,
        "api_requests_this_invocation": request_counts,
        "api_requests_this_invocation_total": sum(request_counts.values()),
        "output_dir": str(args.output_dir),
    }
    _write_json(args.output_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _prepare_seed_population(
    args: argparse.Namespace,
    training_period: EvaluationPeriod,
    protocol: EvoAlphaConfig,
) -> tuple[list[EvaluatedCandidate], FactorScoreReference]:
    seed_dir = args.output_dir / "seed"
    candidates_path = seed_dir / "alpha158_seed_candidates.json"
    reference_path = seed_dir / "alpha158_score_reference.json"
    top_path = seed_dir / "alpha158_top30.json"
    if top_path.is_file() and reference_path.is_file():
        print("[Seed] resume Top 30 and score reference", flush=True)
        return _load_candidates(top_path), load_score_reference(reference_path)

    if candidates_path.is_file() and reference_path.is_file():
        candidates = _load_candidates(candidates_path)
        score_reference = load_score_reference(reference_path)
        print("[Seed] resume evaluated Alpha158 candidates", flush=True)
    else:
        print("[Seed] evaluate 157 Alpha158 factors on 2001-2010", flush=True)

        def progress(index: int, total: int, factor_id: str) -> None:
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[Seed {index}/{total}] {factor_id}", flush=True)

        evaluator = Alpha158SeedEvaluator(
            Alpha158SeedEvaluationConfig(
                panel_path=args.panel,
                membership_path=args.membership,
                training_period=training_period,
                warmup_calendar_days=366,
                output_dir=seed_dir,
            ),
            progress_callback=progress,
        )
        candidates = evaluator.evaluate_seeds()
        score_reference = evaluator.last_score_reference
        if score_reference is None:
            raise RuntimeError("seed evaluator did not create a score reference")
    valid = sorted(
        (
            item
            for item in candidates
            if item.passed_hard_filters and item.composite_score is not None
        ),
        key=lambda item: (-item.composite_score, item.factor_id),
    )
    if len(valid) < protocol.population_size:
        raise ValueError("too few valid Alpha158 seeds for the initial population")
    population = valid[: protocol.population_size]
    _write_json(top_path, [item.model_dump(mode="json") for item in population])
    _write_json(
        seed_dir / "summary.json",
        {
            "evaluated": len(candidates),
            "valid": len(valid),
            "selected": len(population),
            "best_factor_id": population[0].factor_id,
            "best_composite_score": population[0].composite_score,
        },
    )
    return population, score_reference


def _run_generation(
    *,
    args: argparse.Namespace,
    protocol: EvoAlphaConfig,
    training_period: EvaluationPeriod,
    score_reference: FactorScoreReference,
    population: list[EvaluatedCandidate],
    generation: int,
    generation_dir: Path,
    agents: tuple[RefinementAgent, CrossoverAgent, MutationAgent],
) -> tuple[list[EvaluatedCandidate], dict]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Generation {generation}] analyze parent population similarity", flush=True)
    parent_similarity = _load_or_compute_similarity(
        args=args,
        period=training_period,
        candidates=population,
        output_dir=generation_dir,
        prefix=f"generation_{generation:02d}_parent_population",
    )
    plan_path = generation_dir / "parent_plan.json"
    if plan_path.is_file():
        plan = ParentSelectionPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    else:
        plan = ParentSelector(protocol).select(population, parent_similarity, generation)
        _write_json(plan_path, plan.model_dump(mode="json"))
    proposals = _generate_proposals(
        generation=generation,
        population=population,
        plan=plan,
        generation_dir=generation_dir,
        agents=agents,
    )
    evaluations_path = generation_dir / f"generation_{generation:02d}_candidate_evaluations.json"
    if evaluations_path.is_file():
        evaluated = _load_candidates(evaluations_path)
        print(f"[Generation {generation}] resume 28 candidate evaluations", flush=True)
    else:
        print(f"[Generation {generation}] evaluate 28 candidates on training period", flush=True)
        evaluated = TrainingCandidateEvaluator(
            CandidateEvaluationConfig(
                panel_path=args.panel,
                membership_path=args.membership,
                training_period=training_period,
                warmup_calendar_days=366,
                output_dir=generation_dir,
            ),
            score_reference=score_reference,
            known_candidates=population,
        ).evaluate_candidates(proposals, generation=generation)
    all_candidates = [*population, *evaluated]
    print(f"[Generation {generation}] analyze full pool similarity", flush=True)
    pool_similarity = _load_or_compute_similarity(
        args=args,
        period=training_period,
        candidates=all_candidates,
        output_dir=generation_dir,
        prefix=f"generation_{generation:02d}_full_pool",
    )
    update = PopulationUpdater(protocol).update(
        population,
        evaluated,
        pool_similarity,
        generation,
    )
    next_population = list(update.population)
    _write_json(generation_dir / "population_update.json", update.model_dump(mode="json"))
    _write_json(
        generation_dir / "population.json",
        [item.model_dump(mode="json") for item in next_population],
    )
    summary = {
        "generation": generation,
        "evaluated_candidates": len(evaluated),
        "passed_hard_filters": sum(item.passed_hard_filters for item in evaluated),
        "admitted_after_similarity": len(update.admitted_candidate_ids),
        "children_in_next_population": sum(
            item.generation == generation for item in next_population
        ),
        "best_factor_id": next_population[0].factor_id,
        "best_composite_score": next_population[0].composite_score,
    }
    _write_json(generation_dir / "summary.json", summary)
    return next_population, summary


def _generate_proposals(
    *,
    generation: int,
    population: Sequence[EvaluatedCandidate],
    plan: ParentSelectionPlan,
    generation_dir: Path,
    agents: tuple[RefinementAgent, CrossoverAgent, MutationAgent],
):
    refinement_agent, crossover_agent, mutation_agent = agents
    population_by_id = {item.factor_id: item for item in population}
    checkpoint_path = generation_dir / "agent_checkpoint.json"
    checkpoint = _load_agent_checkpoint(checkpoint_path)
    refinements = {
        item["task_id"]: RefinementProposal.model_validate(item)
        for item in checkpoint["refinements"]
    }
    for index, task in enumerate(plan.refinement_tasks, start=1):
        if task.task_id not in refinements:
            print(f"[G{generation} Refinement {index}/12] {task.task_id}", flush=True)
            refinements[task.task_id] = refinement_agent.propose(
                task,
                population_by_id[task.parent_factor_id],
            )
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
    for index, task in enumerate(plan.crossover_tasks, start=1):
        if task.task_id not in intermediates:
            print(f"[G{generation} Crossover {index}/8] {task.task_id}", flush=True)
            intermediates[task.task_id] = crossover_agent.propose(
                task,
                population_by_id[task.parent_factor_ids[0]],
                population_by_id[task.parent_factor_ids[1]],
            )
            checkpoint["intermediates"] = [
                item.model_dump(mode="json") for item in intermediates.values()
            ]
            _write_json(checkpoint_path, checkpoint)
        if task.task_id not in mutations:
            print(f"[G{generation} Mutation {index}/8] {task.task_id}", flush=True)
            mutations[task.task_id] = mutation_agent.propose(intermediates[task.task_id])
            checkpoint["mutations"] = [
                item.model_dump(mode="json") for item in mutations.values()
            ]
            _write_json(checkpoint_path, checkpoint)
    payload = {
        "refinements": [item.model_dump(mode="json") for item in refinements.values()],
        "intermediates": [item.model_dump(mode="json") for item in intermediates.values()],
        "mutations": [item.model_dump(mode="json") for item in mutations.values()],
    }
    _write_json(generation_dir / "agent_proposals.json", payload)
    return [
        *[item.to_candidate_proposal() for item in refinements.values()],
        *[item.to_candidate_proposal() for item in intermediates.values()],
        *[item.to_candidate_proposal() for item in mutations.values()],
    ]


def _load_or_compute_similarity(
    *,
    args: argparse.Namespace,
    period: EvaluationPeriod,
    candidates: Sequence[EvaluatedCandidate],
    output_dir: Path,
    prefix: str,
) -> list[FactorPairSimilarity]:
    path = output_dir / f"{prefix}_pair_similarity.json"
    if path.is_file():
        return [FactorPairSimilarity(**item) for item in _read_json(path)]

    def progress(index: int, total: int, factor_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[Similarity {index}/{total}] {factor_id}", flush=True)

    result = TrainingFactorSimilarityAnalyzer(
        SimilarityConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            training_period=period,
            warmup_calendar_days=366,
            output_dir=output_dir,
            output_prefix=prefix,
        ),
        progress_callback=progress,
    ).analyze_specs(
        [FactorSpec.model_validate(item.artifact["factor_spec"]) for item in candidates]
    )
    return list(result.pairs)


def _evaluate_frozen_validation(
    *,
    panel_path: Path,
    membership_path: Path,
    population: Sequence[EvaluatedCandidate],
    score_reference: FactorScoreReference,
    period: EvaluationPeriod,
    output_dir: Path,
    selection_size: int,
) -> tuple[list[EvaluatedCandidate], list[EvaluatedCandidate]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "validation_population.json"
    top_path = output_dir / "validation_top10.json"
    if all_path.is_file() and top_path.is_file():
        return _load_candidates(all_path), _load_candidates(top_path)
    panel = PanelDataSource(
        DataConfig(
            panel_path=panel_path,
            membership_path=membership_path,
            start_date=period.start_date,
            end_date=period.end_date,
            warmup_calendar_days=366,
            exit_buffer_calendar_days=0,
        )
    ).point_in_time_panel().collect(engine="streaming")
    config = FactorEvaluationConfig()
    labels = build_forward_return_labels(panel, period, config.horizons)
    raw_metrics = []
    failures = {}
    for index, candidate in enumerate(population, start=1):
        try:
            spec = FactorSpec.model_validate(candidate.artifact["factor_spec"])
            values = (
                build_factor_from_spec(panel.lazy(), spec, FactorConfig())
                .select("date", "permno", "factor")
                .collect(engine="streaming")
            )
            raw_metrics.append(
                evaluate_single_factor(
                    values,
                    labels,
                    candidate.factor_id,
                    period,
                    config,
                    direction=candidate.direction,
                )
            )
        except Exception as error:
            failures[candidate.factor_id] = f"{type(error).__name__}: {error}"
        if index == 1 or index % 5 == 0 or index == len(population):
            print(f"[Validation {index}/{len(population)}] {candidate.factor_id}", flush=True)
    scored = {
        item.factor_id: item
        for item in score_factor_metrics(raw_metrics, reference=score_reference)
    }
    results = []
    for candidate in population:
        metrics = scored.get(candidate.factor_id)
        if metrics is None:
            results.append(
                candidate.model_copy(
                    update={
                        "composite_score": None,
                        "passed_hard_filters": False,
                        "metrics": {"execution_error": failures[candidate.factor_id]},
                    }
                )
            )
        else:
            results.append(
                candidate.model_copy(
                    update={
                        "composite_score": metrics.composite_score,
                        "passed_hard_filters": metrics.passed_hard_filters,
                        "metrics": {
                            **metrics.to_dict(),
                            "training_composite_score": candidate.composite_score,
                            "formula_and_direction_frozen": True,
                        },
                    }
                )
            )
    ranked = sorted(
        (
            item
            for item in results
            if item.passed_hard_filters and item.composite_score is not None
        ),
        key=lambda item: (-item.composite_score, item.factor_id),
    )
    if len(ranked) < selection_size:
        raise ValueError("validation produced too few valid factors for Top 10")
    top = ranked[:selection_size]
    _write_json(all_path, [item.model_dump(mode="json") for item in results])
    _write_json(top_path, [item.model_dump(mode="json") for item in top])
    _write_json(
        output_dir / "summary.json",
        {
            "period": asdict(period),
            "formula_and_direction_frozen": True,
            "evaluated": len(results),
            "valid": len(ranked),
            "selected": len(top),
            "best_factor_id": top[0].factor_id,
            "best_composite_score": top[0].composite_score,
        },
    )
    return results, top


def _load_agent_checkpoint(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        return {"refinements": [], "intermediates": [], "mutations": []}
    payload = _read_json(path)
    return {
        "refinements": list(payload.get("refinements", [])),
        "intermediates": list(payload.get("intermediates", [])),
        "mutations": list(payload.get("mutations", [])),
    }


def _load_candidates(path: Path) -> list[EvaluatedCandidate]:
    return [EvaluatedCandidate.model_validate(item) for item in _read_json(path)]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _validate_periods(args: argparse.Namespace) -> None:
    if args.generations < 1:
        raise ValueError("generations must be positive")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    if args.training_end >= args.validation_start:
        raise ValueError("training period must end before validation begins")


if __name__ == "__main__":
    raise SystemExit(main())
