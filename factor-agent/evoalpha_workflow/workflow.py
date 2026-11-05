"""Deterministic outer workflow for EvoAlpha experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from uuid import uuid4

from .config import EvoAlphaConfig
from .state import (
    CandidateProposal,
    EvaluatedCandidate,
    EvoAlphaRunState,
    EvoAlphaStage,
    GenerationSnapshot,
    ProposalType,
    WorkflowEvent,
)
from .tools import EvoAlphaTools


class EvoAlphaWorkflowError(RuntimeError):
    def __init__(self, message: str, state: EvoAlphaRunState) -> None:
        super().__init__(message)
        self.state = state


class EvoAlphaWorkflow:
    def __init__(
        self,
        tools: EvoAlphaTools,
        config: EvoAlphaConfig = EvoAlphaConfig(),
    ) -> None:
        self.tools = tools
        self.config = config

    def run(self, run_id: str | None = None) -> EvoAlphaRunState:
        state = EvoAlphaRunState(run_id=run_id or str(uuid4()))
        try:
            self._run_seed_stage(state)
            for generation in range(1, self.config.generations + 1):
                self._run_generation(state, generation)
            self._run_validation_and_test(state)
            self._transition(state, EvoAlphaStage.COMPLETED, "workflow_completed")
            return state
        except Exception as error:
            state.error = f"{type(error).__name__}: {error}"
            self._transition(
                state,
                EvoAlphaStage.FAILED,
                "workflow_failed",
                error=state.error,
            )
            raise EvoAlphaWorkflowError(state.error, state) from error

    def _run_seed_stage(self, state: EvoAlphaRunState) -> None:
        self._transition(state, EvoAlphaStage.SEED_EVALUATION, "seed_evaluation_started")
        seeds = list(self.tools.seed_evaluator.evaluate_seeds())
        self._validate_unique_ids(seeds, "seed candidates")
        if any(candidate.proposal_type != ProposalType.SEED for candidate in seeds):
            raise ValueError("seed evaluator returned a non-seed candidate")
        state.seed_candidates = seeds
        self._transition(
            state,
            EvoAlphaStage.SEED_SELECTION,
            "seed_evaluation_completed",
            candidate_count=len(seeds),
        )
        state.population = self._select_population(seeds)
        self._append_snapshot(state, generation=0)
        self._record_event(
            state,
            "initial_population_selected",
            population_size=len(state.population),
        )

    def _run_generation(self, state: EvoAlphaRunState, generation: int) -> None:
        state.generation = generation
        self._transition(
            state,
            EvoAlphaStage.OFFSPRING_PROPOSAL,
            "generation_started",
        )
        proposals = list(
            self.tools.offspring_proposer.propose_offspring(
                tuple(state.population),
                generation,
                self.config,
            )
        )
        self._validate_proposals(proposals, state.population, generation)
        state.current_proposals = proposals
        self._transition(
            state,
            EvoAlphaStage.CANDIDATE_EVALUATION,
            "offspring_proposed",
            proposal_count=len(proposals),
        )
        offspring = list(
            self.tools.candidate_evaluator.evaluate_candidates(
                tuple(proposals),
                generation,
            )
        )
        self._validate_evaluated_offspring(proposals, offspring, generation)
        state.current_offspring = offspring
        self._transition(
            state,
            EvoAlphaStage.POPULATION_UPDATE,
            "offspring_evaluated",
            offspring_count=len(offspring),
        )
        state.population = self._select_population([*state.population, *offspring])
        self._append_snapshot(state, generation=generation)
        self._record_event(
            state,
            "population_updated",
            population_size=len(state.population),
        )

    def _run_validation_and_test(self, state: EvoAlphaRunState) -> None:
        self._transition(state, EvoAlphaStage.VALIDATION, "validation_started")
        validation = list(
            self.tools.validation_evaluator.evaluate_validation(
                tuple(state.population)
            )
        )
        self._validate_period_results(state.population, validation, "validation")
        state.validation_results = self._rank_valid_candidates(validation)
        selected = state.validation_results[: self.config.validation_selection_size]
        if len(selected) < self.config.validation_selection_size:
            raise ValueError("validation produced too few valid candidates for final test")
        self._transition(
            state,
            EvoAlphaStage.TEST,
            "validation_completed",
            selected_count=len(selected),
        )
        test_results = list(self.tools.test_evaluator.evaluate_test(tuple(selected)))
        self._validate_period_results(selected, test_results, "test")
        state.test_results = self._rank_valid_candidates(test_results)
        self._record_event(
            state,
            "test_completed",
            tested_count=len(state.test_results),
        )

    def _validate_proposals(
        self,
        proposals: Sequence[CandidateProposal],
        population: Sequence[EvaluatedCandidate],
        generation: int,
    ) -> None:
        if len(proposals) != self.config.offspring_per_generation:
            raise ValueError(
                "offspring proposal count must equal fixed protocol count "
                f"{self.config.offspring_per_generation}"
            )
        self._validate_unique_ids(proposals, "offspring proposals")
        population_ids = {candidate.factor_id for candidate in population}
        proposal_ids = {proposal.factor_id for proposal in proposals}
        overlap = population_ids & proposal_ids
        if overlap:
            raise ValueError(f"offspring ids overlap current population: {sorted(overlap)}")
        for proposal in proposals:
            if proposal.generation != generation:
                raise ValueError("proposal generation does not match current generation")
        expected_counts = {
            ProposalType.REFINEMENT: self.config.refinement_offspring_per_generation,
            ProposalType.CROSSOVER: self.config.crossover_tasks_per_generation,
            ProposalType.MUTATION: self.config.crossover_tasks_per_generation,
        }
        counts = Counter(proposal.proposal_type for proposal in proposals)
        if counts != expected_counts:
            raise ValueError(
                f"proposal type counts do not match fixed protocol: {dict(counts)}"
            )

    def _validate_evaluated_offspring(
        self,
        proposals: Sequence[CandidateProposal],
        offspring: Sequence[EvaluatedCandidate],
        generation: int,
    ) -> None:
        self._validate_unique_ids(offspring, "evaluated offspring")
        proposal_by_id = {proposal.factor_id: proposal for proposal in proposals}
        if set(proposal_by_id) != {candidate.factor_id for candidate in offspring}:
            raise ValueError("candidate evaluator must return every proposal exactly once")
        for candidate in offspring:
            proposal = proposal_by_id[candidate.factor_id]
            if candidate.generation != generation:
                raise ValueError("evaluated candidate generation mismatch")
            if candidate.proposal_type != proposal.proposal_type:
                raise ValueError("evaluated candidate proposal_type mismatch")
            if candidate.parent_factor_ids != proposal.parent_factor_ids:
                raise ValueError("evaluated candidate lineage mismatch")

    @staticmethod
    def _validate_period_results(
        inputs: Sequence[EvaluatedCandidate],
        outputs: Sequence[EvaluatedCandidate],
        period_name: str,
    ) -> None:
        EvoAlphaWorkflow._validate_unique_ids(outputs, f"{period_name} results")
        if {item.factor_id for item in inputs} != {item.factor_id for item in outputs}:
            raise ValueError(
                f"{period_name} evaluator must return every input candidate exactly once"
            )

    def _select_population(
        self,
        candidates: Sequence[EvaluatedCandidate],
    ) -> list[EvaluatedCandidate]:
        ranked = self._rank_valid_candidates(candidates)
        if len(ranked) < self.config.population_size:
            raise ValueError(
                f"need at least {self.config.population_size} valid candidates, "
                f"found {len(ranked)}"
            )
        return ranked[: self.config.population_size]

    @staticmethod
    def _rank_valid_candidates(
        candidates: Sequence[EvaluatedCandidate],
    ) -> list[EvaluatedCandidate]:
        return sorted(
            (candidate for candidate in candidates if candidate.passed_hard_filters),
            key=lambda candidate: (-candidate.composite_score, candidate.factor_id),
        )

    @staticmethod
    def _validate_unique_ids(items: Sequence[object], label: str) -> None:
        identifiers = [getattr(item, "factor_id") for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{label} contain duplicate factor ids")

    def _append_snapshot(self, state: EvoAlphaRunState, generation: int) -> None:
        best = state.population[0]
        state.snapshots.append(
            GenerationSnapshot(
                generation=generation,
                population_factor_ids=tuple(
                    candidate.factor_id for candidate in state.population
                ),
                best_factor_id=best.factor_id,
                best_composite_score=best.composite_score,
            )
        )

    def _transition(
        self,
        state: EvoAlphaRunState,
        stage: EvoAlphaStage,
        event_type: str,
        **details: object,
    ) -> None:
        state.stage = stage
        self._record_event(state, event_type, **details)

    @staticmethod
    def _record_event(
        state: EvoAlphaRunState,
        event_type: str,
        **details: object,
    ) -> None:
        state.events.append(
            WorkflowEvent(
                sequence=len(state.events) + 1,
                event_type=event_type,
                stage=state.stage,
                generation=state.generation,
                details=dict(details),
            )
        )
