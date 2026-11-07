"""Similarity-aware admission and deterministic EvoAlpha population update."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import EvoAlphaConfig
from .similarity import FactorPairSimilarity
from .state import EvaluatedCandidate, ProposalType


class CandidateAdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str
    admitted: bool
    reason: str
    composite_score: float | None
    most_similar_factor_id: str | None = None
    maximum_similarity: float | None = Field(default=None, ge=0.0, le=1.0)


class PopulationUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int = Field(ge=1)
    admission_decisions: tuple[CandidateAdmissionDecision, ...]
    admitted_candidate_ids: tuple[str, ...]
    population: tuple[EvaluatedCandidate, ...]


class PopulationUpdater:
    def __init__(self, config: EvoAlphaConfig = EvoAlphaConfig()) -> None:
        self.config = config

    def update(
        self,
        current_population: Sequence[EvaluatedCandidate],
        candidates: Sequence[EvaluatedCandidate],
        similarities: Sequence[FactorPairSimilarity],
        generation: int,
    ) -> PopulationUpdateResult:
        self._validate_inputs(current_population, candidates)
        similarity_lookup = _similarity_lookup(similarities)
        current_by_id = {item.factor_id: item for item in current_population}
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item.composite_score is None,
                -(item.composite_score or 0.0),
                item.factor_id,
            ),
        )
        accepted_offspring: list[EvaluatedCandidate] = []
        decisions = []
        for candidate in ranked_candidates:
            if not candidate.passed_hard_filters or candidate.composite_score is None:
                decisions.append(
                    CandidateAdmissionDecision(
                        factor_id=candidate.factor_id,
                        admitted=False,
                        reason="failed_hard_filters_or_missing_score",
                        composite_score=candidate.composite_score,
                    )
                )
                continue
            duplicate = _find_exact_formula_duplicate(
                candidate,
                [*current_population, *accepted_offspring],
            )
            if duplicate is not None:
                decisions.append(
                    CandidateAdmissionDecision(
                        factor_id=candidate.factor_id,
                        admitted=False,
                        reason="exact_formula_duplicate",
                        composite_score=candidate.composite_score,
                        most_similar_factor_id=duplicate.factor_id,
                        maximum_similarity=1.0,
                    )
                )
                continue
            if candidate.proposal_type == ProposalType.REFINEMENT:
                decision = self._refinement_decision(
                    candidate,
                    current_by_id,
                    similarity_lookup,
                )
            else:
                references = [*current_population, *accepted_offspring]
                decision = self._crossover_family_decision(
                    candidate,
                    references,
                    similarity_lookup,
                )
            decisions.append(decision)
            if decision.admitted:
                accepted_offspring.append(candidate)

        combined = [*current_population, *accepted_offspring]
        next_population = tuple(
            sorted(
                combined,
                key=lambda item: (-item.composite_score, item.factor_id),
            )[: self.config.population_size]
        )
        if len(next_population) != self.config.population_size:
            raise ValueError("population update did not produce the configured population size")
        return PopulationUpdateResult(
            generation=generation,
            admission_decisions=tuple(decisions),
            admitted_candidate_ids=tuple(item.factor_id for item in accepted_offspring),
            population=next_population,
        )

    def _refinement_decision(
        self,
        candidate: EvaluatedCandidate,
        current_by_id: dict[str, EvaluatedCandidate],
        similarity_lookup: dict[frozenset[str], float],
    ) -> CandidateAdmissionDecision:
        parent_id = candidate.parent_factor_ids[0]
        parent = current_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"refinement parent is not in current population: {parent_id}")
        similarity = _required_similarity(candidate.factor_id, parent_id, similarity_lookup)
        improved = candidate.composite_score > parent.composite_score
        if (
            similarity > self.config.crossover_child_similarity_limit
            and not improved
        ):
            return CandidateAdmissionDecision(
                factor_id=candidate.factor_id,
                admitted=False,
                reason="high_parent_similarity_without_score_improvement",
                composite_score=candidate.composite_score,
                most_similar_factor_id=parent_id,
                maximum_similarity=similarity,
            )
        return CandidateAdmissionDecision(
            factor_id=candidate.factor_id,
            admitted=True,
            reason="refinement_admitted",
            composite_score=candidate.composite_score,
            most_similar_factor_id=parent_id,
            maximum_similarity=similarity,
        )

    def _crossover_family_decision(
        self,
        candidate: EvaluatedCandidate,
        references: Sequence[EvaluatedCandidate],
        similarity_lookup: dict[frozenset[str], float],
    ) -> CandidateAdmissionDecision:
        similarities = [
            (
                reference.factor_id,
                _required_similarity(
                    candidate.factor_id,
                    reference.factor_id,
                    similarity_lookup,
                ),
            )
            for reference in references
        ]
        most_similar_id, maximum = max(similarities, key=lambda item: item[1])
        if maximum > self.config.crossover_child_similarity_limit:
            return CandidateAdmissionDecision(
                factor_id=candidate.factor_id,
                admitted=False,
                reason="crossover_family_similarity_above_limit",
                composite_score=candidate.composite_score,
                most_similar_factor_id=most_similar_id,
                maximum_similarity=maximum,
            )
        return CandidateAdmissionDecision(
            factor_id=candidate.factor_id,
            admitted=True,
            reason="crossover_family_admitted",
            composite_score=candidate.composite_score,
            most_similar_factor_id=most_similar_id,
            maximum_similarity=maximum,
        )

    def _validate_inputs(
        self,
        current_population: Sequence[EvaluatedCandidate],
        candidates: Sequence[EvaluatedCandidate],
    ) -> None:
        if len(current_population) != self.config.population_size:
            raise ValueError("current population size does not match configuration")
        identifiers = [item.factor_id for item in [*current_population, *candidates]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("population update inputs contain duplicate factor ids")
        if any(
            not item.passed_hard_filters or item.composite_score is None
            for item in current_population
        ):
            raise ValueError("current population must contain only valid scored candidates")


def _similarity_lookup(
    similarities: Sequence[FactorPairSimilarity],
) -> dict[frozenset[str], float]:
    lookup = {}
    for pair in similarities:
        key = frozenset((pair.factor_a, pair.factor_b))
        if len(key) != 2 or key in lookup:
            raise ValueError("similarity records contain duplicate or invalid pairs")
        lookup[key] = pair.median_absolute_spearman
    return lookup


def _required_similarity(
    left: str,
    right: str,
    lookup: dict[frozenset[str], float],
) -> float:
    key = frozenset((left, right))
    try:
        return lookup[key]
    except KeyError as error:
        raise ValueError(f"missing similarity for {left} and {right}") from error


def _find_exact_formula_duplicate(
    candidate: EvaluatedCandidate,
    references: Sequence[EvaluatedCandidate],
) -> EvaluatedCandidate | None:
    fingerprint = _formula_fingerprint(candidate)
    if fingerprint is None:
        return None
    return next(
        (
            reference
            for reference in references
            if _formula_fingerprint(reference) == fingerprint
        ),
        None,
    )


def _formula_fingerprint(candidate: EvaluatedCandidate) -> str | None:
    factor_spec = candidate.artifact.get("factor_spec")
    if not isinstance(factor_spec, dict) or "formula" not in factor_spec:
        return None
    return json.dumps(
        factor_spec["formula"],
        sort_keys=True,
        separators=(",", ":"),
    )
