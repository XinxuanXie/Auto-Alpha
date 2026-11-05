"""Replaceable stage interfaces used by the deterministic outer loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import EvoAlphaConfig
from .state import CandidateProposal, EvaluatedCandidate


class SeedEvaluator(Protocol):
    def evaluate_seeds(self) -> Sequence[EvaluatedCandidate]: ...


class OffspringProposer(Protocol):
    def propose_offspring(
        self,
        population: Sequence[EvaluatedCandidate],
        generation: int,
        config: EvoAlphaConfig,
    ) -> Sequence[CandidateProposal]: ...


class CandidateEvaluator(Protocol):
    def evaluate_candidates(
        self,
        proposals: Sequence[CandidateProposal],
        generation: int,
    ) -> Sequence[EvaluatedCandidate]: ...


class ValidationEvaluator(Protocol):
    def evaluate_validation(
        self,
        population: Sequence[EvaluatedCandidate],
    ) -> Sequence[EvaluatedCandidate]: ...


class TestEvaluator(Protocol):
    def evaluate_test(
        self,
        selected_candidates: Sequence[EvaluatedCandidate],
    ) -> Sequence[EvaluatedCandidate]: ...


@dataclass(frozen=True)
class EvoAlphaTools:
    seed_evaluator: SeedEvaluator
    offspring_proposer: OffspringProposer
    candidate_evaluator: CandidateEvaluator
    validation_evaluator: ValidationEvaluator
    test_evaluator: TestEvaluator
