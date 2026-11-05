"""Serializable state models for the deterministic EvoAlpha loop."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvoAlphaStage(StrEnum):
    INIT = "init"
    SEED_EVALUATION = "seed_evaluation"
    SEED_SELECTION = "seed_selection"
    OFFSPRING_PROPOSAL = "offspring_proposal"
    CANDIDATE_EVALUATION = "candidate_evaluation"
    POPULATION_UPDATE = "population_update"
    VALIDATION = "validation"
    TEST = "test"
    COMPLETED = "completed"
    FAILED = "failed"


class ProposalType(StrEnum):
    SEED = "seed"
    REFINEMENT = "refinement"
    CROSSOVER = "crossover"
    MUTATION = "mutation"


class CandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(min_length=1, max_length=200)
    generation: int = Field(ge=1)
    proposal_type: ProposalType
    parent_factor_ids: tuple[str, ...] = Field(min_length=1)
    artifact: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        parent_count = len(self.parent_factor_ids)
        expected = 1 if self.proposal_type in {
            ProposalType.REFINEMENT,
            ProposalType.MUTATION,
        } else 2
        if self.proposal_type == ProposalType.SEED:
            raise ValueError("seed candidates are created by seed evaluation")
        if parent_count != expected:
            raise ValueError(
                f"{self.proposal_type.value} requires {expected} parent factor ids"
            )
        if len(set(self.parent_factor_ids)) != parent_count:
            raise ValueError("parent_factor_ids cannot contain duplicates")
        return self


class EvaluatedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(min_length=1, max_length=200)
    generation: int = Field(ge=0)
    proposal_type: ProposalType
    parent_factor_ids: tuple[str, ...] = ()
    composite_score: float
    passed_hard_filters: bool = True
    direction: int = Field(default=1, ge=-1, le=1)
    artifact: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if not isfinite(self.composite_score):
            raise ValueError("composite_score must be finite")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        if self.proposal_type == ProposalType.SEED:
            if self.generation != 0 or self.parent_factor_ids:
                raise ValueError("seed candidates require generation 0 and no parents")
        elif self.generation < 1:
            raise ValueError("offspring candidates require generation at least 1")
        return self


class GenerationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int = Field(ge=0)
    population_factor_ids: tuple[str, ...]
    best_factor_id: str
    best_composite_score: float


class WorkflowEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    event_type: str
    stage: EvoAlphaStage
    generation: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class EvoAlphaRunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    stage: EvoAlphaStage = EvoAlphaStage.INIT
    generation: int = 0
    seed_candidates: list[EvaluatedCandidate] = Field(default_factory=list)
    population: list[EvaluatedCandidate] = Field(default_factory=list)
    current_proposals: list[CandidateProposal] = Field(default_factory=list)
    current_offspring: list[EvaluatedCandidate] = Field(default_factory=list)
    validation_results: list[EvaluatedCandidate] = Field(default_factory=list)
    test_results: list[EvaluatedCandidate] = Field(default_factory=list)
    snapshots: list[GenerationSnapshot] = Field(default_factory=list)
    events: list[WorkflowEvent] = Field(default_factory=list)
    error: str | None = None
