"""Fixed experiment protocol for the independent EvoAlpha workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvoAlphaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    population_size: int = Field(default=30, ge=1)
    generations: int = Field(default=3, ge=1)
    refinement_offspring_per_generation: int = Field(default=12, ge=0)
    crossover_tasks_per_generation: int = Field(default=8, ge=0)
    parent_pair_similarity_limit: float = Field(default=0.80, ge=0.0, le=1.0)
    crossover_child_similarity_limit: float = Field(default=0.90, ge=0.0, le=1.0)
    random_seed: int = 42
