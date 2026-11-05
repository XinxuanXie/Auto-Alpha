"""Deterministic parent selection for refinement and crossover tasks."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import EvoAlphaConfig
from .similarity import FactorPairSimilarity
from .state import EvaluatedCandidate


class RefinementTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    generation: int = Field(ge=1)
    parent_factor_id: str
    parent_composite_score: float


class CrossoverTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    generation: int = Field(ge=1)
    parent_factor_ids: tuple[str, str]
    parent_composite_scores: tuple[float, float]
    median_absolute_spearman: float = Field(ge=0.0, le=1.0)
    selection_priority: float


class ParentSelectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int = Field(ge=1)
    random_seed: int
    refinement_tasks: tuple[RefinementTask, ...]
    crossover_tasks: tuple[CrossoverTask, ...]


class ParentSelector:
    def __init__(self, config: EvoAlphaConfig = EvoAlphaConfig()) -> None:
        self.config = config

    def select(
        self,
        population: Sequence[EvaluatedCandidate],
        similarities: Sequence[FactorPairSimilarity],
        generation: int,
    ) -> ParentSelectionPlan:
        if generation < 1:
            raise ValueError("generation must be at least 1")
        population_by_id = self._validate_population(population)
        seed = self.config.random_seed + generation
        refinement_parents = self._weighted_sample_without_replacement(
            population,
            self.config.refinement_offspring_per_generation,
            random.Random(seed),
        )
        refinement_tasks = tuple(
            RefinementTask(
                task_id=f"G{generation:02d}-REF-{index:03d}",
                generation=generation,
                parent_factor_id=parent.factor_id,
                parent_composite_score=parent.composite_score,
            )
            for index, parent in enumerate(refinement_parents, start=1)
        )
        crossover_tasks = self._select_crossover_tasks(
            population_by_id,
            similarities,
            generation,
        )
        return ParentSelectionPlan(
            generation=generation,
            random_seed=seed,
            refinement_tasks=refinement_tasks,
            crossover_tasks=crossover_tasks,
        )

    def _validate_population(
        self,
        population: Sequence[EvaluatedCandidate],
    ) -> dict[str, EvaluatedCandidate]:
        if len(population) != self.config.population_size:
            raise ValueError(
                f"parent selection requires population size {self.config.population_size}"
            )
        identifiers = [candidate.factor_id for candidate in population]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("population contains duplicate factor ids")
        if any(
            not candidate.passed_hard_filters
            or candidate.composite_score is None
            for candidate in population
        ):
            raise ValueError("all population candidates must be valid and scored")
        return {candidate.factor_id: candidate for candidate in population}

    @staticmethod
    def _weighted_sample_without_replacement(
        population: Sequence[EvaluatedCandidate],
        sample_size: int,
        rng: random.Random,
    ) -> list[EvaluatedCandidate]:
        if sample_size > len(population):
            raise ValueError("refinement sample size exceeds population size")
        remaining = list(population)
        selected = []
        for _ in range(sample_size):
            weights = [max(candidate.composite_score or 0.0, 0.0) + 1e-12 for candidate in remaining]
            threshold = rng.random() * sum(weights)
            cumulative = 0.0
            selected_index = len(remaining) - 1
            for index, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= threshold:
                    selected_index = index
                    break
            selected.append(remaining.pop(selected_index))
        return selected

    def _select_crossover_tasks(
        self,
        population_by_id: dict[str, EvaluatedCandidate],
        similarities: Sequence[FactorPairSimilarity],
        generation: int,
    ) -> tuple[CrossoverTask, ...]:
        eligible = []
        seen_pairs = set()
        for pair in similarities:
            pair_key = frozenset((pair.factor_a, pair.factor_b))
            if len(pair_key) != 2 or pair_key in seen_pairs:
                raise ValueError("similarity input contains duplicate or invalid pairs")
            seen_pairs.add(pair_key)
            if pair.factor_a not in population_by_id or pair.factor_b not in population_by_id:
                continue
            if pair.median_absolute_spearman > self.config.parent_pair_similarity_limit:
                continue
            left = population_by_id[pair.factor_a]
            right = population_by_id[pair.factor_b]
            mean_score = (left.composite_score + right.composite_score) / 2.0
            priority = (
                self.config.crossover_performance_weight * mean_score
                + self.config.crossover_diversity_weight
                * (1.0 - pair.median_absolute_spearman)
            )
            eligible.append((priority, pair, left, right))
        eligible.sort(
            key=lambda item: (
                -item[0],
                item[1].median_absolute_spearman,
                item[1].factor_a,
                item[1].factor_b,
            )
        )

        usage: Counter[str] = Counter()
        tasks = []
        for priority, pair, left, right in eligible:
            if usage[left.factor_id] >= self.config.max_crossover_tasks_per_parent:
                continue
            if usage[right.factor_id] >= self.config.max_crossover_tasks_per_parent:
                continue
            tasks.append(
                CrossoverTask(
                    task_id=f"G{generation:02d}-CROSS-{len(tasks) + 1:03d}",
                    generation=generation,
                    parent_factor_ids=(left.factor_id, right.factor_id),
                    parent_composite_scores=(
                        left.composite_score,
                        right.composite_score,
                    ),
                    median_absolute_spearman=pair.median_absolute_spearman,
                    selection_priority=priority,
                )
            )
            usage[left.factor_id] += 1
            usage[right.factor_id] += 1
            if len(tasks) == self.config.crossover_tasks_per_generation:
                break
        if len(tasks) != self.config.crossover_tasks_per_generation:
            raise ValueError(
                "not enough eligible crossover pairs under the similarity and usage limits"
            )
        return tuple(tasks)
