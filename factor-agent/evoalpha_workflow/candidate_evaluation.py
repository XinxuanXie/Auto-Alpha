"""Unified training-period evaluator for refinement, crossover and mutation candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Sequence

import polars as pl

from factor_pipeline.config import DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    FactorScoreReference,
    SingleFactorMetrics,
    build_forward_return_labels,
    evaluate_single_factor,
    score_factor_metrics,
)
from factor_pipeline.spec import FactorSpec

from .state import CandidateProposal, EvaluatedCandidate


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    panel_path: Path
    membership_path: Path
    training_period: EvaluationPeriod = EvaluationPeriod(
        start_date=date(2001, 1, 1),
        end_date=date(2001, 12, 31),
    )
    warmup_calendar_days: int = 366
    factor_config: FactorConfig = FactorConfig()
    evaluation_config: FactorEvaluationConfig = FactorEvaluationConfig()
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.warmup_calendar_days < 0:
            raise ValueError("warmup_calendar_days cannot be negative")


class TrainingCandidateEvaluator:
    def __init__(
        self,
        config: CandidateEvaluationConfig,
        score_reference: FactorScoreReference,
        known_candidates: Sequence[EvaluatedCandidate],
    ) -> None:
        self.config = config
        self.score_reference = score_reference
        self.known_candidates = {
            candidate.factor_id: candidate for candidate in known_candidates
        }
        if len(self.known_candidates) != len(known_candidates):
            raise ValueError("known_candidates contain duplicate factor ids")

    def evaluate_candidates(
        self,
        proposals: Sequence[CandidateProposal],
        generation: int,
    ) -> list[EvaluatedCandidate]:
        self._validate_proposals(proposals, generation)
        panel = self._load_training_panel()
        labels = build_forward_return_labels(
            panel,
            self.config.training_period,
            self.config.evaluation_config.horizons,
        )
        successful_metrics: list[SingleFactorMetrics] = []
        failures: dict[str, str] = {}
        proposals_by_id = {proposal.factor_id: proposal for proposal in proposals}
        for proposal in proposals:
            try:
                spec = FactorSpec.model_validate(proposal.artifact["factor_spec"])
                factor_values = (
                    build_factor_from_spec(
                        panel.lazy(),
                        spec,
                        self.config.factor_config,
                    )
                    .select("date", "permno", "factor")
                    .collect(engine="streaming")
                )
                successful_metrics.append(
                    evaluate_single_factor(
                        factor_values=factor_values,
                        labels=labels,
                        factor_id=proposal.factor_id,
                        period=self.config.training_period,
                        config=self.config.evaluation_config,
                    )
                )
            except Exception as error:
                failures[proposal.factor_id] = f"{type(error).__name__}: {error}"

        scored = score_factor_metrics(
            successful_metrics,
            reference=self.score_reference,
        )
        metrics_by_id = {metrics.factor_id: metrics for metrics in scored}
        candidates = []
        for factor_id, proposal in proposals_by_id.items():
            metrics = metrics_by_id.get(factor_id)
            if metrics is None:
                candidates.append(
                    EvaluatedCandidate(
                        factor_id=factor_id,
                        generation=generation,
                        proposal_type=proposal.proposal_type,
                        parent_factor_ids=proposal.parent_factor_ids,
                        composite_score=None,
                        passed_hard_filters=False,
                        artifact=proposal.artifact,
                        metrics={"execution_error": failures[factor_id]},
                    )
                )
                continue
            candidates.append(
                EvaluatedCandidate(
                    factor_id=factor_id,
                    generation=generation,
                    proposal_type=proposal.proposal_type,
                    parent_factor_ids=proposal.parent_factor_ids,
                    composite_score=metrics.composite_score,
                    passed_hard_filters=metrics.passed_hard_filters,
                    direction=metrics.direction,
                    artifact=proposal.artifact,
                    metrics=metrics.to_dict(),
                )
            )

        candidates = self._attach_metric_deltas(candidates)
        if self.config.output_dir is not None:
            self._write_outputs(candidates, generation)
        return candidates

    def _validate_proposals(
        self,
        proposals: Sequence[CandidateProposal],
        generation: int,
    ) -> None:
        identifiers = [proposal.factor_id for proposal in proposals]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate proposals contain duplicate factor ids")
        if any(proposal.generation != generation for proposal in proposals):
            raise ValueError("candidate proposal generation mismatch")
        if any("factor_spec" not in proposal.artifact for proposal in proposals):
            raise ValueError("every candidate proposal requires factor_spec artifact")

    def _load_training_panel(self) -> pl.DataFrame:
        return PanelDataSource(
            DataConfig(
                panel_path=self.config.panel_path,
                membership_path=self.config.membership_path,
                start_date=self.config.training_period.start_date,
                end_date=self.config.training_period.end_date,
                warmup_calendar_days=self.config.warmup_calendar_days,
                exit_buffer_calendar_days=0,
            )
        ).point_in_time_panel().collect(engine="streaming")

    def _attach_metric_deltas(
        self,
        candidates: Sequence[EvaluatedCandidate],
    ) -> list[EvaluatedCandidate]:
        evaluated_by_id = {candidate.factor_id: candidate for candidate in candidates}
        all_candidates = {**self.known_candidates, **evaluated_by_id}
        enriched = []
        for candidate in candidates:
            comparisons = []
            for parent_id in candidate.parent_factor_ids:
                parent = all_candidates.get(parent_id)
                if parent is None:
                    raise ValueError(f"missing evaluated parent for delta: {parent_id}")
                comparisons.append(_comparison(candidate, parent, "direct_parent"))
                if parent.parent_factor_ids:
                    for original_parent_id in parent.parent_factor_ids:
                        original = all_candidates.get(original_parent_id)
                        if original is not None:
                            comparisons.append(
                                _comparison(candidate, original, "original_parent")
                            )
            enriched.append(
                candidate.model_copy(
                    update={
                        "metrics": {
                            **candidate.metrics,
                            "parent_comparisons": comparisons,
                        }
                    }
                )
            )
        return enriched

    def _write_outputs(
        self,
        candidates: Sequence[EvaluatedCandidate],
        generation: int,
    ) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"generation_{generation:02d}_candidate"
        payload = [candidate.model_dump(mode="json") for candidate in candidates]
        (output_dir / f"{prefix}_evaluations.json").write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "factor_id": candidate.factor_id,
                    "proposal_type": candidate.proposal_type.value,
                    "parent_factor_ids": json.dumps(candidate.parent_factor_ids),
                    "composite_score": candidate.composite_score,
                    "passed_hard_filters": candidate.passed_hard_filters,
                    "direction": candidate.direction,
                    "metrics_json": json.dumps(
                        _json_safe(candidate.metrics),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )
        pl.DataFrame(rows).write_parquet(output_dir / f"{prefix}_evaluations.parquet")


def load_score_reference(path: Path) -> FactorScoreReference:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FactorScoreReference(
        rank_ic_values=tuple(payload["rank_ic_values"]),
        rank_icir_values=tuple(payload["rank_icir_values"]),
        monthly_stability_values=tuple(payload["monthly_stability_values"]),
    )


def _comparison(
    child: EvaluatedCandidate,
    parent: EvaluatedCandidate,
    relationship: str,
) -> dict[str, object]:
    metrics = (
        "mean_rank_ic_20d",
        "rank_icir_20d",
        "monthly_stability",
        "cross_horizon_consistency",
    )
    deltas = {}
    for metric in metrics:
        child_value = child.metrics.get(metric)
        parent_value = parent.metrics.get(metric)
        deltas[metric] = _difference(child_value, parent_value)
    deltas["composite_score"] = _difference(
        child.composite_score,
        parent.composite_score,
    )
    return {
        "relationship": relationship,
        "parent_factor_id": parent.factor_id,
        "deltas": deltas,
    }


def _difference(left, right) -> float | None:
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if isfinite(value) else None


def _json_safe(value):
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
