"""Real Alpha158 seed evaluation for the oldest training period."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Callable, Sequence

import polars as pl

from factor_pipeline.alpha158 import executable_alpha158_specs
from factor_pipeline.config import DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    FactorScoreReference,
    SingleFactorMetrics,
    build_forward_return_labels,
    build_score_reference,
    evaluate_single_factor,
    score_factor_metrics,
)
from factor_pipeline.spec import FactorSpec

from .state import EvaluatedCandidate, ProposalType


@dataclass(frozen=True)
class Alpha158SeedEvaluationConfig:
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


class Alpha158SeedEvaluator:
    def __init__(
        self,
        config: Alpha158SeedEvaluationConfig,
        specs: Sequence[FactorSpec] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.config = config
        self.specs = tuple(specs or executable_alpha158_specs())
        self.progress_callback = progress_callback
        self.last_score_reference: FactorScoreReference | None = None

    def evaluate_seeds(self) -> list[EvaluatedCandidate]:
        if not self.specs:
            raise ValueError("seed evaluation requires at least one FactorSpec")
        self._validate_unique_factor_ids()
        panel = self._load_training_panel()
        labels = build_forward_return_labels(
            panel,
            self.config.training_period,
            self.config.evaluation_config.horizons,
        )

        successful_metrics: list[SingleFactorMetrics] = []
        failures: dict[str, str] = {}
        specs_by_id = {spec.factor_id: spec for spec in self.specs}
        for index, spec in enumerate(self.specs, start=1):
            try:
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
                        factor_id=spec.factor_id,
                        period=self.config.training_period,
                        config=self.config.evaluation_config,
                    )
                )
            except Exception as error:
                failures[spec.factor_id] = f"{type(error).__name__}: {error}"
            if self.progress_callback is not None:
                self.progress_callback(index, len(self.specs), spec.factor_id)

        self.last_score_reference = build_score_reference(successful_metrics)
        scored = score_factor_metrics(
            successful_metrics,
            reference=self.last_score_reference,
        )
        metrics_by_id = {metrics.factor_id: metrics for metrics in scored}
        candidates = []
        for factor_id, spec in specs_by_id.items():
            metrics = metrics_by_id.get(factor_id)
            if metrics is None:
                candidates.append(
                    EvaluatedCandidate(
                        factor_id=factor_id,
                        generation=0,
                        proposal_type=ProposalType.SEED,
                        composite_score=None,
                        passed_hard_filters=False,
                        artifact={"factor_spec": spec.model_dump(mode="json")},
                        metrics={"execution_error": failures[factor_id]},
                    )
                )
                continue
            candidates.append(
                EvaluatedCandidate(
                    factor_id=factor_id,
                    generation=0,
                    proposal_type=ProposalType.SEED,
                    composite_score=metrics.composite_score,
                    passed_hard_filters=metrics.passed_hard_filters,
                    direction=metrics.direction,
                    artifact={"factor_spec": spec.model_dump(mode="json")},
                    metrics=metrics.to_dict(),
                )
            )
        if self.config.output_dir is not None:
            self._write_outputs(candidates, self.last_score_reference)
        return candidates

    def _load_training_panel(self) -> pl.DataFrame:
        source = PanelDataSource(
            DataConfig(
                panel_path=self.config.panel_path,
                membership_path=self.config.membership_path,
                start_date=self.config.training_period.start_date,
                end_date=self.config.training_period.end_date,
                warmup_calendar_days=self.config.warmup_calendar_days,
                exit_buffer_calendar_days=0,
            )
        )
        return source.point_in_time_panel().collect(engine="streaming")

    def _validate_unique_factor_ids(self) -> None:
        factor_ids = [spec.factor_id for spec in self.specs]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("seed FactorSpecs contain duplicate factor ids")

    def _write_outputs(
        self,
        candidates: Sequence[EvaluatedCandidate],
        reference: FactorScoreReference,
    ) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "factor_id": candidate.factor_id,
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
        pl.DataFrame(rows).write_parquet(output_dir / "alpha158_seed_metrics.parquet")
        (output_dir / "alpha158_seed_candidates.json").write_text(
            json.dumps(
                _json_safe(
                    [candidate.model_dump(mode="json") for candidate in candidates]
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        (output_dir / "alpha158_score_reference.json").write_text(
            json.dumps(
                _json_safe(asdict(reference)),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )


def _json_safe(value):
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
