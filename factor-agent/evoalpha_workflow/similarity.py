"""Training-period cross-sectional similarity for EvoAlpha parent pairing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import numpy as np
import polars as pl

from factor_pipeline.config import DataConfig, FactorConfig
from factor_pipeline.data import PanelDataSource
from factor_pipeline.factor import build_factor_from_spec
from factor_pipeline.single_factor_evaluation import EvaluationPeriod
from factor_pipeline.spec import FactorSpec


@dataclass(frozen=True)
class SimilarityConfig:
    panel_path: Path
    membership_path: Path
    training_period: EvaluationPeriod = EvaluationPeriod(
        start_date=date(2001, 1, 1),
        end_date=date(2001, 12, 31),
    )
    warmup_calendar_days: int = 366
    minimum_cross_section: int = 50
    parent_pair_similarity_limit: float = 0.80
    factor_config: FactorConfig = FactorConfig()
    output_dir: Path | None = None
    output_prefix: str = "top30"

    def __post_init__(self) -> None:
        if self.warmup_calendar_days < 0:
            raise ValueError("warmup_calendar_days cannot be negative")
        if self.minimum_cross_section < 3:
            raise ValueError("minimum_cross_section must be at least 3")
        if not 0 <= self.parent_pair_similarity_limit <= 1:
            raise ValueError("parent_pair_similarity_limit must be between 0 and 1")
        if not self.output_prefix or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in self.output_prefix
        ):
            raise ValueError("output_prefix contains unsupported characters")


@dataclass(frozen=True)
class FactorPairSimilarity:
    factor_a: str
    factor_b: str
    median_absolute_spearman: float
    median_signed_spearman: float
    valid_days: int
    eligible_for_parent_pairing: bool


@dataclass(frozen=True)
class SimilarityResult:
    factor_ids: tuple[str, ...]
    pairs: tuple[FactorPairSimilarity, ...]
    eligible_pairs: int
    excluded_pairs: int


class TrainingFactorSimilarityAnalyzer:
    def __init__(
        self,
        config: SimilarityConfig,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback

    def analyze_top_candidates(self, candidates_path: Path) -> SimilarityResult:
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        specs = [
            FactorSpec.model_validate(candidate["artifact"]["factor_spec"])
            for candidate in candidates
        ]
        return self.analyze_specs(specs)

    def analyze_specs(self, specs: Sequence[FactorSpec]) -> SimilarityResult:
        if len(specs) < 2:
            raise ValueError("similarity analysis requires at least two factors")
        factor_ids = [spec.factor_id for spec in specs]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("similarity FactorSpecs contain duplicate factor ids")

        panel = self._load_training_panel()
        wide = None
        for index, spec in enumerate(specs, start=1):
            values = (
                build_factor_from_spec(panel.lazy(), spec, self.config.factor_config)
                .filter(
                    pl.col("date").dt.date().is_between(
                        self.config.training_period.start_date,
                        self.config.training_period.end_date,
                    )
                )
                .select("date", "permno", pl.col("factor").alias(spec.factor_id))
                .collect(engine="streaming")
            )
            wide = (
                values
                if wide is None
                else wide.join(values, on=["date", "permno"], how="full", coalesce=True)
            )
            if self.progress_callback is not None:
                self.progress_callback(index, len(specs), spec.factor_id)

        ranked = wide.with_columns(
            [
                pl.col(factor_id)
                .rank(method="average")
                .over("date")
                .alias(factor_id)
                for factor_id in factor_ids
            ]
        ).sort("date", "permno")
        daily_correlations = {
            (left, right): []
            for left_index, left in enumerate(factor_ids)
            for right in factor_ids[left_index + 1 :]
        }
        for daily in ranked.partition_by("date", maintain_order=True):
            for left_index, left in enumerate(factor_ids):
                left_values = daily.get_column(left).to_numpy()
                for right in factor_ids[left_index + 1 :]:
                    right_values = daily.get_column(right).to_numpy()
                    valid = np.isfinite(left_values) & np.isfinite(right_values)
                    if valid.sum() < self.config.minimum_cross_section:
                        continue
                    left_valid = left_values[valid]
                    right_valid = right_values[valid]
                    if np.unique(left_valid).size < 2 or np.unique(right_valid).size < 2:
                        continue
                    correlation = float(np.corrcoef(left_valid, right_valid)[0, 1])
                    if np.isfinite(correlation):
                        daily_correlations[(left, right)].append(correlation)

        pairs = []
        for (left, right), correlations in daily_correlations.items():
            if not correlations:
                raise ValueError(f"no valid similarity dates for {left} and {right}")
            absolute_similarity = float(median(abs(value) for value in correlations))
            pairs.append(
                FactorPairSimilarity(
                    factor_a=left,
                    factor_b=right,
                    median_absolute_spearman=absolute_similarity,
                    median_signed_spearman=float(median(correlations)),
                    valid_days=len(correlations),
                    eligible_for_parent_pairing=(
                        absolute_similarity
                        <= self.config.parent_pair_similarity_limit
                    ),
                )
            )
        pairs.sort(
            key=lambda item: (
                item.median_absolute_spearman,
                item.factor_a,
                item.factor_b,
            )
        )
        eligible = sum(pair.eligible_for_parent_pairing for pair in pairs)
        result = SimilarityResult(
            factor_ids=tuple(factor_ids),
            pairs=tuple(pairs),
            eligible_pairs=eligible,
            excluded_pairs=len(pairs) - eligible,
        )
        if self.config.output_dir is not None:
            self._write_outputs(result)
        return result

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

    def _write_outputs(self, result: SimilarityResult) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.config.output_prefix
        pair_rows = [asdict(pair) for pair in result.pairs]
        pl.DataFrame(pair_rows).write_parquet(output_dir / f"{prefix}_pair_similarity.parquet")
        (output_dir / f"{prefix}_pair_similarity.json").write_text(
            json.dumps(pair_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lookup = {
            (pair.factor_a, pair.factor_b): pair.median_absolute_spearman
            for pair in result.pairs
        }
        matrix_rows = []
        for left in result.factor_ids:
            row = {"factor_id": left}
            for right in result.factor_ids:
                if left == right:
                    row[right] = 1.0
                else:
                    key = (left, right) if (left, right) in lookup else (right, left)
                    row[right] = lookup[key]
            matrix_rows.append(row)
        pl.DataFrame(matrix_rows).write_csv(output_dir / f"{prefix}_similarity_matrix.csv")
        (output_dir / f"{prefix}_similarity_summary.json").write_text(
            json.dumps(
                {
                    "factor_count": len(result.factor_ids),
                    "pair_count": len(result.pairs),
                    "eligible_pairs": result.eligible_pairs,
                    "excluded_pairs": result.excluded_pairs,
                    "similarity_limit": self.config.parent_pair_similarity_limit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
