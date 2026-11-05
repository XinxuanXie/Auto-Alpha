import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from evoalpha_workflow.candidate_evaluation import (
    CandidateEvaluationConfig,
    TrainingCandidateEvaluator,
)
from evoalpha_workflow.state import CandidateProposal, EvaluatedCandidate, ProposalType
from factor_pipeline.config import FactorConfig
from factor_pipeline.single_factor_evaluation import (
    EvaluationPeriod,
    FactorEvaluationConfig,
    FactorScoreReference,
)
from factor_pipeline.spec import FactorSpec


def factor_spec(factor_id: str, window: int) -> FactorSpec:
    return FactorSpec.model_validate(
        {
            "factor_id": factor_id,
            "name": factor_id,
            "description": "Candidate evaluation test factor.",
            "formula": {"op": "return", "field": "close", "window": window},
            "required_fields": ["close"],
            "lookback_days": window,
        }
    )


def known_candidate(factor_id: str, score: float) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        factor_id=factor_id,
        generation=0,
        proposal_type=ProposalType.SEED,
        composite_score=score,
        metrics={
            "mean_rank_ic_20d": 0.01,
            "rank_icir_20d": 0.10,
            "monthly_stability": 0.50,
            "cross_horizon_consistency": 0.50,
        },
    )


class CandidateEvaluationPipelineTests(unittest.TestCase):
    def test_evaluates_candidates_with_fixed_reference_and_parent_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            membership_path = root / "membership.parquet"
            output_dir = root / "results"
            start = datetime(2020, 1, 1)
            rows = []
            for day_index in range(100):
                for stock_index in range(30):
                    close = (
                        20.0
                        + stock_index * 0.35
                        + day_index * (0.03 + stock_index * 0.0007)
                        + np.sin((day_index + stock_index * 1.3) / 7.0)
                    )
                    rows.append(
                        {
                            "date": start + timedelta(days=day_index),
                            "permno": stock_index + 1,
                            "open": close * 0.995,
                            "high": close * 1.01,
                            "low": close * 0.99,
                            "close": close,
                            "volume": 10_000.0 + stock_index * 100 + day_index * 20,
                            "mktcap": 1_000_000.0 + stock_index * 60_000.0,
                        }
                    )
            pl.DataFrame(rows).with_columns(
                pl.col("permno").cast(pl.Int32)
            ).write_parquet(panel_path)
            pl.DataFrame(
                {
                    "permno": pl.Series(range(1, 31), dtype=pl.Int32),
                    "start_date": [start] * 30,
                    "end_date": [start + timedelta(days=99)] * 30,
                }
            ).write_parquet(membership_path)
            crossover_spec = factor_spec("CROSS", 5)
            mutation_spec = factor_spec("MUT", 10)
            proposals = [
                CandidateProposal(
                    factor_id="CROSS",
                    generation=1,
                    proposal_type=ProposalType.CROSSOVER,
                    parent_factor_ids=("PARENT-A", "PARENT-B"),
                    artifact={"factor_spec": crossover_spec.model_dump(mode="json")},
                ),
                CandidateProposal(
                    factor_id="MUT",
                    generation=1,
                    proposal_type=ProposalType.MUTATION,
                    parent_factor_ids=("CROSS",),
                    artifact={"factor_spec": mutation_spec.model_dump(mode="json")},
                ),
            ]
            evaluator = TrainingCandidateEvaluator(
                CandidateEvaluationConfig(
                    panel_path=panel_path,
                    membership_path=membership_path,
                    training_period=EvaluationPeriod(
                        start_date=date(2020, 2, 1),
                        end_date=date(2020, 4, 9),
                    ),
                    warmup_calendar_days=31,
                    factor_config=FactorConfig(minimum_cross_section=10),
                    evaluation_config=FactorEvaluationConfig(
                        minimum_cross_section=10,
                    ),
                    output_dir=output_dir,
                ),
                score_reference=FactorScoreReference(
                    rank_ic_values=(-0.10, 0.0, 0.10),
                    rank_icir_values=(-1.0, 0.0, 1.0),
                    monthly_stability_values=(0.0, 0.5, 1.0),
                ),
                known_candidates=[
                    known_candidate("PARENT-A", 0.60),
                    known_candidate("PARENT-B", 0.55),
                ],
            )

            evaluated = evaluator.evaluate_candidates(proposals, generation=1)

            self.assertEqual(len(evaluated), 2)
            mutation = next(item for item in evaluated if item.factor_id == "MUT")
            comparisons = mutation.metrics["parent_comparisons"]
            self.assertEqual(
                {item["relationship"] for item in comparisons},
                {"direct_parent", "original_parent"},
            )
            self.assertEqual(len(comparisons), 3)
            self.assertTrue(
                (output_dir / "generation_01_candidate_evaluations.json").is_file()
            )
            self.assertTrue(
                (output_dir / "generation_01_candidate_evaluations.parquet").is_file()
            )

    def test_rejects_duplicate_candidate_ids(self):
        proposal = CandidateProposal(
            factor_id="DUPLICATE",
            generation=1,
            proposal_type=ProposalType.REFINEMENT,
            parent_factor_ids=("PARENT",),
            artifact={"factor_spec": factor_spec("DUPLICATE", 5).model_dump(mode="json")},
        )
        evaluator = object.__new__(TrainingCandidateEvaluator)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluator._validate_proposals([proposal, proposal], generation=1)


if __name__ == "__main__":
    unittest.main()
