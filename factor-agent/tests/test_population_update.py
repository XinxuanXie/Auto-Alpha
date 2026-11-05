import unittest

from evoalpha_workflow.population import PopulationUpdater
from evoalpha_workflow.similarity import FactorPairSimilarity
from evoalpha_workflow.state import EvaluatedCandidate, ProposalType


def candidate(
    factor_id: str,
    score: float,
    proposal_type: ProposalType = ProposalType.SEED,
    parents: tuple[str, ...] = (),
) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        factor_id=factor_id,
        generation=0 if proposal_type == ProposalType.SEED else 1,
        proposal_type=proposal_type,
        parent_factor_ids=parents,
        composite_score=score,
    )


def pair(left: str, right: str, similarity: float) -> FactorPairSimilarity:
    return FactorPairSimilarity(
        factor_a=left,
        factor_b=right,
        median_absolute_spearman=similarity,
        median_signed_spearman=similarity,
        valid_days=248,
        eligible_for_parent_pairing=similarity <= 0.8,
    )


class PopulationUpdaterTests(unittest.TestCase):
    def test_applies_refinement_and_crossover_family_similarity_rules(self):
        population = [candidate(f"P-{index:02d}", 0.50 - index / 100) for index in range(30)]
        refinement = candidate(
            "REF",
            0.60,
            ProposalType.REFINEMENT,
            ("P-00",),
        )
        crossover = candidate(
            "CROSS",
            0.59,
            ProposalType.CROSSOVER,
            ("P-00", "P-01"),
        )
        mutation = candidate(
            "MUT",
            0.58,
            ProposalType.MUTATION,
            ("CROSS",),
        )
        candidates = [refinement, crossover, mutation]
        similarities = []
        for child in candidates:
            for parent in population:
                value = 0.95 if child.factor_id == "REF" and parent.factor_id == "P-00" else 0.20
                similarities.append(pair(child.factor_id, parent.factor_id, value))
        similarities.extend(
            [
                pair("REF", "CROSS", 0.30),
                pair("REF", "MUT", 0.30),
                pair("CROSS", "MUT", 0.99),
            ]
        )

        result = PopulationUpdater().update(
            population,
            candidates,
            similarities,
            generation=1,
        )

        decisions = {item.factor_id: item for item in result.admission_decisions}
        self.assertTrue(decisions["REF"].admitted)
        self.assertTrue(decisions["CROSS"].admitted)
        self.assertFalse(decisions["MUT"].admitted)
        self.assertEqual(
            decisions["MUT"].reason,
            "crossover_family_similarity_above_limit",
        )
        self.assertEqual(len(result.population), 30)
        self.assertIn("REF", {item.factor_id for item in result.population})
        self.assertIn("CROSS", {item.factor_id for item in result.population})

    def test_rejects_high_similarity_refinement_without_improvement(self):
        population = [candidate(f"P-{index:02d}", 0.80 - index / 100) for index in range(30)]
        refinement = candidate(
            "REF",
            0.70,
            ProposalType.REFINEMENT,
            ("P-00",),
        )
        result = PopulationUpdater().update(
            population,
            [refinement],
            [pair("REF", "P-00", 0.95)],
            generation=1,
        )

        self.assertFalse(result.admission_decisions[0].admitted)
        self.assertEqual(
            result.admission_decisions[0].reason,
            "high_parent_similarity_without_score_improvement",
        )


if __name__ == "__main__":
    unittest.main()
