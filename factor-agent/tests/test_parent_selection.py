import unittest
from collections import Counter

from evoalpha_workflow import EvoAlphaConfig
from evoalpha_workflow.parent_selection import ParentSelector
from evoalpha_workflow.similarity import FactorPairSimilarity
from evoalpha_workflow.state import EvaluatedCandidate, ProposalType


def make_population():
    return [
        EvaluatedCandidate(
            factor_id=f"F-{index:02d}",
            generation=0,
            proposal_type=ProposalType.SEED,
            composite_score=1.0 - index / 100,
        )
        for index in range(30)
    ]


def make_similarities(population):
    pairs = []
    for left_index, left in enumerate(population):
        for right_index, right in enumerate(population[left_index + 1 :], start=left_index + 1):
            similarity = ((left_index * 17 + right_index * 11) % 75) / 100
            pairs.append(
                FactorPairSimilarity(
                    factor_a=left.factor_id,
                    factor_b=right.factor_id,
                    median_absolute_spearman=similarity,
                    median_signed_spearman=similarity,
                    valid_days=248,
                    eligible_for_parent_pairing=True,
                )
            )
    return pairs


class ParentSelectorTests(unittest.TestCase):
    def test_selects_reproducible_fixed_parent_plan(self):
        population = make_population()
        similarities = make_similarities(population)
        selector = ParentSelector()

        first = selector.select(population, similarities, generation=1)
        second = selector.select(population, similarities, generation=1)

        self.assertEqual(first, second)
        self.assertEqual(len(first.refinement_tasks), 12)
        self.assertEqual(len(first.crossover_tasks), 8)
        self.assertEqual(
            len({task.parent_factor_id for task in first.refinement_tasks}),
            12,
        )
        usage = Counter(
            parent
            for task in first.crossover_tasks
            for parent in task.parent_factor_ids
        )
        self.assertLessEqual(max(usage.values()), 2)
        self.assertTrue(
            all(task.median_absolute_spearman <= 0.80 for task in first.crossover_tasks)
        )
        self.assertEqual(first.random_seed, 43)

    def test_rejects_insufficient_eligible_crossover_pairs(self):
        population = make_population()
        only_pair = FactorPairSimilarity(
            factor_a=population[0].factor_id,
            factor_b=population[1].factor_id,
            median_absolute_spearman=0.1,
            median_signed_spearman=0.1,
            valid_days=248,
            eligible_for_parent_pairing=True,
        )

        with self.assertRaisesRegex(ValueError, "not enough eligible crossover pairs"):
            ParentSelector().select(population, [only_pair], generation=1)

    def test_config_requires_crossover_weights_to_sum_to_one(self):
        with self.assertRaises(ValueError):
            EvoAlphaConfig(
                crossover_performance_weight=0.8,
                crossover_diversity_weight=0.3,
            )


if __name__ == "__main__":
    unittest.main()
