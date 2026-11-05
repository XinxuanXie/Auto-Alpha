import json
import unittest
from pathlib import Path

from evoalpha_workflow.crossover import (
    CrossoverAgent,
    MutationAgent,
    RuleBasedMockCrossoverClient,
    RuleBasedMockMutationClient,
)
from evoalpha_workflow.parent_selection import ParentSelectionPlan
from evoalpha_workflow.state import EvaluatedCandidate, ProposalType
from factor_pipeline.spec import FactorSpec, FormulaNode


RESULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "evoalpha_seed_2001"


class StaticClient:
    def __init__(self, response):
        self.response = response

    def complete(self, system_prompt, user_prompt):
        del system_prompt, user_prompt
        return self.response


class CrossoverAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = {
            candidate.factor_id: candidate
            for candidate in (
                EvaluatedCandidate.model_validate(item)
                for item in json.loads(
                    (RESULT_ROOT / "alpha158_top30.json").read_text(encoding="utf-8")
                )
            )
        }
        cls.plan = ParentSelectionPlan.model_validate_json(
            (RESULT_ROOT / "generation_01_parent_plan.json").read_text(
                encoding="utf-8"
            )
        )

    def test_mock_generates_eight_intermediates_and_mutations(self):
        crossover_agent = CrossoverAgent(RuleBasedMockCrossoverClient())
        mutation_agent = MutationAgent(RuleBasedMockMutationClient())
        intermediates = []
        mutations = []

        for task in self.plan.crossover_tasks:
            intermediate = crossover_agent.propose(
                task,
                self.population[task.parent_factor_ids[0]],
                self.population[task.parent_factor_ids[1]],
            )
            intermediates.append(intermediate)
            mutations.append(mutation_agent.propose(intermediate))

        self.assertEqual(len(intermediates), 8)
        self.assertEqual(len(mutations), 8)
        self.assertEqual(len({item.intermediate_factor_id for item in intermediates}), 8)
        self.assertEqual(len({item.child_factor_id for item in mutations}), 8)
        self.assertTrue(
            all(
                item.to_candidate_proposal().proposal_type == ProposalType.CROSSOVER
                for item in intermediates
            )
        )
        self.assertTrue(
            all(
                item.to_candidate_proposal().proposal_type == ProposalType.MUTATION
                for item in mutations
            )
        )

    def test_negative_parent_direction_is_embedded_in_formula(self):
        task = self.plan.crossover_tasks[0]
        parent_a = self.population[task.parent_factor_ids[0]]
        parent_b = self.population[task.parent_factor_ids[1]]
        proposal = CrossoverAgent(RuleBasedMockCrossoverClient()).propose(
            task,
            parent_a,
            parent_b,
        )
        operators = []

        def visit(node):
            operators.append(node.op.value)
            for argument in node.args:
                visit(argument)

        visit(proposal.factor_spec.formula)
        expected_negations = sum(parent.direction == -1 for parent in (parent_a, parent_b))
        self.assertGreaterEqual(operators.count("negate"), expected_negations)

    def test_rejects_crossover_that_omits_second_parent(self):
        task = self.plan.crossover_tasks[0]
        parent_a = self.population[task.parent_factor_ids[0]]
        parent_b = self.population[task.parent_factor_ids[1]]
        client = RuleBasedMockCrossoverClient()
        response = client.complete(
            *CrossoverAgent.build_prompts(
                task,
                parent_a,
                parent_b,
                FactorSpec.model_validate(parent_a.artifact["factor_spec"]),
                FactorSpec.model_validate(parent_b.artifact["factor_spec"]),
            )
        )
        formula = response["factor_spec"]["formula"]
        formula["args"][1] = formula["args"][0]
        validated = FormulaNode.model_validate(formula)
        response["factor_spec"]["required_fields"] = sorted(
            field.value for field in validated.referenced_fields()
        )
        response["factor_spec"]["lookback_days"] = validated.required_history()

        with self.assertRaisesRegex(ValueError, "parent B"):
            CrossoverAgent(StaticClient(response)).propose(task, parent_a, parent_b)


if __name__ == "__main__":
    unittest.main()
