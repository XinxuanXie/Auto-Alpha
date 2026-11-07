import json
import unittest
from pathlib import Path

from evoalpha_workflow.parent_selection import ParentSelectionPlan
from evoalpha_workflow.refinement import (
    RefinementAgent,
    RuleBasedMockRefinementClient,
    _same_shape_differences,
)
from evoalpha_workflow.state import EvaluatedCandidate
from factor_pipeline.spec import FactorSpec


RESULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "evoalpha_seed_2001"


class StaticClient:
    def __init__(self, response):
        self.response = response

    def complete(self, system_prompt, user_prompt):
        del system_prompt, user_prompt
        return self.response


class RefinementAgentTests(unittest.TestCase):
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

    def test_mock_generates_all_twelve_valid_refinements(self):
        agent = RefinementAgent(RuleBasedMockRefinementClient())

        proposals = [
            agent.propose(task, self.population[task.parent_factor_id])
            for task in self.plan.refinement_tasks
        ]

        self.assertEqual(len(proposals), 12)
        self.assertEqual(len({proposal.child_factor_id for proposal in proposals}), 12)
        self.assertTrue(all(proposal.generation == 1 for proposal in proposals))
        candidate = proposals[0].to_candidate_proposal()
        self.assertEqual(candidate.parent_factor_ids, (proposals[0].parent_factor_id,))

    def test_prompt_contains_parent_spec_metrics_and_constraints(self):
        task = self.plan.refinement_tasks[0]
        parent = self.population[task.parent_factor_id]
        parent_spec = parent.artifact["factor_spec"]

        system_prompt, user_prompt = RefinementAgent.build_prompts(
            task,
            parent,
            FactorSpec.model_validate(parent_spec),
        )
        payload = json.loads(user_prompt)

        self.assertIn("exactly one primary change", system_prompt)
        self.assertEqual(payload["task"]["task_id"], task.task_id)
        self.assertEqual(
            payload["parent_factor_spec"]["factor_id"],
            parent.factor_id,
        )
        self.assertIn("parent_training_metrics", payload)
        self.assertTrue(payload["constraints"]["one_primary_change_only"])

    def test_canonicalizes_response_with_more_than_one_window_change(self):
        task = self.plan.refinement_tasks[0]
        parent = self.population[task.parent_factor_id]
        response = RuleBasedMockRefinementClient().complete(
            *RefinementAgent.build_prompts(
                task,
                parent,
                FactorSpec.model_validate(parent.artifact["factor_spec"]),
            )
        )
        formula = response["factor_spec"]["formula"]
        formula["args"][0]["args"][1]["window"] = 2
        response["factor_spec"]["lookback_days"] = 61

        proposal = RefinementAgent(StaticClient(response)).propose(task, parent)

        parent_spec = FactorSpec.model_validate(parent.artifact["factor_spec"])
        self.assertEqual(
            _same_shape_differences(parent_spec.formula, proposal.factor_spec.formula),
            ["window"],
        )

    def test_window_refinement_discards_unrelated_formula_changes(self):
        task = self.plan.refinement_tasks[0]
        parent = self.population[task.parent_factor_id]
        response = RuleBasedMockRefinementClient().complete(
            *RefinementAgent.build_prompts(
                task,
                parent,
                FactorSpec.model_validate(parent.artifact["factor_spec"]),
            )
        )
        formula = response["factor_spec"]["formula"]
        changed_node = formula
        while changed_node.get("window") is None:
            changed_node = changed_node["args"][0]
        changed_node["op"] = "mean" if changed_node["op"] != "mean" else "std"

        proposal = RefinementAgent(StaticClient(response)).propose(task, parent)

        parent_spec = FactorSpec.model_validate(parent.artifact["factor_spec"])
        self.assertEqual(
            _same_shape_differences(parent_spec.formula, proposal.factor_spec.formula),
            ["window"],
        )

    def test_derives_factor_metadata_from_generated_formula(self):
        task = self.plan.refinement_tasks[0]
        parent = self.population[task.parent_factor_id]
        response = RuleBasedMockRefinementClient().complete(
            *RefinementAgent.build_prompts(
                task,
                parent,
                FactorSpec.model_validate(parent.artifact["factor_spec"]),
            )
        )
        response["factor_spec"]["required_fields"] = ["open"]
        response["factor_spec"]["lookback_days"] = 504

        proposal = RefinementAgent(StaticClient(response)).propose(task, parent)

        self.assertEqual(
            set(proposal.factor_spec.required_fields),
            proposal.factor_spec.formula.referenced_fields(),
        )
        self.assertEqual(
            proposal.factor_spec.lookback_days,
            proposal.factor_spec.formula.required_history(),
        )


if __name__ == "__main__":
    unittest.main()
