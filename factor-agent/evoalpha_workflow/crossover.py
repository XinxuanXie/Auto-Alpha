"""Structured Crossover and Mutation agent contracts with deterministic checks."""

from __future__ import annotations

import copy
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from factor_pipeline.spec import FactorSpec, FormulaNode, FormulaOperator

from .parent_selection import CrossoverTask
from .refinement import (
    FormulaRefinementChecker,
    RefinementChangeType,
    _change_first_window,
    _formula_depth,
)
from .state import CandidateProposal, EvaluatedCandidate, ProposalType


class CrossoverProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    generation: int = Field(ge=1)
    intermediate_factor_id: str = Field(min_length=1, max_length=200)
    parent_factor_ids: tuple[str, str]
    complementarity_analysis: str = Field(min_length=30, max_length=2_000)
    integration_logic: str = Field(min_length=20, max_length=1_500)
    expected_metric_effects: dict[str, str] = Field(min_length=1)
    factor_spec: FactorSpec

    def to_candidate_proposal(self) -> CandidateProposal:
        return CandidateProposal(
            factor_id=self.intermediate_factor_id,
            generation=self.generation,
            proposal_type=ProposalType.CROSSOVER,
            parent_factor_ids=self.parent_factor_ids,
            artifact={
                "factor_spec": self.factor_spec.model_dump(mode="json"),
                "crossover": self.model_dump(mode="json", exclude={"factor_spec"}),
            },
        )


class MutationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    generation: int = Field(ge=1)
    child_factor_id: str = Field(min_length=1, max_length=200)
    intermediate_factor_id: str = Field(min_length=1, max_length=200)
    diagnosis: str = Field(min_length=20, max_length=2_000)
    mutation_type: RefinementChangeType
    mutation_description: str = Field(min_length=10, max_length=1_000)
    expected_metric_effects: dict[str, str] = Field(min_length=1)
    factor_spec: FactorSpec

    def to_candidate_proposal(self) -> CandidateProposal:
        return CandidateProposal(
            factor_id=self.child_factor_id,
            generation=self.generation,
            proposal_type=ProposalType.MUTATION,
            parent_factor_ids=(self.intermediate_factor_id,),
            artifact={
                "factor_spec": self.factor_spec.model_dump(mode="json"),
                "mutation": self.model_dump(mode="json", exclude={"factor_spec"}),
            },
        )


class CrossoverModelClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class MutationModelClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class CrossoverAgent:
    def __init__(self, client: CrossoverModelClient) -> None:
        self.client = client

    def propose(
        self,
        task: CrossoverTask,
        parent_a: EvaluatedCandidate,
        parent_b: EvaluatedCandidate,
    ) -> CrossoverProposal:
        self._validate_task_parents(task, parent_a, parent_b)
        spec_a = _parent_spec(parent_a)
        spec_b = _parent_spec(parent_b)
        system_prompt, user_prompt = self.build_prompts(
            task,
            parent_a,
            parent_b,
            spec_a,
            spec_b,
        )
        proposal = CrossoverProposal.model_validate(
            self.client.complete(system_prompt, user_prompt)
        )
        self._validate_identity(task, proposal)
        CrossoverFormulaChecker.validate(
            parent_a,
            parent_b,
            spec_a,
            spec_b,
            proposal.factor_spec,
        )
        return proposal

    @staticmethod
    def build_prompts(
        task: CrossoverTask,
        parent_a: EvaluatedCandidate,
        parent_b: EvaluatedCandidate,
        spec_a: FactorSpec,
        spec_b: FactorSpec,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a quantitative factor crossover specialist. Explain why two selected "
            "parents contain complementary information, then create one intermediate formula "
            "that materially uses both oriented parent signals. Preserve the supplied parent "
            "directions, use only the structured DSL, avoid future information, and return "
            "one JSON object without markdown or Python code."
        )
        payload = {
            "task": task.model_dump(mode="json"),
            "parent_a": {
                "factor_spec": spec_a.model_dump(mode="json"),
                "training_metrics": parent_a.metrics,
                "training_direction": parent_a.direction,
            },
            "parent_b": {
                "factor_spec": spec_b.model_dump(mode="json"),
                "training_metrics": parent_b.metrics,
                "training_direction": parent_b.direction,
            },
            "constraints": {
                "must_use_both_oriented_parent_formulas": True,
                "maximum_depth_increase_over_deeper_parent": 3,
                "intermediate_factor_id": f"{task.task_id}-INTERMEDIATE",
                "allowed_root_operators": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "where",
                ],
            },
            "required_output_fields": [
                "task_id",
                "generation",
                "intermediate_factor_id",
                "parent_factor_ids",
                "complementarity_analysis",
                "integration_logic",
                "expected_metric_effects",
                "factor_spec",
            ],
        }
        return system_prompt, json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _validate_task_parents(
        task: CrossoverTask,
        parent_a: EvaluatedCandidate,
        parent_b: EvaluatedCandidate,
    ) -> None:
        if task.parent_factor_ids != (parent_a.factor_id, parent_b.factor_id):
            raise ValueError("crossover task parents do not match supplied candidates")

    @staticmethod
    def _validate_identity(task: CrossoverTask, proposal: CrossoverProposal) -> None:
        if proposal.task_id != task.task_id:
            raise ValueError("crossover response task_id mismatch")
        if proposal.generation != task.generation:
            raise ValueError("crossover response generation mismatch")
        if proposal.parent_factor_ids != task.parent_factor_ids:
            raise ValueError("crossover response parent ids mismatch")
        expected_id = f"{task.task_id}-INTERMEDIATE"
        if proposal.intermediate_factor_id != expected_id:
            raise ValueError(f"intermediate_factor_id must equal {expected_id}")
        if proposal.factor_spec.factor_id != expected_id:
            raise ValueError("FactorSpec factor_id must equal intermediate_factor_id")


class MutationAgent:
    def __init__(self, client: MutationModelClient) -> None:
        self.client = client

    def propose(self, crossover: CrossoverProposal) -> MutationProposal:
        system_prompt, user_prompt = self.build_prompts(crossover)
        proposal = MutationProposal.model_validate(
            self.client.complete(system_prompt, user_prompt)
        )
        expected_id = f"{crossover.task_id}-MUTATION"
        if proposal.task_id != crossover.task_id:
            raise ValueError("mutation response task_id mismatch")
        if proposal.generation != crossover.generation:
            raise ValueError("mutation response generation mismatch")
        if proposal.intermediate_factor_id != crossover.intermediate_factor_id:
            raise ValueError("mutation response intermediate factor mismatch")
        if proposal.child_factor_id != expected_id:
            raise ValueError(f"mutation child_factor_id must equal {expected_id}")
        if proposal.factor_spec.factor_id != expected_id:
            raise ValueError("mutation FactorSpec id must equal child_factor_id")
        FormulaRefinementChecker.validate(
            crossover.factor_spec,
            proposal.factor_spec,
            proposal.mutation_type,
        )
        return proposal

    @staticmethod
    def build_prompts(crossover: CrossoverProposal) -> tuple[str, str]:
        system_prompt = (
            "You are a quantitative factor mutation specialist. Apply exactly one small, "
            "auditable mutation to the supplied crossover intermediate factor. Preserve its "
            "combined parent mechanism, use only the structured DSL, and return one JSON "
            "object without markdown or Python code."
        )
        payload = {
            "crossover_intermediate": crossover.model_dump(mode="json"),
            "constraints": {
                "one_primary_change_only": True,
                "maximum_formula_depth_increase": 2,
                "child_factor_id": f"{crossover.task_id}-MUTATION",
                "allowed_mutation_types": [
                    item.value for item in RefinementChangeType
                ],
            },
            "required_output_fields": [
                "task_id",
                "generation",
                "child_factor_id",
                "intermediate_factor_id",
                "diagnosis",
                "mutation_type",
                "mutation_description",
                "expected_metric_effects",
                "factor_spec",
            ],
        }
        return system_prompt, json.dumps(payload, ensure_ascii=False, sort_keys=True)


class CrossoverFormulaChecker:
    @classmethod
    def validate(
        cls,
        parent_a: EvaluatedCandidate,
        parent_b: EvaluatedCandidate,
        spec_a: FactorSpec,
        spec_b: FactorSpec,
        child: FactorSpec,
    ) -> None:
        if spec_a.portfolio_rules != spec_b.portfolio_rules:
            raise ValueError("crossover parents must share portfolio rules")
        if spec_a.universe_filters != spec_b.universe_filters:
            raise ValueError("crossover parents must share universe filters")
        if child.portfolio_rules != spec_a.portfolio_rules:
            raise ValueError("crossover must preserve portfolio rules")
        if child.universe_filters != spec_a.universe_filters:
            raise ValueError("crossover must preserve universe filters")
        if child.rebalance_frequency != spec_a.rebalance_frequency:
            raise ValueError("crossover must preserve rebalance frequency")
        if not child.higher_is_better:
            raise ValueError("oriented crossover output must use higher_is_better=true")
        if child.formula.op not in {
            FormulaOperator.ADD,
            FormulaOperator.SUBTRACT,
            FormulaOperator.MULTIPLY,
            FormulaOperator.DIVIDE,
            FormulaOperator.WHERE,
        }:
            raise ValueError("crossover formula requires an explicit integration root")
        oriented_a = _oriented_formula(spec_a.formula, parent_a.direction)
        oriented_b = _oriented_formula(spec_b.formula, parent_b.direction)
        if not _contains_subtree(child.formula, oriented_a):
            raise ValueError("crossover formula does not retain oriented parent A")
        if not _contains_subtree(child.formula, oriented_b):
            raise ValueError("crossover formula does not retain oriented parent B")
        maximum_depth = max(_formula_depth(spec_a.formula), _formula_depth(spec_b.formula)) + 3
        if _formula_depth(child.formula) > maximum_depth:
            raise ValueError("crossover formula exceeds maximum depth increase")
        required_history = max(
            child.formula.required_history(),
            child.universe_filters.required_history()
            if child.universe_filters is not None
            else 0,
        )
        if child.lookback_days != required_history:
            raise ValueError("crossover lookback_days must equal derived required history")


class RuleBasedMockCrossoverClient:
    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del system_prompt
        request = json.loads(user_prompt)
        task = request["task"]
        parent_a = request["parent_a"]
        parent_b = request["parent_b"]
        formula_a = _oriented_formula_dict(
            parent_a["factor_spec"]["formula"],
            parent_a["training_direction"],
        )
        formula_b = _oriented_formula_dict(
            parent_b["factor_spec"]["formula"],
            parent_b["training_direction"],
        )
        combined = {
            "op": "add",
            "args": [
                {"op": "zscore", "args": [formula_a]},
                {"op": "zscore", "args": [formula_b]},
            ],
        }
        formula = FormulaNode.model_validate(combined)
        intermediate_id = request["constraints"]["intermediate_factor_id"]
        spec_a = parent_a["factor_spec"]
        factor_spec = {
            **copy.deepcopy(spec_a),
            "factor_id": intermediate_id,
            "name": f"Crossover {task['parent_factor_ids'][0]} and {task['parent_factor_ids'][1]}",
            "description": "Mock normalized sum of two oriented parent signals.",
            "formula": formula.model_dump(mode="json"),
            "required_fields": sorted(
                field.value for field in formula.referenced_fields()
            ),
            "lookback_days": formula.required_history(),
            "higher_is_better": True,
        }
        return {
            "task_id": task["task_id"],
            "generation": task["generation"],
            "intermediate_factor_id": intermediate_id,
            "parent_factor_ids": task["parent_factor_ids"],
            "complementarity_analysis": (
                "The selected parents have strong individual training scores and low absolute "
                "cross-sectional similarity, so they may contribute distinct predictive signals."
            ),
            "integration_logic": (
                "Orient each parent by its training direction, normalize both cross-sectionally, "
                "and add them with equal weight."
            ),
            "expected_metric_effects": {
                "rank_icir_20d": "improve stability through complementary information",
                "monthly_stability": "reduce dependence on one parent mechanism",
            },
            "factor_spec": factor_spec,
        }


class RuleBasedMockMutationClient:
    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del system_prompt
        request = json.loads(user_prompt)
        crossover = request["crossover_intermediate"]
        child_id = request["constraints"]["child_factor_id"]
        child_spec = copy.deepcopy(crossover["factor_spec"])
        child_spec["factor_id"] = child_id
        child_spec["name"] = f"Mutated {crossover['intermediate_factor_id']}"
        child_spec["description"] = "Mock single-step mutation of crossover intermediate."
        formula = child_spec["formula"]
        if _change_first_window(formula):
            mutation_type = RefinementChangeType.WINDOW
            description = "Change one rolling window in the crossover intermediate."
        else:
            child_spec["formula"] = {"op": "zscore", "args": [formula]}
            mutation_type = RefinementChangeType.NORMALIZATION
            description = "Add one cross-sectional normalization layer."
        validated_formula = FormulaNode.model_validate(child_spec["formula"])
        child_spec["required_fields"] = sorted(
            field.value for field in validated_formula.referenced_fields()
        )
        child_spec["lookback_days"] = validated_formula.required_history()
        return {
            "task_id": crossover["task_id"],
            "generation": crossover["generation"],
            "child_factor_id": child_id,
            "intermediate_factor_id": crossover["intermediate_factor_id"],
            "diagnosis": (
                "The crossover intermediate combines both parents, while one local parameter "
                "can be varied to test whether its horizon is overly rigid."
            ),
            "mutation_type": mutation_type.value,
            "mutation_description": description,
            "expected_metric_effects": {
                "mean_rank_ic_20d": "retain the combined signal while testing a nearby variant",
                "rank_icir_20d": "preserve or improve temporal stability",
            },
            "factor_spec": child_spec,
        }


def _parent_spec(parent: EvaluatedCandidate) -> FactorSpec:
    try:
        return FactorSpec.model_validate(parent.artifact["factor_spec"])
    except KeyError as error:
        raise ValueError("crossover parent is missing factor_spec") from error


def _oriented_formula(formula: FormulaNode, direction: int) -> FormulaNode:
    if direction == 1:
        return formula
    return FormulaNode(op=FormulaOperator.NEGATE, args=[formula])


def _oriented_formula_dict(formula: dict[str, Any], direction: int) -> dict[str, Any]:
    return copy.deepcopy(formula) if direction == 1 else {
        "op": "negate",
        "args": [copy.deepcopy(formula)],
    }


def _contains_subtree(node: FormulaNode, target: FormulaNode) -> bool:
    return node == target or any(_contains_subtree(argument, target) for argument in node.args)
