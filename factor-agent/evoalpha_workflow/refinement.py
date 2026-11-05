"""Structured Refinement Agent contract with deterministic validation."""

from __future__ import annotations

import copy
import json
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from factor_pipeline.spec import FactorSpec, FormulaNode, FormulaOperator

from .parent_selection import RefinementTask
from .state import CandidateProposal, EvaluatedCandidate, ProposalType


class RefinementChangeType(StrEnum):
    WINDOW = "window"
    OPERATOR = "operator"
    NORMALIZATION = "normalization"
    CONFIRMATION = "confirmation"
    RISK_ADJUSTMENT = "risk_adjustment"


class RefinementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    generation: int = Field(ge=1)
    child_factor_id: str = Field(min_length=1, max_length=200)
    parent_factor_id: str = Field(min_length=1, max_length=200)
    diagnosis: str = Field(min_length=20, max_length=2_000)
    change_type: RefinementChangeType
    change_description: str = Field(min_length=10, max_length=1_000)
    preserved_core_logic: str = Field(min_length=10, max_length=1_000)
    expected_metric_effects: dict[str, str] = Field(min_length=1)
    factor_spec: FactorSpec

    def to_candidate_proposal(self) -> CandidateProposal:
        return CandidateProposal(
            factor_id=self.child_factor_id,
            generation=self.generation,
            proposal_type=ProposalType.REFINEMENT,
            parent_factor_ids=(self.parent_factor_id,),
            artifact={
                "factor_spec": self.factor_spec.model_dump(mode="json"),
                "refinement": self.model_dump(mode="json", exclude={"factor_spec"}),
            },
        )


class RefinementModelClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class RefinementAgent:
    def __init__(self, client: RefinementModelClient) -> None:
        self.client = client

    def propose(
        self,
        task: RefinementTask,
        parent: EvaluatedCandidate,
    ) -> RefinementProposal:
        parent_spec = _parent_spec(parent)
        if task.parent_factor_id != parent.factor_id:
            raise ValueError("refinement task parent does not match candidate")
        system_prompt, user_prompt = self.build_prompts(task, parent, parent_spec)
        proposal = RefinementProposal.model_validate(
            self.client.complete(system_prompt, user_prompt)
        )
        self._validate_identity(task, parent, proposal)
        FormulaRefinementChecker.validate(
            parent_spec,
            proposal.factor_spec,
            proposal.change_type,
        )
        return proposal

    @staticmethod
    def build_prompts(
        task: RefinementTask,
        parent: EvaluatedCandidate,
        parent_spec: FactorSpec,
    ) -> tuple[str, str]:
        operators = [operator.value for operator in FormulaOperator]
        system_prompt = (
            "You are a quantitative factor refinement specialist. Improve one existing "
            "single-factor formula through exactly one primary change. Preserve its core "
            "economic mechanism, use only the supplied OHLCV-compatible structured DSL, "
            "avoid future information, and return one JSON object matching the requested "
            "schema. Do not return Python code or markdown."
        )
        payload = {
            "task": task.model_dump(mode="json"),
            "parent_factor_spec": parent_spec.model_dump(mode="json"),
            "parent_training_metrics": parent.metrics,
            "allowed_change_types": [item.value for item in RefinementChangeType],
            "available_operators": operators,
            "constraints": {
                "one_primary_change_only": True,
                "maximum_formula_depth_increase": 2,
                "preserve_direction_flag": parent_spec.higher_is_better,
                "preserve_rebalance_and_portfolio_rules": True,
                "child_factor_id": f"{task.task_id}-CHILD",
            },
            "required_output_fields": [
                "task_id",
                "generation",
                "child_factor_id",
                "parent_factor_id",
                "diagnosis",
                "change_type",
                "change_description",
                "preserved_core_logic",
                "expected_metric_effects",
                "factor_spec",
            ],
        }
        return system_prompt, json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _validate_identity(
        task: RefinementTask,
        parent: EvaluatedCandidate,
        proposal: RefinementProposal,
    ) -> None:
        if proposal.task_id != task.task_id:
            raise ValueError("refinement response task_id mismatch")
        if proposal.generation != task.generation:
            raise ValueError("refinement response generation mismatch")
        if proposal.parent_factor_id != parent.factor_id:
            raise ValueError("refinement response parent_factor_id mismatch")
        expected_child_id = f"{task.task_id}-CHILD"
        if proposal.child_factor_id != expected_child_id:
            raise ValueError(f"child_factor_id must equal {expected_child_id}")
        if proposal.factor_spec.factor_id != proposal.child_factor_id:
            raise ValueError("FactorSpec factor_id must equal child_factor_id")


class FormulaRefinementChecker:
    @classmethod
    def validate(
        cls,
        parent: FactorSpec,
        child: FactorSpec,
        change_type: RefinementChangeType,
    ) -> None:
        if parent.formula == child.formula:
            raise ValueError("refinement must change the parent formula")
        if _formula_depth(child.formula) > _formula_depth(parent.formula) + 2:
            raise ValueError("refinement formula depth may increase by at most two")
        if child.higher_is_better != parent.higher_is_better:
            raise ValueError("refinement must preserve higher_is_better")
        if child.rebalance_frequency != parent.rebalance_frequency:
            raise ValueError("refinement must preserve rebalance frequency")
        if child.portfolio_rules != parent.portfolio_rules:
            raise ValueError("refinement must preserve portfolio rules")
        if child.universe_filters != parent.universe_filters:
            raise ValueError("refinement must preserve universe filters")
        required_history = max(
            child.formula.required_history(),
            child.universe_filters.required_history()
            if child.universe_filters is not None
            else 0,
        )
        if child.lookback_days != required_history:
            raise ValueError("child lookback_days must equal derived required history")

        if change_type == RefinementChangeType.WINDOW:
            differences = _same_shape_differences(parent.formula, child.formula)
            if differences != ["window"]:
                raise ValueError("window refinement must change exactly one window")
        elif change_type == RefinementChangeType.OPERATOR:
            differences = _same_shape_differences(parent.formula, child.formula)
            if differences != ["op"]:
                raise ValueError("operator refinement must change exactly one operator")
        elif change_type == RefinementChangeType.NORMALIZATION:
            if child.formula.op not in {FormulaOperator.RANK, FormulaOperator.ZSCORE}:
                raise ValueError("normalization refinement must add rank or zscore")
            if len(child.formula.args) != 1 or child.formula.args[0] != parent.formula:
                raise ValueError("normalization refinement must wrap the parent formula")
        elif change_type == RefinementChangeType.CONFIRMATION:
            if child.formula.op not in {FormulaOperator.MULTIPLY, FormulaOperator.WHERE}:
                raise ValueError("confirmation refinement must use multiply or where")
            if not _contains_subtree(child.formula, parent.formula):
                raise ValueError("confirmation refinement must retain the parent formula")
        elif change_type == RefinementChangeType.RISK_ADJUSTMENT:
            if child.formula.op != FormulaOperator.DIVIDE:
                raise ValueError("risk adjustment must divide the parent signal")
            if not _contains_subtree(child.formula, parent.formula):
                raise ValueError("risk adjustment must retain the parent formula")


class RuleBasedMockRefinementClient:
    """Deterministic stand-in used before connecting a real LLM API."""

    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del system_prompt
        request = json.loads(user_prompt)
        task = request["task"]
        parent_spec = request["parent_factor_spec"]
        child_id = request["constraints"]["child_factor_id"]
        child_spec = copy.deepcopy(parent_spec)
        child_spec["factor_id"] = child_id
        child_spec["name"] = f"Refined {parent_spec['name']}"
        child_spec["description"] = (
            f"Mock refinement of {parent_spec['factor_id']} for interface validation."
        )
        formula = child_spec["formula"]
        changed_window = _change_first_window(formula)
        if changed_window:
            change_type = RefinementChangeType.WINDOW
            change_description = "Change one rolling window while preserving the formula structure."
        else:
            child_spec["formula"] = {"op": "zscore", "args": [formula]}
            change_type = RefinementChangeType.NORMALIZATION
            change_description = "Add one cross-sectional z-score normalization layer."
        validated_formula = FormulaNode.model_validate(child_spec["formula"])
        child_spec["required_fields"] = sorted(
            field.value for field in validated_formula.referenced_fields()
        )
        child_spec["lookback_days"] = validated_formula.required_history()
        return {
            "task_id": task["task_id"],
            "generation": task["generation"],
            "child_factor_id": child_id,
            "parent_factor_id": task["parent_factor_id"],
            "diagnosis": (
                "The parent has a coherent signal but its fixed transformation may be "
                "sensitive to the selected horizon or cross-sectional scale."
            ),
            "change_type": change_type.value,
            "change_description": change_description,
            "preserved_core_logic": (
                "The original fields, direction and primary economic signal remain unchanged."
            ),
            "expected_metric_effects": {
                "mean_rank_ic_20d": "preserve or improve predictive strength",
                "rank_icir_20d": "improve stability across training dates",
            },
            "factor_spec": child_spec,
        }


def _parent_spec(parent: EvaluatedCandidate) -> FactorSpec:
    try:
        payload = parent.artifact["factor_spec"]
    except KeyError as error:
        raise ValueError("parent candidate is missing factor_spec artifact") from error
    return FactorSpec.model_validate(payload)


def _formula_depth(node: FormulaNode) -> int:
    return 1 + max((_formula_depth(argument) for argument in node.args), default=0)


def _contains_subtree(node: FormulaNode, target: FormulaNode) -> bool:
    return node == target or any(_contains_subtree(argument, target) for argument in node.args)


def _same_shape_differences(parent: FormulaNode, child: FormulaNode) -> list[str]:
    differences = []

    def compare(left: FormulaNode, right: FormulaNode) -> None:
        for field_name in (
            "op",
            "field",
            "value",
            "window",
            "min_periods",
            "quantile",
            "ddof",
        ):
            if getattr(left, field_name) != getattr(right, field_name):
                differences.append(field_name)
        if len(left.args) != len(right.args):
            differences.append("args")
            return
        for left_argument, right_argument in zip(left.args, right.args, strict=True):
            compare(left_argument, right_argument)

    compare(parent, child)
    return differences


def _change_first_window(formula: dict[str, Any]) -> bool:
    if formula.get("window") is not None:
        window = formula["window"]
        formula["window"] = {5: 10, 10: 20, 20: 30, 30: 60, 60: 30}.get(
            window,
            max(1, window + 5),
        )
        return True
    for argument in formula.get("args", []):
        if _change_first_window(argument):
            return True
    return False
