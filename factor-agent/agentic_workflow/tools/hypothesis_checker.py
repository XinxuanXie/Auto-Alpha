"""Deterministic validation for research hypotheses."""

import sqlite3
from dataclasses import dataclass

from agentic_workflow.hypothesis import hypothesis_signature
from agentic_workflow.state import WorkflowState
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.base import ToolResult


@dataclass(frozen=True)
class HypothesisRules:
    prediction_horizon_days: int = 5
    supported_concepts: frozenset[str] = frozenset(
        {
            "intraday_range",
            "liquidity",
            "market_cap",
            "price_momentum",
            "price_reversal",
            "price_volume_interaction",
            "recent_return",
            "volatility",
            "volume_abnormality",
            "volume_trend",
        }
    )
    # 禁止使用含未来信息的概念
    leakage_concepts: frozenset[str] = frozenset(
        {
            "forward_return",
            "future_price",
            "future_return",
            "future_volume",
            "label",
            "next_day_price",
            "target",
        }
    )


class RuleBasedHypothesisCheckerTool:
    name = "rule_based_hypothesis_checker"

    def __init__(
        self,
        repository: SQLiteResearchRepository,
        rules: HypothesisRules | None = None,
    ) -> None:
        self.repository = repository
        self.rules = rules or HypothesisRules()

    def run(self, state: WorkflowState) -> ToolResult:
        hypothesis = state["hypothesis"]
        if not isinstance(hypothesis, dict):
            return ToolResult.fail("A structured hypothesis is required before checking")

        issues: list[dict[str, str]] = []
        required_text_fields = (
            "hypothesis_id",
            "title",
            "claim",
            "economic_rationale",
            "prediction_direction",
        )
        for field in required_text_fields:
            value = hypothesis.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(
                    {
                        "code": "missing_or_invalid_field",
                        "message": f"{field} must be a non-empty string",
                    }
                )

        direction = hypothesis.get("prediction_direction")
        if direction not in {"positive", "negative"}:
            issues.append(
                {
                    "code": "invalid_prediction_direction",
                    "message": "prediction_direction must be positive or negative",
                }
            )

        horizon = hypothesis.get("prediction_horizon_days")
        if horizon != self.rules.prediction_horizon_days:
            issues.append(
                {
                    "code": "unsupported_prediction_horizon",
                    "message": (
                        "prediction_horizon_days must equal "
                        f"{self.rules.prediction_horizon_days}"
                    ),
                }
            )

        # 数据支持
        concepts = hypothesis.get("required_concepts")
        if not isinstance(concepts, list) or not concepts or not all(
            isinstance(concept, str) and concept.strip() for concept in concepts
        ):
            issues.append(
                {
                    "code": "invalid_required_concepts",
                    "message": "required_concepts must be a non-empty list of strings",
                }
            )
            normalized_concepts: set[str] = set()
        else:
            normalized_concepts = {
                "_".join(concept.casefold().split()) for concept in concepts
            }
        
        # 避免数据泄漏和不支持的概念
        leakage = sorted(normalized_concepts & self.rules.leakage_concepts)
        if leakage:
            issues.append(
                {
                    "code": "future_information_leakage",
                    "message": f"Leakage concepts are not allowed: {', '.join(leakage)}",
                }
            )

        unsupported = sorted(
            normalized_concepts
            - self.rules.supported_concepts
            - self.rules.leakage_concepts
        )
        if unsupported:
            issues.append(
                {
                    "code": "unsupported_data_concept",
                    "message": (
                        "Concepts are not available from OHLCV and market cap: "
                        f"{', '.join(unsupported)}"
                    ),
                }
            )

        # 检查是否存在完全重复的假设
        duplicate = False
        if not issues:
            try:
                duplicate = self.repository.contains_hypothesis(hypothesis)
            except sqlite3.Error as error:
                return ToolResult.fail(
                    f"Failed to query hypothesis memory: {error}"
                )
            if duplicate:
                issues.append(
                    {
                        "code": "exact_duplicate_hypothesis",
                        "message": "The same normalized hypothesis already exists",
                    }
                )

        hypothesis_check = {
            "accepted": not issues,
            "issues": issues,
            "checks": {
                "schema_valid": not any(
                    issue["code"]
                    in {
                        "missing_or_invalid_field",
                        "invalid_prediction_direction",
                        "unsupported_prediction_horizon",
                        "invalid_required_concepts",
                    }
                    for issue in issues
                ),
                "data_supported": not any(
                    issue["code"] == "unsupported_data_concept"
                    for issue in issues
                ),
                "leakage_free": not leakage,
                "exact_duplicate": duplicate,
            },
            "signature": hypothesis_signature(hypothesis),
        }
        return ToolResult.ok({"hypothesis_check": hypothesis_check})
