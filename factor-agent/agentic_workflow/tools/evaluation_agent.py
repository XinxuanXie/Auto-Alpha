"""Deterministic acceptance rules with OpenAI-compatible evaluation diagnosis."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agentic_workflow.state import ManagerAction, WorkflowState
from agentic_workflow.tools.base import ToolResult


@dataclass(frozen=True)
class EvaluationThresholds:
    minimum_rank_ic: float = 0.02
    minimum_net_return: float = 0.0


class EvaluationDiagnosis(BaseModel):
    diagnosis: str = Field(min_length=1, max_length=1_500)
    evidence: list[str] = Field(min_length=1, max_length=6)
    recommended_action: ManagerAction
    revision_guidance: list[str] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


def build_evaluation_context(
    state: WorkflowState,
    rule_evaluation: dict[str, Any],
    allowed_recommendations: tuple[ManagerAction, ...],
) -> dict[str, Any]:
    return {
        "research_goal": state["research_goal"],
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "attempt": state["attempt"],
        "max_attempts_per_hypothesis": state[
            "max_attempts_per_hypothesis"
        ],
        "hypothesis": state["hypothesis"],
        "factor_spec": state["factor_spec"],
        "implementation": state["implementation"],
        "backtest_result": state["backtest_result"],
        "rule_evaluation": rule_evaluation,
        "allowed_recommendations": [
            action.value for action in allowed_recommendations
        ],
    }


def build_system_prompt() -> str:
    return (
        "你是自动化量化研究系统中的 Evaluation Agent。"
        "确定性规则已经完成验收判断，rule_evaluation.status 是不可更改的最终验收状态。"
        "你的职责是结合研究假设、因子定义和回测结果解释成功或失败原因，"
        "列出直接由输入指标支持的证据，并提出下一步研究建议。"
        "recommended_action 必须来自 allowed_recommendations。"
        "因子未通过硬性门槛时不能建议 accept_factor。"
        "重点分析预测能力、方向一致性、收益、风险、换手率、交易成本和稳健性。"
        "不要虚构输入中不存在的分组回测、样本外检验或统计显著性结果。"
        "仅返回符合 output_schema 的 JSON 对象。"
    )


class EvaluationAgentTool:
    name = "evaluation_agent"

    def __init__(
        self,
        model: str | None = None,
        thresholds: EvaluationThresholds | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.thresholds = thresholds or EvaluationThresholds()
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for the Evaluation Agent"
                )
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "OPENAI_BASE_URL",
                    "https://hkust.azure-api.net/hkust-genai/v1",
                ),
                default_headers={"api-key": api_key},
            )
        self.client = client

    def run(self, state: WorkflowState) -> ToolResult:
        backtest_result = state["backtest_result"]
        if not isinstance(backtest_result, dict):
            return ToolResult.fail(
                "A structured backtest result is required before evaluation"
            )

        try:
            rule_evaluation = self._evaluate_rules(backtest_result)
            allowed_recommendations = self._allowed_recommendations(
                rule_evaluation["status"],
                can_adjust=(
                    state["attempt"]
                    < state["max_attempts_per_hypothesis"]
                ),
            )
            request = {
                "task": "解释确定性验收结果并提出下一步研究建议。",
                "evaluation_context": build_evaluation_context(
                    state,
                    rule_evaluation,
                    allowed_recommendations,
                ),
                "output_schema": EvaluationDiagnosis.model_json_schema(),
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            request, ensure_ascii=False, default=str
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            if not response.choices:
                raise ValueError("Evaluation Agent returned no response choices")
            response_content = response.choices[0].message.content or ""
            if not response_content:
                raise ValueError("Evaluation Agent returned empty content")
            diagnosis = EvaluationDiagnosis.model_validate_json(response_content)
            if diagnosis.recommended_action not in allowed_recommendations:
                raise ValueError(
                    "Evaluation Agent selected a recommendation outside the "
                    f"allowed set: {diagnosis.recommended_action.value}"
                )
        except Exception as error:
            return ToolResult.fail(f"Evaluation Agent encountered an error: {error}")

        evaluation = {
            **rule_evaluation,
            **diagnosis.model_dump(mode="json"),
            "model": self.model,
        }
        return ToolResult.ok(
            {
                "evaluation": evaluation,
                "next_action": diagnosis.recommended_action,
            }
        )

    def _evaluate_rules(self, backtest_result: dict[str, Any]) -> dict[str, Any]:
        rank_ic = backtest_result.get(
            "rank_ic", backtest_result.get("mean_rank_ic")
        )
        net_return = backtest_result.get(
            "net_return", backtest_result.get("net_excess_return")
        )
        if not isinstance(rank_ic, (int, float)) or not isinstance(
            net_return, (int, float)
        ):
            raise ValueError(
                "Backtest result must contain numeric rank_ic and net_return"
            )

        rank_ic_passed = rank_ic > self.thresholds.minimum_rank_ic
        net_return_passed = net_return > self.thresholds.minimum_net_return
        passed = rank_ic_passed and net_return_passed
        reasons = []
        if rank_ic_passed:
            reasons.append("Rank IC exceeds the deterministic threshold")
        else:
            reasons.append("Rank IC does not exceed the deterministic threshold")
        if net_return_passed:
            reasons.append("Net return is positive after costs")
        else:
            reasons.append("Net return is not positive after costs")

        return {
            "status": "PASS" if passed else "FAIL_HYPOTHESIS",
            "reasons": reasons,
            "rule_checks": {
                "rank_ic": {
                    "observed": float(rank_ic),
                    "required": f"> {self.thresholds.minimum_rank_ic}",
                    "passed": rank_ic_passed,
                },
                "net_return": {
                    "observed": float(net_return),
                    "required": f"> {self.thresholds.minimum_net_return}",
                    "passed": net_return_passed,
                },
            },
        }

    @staticmethod
    def _allowed_recommendations(
        status: str,
        *,
        can_adjust: bool,
    ) -> tuple[ManagerAction, ...]:
        if status == "PASS":
            actions = (
                ManagerAction.ACCEPT_FACTOR,
                ManagerAction.ADJUST_FACTOR,
                ManagerAction.NEW_HYPOTHESIS,
                ManagerAction.STOP_RESEARCH,
            )
        else:
            actions = (
                ManagerAction.NEW_HYPOTHESIS,
                ManagerAction.REVISE_HYPOTHESIS,
                ManagerAction.ADJUST_FACTOR,
                ManagerAction.REJECT_FACTOR,
                ManagerAction.STOP_RESEARCH,
            )
        if can_adjust:
            return actions
        return tuple(
            action
            for action in actions
            if action != ManagerAction.ADJUST_FACTOR
        )
