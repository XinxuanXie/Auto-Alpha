"""Outer Manager Agent policies for research-loop decisions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

from agentic_workflow.state import ManagerAction, WorkflowState


class ManagerDecision(BaseModel): # 只保存manager的决策结果，不保存api和policy的运行信息
    action: ManagerAction
    rationale: str = Field(min_length=1, max_length=1_500)
    instructions: str = Field(min_length=1, max_length=1_500)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True) # 对象创建后无法更改
class ManagerPolicyResult: 
    decision: ManagerDecision
    policy: str
    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None

    def to_state_payload(self) -> dict[str, Any]: #把result对象转换为字典形式，方便存储到state中
        payload = self.decision.model_dump(mode="json")
        payload.update(
            {
                "policy": self.policy,
                "model": self.model,
                "response_id": self.response_id,
                "usage": self.usage,
            }
        )
        return payload


class ManagerPolicy(Protocol): # 规定了ManagerPolicy的接口，要求实现decide方法
    name: str

    def decide(
        self,
        state: WorkflowState,
        allowed_actions: Sequence[ManagerAction],
    ) -> ManagerPolicyResult:
        """Choose one allowed research action from the accumulated evidence."""


class RuleBasedManagerPolicy: # 用于测试工作流的policy 不调用llm
    name = "rule_based_manager"

    def decide(
        self,
        state: WorkflowState,
        allowed_actions: Sequence[ManagerAction],
    ) -> ManagerPolicyResult:
        recommended = state["next_action"]
        action = (
            recommended
            if recommended in allowed_actions
            else ManagerAction.STOP_RESEARCH
        )
        return ManagerPolicyResult(
            decision=ManagerDecision(
                action=action,
                rationale="采用当前 Evaluation Tool 的确定性建议。",
                instructions="按照既定工作流执行下一动作。",
                confidence=1.0,
            ),
            policy=self.name,
        )


class OpenAIManagerPolicy: #接入api的manager policy，调用llm来做决策
    name = "openai_manager"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for the OpenAI Manager Agent"
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

    def decide(
        self,
        state: WorkflowState,
        allowed_actions: Sequence[ManagerAction],
    ) -> ManagerPolicyResult:
        evidence = {
            "research_goal": state["research_goal"],
            "iteration": state["iteration"],
            "max_iterations": state["max_iterations"],
            "attempt": state["attempt"],
            "max_attempts_per_hypothesis": state[
                "max_attempts_per_hypothesis"
            ],
            "hypothesis": state["hypothesis"],
            "hypothesis_candidates": state["hypothesis_candidates"],
            "hypothesis_selection": state["hypothesis_selection"],
            "hypothesis_check": state["hypothesis_check"],
            "factor_spec": state["factor_spec"],
            "implementation": state["implementation"],
            "backtest_result": state["backtest_result"],
            "evaluation": state["evaluation"],
            "evaluation_recommendation": (
                state["next_action"].value
                if state["next_action"] is not None
                else None
            ),
            "allowed_actions": [action.value for action in allowed_actions],
        }
        request = {
            "evidence": evidence,
            "output_schema": ManagerDecision.model_json_schema(),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是量化因子研究系统的外层 Manager Agent。"
                        "你负责根据完整实验依据选择下一研究动作，不负责直接生成代码。"
                        "action 必须来自 allowed_actions。"
                        "优先考虑数据泄漏、样本外稳健性、交易成本、换手率、"
                        "预测能力和剩余迭代预算。"
                        "rationale 应说明证据，instructions 应明确指导下一轮内部模块。"
                        "仅返回符合 output_schema 的 JSON 对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, default=str),
                },
            ],
            response_format={"type": "json_object"},
        )
        if not response.choices:
            raise RuntimeError("OpenAI Manager returned no response choices")
        response_content = response.choices[0].message.content
        if not response_content:
            raise RuntimeError("OpenAI Manager returned no structured decision")
        decision = ManagerDecision.model_validate_json(response_content)
        if decision.action not in allowed_actions:
            raise RuntimeError(
                f"OpenAI Manager selected disallowed action: {decision.action.value}"
            )
        usage = None
        if getattr(response, "usage", None) is not None:
            usage_object = response.usage
            usage = (
                usage_object.model_dump(mode="json")
                if hasattr(usage_object, "model_dump")
                else asdict(usage_object)
            )
        return ManagerPolicyResult(
            decision=decision,
            policy=self.name,
            model=self.model,
            response_id=getattr(response, "id", None),
            usage=usage,
        )


def allowed_manager_actions(state: WorkflowState) -> tuple[ManagerAction, ...]:
    evaluation = state["evaluation"] or {}
    status = evaluation.get("status")
    if status == "REJECTED_HYPOTHESIS":
        actions = (
            ManagerAction.NEW_HYPOTHESIS,
            ManagerAction.STOP_RESEARCH,
        )
    elif status == "PASS":
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
    if state["attempt"] >= state["max_attempts_per_hypothesis"]:
        attempt_actions = {
            ManagerAction.RETRY_IMPLEMENTATION,
            ManagerAction.ADJUST_FACTOR,
        }
        actions = tuple(
            action for action in actions if action not in attempt_actions
        )
    if state["iteration"] < state["max_iterations"]:
        return actions
    iterative_actions = {
        ManagerAction.NEW_HYPOTHESIS,
        ManagerAction.REVISE_HYPOTHESIS,
        ManagerAction.RETRY_IMPLEMENTATION,
        ManagerAction.ADJUST_FACTOR,
    }
    return tuple(action for action in actions if action not in iterative_actions)
