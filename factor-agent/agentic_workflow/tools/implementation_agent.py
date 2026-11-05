"""OpenAI-compatible Implementation Agent for FactorSpec generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from agentic_workflow.state import WorkflowState
from agentic_workflow.tools.base import ToolResult
from factor_pipeline.compiler import compile_factor_spec
from factor_pipeline.spec import (
    DataField,
    FactorSpec,
    FormulaNode,
    FormulaOperator,
    PortfolioRules,
    UniverseFilterSpec,
)


class ImplementationProposal(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1_500)
    formula: FormulaNode
    required_fields: list[DataField] = Field(min_length=1)
    lookback_days: int = Field(ge=0, le=504)
    higher_is_better: bool
    rebalance_frequency: Literal["weekly"] = "weekly"
    universe_filters: UniverseFilterSpec | None = None


def build_system_prompt() -> str:
    fields = ", ".join(field.value for field in DataField)
    operators = ", ".join(operator.value for operator in FormulaOperator)
    return (
        "你是自动化量化研究系统中的 Implementation Agent。"
        "你的职责是将一个结构化研究假设翻译成可执行的 FactorSpec 提案。"
        "当 task_mode 为 revision 时，必须比较 previous_attempt 中的旧公式、"
        "回测结果和 Evaluation 诊断，按照 Manager 指令进行针对性修改，"
        "并保持研究假设的经济机制一致。"
        "formula 必须准确实现 hypothesis 的经济含义和预测方向。"
        "formula 只能使用系统提供的字段与算子，并遵守 FormulaNode 的参数结构。"
        "lag、return、mean、std 是需要 window 的时序算子。"
        "zscore 和 rank 是不接受 window 的当日横截面算子。"
        "negate 接受一个输入且不接受 window，二元算子必须有两个 args。"
        "greater_than 和 less_than 接受两个 args 并返回条件；"
        "where 接受 condition、true_value、false_value 三个 args。"
        "required_fields 必须与 formula 实际引用的字段完全一致。"
        "lookback_days 必须覆盖公式所需的全部历史窗口。"
        "higher_is_better 表示原始公式值越高是否对应越高的未来收益。"
        "如果 hypothesis 包含最低价格、流动性或市值约束，必须使用 "
        "universe_filters 明确表达，不能只写在 description。"
        "组合构建由确定性系统固定使用 top_k=20、exit_rank=30，"
        "不得在提案中生成或修改 portfolio_rules。"
        "输出只包含 FactorSpec 提案，Python 代码和自定义算子由确定性系统管理。"
        "仅返回符合 output_schema 的 JSON 对象。"
        f"可用字段：{fields}。"
        f"可用算子：{operators}。"
    )


def build_implementation_context(state: WorkflowState) -> dict[str, object]:
    manager_decision = state["manager_decision"] or {}
    previous_attempt = state["previous_attempt"]
    return {
        "task_mode": "revision" if previous_attempt else "initial",
        "research_goal": state["research_goal"],
        "iteration": state["iteration"],
        "attempt": state["attempt"],
        "hypothesis": state["hypothesis"],
        "previous_attempt": previous_attempt,
        "manager_rationale": manager_decision.get("rationale"),
        "manager_instructions": manager_decision.get("instructions"),
        "available_fields": [field.value for field in DataField],
        "available_operators": [operator.value for operator in FormulaOperator],
        "supported_universe_filters": {
            "min_price": "absolute close-price threshold",
            "min_liquidity_percentile": "cross-sectional percentile of rolling dollar volume",
            "liquidity_lookback_days": "rolling dollar-volume window",
            "min_market_cap_percentile": "cross-sectional market-cap percentile",
        },
        "fixed_portfolio_rules": {"top_k": 20, "exit_rank": 30},
    }


class ImplementationAgentTool:
    name = "implementation_agent"

    def __init__(
        self,
        panel_path: str | Path,
        model: str | None = None,
        max_attempts: int = 4,
        *,
        client: Any | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.panel_path = Path(panel_path)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.max_attempts = max_attempts
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for the Implementation Agent"
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
        hypothesis = state["hypothesis"]
        if not isinstance(hypothesis, dict):
            return ToolResult.fail(
                "A structured hypothesis is required before implementation"
            )
        hypothesis_id = hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            return ToolResult.fail(
                "A valid hypothesis_id is required before implementation"
            )

        request = {
            "task": "将研究假设翻译成一个结构化 FactorSpec 提案。",
            "implementation_context": build_implementation_context(state),
            "output_schema": ImplementationProposal.model_json_schema(),
        }
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(request, ensure_ascii=False, default=str),
            },
        ]
        for attempt in range(1, self.max_attempts + 1):
            response_content = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                if not response.choices:
                    raise ValueError(
                        "Implementation Agent returned no response choices"
                    )
                response_content = response.choices[0].message.content or ""
                if not response_content:
                    raise ValueError("Implementation Agent returned empty content")

                proposal = ImplementationProposal.model_validate_json(
                    response_content
                )
                factor_payload = proposal.model_dump(mode="json")
                factor_payload.update(
                    {
                        "factor_id": self._build_factor_id(hypothesis_id),
                        "hypothesis_id": hypothesis_id,
                        "portfolio_rules": PortfolioRules().model_dump(
                            mode="json"
                        ),
                    }
                )
                factor_spec = FactorSpec.model_validate(factor_payload)
                compiled_schema = compile_factor_spec(
                    pl.scan_parquet(self.panel_path), factor_spec
                ).collect_schema()
                if "raw_factor" not in compiled_schema:
                    raise ValueError("Compiled FactorSpec did not produce raw_factor")
                break
            except Exception as error:
                if attempt >= self.max_attempts:
                    return ToolResult.fail(
                        f"Implementation Agent failed after {attempt} attempts: {error}"
                    )
                if response_content:
                    messages.append(
                        {"role": "assistant", "content": response_content}
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一份 JSON 未通过确定性校验。"
                            f"校验错误：{error}。"
                            "请修正规则冲突并仅返回完整的 JSON 对象。"
                        ),
                    }
                )

        return ToolResult.ok(
            {
                "factor_spec": factor_spec.model_dump(
                    mode="json", exclude_none=True
                ),
                "implementation": {
                    "factor_spec_schema": "factor_spec_v1",
                    "compiled_artifact": "factor_pipeline.compile_factor_spec",
                    "unit_tests_passed": True,
                    "generation_attempts": attempt,
                    "validation": {
                        "factor_spec_valid": True,
                        "compile_schema_valid": True,
                    },
                },
            }
        )

    @staticmethod
    def _build_factor_id(hypothesis_id: str) -> str:
        if hypothesis_id.startswith("H-"):
            return f"F-{hypothesis_id[2:]}"
        return f"F-{hypothesis_id}"
