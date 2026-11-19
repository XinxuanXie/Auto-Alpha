"""Tests for the V2.1 repair-path fixes.

Covers the Smoke-run failure in repair_experiments:
DeepSeek returned {"op": "add", "field": "close", "args": []},
which is valid JSON but fails FormulaNode validation
("add requires exactly two args"). The old retry loop fed the
LLM only loc + message, without the offending node, and one
unrepairable experiment aborted the whole run.
"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np
import polars as pl
from openai import BadRequestError
from pydantic import ValidationError

from factor_pipeline.spec import FormulaNode
from hypothesis_workflow_v2.contracts import (
    CheckIssue,
    DevelopedHypothesis,
    ExperimentVariantType,
    FactorExperimentSpec,
    FactorFormulaSpecV2,
    MeasurementPlan,
    MeasurementProxy,
    ResearchDirection,
    SplitName,
)
from hypothesis_workflow_v2.real_backtest import (
    RealBacktestConfig,
    RealV2BacktestEngine,
    _SplitData,
)
from hypothesis_workflow_v2.state import V2Stage
from hypothesis_workflow_v2.__main__ import _checkpoint_wal
from hypothesis_workflow_v2.real_tools import (
    DEGENERATE_SIGNAL_MIN_VARIED_DATES,
    DUPLICATE_REPAIR_RULES,
    DUPLICATE_SIGNAL_MIN_CORRELATION,
    EXPERIMENT_DIVERSITY_RULES,
    FORMULA_NODE_SHAPE_RULES,
    MAX_MEASUREMENT_PROXIES,
    MEASUREMENT_PROXY_RULES,
    MECHANISM_FIRST_LITERATURE_RULES,
    MIN_PREDICTION_HORIZON_DAYS,
    PANEL_ZERO_VOLUME_NOTE,
    RealV2Tools,
    TURNOVER_RULES,
    _ExperimentBatch,
    _ExperimentDraft,
    _ExperimentRepairBatch,
    _MeasurementPlanDraft,
    _count_varied_cross_section_dates,
    _derived_proxy_ids,
    _direction_focus,
    _direction_literature,
    _normalize_experiment_batch,
    _normalize_formula_node,
    _normalize_represented_proxies,
    _rank_correlation,
    _salvage_json,
    _truncated_error_input,
    _validate_measurement_plan,
)


def _make_experiment(experiment_id: str) -> FactorExperimentSpec:
    factor = FactorFormulaSpecV2(
        factor_id=f"F-{experiment_id}",
        hypothesis_id="H-TEST-01",
        name="test factor",
        description="test factor for repair isolation",
        formula=FormulaNode.model_validate(
            {"op": "mean", "field": "close", "window": 20}
        ),
        required_fields=["close"],
        lookback_days=19,
    )
    return FactorExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="H-TEST-01",
        variation_reason="test variation reason",
        measurement_plan_id="MP-TEST-01",
        variant_type=ExperimentVariantType.BASELINE,
        represented_proxy_ids=["P1"],
        factor=factor,
        prediction_horizon_days=5,
    )


def _make_plan() -> MeasurementPlan:
    return MeasurementPlan(
        plan_id="MP-TEST-01",
        hypothesis_id="H-TEST-01",
        mechanism_summary="test mechanism summary",
        proxies=[
            MeasurementProxy(
                proxy_id="P1",
                mechanism_component="test mechanism component",
                observable_definition="test observable definition",
                required_fields=["close"],
                limitations=["test limitation"],
            )
        ],
        baseline_logic="test baseline logic",
        full_mechanism_logic="test full mechanism logic",
    )


def _repair_batch_json(formula: dict) -> str:
    return json.dumps(
        {
            "experiments": [
                {
                    "name": "repaired factor",
                    "description": "repaired factor description",
                    "variation_reason": "repaired variation reason",
                    "formula": formula,
                }
            ]
        }
    )


class NormalizeFormulaNodeTest(unittest.TestCase):
    def test_binary_add_field_with_one_arg_is_restored(self) -> None:
        normalized = _normalize_formula_node(
            {
                "op": "add",
                "field": "close",
                "args": [{"op": "field", "field": "volume"}],
            }
        )
        node = FormulaNode.model_validate(normalized)
        self.assertEqual(node.op.value, "add")
        self.assertIsNone(node.field)
        self.assertEqual(len(node.args), 2)
        self.assertEqual(node.args[0].field.value, "close")
        self.assertEqual(node.args[1].field.value, "volume")

    def test_binary_add_field_with_no_args_stays_invalid(self) -> None:
        # Cannot invent a second operand: this must still fail validation
        # and go through the retry feedback path, not be silently fixed.
        normalized = _normalize_formula_node(
            {"op": "add", "field": "close", "args": []}
        )
        with self.assertRaises(ValidationError):
            FormulaNode.model_validate(normalized)

    def test_binary_add_field_with_two_args_drops_field(self) -> None:
        normalized = _normalize_formula_node(
            {
                "op": "add",
                "field": "close",
                "args": [
                    {"op": "field", "field": "close"},
                    {"op": "constant", "value": 1.0},
                ],
            }
        )
        node = FormulaNode.model_validate(normalized)
        self.assertIsNone(node.field)
        self.assertEqual(len(node.args), 2)

    def test_ternary_where_field_with_two_args_is_dropped(self) -> None:
        # where 的第一个参数必须是布尔条件，一个 bare field 永远当不了
        # 条件，所以 normalize 不把它还原成第一个 args，而是直接去掉，
        # 让校验报“需要 3 个 args”，由重试反馈处理。
        normalized = _normalize_formula_node(
            {
                "op": "where",
                "field": "close",
                "args": [
                    {"op": "field", "field": "volume"},
                    {"op": "constant", "value": 1.0},
                ],
            }
        )
        self.assertIsNone(normalized.get("field"))
        self.assertEqual(len(normalized["args"]), 2)
        with self.assertRaises(ValidationError):
            FormulaNode.model_validate(normalized)

    def test_ternary_where_two_args_with_condition_is_completed(self) -> None:
        # Patch 14 (a5a57aea crash): DeepSeek wrote where(cond, X) with
        # only 2 args three times. That is the "filter" idiom — X where
        # cond holds, else 0 — so normalize completes it with 0.0.
        normalized = _normalize_formula_node(
            {
                "op": "where",
                "args": [
                    {
                        "op": "greater_than",
                        "args": [
                            {"op": "field", "field": "close"},
                            {"op": "constant", "value": 0.0},
                        ],
                    },
                    {"op": "field", "field": "volume"},
                ],
            }
        )
        self.assertEqual(len(normalized["args"]), 3)
        self.assertEqual(normalized["args"][2], {"op": "constant", "value": 0.0})
        FormulaNode.model_validate(normalized)

    def test_ternary_where_two_args_without_condition_stays_invalid(self) -> None:
        # No condition to complete from: a two-arg where whose first arg
        # is not a comparison cannot be recovered, it must still fail
        # validation and go through retry feedback.
        normalized = _normalize_formula_node(
            {
                "op": "where",
                "args": [
                    {"op": "field", "field": "close"},
                    {"op": "constant", "value": 1.0},
                ],
            }
        )
        self.assertEqual(len(normalized["args"]), 2)
        with self.assertRaises(ValidationError):
            FormulaNode.model_validate(normalized)

    def test_ternary_where_field_with_two_args_condition_is_completed(
        self,
    ) -> None:
        # Field dropped first (a bare field can never be the condition),
        # then the remaining [comparison, value] pair is completed.
        normalized = _normalize_formula_node(
            {
                "op": "where",
                "field": "close",
                "args": [
                    {
                        "op": "less_than",
                        "args": [
                            {"op": "field", "field": "volume"},
                            {"op": "constant", "value": 1.0},
                        ],
                    },
                    {"op": "field", "field": "close"},
                ],
            }
        )
        self.assertIsNone(normalized.get("field"))
        self.assertEqual(len(normalized["args"]), 3)
        FormulaNode.model_validate(normalized)

    def test_unary_mean_field_with_one_arg_still_drops_field(self) -> None:
        normalized = _normalize_formula_node(
            {
                "op": "mean",
                "field": "close",
                "window": 20,
                "args": [{"op": "field", "field": "close"}],
            }
        )
        node = FormulaNode.model_validate(normalized)
        self.assertIsNone(node.field)
        self.assertEqual(len(node.args), 1)

    def test_return_and_lag_drop_min_periods(self) -> None:
        # Patch 12 (ec610c9d crash): DeepSeek put min_periods on return
        # nodes three attempts in a row. min_periods on return/lag is
        # meaningless, so normalize drops it instead of failing the batch.
        for op in ("return", "lag"):
            normalized = _normalize_formula_node(
                {"op": op, "field": "close", "window": 20, "min_periods": 15}
            )
            node = FormulaNode.model_validate(normalized)
            self.assertIsNone(node.min_periods)
            self.assertEqual(node.window, 20)

    def test_field_and_constant_drop_rolling_parameters(self) -> None:
        normalized = _normalize_formula_node(
            {"op": "field", "field": "close", "window": 20, "min_periods": 10}
        )
        node = FormulaNode.model_validate(normalized)
        self.assertIsNone(node.window)
        self.assertIsNone(node.min_periods)
        normalized = _normalize_formula_node(
            {"op": "constant", "value": 1.0, "ddof": 1, "quantile": 0.5}
        )
        node = FormulaNode.model_validate(normalized)
        self.assertIsNone(node.ddof)
        self.assertIsNone(node.quantile)

    def test_rolling_operators_keep_min_periods_but_clamp_to_window(
        self,
    ) -> None:
        kept = _normalize_formula_node(
            {
                "op": "mean",
                "args": [{"op": "field", "field": "close"}],
                "window": 20,
                "min_periods": 15,
            }
        )
        node = FormulaNode.model_validate(kept)
        self.assertEqual(node.min_periods, 15)
        clamped = _normalize_formula_node(
            {
                "op": "mean",
                "args": [{"op": "field", "field": "close"}],
                "window": 20,
                "min_periods": 25,
            }
        )
        node = FormulaNode.model_validate(clamped)
        self.assertEqual(node.min_periods, 20)

    def test_non_std_drops_ddof_and_non_quantile_drops_quantile(self) -> None:
        node = FormulaNode.model_validate(
            _normalize_formula_node(
                {
                    "op": "correlation",
                    "window": 20,
                    "ddof": 1,
                    "args": [
                        {"op": "return", "field": "close", "window": 1},
                        {"op": "return", "field": "volume", "window": 1},
                    ],
                }
            )
        )
        self.assertIsNone(node.ddof)
        node = FormulaNode.model_validate(
            _normalize_formula_node(
                {
                    "op": "mean",
                    "args": [{"op": "field", "field": "close"}],
                    "window": 20,
                    "quantile": 0.5,
                }
            )
        )
        self.assertIsNone(node.quantile)


class SalvageJsonTest(unittest.TestCase):
    def test_missing_comma_between_members_is_inserted(self) -> None:
        salvaged = _salvage_json('{"a": 1 "b": 2}')
        self.assertIsNotNone(salvaged)
        self.assertEqual(json.loads(salvaged), {"a": 1, "b": 2})

    def test_missing_comma_between_array_elements_is_inserted(self) -> None:
        salvaged = _salvage_json('{"a": [1 2, 3]}')
        self.assertIsNotNone(salvaged)
        self.assertEqual(json.loads(salvaged), {"a": [1, 2, 3]})

    def test_trailing_comma_is_removed(self) -> None:
        salvaged = _salvage_json('{"a": 1,}')
        self.assertIsNotNone(salvaged)
        self.assertEqual(json.loads(salvaged), {"a": 1})

    def test_missing_colon_is_inserted(self) -> None:
        salvaged = _salvage_json('{"a" 1}')
        self.assertIsNotNone(salvaged)
        self.assertEqual(json.loads(salvaged), {"a": 1})

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(_salvage_json("not json at all {{{"))
        self.assertIsNone(_salvage_json('{"a": }'))

    def test_valid_json_passes_through(self) -> None:
        self.assertEqual(_salvage_json('{"a": 1}'), '{"a": 1}')

    def test_batch_with_missing_comma_validates_after_salvage(self) -> None:
        # Patch 15 (a5a57aea resume crash): a missing comma deep inside
        # a formula node killed the whole batch three times. Salvage
        # repairs it, and the salvaged batch still goes through schema
        # validation afterwards.
        batch = {
            "experiments": [
                {
                    "name": "t",
                    "description": "t",
                    "variation_reason": "crash replay reason",
                    "variant_type": "baseline",
                    "represented_proxy_ids": ["P1"],
                    "formula": {"op": "return", "field": "close", "window": 20},
                    "prediction_horizon_days": 20,
                },
                {
                    "name": "t",
                    "description": "t",
                    "variation_reason": "crash replay reason",
                    "variant_type": "full_mechanism",
                    "represented_proxy_ids": ["P1"],
                    "formula": {
                        "op": "mean",
                        "window": 20,
                        "args": [
                            {"op": "return", "field": "close", "window": 5}
                        ],
                    },
                    "prediction_horizon_days": 20,
                },
            ]
        }
        text = json.dumps(batch)
        broken = text.replace('"window": 20, "args"', '"window": 20 "args"', 1)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(broken)
        salvaged = _salvage_json(broken)
        self.assertIsNotNone(salvaged)
        validated = _ExperimentBatch.model_validate(
            _normalize_experiment_batch(json.loads(salvaged))
        )
        self.assertEqual(len(validated.experiments), 2)


class ShapeRulesTest(unittest.TestCase):
    def test_rules_state_binary_arity_and_wrong_example(self) -> None:
        self.assertIn("恰好有 2 个 args", FORMULA_NODE_SHAPE_RULES)
        self.assertIn("field 不是操作数", FORMULA_NODE_SHAPE_RULES)
        self.assertIn('"op": "add"', FORMULA_NODE_SHAPE_RULES)

    def test_truncated_error_input_is_bounded(self) -> None:
        preview = _truncated_error_input({"op": "add", "field": "close"}, limit=10)
        self.assertTrue(preview.endswith("…(truncated)"))
        self.assertLessEqual(len(preview), 10 + len("…(truncated)"))


class ValidatedJsonCompletionTest(unittest.TestCase):
    def _tools(self) -> RealV2Tools:
        return RealV2Tools.__new__(RealV2Tools)

    def test_retry_feedback_includes_offending_node_then_succeeds(self) -> None:
        tools = self._tools()
        invalid = _repair_batch_json({"op": "add", "field": "close", "args": []})
        valid = _repair_batch_json(
            {
                "op": "add",
                "args": [
                    {"op": "field", "field": "close"},
                    {"op": "constant", "value": 1.0},
                ],
            }
        )
        seen_payloads: list[dict] = []

        def fake_completion(
            client, model_name, system_prompt, payload, **kwargs
        ) -> str:
            seen_payloads.append(payload)
            return invalid if len(seen_payloads) == 1 else valid

        tools._completion_content = fake_completion  # type: ignore[method-assign]
        result = tools._validated_json_completion(
            None,  # type: ignore[arg-type]
            "test-model",
            "test prompt",
            {"output_schema": {}},
            _ExperimentRepairBatch,
            provider_name="Test",
        )
        self.assertEqual(len(result.experiments), 1)
        self.assertEqual(result.experiments[0].formula.op.value, "add")
        # The second attempt must have been told what was wrong, including
        # the offending node, not just loc + message.
        self.assertEqual(len(seen_payloads), 2)
        retry_payload = json.dumps(seen_payloads[1], ensure_ascii=False, default=str)
        self.assertIn("add", retry_payload)
        self.assertIn("validation_errors", seen_payloads[1])
        self.assertIn(
            "恰好有 2 个 args", seen_payloads[1]["correction_instruction"]
        )

    def test_three_invalid_attempts_raise_value_error(self) -> None:
        tools = self._tools()
        invalid = _repair_batch_json({"op": "add", "field": "close", "args": []})
        tools._completion_content = (  # type: ignore[method-assign]
            lambda *args, **kwargs: invalid
        )
        with self.assertRaises(ValueError) as context:
            tools._validated_json_completion(
                None,  # type: ignore[arg-type]
                "test-model",
                "test prompt",
                {"output_schema": {}},
                _ExperimentRepairBatch,
                provider_name="Test",
            )
        self.assertIn("after three attempts", str(context.exception))


class RepairExperimentsIsolationTest(unittest.TestCase):
    def test_one_failed_repair_does_not_abort_the_other(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        rejected = [_make_experiment("E-TEST-01"), _make_experiment("E-TEST-02")]
        issues = {
            "E-TEST-01": [
                CheckIssue(code="formula_compile_failed", message="bad formula")
            ],
            "E-TEST-02": [
                CheckIssue(code="formula_compile_failed", message="bad formula")
            ],
        }
        good_batch = _ExperimentRepairBatch.model_validate_json(
            _repair_batch_json(
                {
                    "op": "add",
                    "args": [
                        {"op": "field", "field": "close"},
                        {"op": "constant", "value": 1.0},
                    ],
                }
            )
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            side_effect=[ValueError("schema-invalid JSON"), good_batch]
        )
        repaired = tools.repair_experiments(
            rejected, issues, [], [_make_plan()]
        )
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].experiment_id, "E-TEST-02-R1")
        self.assertEqual(tools._deepseek_json.call_count, 2)


class CheckpointSerdeTest(unittest.TestCase):
    def test_v2stage_roundtrips_with_allowlisted_serde(self) -> None:
        # Mirrors __main__.py: SqliteSaver(connection, serde=JsonPlusSerializer(
        # allowed_msgpack_modules=[(V2Stage.__module__, V2Stage.__name__)])).
        # Without the allowlist, LangGraph warns
        # "Deserializing unregistered type ... V2Stage" and will block it
        # in a future version.
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[(V2Stage.__module__, V2Stage.__name__)]
        )
        data_type, data = serde.dumps_typed(V2Stage.REPAIR_EXPERIMENTS)
        restored = serde.loads_typed((data_type, data))
        self.assertEqual(restored, V2Stage.REPAIR_EXPERIMENTS)


def _make_proxy(
    proxy_id: str,
    definition: str,
    required_fields: list[str],
) -> MeasurementProxy:
    return MeasurementProxy(
        proxy_id=proxy_id,
        mechanism_component="test mechanism component",
        observable_definition=definition,
        required_fields=required_fields,
        limitations=["test limitation"],
    )


def _make_plan_draft(
    proxies: list[MeasurementProxy],
    full_logic: str = "full logic using P1",
    baseline_logic: str = "baseline logic using P1",
) -> _MeasurementPlanDraft:
    return _MeasurementPlanDraft(
        mechanism_summary="test mechanism summary",
        proxies=proxies,
        baseline_logic=baseline_logic,
        full_mechanism_logic=full_logic,
        unobservable_aspects=[],
    )


def _make_hypothesis() -> DevelopedHypothesis:
    return DevelopedHypothesis(
        hypothesis_id="H-TEST-01",
        idea_id="I-TEST-01",
        direction=ResearchDirection.MOMENTUM,
        title="test hypothesis",
        claim="test claim for plan validation",
        economic_rationale="test economic rationale",
        core_mechanism="test core mechanism",
        counterargument="test counterargument",
        falsification_criteria=["test falsification criterion"],
        distinction_from_siblings="test distinction",
        prediction_horizons_days=[5],
        required_concepts=["price_momentum"],
    )


class ValidateMeasurementPlanTest(unittest.TestCase):
    def test_clean_plan_has_no_violations(self) -> None:
        draft = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "过去20日收盘价 close 相对过去60日均值的偏离",
                    ["close"],
                ),
                _make_proxy(
                    "P2",
                    "过去20日成交量 volume 相对过去60日均值的比率",
                    ["volume"],
                ),
            ]
        )
        self.assertEqual(_validate_measurement_plan(draft), [])

    def test_too_many_proxies_is_a_violation(self) -> None:
        proxies = [
            _make_proxy(f"P{i}", f"过去20日收盘价 close 的统计量 {i}", ["close"])
            for i in range(1, MAX_MEASUREMENT_PROXIES + 2)
        ]
        violations = _validate_measurement_plan(_make_plan_draft(proxies))
        self.assertTrue(any("超过上限" in item for item in violations))

    def test_future_data_in_definition_is_a_violation(self) -> None:
        # Price-Volume P3 in run 13ac24bc: "计算未来K日内收盘价是否持续
        # 位于原边界之外" — the DSL forbids future data.
        draft = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "计算未来5日收盘价 close 是否持续在边界之外",
                    ["close"],
                )
            ]
        )
        violations = _validate_measurement_plan(draft)
        self.assertTrue(any("未来" in item for item in violations))

    def test_market_wide_statistic_is_a_violation(self) -> None:
        # Liquidity P1 in run 13ac24bc: "全市场换手率中位数" — no
        # cross-sectional median operator exists in the DSL.
        draft = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "全市场换手率中位数相对其20日均值的偏离",
                    ["volume", "mktcap"],
                )
            ]
        )
        violations = _validate_measurement_plan(draft)
        self.assertTrue(any("全市场" in item for item in violations))

    def test_required_fields_must_match_definition(self) -> None:
        # Reversal P4 in run 13ac24bc: required_fields lists open, but its
        # own definition CLV=(close-low)/(high-low) never uses open.
        draft = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "构造收盘相对日内区间位置 CLV=(close-low)/(high-low)",
                    ["close", "high", "low", "open"],
                )
            ]
        )
        violations = _validate_measurement_plan(draft)
        self.assertTrue(any("open" in item for item in violations))
        # And the other direction: a field the definition uses but the
        # list omits must also be flagged.
        draft = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "过去20日成交量 volume 相对过去60日均值的比率",
                    ["close"],
                )
            ]
        )
        violations = _validate_measurement_plan(draft)
        self.assertTrue(any("volume" in item for item in violations))

    def test_composite_proxy_inherits_referenced_proxy_fields(self) -> None:
        # Price-Volume P4 in run 13ac24bc is "P1+P2 combined": its
        # definition names P1 and P2 instead of spelling out the fields,
        # so its required_fields must be judged against the union.
        draft = _make_plan_draft(
            [
                _make_proxy("P1", "当日成交量 volume 相对过去20日均值的比率", ["volume"]),
                _make_proxy("P2", "基于 high、low、close 字段的突破", ["close", "high", "low"]),
                _make_proxy(
                    "P3",
                    "结合 P1 和 P2，定义放量突破事件",
                    ["close", "high", "low", "volume"],
                ),
            ],
            full_logic="full logic using P1, P2 and P3",
            baseline_logic="baseline logic using P1 and P2",
        )
        self.assertEqual(_validate_measurement_plan(draft), [])

    def test_logic_referencing_undefined_proxy_is_a_violation(self) -> None:
        draft = _make_plan_draft(
            [_make_proxy("P1", "过去20日收盘价 close 的偏离", ["close"])],
            full_logic="full logic using P1 and P9",
        )
        violations = _validate_measurement_plan(draft)
        self.assertTrue(any("P9" in item for item in violations))

    def test_rules_state_proxy_cap_and_future_ban(self) -> None:
        self.assertIn("最多 5 个", MEASUREMENT_PROXY_RULES)
        self.assertIn("不准出现“未来”", MEASUREMENT_PROXY_RULES)

    def test_shape_rules_state_where_condition_and_correlation_ddof(self) -> None:
        # f1bfbaee 的两个编译失败：correlation 被传了编译器不支持的 ddof；
        # where 的条件用了数值序列而不是比较结果。
        self.assertIn("correlation 不接受 ddof", FORMULA_NODE_SHAPE_RULES)
        self.assertIn("where 的第一个参数必须是布尔条件", FORMULA_NODE_SHAPE_RULES)


class DesignMeasurementPlansTest(unittest.TestCase):
    def test_invalid_draft_is_repaired_once(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        invalid = _make_plan_draft(
            [
                _make_proxy(f"P{i}", f"过去20日收盘价 close 的统计量 {i}", ["close"])
                for i in range(1, MAX_MEASUREMENT_PROXIES + 2)
            ]
        )
        valid = _make_plan_draft(
            [_make_proxy("P1", "过去20日收盘价 close 的偏离", ["close"])]
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            side_effect=[invalid, valid]
        )
        plans = tools.design_measurement_plans([_make_hypothesis()])
        self.assertEqual(len(plans), 1)
        self.assertLessEqual(len(plans[0].proxies), MAX_MEASUREMENT_PROXIES)
        self.assertEqual(plans[0].unobservable_aspects, [])
        self.assertEqual(tools._deepseek_json.call_count, 2)
        # The repair call must have been told what was wrong.
        repair_payload = tools._deepseek_json.call_args_list[1][0][1]
        self.assertIn("plan_violations", repair_payload)

    def test_unfixable_draft_keeps_fewest_violations_and_records_them(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        bad = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "计算未来5日收盘价 close 是否持续在边界之外",
                    ["close"],
                )
            ]
        )
        worse = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "计算未来5日收盘价 close 是否持续在边界之外",
                    ["close"],
                ),
                _make_proxy(
                    "P2",
                    "全市场换手率中位数相对其20日均值的偏离",
                    ["volume", "mktcap"],
                ),
            ]
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            side_effect=[worse, bad]
        )
        plans = tools.design_measurement_plans([_make_hypothesis()])
        self.assertEqual(len(plans), 1)
        # The repair (1 violation) beats the first draft (2 violations),
        # so it is kept — but its remaining violation must still be
        # recorded instead of silently dropped.
        self.assertEqual(len(plans[0].proxies), 1)
        self.assertTrue(
            any(item.startswith("PLAN_CHECK:") for item in plans[0].unobservable_aspects)
        )

    def test_failed_repair_call_does_not_abort_the_run(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        invalid = _make_plan_draft(
            [
                _make_proxy(
                    "P1",
                    "计算未来5日收盘价 close 是否持续在边界之外",
                    ["close"],
                )
            ]
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            side_effect=[invalid, ValueError("schema-invalid JSON")]
        )
        plans = tools.design_measurement_plans([_make_hypothesis()])
        self.assertEqual(len(plans), 1)
        self.assertTrue(
            any(item.startswith("PLAN_CHECK:") for item in plans[0].unobservable_aspects)
        )


def _make_experiment_draft(
    variant: ExperimentVariantType, horizon: int
) -> _ExperimentDraft:
    formula = (
        {"op": "field", "field": "close"}
        if variant == ExperimentVariantType.BASELINE
        else {
            "op": "multiply",
            "args": [
                {"op": "field", "field": "close"},
                {"op": "constant", "value": 1.0},
            ],
        }
    )
    return _ExperimentDraft(
        name="test experiment",
        description="test experiment description",
        variation_reason="test variation reason",
        variant_type=variant,
        represented_proxy_ids=["P1"],
        formula=formula,  # type: ignore[arg-type]
        prediction_horizon_days=horizon,
    )


class GenerateExperimentsHorizonTest(unittest.TestCase):
    def _run_generate(
        self, horizons: list[int], draft_horizons: list[int]
    ) -> list[FactorExperimentSpec]:
        tools = RealV2Tools.__new__(RealV2Tools)
        tools.max_experiments_per_hypothesis = 2
        tools.cache_dir = None
        hypothesis = _make_hypothesis()
        hypothesis.prediction_horizons_days = horizons
        plan = MeasurementPlan(
            plan_id="MP-TEST-01",
            hypothesis_id=hypothesis.hypothesis_id,
            **_make_plan_draft(
                [_make_proxy("P1", "过去20日收盘价 close 的偏离", ["close"])]
            ).model_dump(),
        )
        batch = _ExperimentBatch(
            experiments=[
                _make_experiment_draft(ExperimentVariantType.BASELINE, draft_horizons[0]),
                _make_experiment_draft(
                    ExperimentVariantType.FULL_MECHANISM, draft_horizons[1]
                ),
            ]
        )
        tools._deepseek_json = Mock(return_value=batch)  # type: ignore[method-assign]
        self._deepseek_mock = tools._deepseek_json
        return tools.generate_experiments([hypothesis], [plan])

    def test_short_horizon_is_raised_to_the_floor(self) -> None:
        # 13ac24bc 重跑出来的 Reversal 基线选了 1 日持有，turnover 1.0，
        # 10 年毛收益 7.52 被成本吃成净收益 -0.31。
        specs = self._run_generate([1, 3, 5], [1, 3])
        self.assertEqual(
            [spec.prediction_horizon_days for spec in specs],
            [MIN_PREDICTION_HORIZON_DAYS, MIN_PREDICTION_HORIZON_DAYS],
        )

    def test_horizon_without_any_eligible_value_uses_the_longest(self) -> None:
        specs = self._run_generate([1, 3], [1, 3])
        self.assertEqual(
            [spec.prediction_horizon_days for spec in specs], [3, 3]
        )

    def test_generate_prompt_states_the_horizon_floor(self) -> None:
        # The prompt is built inline in generate_experiments; the floor
        # constant is the single source of truth, so a change to it would
        # also change the prompt. This guards the constant itself.
        self.assertGreaterEqual(MIN_PREDICTION_HORIZON_DAYS, 5)

    def test_generate_prompt_warns_that_zero_volume_never_occurs(self) -> None:
        # 0f45e744 crashed on mean(where(volume<1, 1, 0), 20): the panel
        # has 0 rows with volume<1, so that term is always 0 and the whole
        # factor is constant. The note must actually reach the LLM prompt,
        # not just exist as a constant.
        self.assertIn("volume<1", PANEL_ZERO_VOLUME_NOTE)
        self.assertIn("0 行", PANEL_ZERO_VOLUME_NOTE)
        self._run_generate([5, 10], [5, 10])
        prompt = self._deepseek_mock.call_args[0][0]
        self.assertIn("volume<1", prompt)
        self.assertIn("零成交量", prompt)


class CheckpointWalTest(unittest.TestCase):
    def test_wal_checkpoint_makes_the_db_file_self_contained(self) -> None:
        # 13ac24bc 的教训：SqliteSaver 开 WAL，checkpoint 先落在 -wal 里；
        # 只拷 checkpoints.sqlite（没拷 -wal）就丢了 generate 之后的所有
        # checkpoint，resume 被迫重跑 generate，实验全换了。
        import shutil
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "checkpoints.sqlite"
            connection = sqlite3.connect(db_path, check_same_thread=False)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE t (v TEXT)")
                connection.execute("INSERT INTO t VALUES ('after-generate')")
                connection.commit()
                # 只拷主文件（模拟用户只发了 checkpoints.sqlite）。
                copy_dir = Path(tmp) / "copy-before"
                copy_dir.mkdir()
                shutil.copy(db_path, copy_dir / "checkpoints.sqlite")
                with sqlite3.connect(
                    f"file:{copy_dir / 'checkpoints.sqlite'}?mode=ro", uri=True
                ) as copy_conn:
                    try:
                        rows = copy_conn.execute("SELECT v FROM t").fetchall()
                    except sqlite3.OperationalError:
                        # 表本身都还在 -wal 里，主文件里根本没有这张表。
                        rows = None
                self.assertIsNone(rows)
                # 退出时做的 WAL checkpoint：再拷一次就全在主文件里了。
                _checkpoint_wal(connection)
                copy_dir2 = Path(tmp) / "copy-after"
                copy_dir2.mkdir()
                shutil.copy(db_path, copy_dir2 / "checkpoints.sqlite")
                with sqlite3.connect(
                    f"file:{copy_dir2 / 'checkpoints.sqlite'}?mode=ro", uri=True
                ) as copy_conn:
                    rows = copy_conn.execute("SELECT v FROM t").fetchall()
                self.assertEqual(rows, [("after-generate",)])
            finally:
                connection.close()


class RankCorrelationTest(unittest.TestCase):
    def test_identical_and_negated_vectors(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.normal(size=2000)
        corr, overlap = _rank_correlation(base, base.copy())
        self.assertAlmostEqual(corr, 1.0)
        self.assertEqual(overlap, 2000)
        corr, _ = _rank_correlation(base, -base)
        self.assertAlmostEqual(corr, -1.0)

    def test_unrelated_vectors_are_not_flagged(self) -> None:
        rng = np.random.default_rng(7)
        corr, _ = _rank_correlation(
            rng.normal(size=2000), rng.normal(size=2000)
        )
        self.assertIsNotNone(corr)
        assert corr is not None
        self.assertLess(abs(corr), DUPLICATE_SIGNAL_MIN_CORRELATION)

    def test_too_few_overlapping_or_constant_vectors_return_none(self) -> None:
        rng = np.random.default_rng(7)
        short = rng.normal(size=100)
        corr, overlap = _rank_correlation(short, short.copy())
        self.assertIsNone(corr)
        self.assertEqual(overlap, 100)
        corr, _ = _rank_correlation(
            np.ones(2000), rng.normal(size=2000)
        )
        self.assertIsNone(corr)


def _return_formula(window: int) -> dict:
    return {
        "op": "return",
        "args": [{"op": "field", "field": "close"}],
        "window": window,
    }


def _volume_scaled_return_formula(window: int) -> dict:
    """Baseline return times a volume ratio that is ~1 on the synthetic
    panel: the same signal as _return_formula written differently."""
    return {
        "op": "multiply",
        "args": [
            _return_formula(window),
            {
                "op": "divide",
                "args": [
                    {"op": "field", "field": "volume"},
                    {"op": "mean", "field": "volume", "window": 20},
                ],
            },
        ],
    }


def _make_formula_experiment(
    experiment_id: str,
    hypothesis_id: str,
    plan_id: str,
    formula: dict,
    proxy_ids: list[str],
    variant: ExperimentVariantType = ExperimentVariantType.BASELINE,
) -> FactorExperimentSpec:
    node = FormulaNode.model_validate(formula)
    factor = FactorFormulaSpecV2(
        factor_id=f"F-{experiment_id}",
        hypothesis_id=hypothesis_id,
        name="test factor",
        description="test factor for signal dedup",
        formula=node,
        required_fields=sorted(
            field.value for field in node.referenced_fields()
        ),
        lookback_days=node.required_history(),
    )
    return FactorExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        variation_reason="test variation reason",
        measurement_plan_id=plan_id,
        variant_type=variant,
        represented_proxy_ids=proxy_ids,
        factor=factor,
        prediction_horizon_days=5,
    )


def _make_formula_plan(
    plan_id: str, hypothesis_id: str, required_fields: list[str]
) -> MeasurementPlan:
    return MeasurementPlan(
        plan_id=plan_id,
        hypothesis_id=hypothesis_id,
        mechanism_summary="test mechanism summary",
        proxies=[
            MeasurementProxy(
                proxy_id="P1",
                mechanism_component="test mechanism component",
                observable_definition="test observable definition",
                required_fields=required_fields,
                limitations=["test limitation"],
            )
        ],
        baseline_logic="test baseline logic",
        full_mechanism_logic="test full mechanism logic",
    )


def _write_synthetic_panel(path: Path) -> None:
    """Small panel for preflight dedup tests: 2008-2010, 100 permnos.

    Volume is nearly constant on purpose: volume/mean(volume, 20) is then
    ~1, so a volume-ratio-scaled 5-day return is the same signal as the
    plain 5-day return written differently — exactly the collapse the
    dedup check must catch.
    """
    rng = np.random.default_rng(7)
    all_dates = pl.date_range(
        date(2008, 1, 1), date(2010, 12, 31), "1d", eager=True
    )
    dates = all_dates.filter(all_dates.dt.weekday() < 5).to_list()
    n_dates = len(dates)
    n_permnos = 100
    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, size=(n_dates, n_permnos)), axis=0)
    )
    volume = 1000.0 * (1.0 + rng.normal(0.0, 0.001, size=(n_dates, n_permnos)))
    frame = pl.DataFrame(
        {
            "date": [d for d in dates for _ in range(n_permnos)],
            "permno": list(range(1, n_permnos + 1)) * n_dates,
            "open": close.ravel() * (1.0 + rng.normal(0, 0.001, n_dates * n_permnos)),
            "high": close.ravel() * (1.0 + np.abs(rng.normal(0, 0.002, n_dates * n_permnos))),
            "low": close.ravel() * (1.0 - np.abs(rng.normal(0, 0.002, n_dates * n_permnos))),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "mktcap": np.full(n_dates * n_permnos, 1e9),
        }
    )
    frame.write_parquet(path)


class PreflightDuplicateSignalTest(unittest.TestCase):
    def _tools(self, panel_path: Path) -> RealV2Tools:
        tools = RealV2Tools.__new__(RealV2Tools)
        tools.panel_path = panel_path
        tools.max_experiments_per_hypothesis = 5
        tools.backtest_engine = Mock()
        tools.backtest_engine.validate_sample_capacity = Mock(return_value=[])
        return tools

    def test_negated_and_volume_scaled_copies_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            experiments = [
                _make_formula_experiment(
                    "E-TEST-A", "H-TEST-01", "MP-TEST-01",
                    _return_formula(5), ["P1"],
                ),
                _make_formula_experiment(
                    "E-TEST-B", "H-TEST-02", "MP-TEST-02",
                    {"op": "negate", "args": [_return_formula(5)]}, ["P1"],
                ),
                _make_formula_experiment(
                    "E-TEST-C", "H-TEST-02", "MP-TEST-02",
                    _return_formula(20), ["P1"],
                ),
                _make_formula_experiment(
                    "E-TEST-D", "H-TEST-03", "MP-TEST-03",
                    {
                        "op": "multiply",
                        "args": [
                            {
                                "op": "divide",
                                "args": [
                                    {"op": "field", "field": "volume"},
                                    {
                                        "op": "mean",
                                        "field": "volume",
                                        "window": 20,
                                    },
                                ],
                            },
                            _return_formula(5),
                        ],
                    },
                    ["P1"],
                ),
            ]
            plans = [
                _make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"]),
                _make_formula_plan("MP-TEST-02", "H-TEST-02", ["close"]),
                _make_formula_plan(
                    "MP-TEST-03", "H-TEST-03", ["close", "volume"]
                ),
            ]
            results = {
                item.experiment_id: item
                for item in tools.preflight_experiments(experiments, plans, 0)
            }
            self.assertTrue(results["E-TEST-A"].accepted)
            # B is the same signal with a minus sign; C is a genuinely
            # different horizon (5d vs 20d returns correlate ~0.5).
            self.assertFalse(results["E-TEST-B"].accepted)
            self.assertEqual(
                [issue.code for issue in results["E-TEST-B"].issues],
                ["duplicate_signal"],
            )
            self.assertIn("E-TEST-A", results["E-TEST-B"].issues[0].message)
            self.assertTrue(results["E-TEST-C"].accepted)
            # D is textually different but the volume ratio is ~1, so it
            # is the same signal as A written differently.
            self.assertFalse(results["E-TEST-D"].accepted)
            self.assertIn(
                "duplicate_signal",
                [issue.code for issue in results["E-TEST-D"].issues],
            )
            self.assertIn("E-TEST-A", results["E-TEST-D"].issues[0].message)

    def test_duplicate_of_rejected_experiment_does_not_flag_the_clean_one(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            # E references a plan that does not exist, so it is rejected
            # for lineage; its signal is return(close, 5) times 1.0, i.e.
            # identical to A. A must not be killed for duplicating a
            # doomed experiment — round 1 re-checks the pair if E's repair
            # survives.
            bad = _make_formula_experiment(
                "E-TEST-E", "H-TEST-01", "MP-MISSING",
                {
                    "op": "multiply",
                    "args": [
                        _return_formula(5),
                        {"op": "constant", "value": 1.0},
                    ],
                },
                ["P1"],
            )
            good = _make_formula_experiment(
                "E-TEST-A", "H-TEST-01", "MP-TEST-01",
                _return_formula(5), ["P1"],
            )
            results = {
                item.experiment_id: item
                for item in tools.preflight_experiments(
                    [bad, good], [_make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"])], 0
                )
            }
            self.assertFalse(results["E-TEST-E"].accepted)
            self.assertEqual(
                [issue.code for issue in results["E-TEST-E"].issues],
                ["invalid_measurement_plan_lineage"],
            )
            self.assertTrue(results["E-TEST-A"].accepted)

    def test_full_mechanism_exempt_from_duplicate_of_own_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            # The full is its baseline times a ~1 volume ratio: rank corr
            # ~1.0 with the baseline. A full IS its baseline plus
            # modulation, so that correlation is the design, not
            # duplication — the backtest must decide whether the
            # modulation adds value.
            baseline = _make_formula_experiment(
                "E-TEST-A", "H-TEST-01", "MP-TEST-01",
                _return_formula(5), ["P1"],
            )
            full = _make_formula_experiment(
                "E-TEST-F", "H-TEST-01", "MP-TEST-01",
                _volume_scaled_return_formula(5), ["P1"],
                variant=ExperimentVariantType.FULL_MECHANISM,
            )
            results = {
                item.experiment_id: item
                for item in tools.preflight_experiments(
                    [baseline, full],
                    [_make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"])],
                    0,
                )
            }
            self.assertTrue(results["E-TEST-A"].accepted)
            self.assertTrue(results["E-TEST-F"].accepted)
            self.assertEqual(results["E-TEST-F"].issues, [])

    def test_full_mechanism_still_flagged_against_other_hypothesis_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            # Same pair of formulas, but the full belongs to another
            # hypothesis: the exemption is only for a full against its
            # OWN baseline. The 0.90 threshold is unchanged here.
            baseline = _make_formula_experiment(
                "E-TEST-A", "H-TEST-01", "MP-TEST-01",
                _return_formula(5), ["P1"],
            )
            full = _make_formula_experiment(
                "E-TEST-F", "H-TEST-02", "MP-TEST-02",
                _volume_scaled_return_formula(5), ["P1"],
                variant=ExperimentVariantType.FULL_MECHANISM,
            )
            results = {
                item.experiment_id: item
                for item in tools.preflight_experiments(
                    [baseline, full],
                    [
                        _make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"]),
                        _make_formula_plan("MP-TEST-02", "H-TEST-02", ["close"]),
                    ],
                    0,
                )
            }
            self.assertTrue(results["E-TEST-A"].accepted)
            self.assertFalse(results["E-TEST-F"].accepted)
            self.assertIn(
                "duplicate_signal",
                [issue.code for issue in results["E-TEST-F"].issues],
            )


class CompletionRetryTest(unittest.TestCase):
    def _bad_request(self) -> BadRequestError:
        response = httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://example.test/v1/chat/completions"
            ),
        )
        return BadRequestError("Error code: 400", response=response, body=None)

    def _ok_response(self, content: str):
        choice = Mock()
        choice.finish_reason = "stop"
        choice.message.content = content
        response = Mock()
        response.choices = [choice]
        return response

    def test_transient_bad_request_is_retried_once(self) -> None:
        client = Mock()
        client.chat.completions.create = Mock(
            side_effect=[
                self._bad_request(),
                self._ok_response('{"ok": true}'),
            ]
        )
        with patch(
            "hypothesis_workflow_v2.real_tools.time.sleep"
        ) as sleep:
            content = RealV2Tools._completion_content(
                client,
                "gpt-5.6-luna",
                "system",
                {"idea": "x"},
                max_tokens=10,
                provider_name="Test",
                response_format={"type": "json_object"},
            )
        self.assertEqual(content, '{"ok": true}')
        sleep.assert_called_once_with(5)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_persistent_bad_request_is_raised(self) -> None:
        client = Mock()
        client.chat.completions.create = Mock(side_effect=self._bad_request())
        with patch(
            "hypothesis_workflow_v2.real_tools.time.sleep"
        ) as sleep:
            with self.assertRaises(BadRequestError):
                RealV2Tools._completion_content(
                    client,
                    "gpt-5.6-luna",
                    "system",
                    {"idea": "x"},
                    max_tokens=10,
                    provider_name="Test",
                )
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(sleep.call_count, 3)


class RepairDuplicateReferenceTest(unittest.TestCase):
    def test_repair_prompt_includes_cross_hypothesis_duplicate(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        accepted = [
            _make_formula_experiment(
                "E-TEST-A", "H-TEST-01", "MP-TEST-01",
                _return_formula(5), ["P1"],
            )
        ]
        rejected = [
            _make_formula_experiment(
                "E-TEST-B", "H-TEST-02", "MP-TEST-02",
                {"op": "negate", "args": [_return_formula(5)]}, ["P1"],
            )
        ]
        issues = {
            "E-TEST-B": [
                CheckIssue(
                    code="duplicate_signal",
                    message="nearly identical to experiment E-TEST-A",
                )
            ]
        }
        tools._duplicate_pairs = {"E-TEST-B": ("E-TEST-A", 0.99)}
        good_batch = _ExperimentRepairBatch.model_validate_json(
            _repair_batch_json(_return_formula(20))
        )
        seen_payloads: list[dict] = []

        def fake_deepseek_json(system_prompt, payload, model):
            seen_payloads.append(payload)
            return good_batch

        tools._deepseek_json = fake_deepseek_json  # type: ignore[method-assign]
        repaired = tools.repair_experiments(
            rejected,
            issues,
            accepted,
            [
                _make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"]),
                _make_formula_plan("MP-TEST-02", "H-TEST-02", ["close"]),
            ],
        )
        self.assertEqual(len(repaired), 1)
        self.assertEqual(len(seen_payloads), 1)
        existing_ids = {
            item["experiment_id"]
            for item in seen_payloads[0]["existing_experiments"]
        }
        # The duplicate reference lives in another hypothesis: without it
        # the LLM would only see same-hypothesis experiments and could not
        # tell what to differentiate from.
        self.assertIn("E-TEST-A", existing_ids)


def _make_multi_proxy_plan(
    plan_id: str, hypothesis_id: str, proxy_fields: list[list[str]]
) -> MeasurementPlan:
    return MeasurementPlan(
        plan_id=plan_id,
        hypothesis_id=hypothesis_id,
        mechanism_summary="test mechanism summary",
        proxies=[
            MeasurementProxy(
                proxy_id=f"P{index + 1}",
                mechanism_component="test mechanism component",
                observable_definition="test observable definition",
                required_fields=fields,
                limitations=["test limitation"],
            )
            for index, fields in enumerate(proxy_fields)
        ],
        baseline_logic="test baseline logic",
        full_mechanism_logic="test full mechanism logic",
    )


def _all_fields_formula() -> dict:
    """Uses close, volume, mktcap, high and low in one formula."""
    return {
        "op": "add",
        "args": [
            {
                "op": "multiply",
                "args": [
                    {"op": "field", "field": "close"},
                    {"op": "field", "field": "volume"},
                ],
            },
            {
                "op": "divide",
                "args": [
                    {"op": "field", "field": "mktcap"},
                    {
                        "op": "subtract",
                        "args": [
                            {"op": "field", "field": "high"},
                            {"op": "field", "field": "low"},
                        ],
                    },
                ],
            },
        ],
    }


class PreflightDerivedCoverageTest(unittest.TestCase):
    def _tools(self, panel_path: Path) -> RealV2Tools:
        tools = RealV2Tools.__new__(RealV2Tools)
        tools.panel_path = panel_path
        tools.max_experiments_per_hypothesis = 5
        tools.backtest_engine = Mock()
        tools.backtest_engine.validate_sample_capacity = Mock(return_value=[])
        return tools

    def test_full_declaring_subset_but_covering_all_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            plan = _make_multi_proxy_plan(
                "MP-TEST-01",
                "H-TEST-01",
                [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
            )
            # 3d2eeec5's dominant failure: the formula implements every
            # proxy but the declaration lists only two. Coverage is judged
            # by the formula, so this must pass.
            experiment = _make_formula_experiment(
                "E-TEST-FULL",
                "H-TEST-01",
                "MP-TEST-01",
                _all_fields_formula(),
                ["P1", "P2"],
                variant=ExperimentVariantType.FULL_MECHANISM,
            )
            results = tools.preflight_experiments([experiment], [plan], 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].accepted, results[0].issues)

    def test_full_missing_a_field_is_rejected_with_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            plan = _make_multi_proxy_plan(
                "MP-TEST-01",
                "H-TEST-01",
                [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
            )
            experiment = _make_formula_experiment(
                "E-TEST-FULL",
                "H-TEST-01",
                "MP-TEST-01",
                _return_formula(5),
                ["P1", "P2", "P3", "P4", "P5"],
                variant=ExperimentVariantType.FULL_MECHANISM,
            )
            results = tools.preflight_experiments([experiment], [plan], 0)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].accepted)
            by_code = {
                issue.code: issue.message for issue in results[0].issues
            }
            self.assertIn("incomplete_full_mechanism", by_code)
            self.assertIn("P4", by_code["incomplete_full_mechanism"])
            self.assertIn("P5", by_code["incomplete_full_mechanism"])
            self.assertIn("measurement_proxy_not_implemented", by_code)
            self.assertIn(
                "mktcap", by_code["measurement_proxy_not_implemented"]
            )

    def test_non_full_declaration_is_not_forced_to_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = self._tools(panel_path)
            plan = _make_multi_proxy_plan(
                "MP-TEST-01",
                "H-TEST-01",
                [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
            )
            # A baseline that incidentally uses every field is still a
            # baseline: its declaration states intent, and preflight
            # does not rewrite it or complain about the extra coverage.
            experiment = _make_formula_experiment(
                "E-TEST-BASE",
                "H-TEST-01",
                "MP-TEST-01",
                _all_fields_formula(),
                ["P1"],
            )
            results = tools.preflight_experiments([experiment], [plan], 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].accepted, results[0].issues)


class NormalizeRepresentedProxiesTest(unittest.TestCase):
    def _plan(self) -> MeasurementPlan:
        return _make_multi_proxy_plan(
            "MP-TEST-01",
            "H-TEST-01",
            [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
        )

    def test_full_declaration_is_rewritten_to_derived(self) -> None:
        plan = self._plan()
        experiment = _make_formula_experiment(
            "E-TEST-FULL",
            "H-TEST-01",
            "MP-TEST-01",
            _all_fields_formula(),
            ["P1"],
            variant=ExperimentVariantType.FULL_MECHANISM,
        )
        _normalize_represented_proxies([experiment], [plan])
        self.assertEqual(
            experiment.represented_proxy_ids, ["P1", "P2", "P3", "P4", "P5"]
        )
        self.assertEqual(_derived_proxy_ids(experiment, plan), set(experiment.represented_proxy_ids))

    def test_non_full_declaration_is_rewritten_to_derived(self) -> None:
        # Patch 13: ec610c9d lost 9 experiments to
        # measurement_proxy_not_implemented — all repaired -R1s whose
        # formulas no longer implement the inherited declaration. The
        # declaration states what the formula implements, so every
        # variant is derived now, not just full mechanisms.
        plan = self._plan()
        experiment = _make_formula_experiment(
            "E-TEST-BASE",
            "H-TEST-01",
            "MP-TEST-01",
            _all_fields_formula(),
            ["P1"],
        )
        _normalize_represented_proxies([experiment], [plan])
        self.assertEqual(
            experiment.represented_proxy_ids, ["P1", "P2", "P3", "P4", "P5"]
        )

    def test_non_full_with_subset_formula_derives_subset(self) -> None:
        # A volume-only formula declaring [P1..P4] is relabeled to the
        # proxies it actually implements (P2 = volume), not rejected.
        plan = self._plan()
        experiment = _make_formula_experiment(
            "E-TEST-ABL",
            "H-TEST-01",
            "MP-TEST-01",
            {
                "op": "mean",
                "window": 20,
                "args": [{"op": "field", "field": "volume"}],
            },
            ["P1", "P2", "P3", "P4"],
            variant=ExperimentVariantType.ABLATION,
        )
        _normalize_represented_proxies([experiment], [plan])
        self.assertEqual(experiment.represented_proxy_ids, ["P2"])

    def test_full_with_empty_derivation_is_left_alone(self) -> None:
        plan = _make_multi_proxy_plan("MP-TEST-01", "H-TEST-01", [["mktcap"]])
        experiment = _make_formula_experiment(
            "E-TEST-FULL",
            "H-TEST-01",
            "MP-TEST-01",
            _return_formula(5),
            ["P1"],
            variant=ExperimentVariantType.FULL_MECHANISM,
        )
        _normalize_represented_proxies([experiment], [plan])
        self.assertEqual(experiment.represented_proxy_ids, ["P1"])


class RepairFullNormalizationTest(unittest.TestCase):
    def test_repaired_full_inherits_derived_declaration(self) -> None:
        tools = RealV2Tools.__new__(RealV2Tools)
        plan = _make_multi_proxy_plan(
            "MP-TEST-01",
            "H-TEST-01",
            [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
        )
        rejected = [
            _make_formula_experiment(
                "E-TEST-FULL",
                "H-TEST-01",
                "MP-TEST-01",
                _return_formula(5),
                ["P1", "P2"],
                variant=ExperimentVariantType.FULL_MECHANISM,
            )
        ]
        issues = {
            "E-TEST-FULL": [
                CheckIssue(
                    code="incomplete_full_mechanism",
                    message="Full-mechanism experiment does not implement proxies P3, P4, P5",
                )
            ]
        }
        good_batch = _ExperimentRepairBatch.model_validate_json(
            _repair_batch_json(_all_fields_formula())
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            return_value=good_batch
        )
        repaired = tools.repair_experiments(rejected, issues, [], [plan])
        self.assertEqual(len(repaired), 1)
        # The repair inherits the old incomplete declaration; the new
        # formula covers every proxy, so the stored declaration must be
        # rewritten to match it instead of carrying the old list forward.
        self.assertEqual(
            repaired[0].represented_proxy_ids, ["P1", "P2", "P3", "P4", "P5"]
        )

    def test_repaired_ablation_inherits_derived_declaration(self) -> None:
        # Patch 13: the ec610c9d rejections were repaired ablations whose
        # volume-only formulas still carried the old [P1..P4] declaration.
        # The repaired declaration must be derived from the new formula.
        tools = RealV2Tools.__new__(RealV2Tools)
        plan = _make_multi_proxy_plan(
            "MP-TEST-01",
            "H-TEST-01",
            [["close"], ["volume"], ["close", "volume"], ["mktcap"], ["high", "low"]],
        )
        rejected = [
            _make_formula_experiment(
                "E-TEST-ABL",
                "H-TEST-01",
                "MP-TEST-01",
                _return_formula(5),
                ["P1", "P2", "P3", "P4"],
                variant=ExperimentVariantType.ABLATION,
            )
        ]
        issues = {
            "E-TEST-ABL": [
                CheckIssue(
                    code="duplicate_signal",
                    message="Signal duplicates E-TEST-OTHER",
                )
            ]
        }
        volume_only = {
            "op": "mean",
            "window": 20,
            "args": [{"op": "field", "field": "volume"}],
        }
        good_batch = _ExperimentRepairBatch.model_validate_json(
            _repair_batch_json(volume_only)
        )
        tools._deepseek_json = Mock(  # type: ignore[method-assign]
            return_value=good_batch
        )
        repaired = tools.repair_experiments(rejected, issues, [], [plan])
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].represented_proxy_ids, ["P2"])


class LiteratureGroundingTest(unittest.TestCase):
    def test_every_direction_has_literature_with_a_citation(self) -> None:
        for direction in ResearchDirection:
            text = _direction_literature(direction)
            self.assertTrue(text, f"{direction} has no literature brief")
            # Each brief must anchor on at least one real paper: a year.
            self.assertRegex(text, r"(19|20)\d\d")
            # And name an author, not just a vague mechanism.
            self.assertRegex(text, r"[A-Z][a-z]+")

    def test_mechanism_first_allows_zero_to_two_citations(self) -> None:
        # Patch 9: mechanism first, literature as support — a hypothesis may
        # cite 0-2 papers, and no single paper has to anchor it.
        self.assertIn("0 到 2 篇", MECHANISM_FIRST_LITERATURE_RULES)
        self.assertIn("机制优先", MECHANISM_FIRST_LITERATURE_RULES)
        self.assertIn("硬锚定", MECHANISM_FIRST_LITERATURE_RULES)
        # The old hard rule is gone from the prompts: neither the idea nor
        # the expansion prompt may demand one anchored paper any more.
        import inspect

        source = inspect.getsource(RealV2Tools.generate_research_ideas)
        source += inspect.getsource(RealV2Tools.expand_hypotheses)
        self.assertNotIn("没有文献锚点的假设不准输出", source)
        self.assertNotIn("必须锚定其中一个文献机制", source)

    def test_dedup_threshold_is_0_90_after_the_4fa040cd_miss(self) -> None:
        # The missed pair REVERSAL-02-01 vs PRICE_VOLUME-02-04 correlates
        # at 0.944 on the dedup sample; 0.95 let it through.
        self.assertEqual(DUPLICATE_SIGNAL_MIN_CORRELATION, 0.90)

    def test_diversity_rules_require_a_different_primary_signal(self) -> None:
        self.assertIn("主信号", EXPERIMENT_DIVERSITY_RULES)
        self.assertIn("baseline", EXPERIMENT_DIVERSITY_RULES)
        self.assertIn("只取负", EXPERIMENT_DIVERSITY_RULES)

    def test_duplicate_repair_rules_require_switching_signal_family(
        self,
    ) -> None:
        self.assertIn("换主信号族", DUPLICATE_REPAIR_RULES)
        self.assertIn("0.90", DUPLICATE_REPAIR_RULES)
        self.assertIn("只改窗口", DUPLICATE_REPAIR_RULES)

    def test_overnight_direction_brief_names_dsl_construction(self) -> None:
        # Patch 11: the new overnight-intraday direction must give the LLM
        # a concrete, compilable construction, not just a paper name —
        # overnight = open / lag(close, 1) - 1, intraday = close / open - 1.
        text = _direction_literature(ResearchDirection.OVERNIGHT_INTRADAY)
        self.assertIn("open", text)
        self.assertIn("lag", text)
        self.assertIn("隔夜", text)
        self.assertIn("日内", text)
        self.assertIn("2019", text)
        # Every direction must have a focus string too (KeyError otherwise).
        self.assertTrue(_direction_focus(ResearchDirection.OVERNIGHT_INTRADAY))

    def test_turnover_rules_require_slow_signals(self) -> None:
        # Patch 11: 07-08 block — positive IC, negative net return on every
        # variant. Generation must prefer slow signals from the start.
        self.assertIn("10bps", TURNOVER_RULES)
        self.assertIn("20 日", TURNOVER_RULES)
        self.assertIn("每天翻转", TURNOVER_RULES)
        import inspect

        source = inspect.getsource(RealV2Tools.generate_experiments)
        self.assertIn("TURNOVER_RULES", source)

    def test_review_checks_calendar_and_turnover(self) -> None:
        # Patch 11: the critical review must ask whether the edge is only a
        # calendar effect (December tax-loss / January rebound killed 03-02)
        # and whether the implied trading frequency survives 10bps costs.
        import inspect

        source = inspect.getsource(RealV2Tools.analyze_insights)
        self.assertIn("日历", source)
        self.assertIn("换手成本", source)


class DegenerateSignalTest(unittest.TestCase):
    def test_counts_only_dates_with_two_distinct_values(self) -> None:
        values = np.array([1.0, 1.0, 2.0, 3.0, np.nan, np.nan])
        dates = [
            date(2008, 1, 2),
            date(2008, 1, 2),
            date(2008, 1, 3),
            date(2008, 1, 3),
            date(2008, 1, 4),
            date(2008, 1, 4),
        ]
        # 2008-01-02 is constant, 2008-01-03 varies, 2008-01-04 has no
        # finite value at all.
        self.assertEqual(
            _count_varied_cross_section_dates(values, dates), 1
        )
        self.assertEqual(
            _count_varied_cross_section_dates(
                np.ones(4), [date(2008, 1, 2)] * 4
            ),
            0,
        )

    def test_zero_volume_term_is_rejected_as_degenerate(self) -> None:
        # 0f45e744's REVERSAL-03-02, reduced to the crashing shape:
        # negate(return(close, 5)) times mean(where(volume<1, 1, 0), 20).
        # The synthetic panel's volume is ~1000, so volume<1 never holds,
        # the mean term is always 0, and the whole factor is constant —
        # exactly the factor train crashed on. A plain 5-day return in the
        # same batch must still pass.
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            _write_synthetic_panel(panel_path)
            tools = RealV2Tools.__new__(RealV2Tools)
            tools.panel_path = panel_path
            tools.max_experiments_per_hypothesis = 5
            tools.backtest_engine = Mock()
            tools.backtest_engine.validate_sample_capacity = Mock(return_value=[])
            degenerate_formula = {
                "op": "multiply",
                "args": [
                    {"op": "negate", "args": [_return_formula(5)]},
                    {
                        "op": "mean",
                        "window": 20,
                        "args": [
                            {
                                "op": "where",
                                "args": [
                                    {
                                        "op": "less_than",
                                        "args": [
                                            {"op": "field", "field": "volume"},
                                            {"op": "constant", "value": 1.0},
                                        ],
                                    },
                                    {"op": "constant", "value": 1.0},
                                    {"op": "constant", "value": 0.0},
                                ],
                            }
                        ],
                    },
                ],
            }
            experiments = [
                _make_formula_experiment(
                    "E-GOOD", "H-TEST-01", "MP-TEST-01",
                    _return_formula(5), ["P1"],
                ),
                _make_formula_experiment(
                    "E-DEGENERATE", "H-TEST-02", "MP-TEST-02",
                    degenerate_formula, ["P1"],
                ),
            ]
            plans = [
                _make_formula_plan("MP-TEST-01", "H-TEST-01", ["close"]),
                _make_formula_plan(
                    "MP-TEST-02", "H-TEST-02", ["close", "volume"]
                ),
            ]
            results = {
                item.experiment_id: item
                for item in tools.preflight_experiments(experiments, plans, 0)
            }
            self.assertTrue(results["E-GOOD"].accepted)
            self.assertFalse(results["E-DEGENERATE"].accepted)
            self.assertEqual(
                [issue.code for issue in results["E-DEGENERATE"].issues],
                ["degenerate_signal"],
            )
            self.assertGreaterEqual(DEGENERATE_SIGNAL_MIN_VARIED_DATES, 10)


class BacktestIsolationTest(unittest.TestCase):
    def test_one_failing_experiment_does_not_abort_train(self) -> None:
        # 0f45e744: a single experiment raised "factor produced no valid
        # Rank IC dates" and the 10 train results already computed were
        # lost with it. evaluate() must drop only the bad experiment.
        engine = RealV2BacktestEngine.__new__(RealV2BacktestEngine)
        engine.config = RealBacktestConfig(
            panel_path=Path("panel.parquet"),
            membership_path=Path("membership.parquet"),
        )
        engine._split_cache = {}
        engine.failed_experiments = []
        engine._load_split = lambda split: _SplitData(  # type: ignore[method-assign]
            factor_panel=pl.DataFrame(),
            evaluation_panel=pl.DataFrame(),
            price_panel=pl.DataFrame(),
            calendar=[date(2001, 1, 2)],
        )

        def _factor_values(panel, experiment, period):
            if experiment.experiment_id == "E-BAD":
                raise ValueError("factor produced no valid Rank IC dates")
            return pl.DataFrame()

        engine._factor_values = _factor_values  # type: ignore[method-assign]
        engine._rank_ic_metrics = (  # type: ignore[method-assign]
            lambda factor_values, labels, horizon: {
                "mean_rank_ic": 0.03,
                "rank_ics": np.array([0.03, 0.02, 0.04]),
                "valid_ic_dates": 3,
                "median_stock_coverage": 0.9,
            }
        )
        engine._overlapping_cohort_metrics = (  # type: ignore[method-assign]
            lambda *args, **kwargs: {
                "gross_return": 0.5,
                "net_return": 0.4,
                "annualized_return": 0.1,
                "annualized_volatility": 0.2,
                "sharpe": 1.0,
                "max_drawdown": -0.01,
                "turnover": 0.1,
            }
        )
        experiments = [
            _make_formula_experiment(
                "E-GOOD", "H-TEST-01", "MP-TEST-01",
                _return_formula(5), ["P1"],
            ),
            _make_formula_experiment(
                "E-BAD", "H-TEST-02", "MP-TEST-02",
                _return_formula(5), ["P1"],
            ),
        ]
        with patch(
            "hypothesis_workflow_v2.real_backtest.build_forward_return_labels",
            return_value=pl.DataFrame(),
        ):
            metrics = engine.evaluate(experiments, SplitName.TRAIN)
        self.assertEqual(
            [item.experiment_id for item in metrics], ["E-GOOD"]
        )
        self.assertEqual(
            engine.failed_experiments,
            [("E-BAD", "factor produced no valid Rank IC dates")],
        )


if __name__ == "__main__":
    unittest.main()
