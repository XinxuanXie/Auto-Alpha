# Hypothesis Workflow V2.1：数据契约与状态图

## 核心契约

### Research 与批判性开发

`ResearchIdea` 保存方向、自由文本、模型和 Prompt 版本。

`InsightAnalysis` 保存：

```text
decision: accept | revise | replace
supporting_arguments
objections
alternative_explanations
final_research_idea
analysis_text
```

`DevelopedHypothesis` 保存 claim、经济逻辑、核心机制、反驳、证伪标准、同批差异、预测周期和研究概念。预测周期自动去重排序，概念名统一规范化。

### Measurement Plan

```text
MeasurementPlan
├── plan_id
├── hypothesis_id
├── mechanism_summary
├── proxies[]
│   ├── proxy_id
│   ├── mechanism_component
│   ├── observable_definition
│   ├── required_fields
│   └── limitations
├── baseline_logic
├── full_mechanism_logic
└── unobservable_aspects
```

同一计划内 `proxy_id` 必须唯一。代理字段使用 `DataField` 枚举并自动去重排序。

### Formula 与实验

`FactorFormulaSpecV2` 保存公式 AST、真实引用字段和推导后的最大历史长度。字段和回看期必须与公式结构完全一致。

`FactorExperimentSpec` 在公式之外增加：

```text
measurement_plan_id
variant_type: baseline | full_mechanism | ablation | robustness
represented_proxy_ids
prediction_horizon_days
```

Factor、Experiment 和 Hypothesis 的 ID 必须一致；代理编号不能重复。

### 检查与回测

`HypothesisCheckResult` 和 `PreflightResult` 保存 `accepted`、结构化 `issues` 和 0/1 修正轮次。通过项不能携带问题，失败项必须至少有一个问题。

`SplitMetrics` 保存 Rank IC、ICIR、Newey–West、FDR、收益、波动、Sharpe、回撤、换手和覆盖率。训练使用 `direction_source=train`，验证与测试使用 `frozen_train`。

## 状态图

```text
INIT
→ LOAD_CONTEXT
→ GENERATE_RESEARCH_IDEAS
→ ANALYZE_INSIGHTS
→ EXPAND_HYPOTHESES
→ CHECK_HYPOTHESES
   ├─ rejected and repair unused → REPAIR_HYPOTHESES → CHECK_HYPOTHESES
   └─ accepted → DESIGN_MEASUREMENT_PLANS
→ GENERATE_EXPERIMENTS
→ PREFLIGHT_EXPERIMENTS
   ├─ rejected and repair unused → REPAIR_EXPERIMENTS → PREFLIGHT_EXPERIMENTS
   └─ accepted → RUN_TRAIN
→ FILTER_TRAIN
   ├─ passed → RUN_VALIDATION
   └─ none → BUILD_MEMORY
→ FILTER_VALIDATION
   ├─ passed → RUN_TEST
   └─ none → BUILD_MEMORY
→ BUILD_MEMORY
→ RECORD_RUN
→ COMPLETED
```

任一关键生成阶段没有有效产物时进入 `FAILED`。

## Artifact

LangGraph State 只保存运行状态、候选 ID 和 `ArtifactRef`。阶段产物使用原子 JSON 写入并附带 SHA-256：

```text
research_context.json
research_ideas.json
insight_analyses.json
hypotheses.json
hypothesis_checks.json
measurement_plans.json
experiment_specs.json
preflight_results.json
train_metrics.json
validation_metrics.json
test_metrics.json
memory_summary.json
run_summary.json
```

## 修正语义

- 假设修正只处理失败假设，并接收该假设的真实 `CheckIssue`；
- 实验修正只处理失败实验，并接收真实 `CheckIssue`、Measurement Plan 和同假设已通过公式；
- 两类业务修正各最多一次；
- HTTP 重试和 JSON schema 重试属于传输/格式恢复，不占业务修正次数。

## 版本边界

V2.1 的 `InsightAnalysis`、`DevelopedHypothesis` 和 `FactorExperimentSpec` 增加必填字段，并新增 `MeasurementPlan`。V2.0 checkpoint 不满足新契约，应作为历史运行只读保留。新实验使用新 `run_id`。
