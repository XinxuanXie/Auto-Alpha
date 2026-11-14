# Hypothesis Workflow V2.1 文件指南

## 阅读顺序

建议依次阅读：

1. `contracts.py`：阶段之间传递什么；
2. `state.py`：有哪些状态；
3. `manager.py`：状态如何连接；
4. `tools.py`：Manager 依赖哪些能力；
5. `real_tools.py`：真实 Agent、Checker 和适配逻辑；
6. `real_backtest.py`：如何评价单因子；
7. `artifacts.py`：阶段产物如何保存；
8. `__main__.py`：如何组装和运行。

## 文件职责

### `__init__.py`

模块公共出口。轻量 contracts、Manager 和 Mock 工具直接导出；`RealV2Tools` 延迟导入，避免运行 Mock 测试时加载 Polars、OpenAI 客户端和因子库。

### `__main__.py`

CLI 入口，负责参数、`.env`、工具实例、Artifact Repository、SQLite checkpoint 和 Manager 组装。

| 模式 | 假设上限/方向 | 实验上限/假设 | 用途 |
|---|---:|---:|---|
| `mock` | 固定模拟 | 固定模拟 | 无 API 回归测试 |
| `real-smoke` | 1 | 2 | 低成本真实链路 |
| `real-full` | 4 | 5 | 正式 V2.1 实验 |

### `contracts.py`

Pydantic 数据契约：

- `ResearchIdea`：抽象金融洞察；
- `InsightAnalysis`：支持、反驳、替代解释及 ACCEPT/REVISE/REPLACE；
- `DevelopedHypothesis`：机制、反驳、证伪标准和预测周期；
- `MeasurementProxy` / `MeasurementPlan`：机制到可观测代理的映射；
- `FactorFormulaSpecV2`：公式 AST、字段和历史长度；
- `FactorExperimentSpec`：Measurement Plan 谱系、实验类型和公式；
- `HypothesisCheckResult` / `PreflightResult`：确定性检查结果；
- `SplitMetrics`：训练、验证和测试指标；
- `ArtifactRef`：Artifact 路径、校验和和数量。

### `state.py`

定义 `V2Stage` 和轻量 `V2WorkflowState`。V2.1 新增 `DESIGN_MEASUREMENT_PLANS`。State 只保存运行状态、Artifact 引用、修正机会、候选 ID、事件和错误，不保存大型候选正文。

### `manager.py`

`build_manager()` 创建固定 LangGraph：

```text
Research → Critical Review → Hypothesis → Check/Repair
→ Measurement Plan → FactorSpec → Preflight/Repair
→ Train → Validation → Test → Memory → Record
```

Manager 只编排和路由，不调用 LLM 自由决策。修正节点从 Checker Artifact 读取实际 `CheckIssue`，Measurement Plan 同时传给首次 Preflight 和修正 Preflight。

### `tools.py`

`V2Tools` Protocol 定义 Manager 所需方法。`MockV2Tools` 提供确定性实现，覆盖：

- 四方向成功；
- 一次假设修正；
- Measurement Plan；
- Baseline 与 Full Mechanism；
- 一次实验修正；
- 训练、验证、测试筛选。

### `real_tools.py`

真实工具链：

- DeepSeek 生成四路 Research Idea；
- 学校 OpenAI 执行 Critical Review 和 Hypothesis Expansion；
- DeepSeek 生成 Measurement Plan 和 FactorSpec；
- Hypothesis Checker 检查概念、泄漏和完全重复；
- Preflight 检查 Measurement Plan 谱系、DSL、真实字段、样本容量和公式执行；
- 修正 Agent 接收真实问题并最多修正一次；
- JSON mode、Pydantic 错误反馈、空响应重试和 FactorSpec cache。

### `real_backtest.py`

真实单因子评估：历史时点 S&P 500、日频 Rank IC、训练方向冻结、Newey–West、BH-FDR、Top 20 重叠 cohort、10 bps 成本及组合指标。

固定分块：训练 2001–2010，验证 2011–2012，测试 2013–2015，2016–2025 保留。

### `artifacts.py`

JSON/Parquet 原子写入、SHA-256、路径限制和读取校验。运行目录增加 `measurement_plans.json`。

### `README.md`

面向使用者的命令、模式、恢复边界和测试说明。

## Checkpoint 与恢复

SQLite checkpoint 在节点完成后保存 State：

- 已完成节点不会重复执行；
- 当前未完成节点从开头重跑；
- 已缓存的 FactorSpec 可复用；
- 回测节点中断时重跑该回测节点。

V2.1 新运行恢复命令：

```bash
conda run -n autoalpha-agent python -m hypothesis_workflow_v2 \
  --mode real-full \
  --env-file ../.env \
  --run-id <V2.1-run-id> \
  --resume
```

V2.0 Run `70778337-b7c6-4a41-a250-dc62a0748441` 缺少 V2.1 必填契约和 Measurement Plan，只用于历史复盘，不从 V2.1 代码恢复。

## 主要 Artifact

```text
.runtime/hypothesis_workflow_v2_real_full/<run-id>/
├── research_context.json
├── research_ideas.json
├── insight_analyses.json
├── hypotheses.json
├── hypothesis_checks.json
├── measurement_plans.json
├── experiment_specs.json
├── preflight_results.json
├── train_metrics.json
├── validation_metrics.json
├── test_metrics.json
├── memory_summary.json
└── run_summary.json
```
