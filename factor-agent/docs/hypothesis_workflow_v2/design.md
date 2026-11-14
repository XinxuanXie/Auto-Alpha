# Hypothesis Workflow V2.1 设计

> 当前版本：2026-11-14。V2.1 直接升级现有 `hypothesis_workflow_v2`，保留 V1、EvoAlpha 和 V2.0 历史运行产物。

## 目标

V2.0 已证明四路 Research、LLM 假设深化、FactorSpec、真实回测和 checkpoint 可以完整运行。正式 Run `70778337-b7c6-4a41-a250-dc62a0748441` 共生成 14 个假设和 84 个实验，训练阶段有 10 个候选通过，验证阶段 0 个通过。

日志显示主要瓶颈位于假设到公式的语义传递：

- Development Agent 倾向顺着 Research Idea 扩写，缺少反驳和替代解释；
- 同一方向的假设经常只是周期、状态或分组变化，经济机制高度相似；
- 丰富的金融机制在 FactorSpec 阶段被压缩成简单技术指标；
- 修正模块没有稳定接收 Checker 的真实错误信息。

V2.1 因此把主链改为：

```text
Research Idea
→ Critical Development Review
→ Distinct Hypotheses
→ Deterministic Hypothesis Check
→ Measurement Plan
→ Baseline / Full / Ablation Experiments
→ Deterministic Preflight
→ Train → Validation → Test
→ Memory → Record
```

## 四个固定研究方向

每轮固定运行四个相互独立的 Research Agent，每个方向输出一个抽象金融洞察。

| 方向 | 固定研究范围 |
|---|---|
| Momentum | 信息扩散、趋势延续与拥挤过滤 |
| Reversal | 过度反应、价格冲击与卖压衰竭 |
| Liquidity–Volatility | 流动性变化、风险补偿与波动状态 |
| Price–Volume | 价格与成交量的确认、背离和非线性交互 |

Research Agent 使用 DeepSeek，接收 OHLCV/市值边界和完整因子定义库。它只讨论市场现象、经济或行为机制、横截面收益差异及成立条件，不生成字段、窗口、公式或回测结论。

## 批判性 Hypothesis Development

同一个学校 OpenAI Agent 分两步处理每个 Research Idea。

### 1. Critical Development Review

Agent 必须同时给出：

- `supporting_arguments`：支持机制；
- `objections`：最强反对理由；
- `alternative_explanations`：竞争解释；
- `decision`：`ACCEPT`、`REVISE` 或 `REPLACE`；
- `final_research_idea`：批判后唯一保留的研究洞察。

`REPLACE` 允许否定原 Idea，但新 Idea 必须留在同一研究方向。该阶段保持纯金融分析，不讨论数据字段、DSL 或公式。

### 2. Hypothesis Expansion

Agent 只读取 `final_research_idea`，生成 1–4 个假设。允许仅输出一个高质量假设。

每个假设必须包含：

- claim 与 economic rationale；
- core mechanism；
- strongest counterargument；
- falsification criteria；
- 与同批其他假设的实质区别；
- 预测周期和所需研究概念。

不同假设必须代表不同经济机制或竞争解释。更换预测周期、市场状态、流动性分组或规模分组不构成独立假设。

## Hypothesis Checker 与一次修正

确定性 Checker 检查不支持概念、未来信息和本轮完全重复。失败项只允许修正一次。修正 Agent 接收对应假设的结构化 `CheckIssue`，已通过项保持不变。

## Measurement Plan

V2.1 在假设与公式之间新增测量设计层，防止抽象机制被直接压缩为方便计算的技术指标。

每个 `MeasurementPlan` 包含：

- 机制摘要；
- 一个或多个稳定编号的代理变量 `P1`、`P2`……；
- 每个代理对应的机制组件、可观测定义、所需字段和局限；
- 最简 baseline 的测量逻辑；
- 完整机制的测量逻辑；
- 当前 OHLCV/市值无法忠实观测的部分。

Measurement Planning Agent 使用 DeepSeek，只能使用当前真实字段，并必须承认代理变量的测量误差。

## FactorSpec 实验族

FactorSpec Agent 根据一个 Hypothesis 和对应 Measurement Plan 生成 2–5 个实验：

1. 恰当的 `baseline`；
2. 覆盖全部代理的 `full_mechanism`；
3. 可选的 1–3 个 `ablation` 或 `robustness`。

每个实验显式记录：

- `measurement_plan_id`；
- `variant_type`；
- `represented_proxy_ids`；
- 公式 AST、预测周期和变化理由。

系统验证公式真实引用的字段覆盖其声明代理需要的字段；`full_mechanism` 必须覆盖计划中的全部代理。仅改变预测周期不能构成新实验。

## Preflight 与一次修正

Preflight 对首次生成和修正后的公式执行同一套确定性检查：

- Measurement Plan、Hypothesis 和 Experiment 的 ID 谱系一致；
- 代理编号属于对应计划；
- Full Mechanism 覆盖全部代理；
- 公式字段覆盖声明代理所需字段；
- DSL 结构、字段、窗口和参数合法；
- 公式可在真实样本上编译并产生有限值；
- 样本容量满足预测周期；
- 本轮不存在公式与周期完全相同的实验；
- 每个假设最多 5 个实验。

失败实验只修正一次。修正 Agent 接收真实 `CheckIssue`、对应 Measurement Plan 以及同假设已经通过的公式，针对错误进行修正。

## Manager 的固定职责

LangGraph Manager 负责固定编排和失败路由，不使用 LLM 自由选择研究方向、修改假设或设计公式。

```text
START
→ Load Context
→ 4 × Research
→ Critical Review
→ Hypothesis Expansion
→ Hypothesis Check / Repair once
→ Measurement Planning
→ FactorSpec Generation
→ Preflight / Repair once
→ Train Filter
→ Validation Filter
→ Test Report
→ Memory
→ Record
→ END
```

四个研究方向全部失败、所有假设失效或所有实验 Preflight 失败时进入 `FAILED`。训练或验证无候选通过属于正常研究结果，流程仍写入 Memory 和 Run Record 后进入 `COMPLETED`。

## 数据与评价协议

```text
Warm-up：2000
训练集：2001–2010
验证集：2011–2012
测试集：2013–2015
长期保留集：2016–2025
```

- 历史时点 S&P 500 成分股；
- 日频横截面 Rank IC；
- 因子方向只在训练集确定，随后冻结；
- Mean Rank IC > 0.02 且成本后净收益 > 0 才进入下一分块；
- 测试集只报告，不再筛选或调参；
- Top 20、下一交易日开盘进入、持有 `h` 日、重叠 cohort、每个 cohort 使用 `1/h` 资金；
- 交易成本 10 bps；
- Newey–West 与 BH-FDR 用于报告，不进入首版 PASS 门槛。

## 规模与模型

正式模式的理论上限：

```text
4 个方向 × 每方向 1 个 Idea × 每 Idea 1–4 个 Hypothesis
× 每 Hypothesis 2–5 个 Experiment
= 每轮 8–80 个实验
```

- Research：低成本 DeepSeek；
- Critical Review 与 Hypothesis Expansion：学校 OpenAI；
- Measurement Plan 与 FactorSpec：低成本 DeepSeek；
- Checker、回测、筛选、统计、记录和 Manager 路由：确定性代码。

## 版本兼容

V2.1 增加了必填契约字段和 `measurement_plans.json`。V2.0 的 Artifact 保留用于复盘，但不能使用 V2.1 代码从旧 checkpoint 继续运行。V2.1 真实实验必须创建新的 `run_id`。

V1 与 EvoAlpha 仍是独立对照工作流，没有被 V2.1 覆盖。
