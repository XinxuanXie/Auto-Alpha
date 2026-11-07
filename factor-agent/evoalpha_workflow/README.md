# EvoAlpha Workflow

本目录承载独立的 EvoAlpha 式单因子进化工作流，不修改或替换 `agentic_workflow/`。

## 边界

```text
agentic_workflow/
  原 Manager、Research Agent、Implementation Agent 和 Memory 工作流

evoalpha_workflow/
  新的种子评价、父代选择、Refinement、Crossover、Mutation 和 Population Update

factor_pipeline/
  两条工作流共享的 FactorSpec、Polars Compiler、数据清洗、单因子评价和回测基础设施
```

旧工作流和 EvoAlpha 工作流拥有独立入口、状态和测试。共享层只提供确定性基础能力，不包含研究路线决策。

## 当前模块

```text
evoalpha_workflow/
├── config.py
├── state.py
├── tools.py
├── workflow.py
├── seed_evaluation.py
├── run_seed_evaluation.py
├── similarity.py
├── run_similarity.py
├── parent_selection.py
├── run_parent_selection.py
├── refinement.py
├── run_refinement_mock.py
├── crossover.py
├── run_crossover_mock.py
├── candidate_evaluation.py
├── run_candidate_evaluation.py
├── llm_client.py
├── population.py
├── run_generation_one_real.py
└── run_three_generation_deepseek.py
```

当前已实现真实 Alpha158 Seed Evaluation、父代选择、真实 LLM
Refinement/Crossover/Mutation、统一训练期评价、相关性过滤和 Top 30 种群更新。

## 固定主循环

```text
Seed Evaluation
→ Select Top 30
→ [Proposal → Evaluation → Population Update] × 3 generations
→ Validation of final Top 30
→ Select validation Top 10
→ Final Test once
```

每代协议固定为：

```text
12 Refinement
+ 8 Crossover intermediates
+ 8 Mutations
= 28 evaluated offspring
```

`workflow.py` 只负责编排、验证数量与谱系、更新种群和记录事件。具体实现通过 `tools.py` 的接口注入，后续可将 Refinement、Crossover 和 Mutation 替换为 Agent，而不改变外层循环。

## 运行 Alpha158 Seed Evaluation

在仓库根目录执行：

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_seed_evaluation
```

默认设置：

- 2000 年作为 warm-up；
- 2001 年作为训练与初始种子评分区间；
- 2002 年不进入 Seed Evaluation；
- 输入为历史时点成分股面板；
- 评价 157 个非 VWAP Alpha158；
- 输出逐因子指标、固定评分参考和初始 Top 30。

## 运行 Top 30 相关性分析

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_similarity
```

相关性定义为每日横截面 Spearman 绝对值的训练期中位数。默认将相似度高于 `0.80` 的组合排除出双父代候选池，同时保留有符号相关性用于解释。

## 生成父代计划

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_parent_selection
```

Refinement 使用综合分数加权且不放回抽样，每代选择 12 个不同父代。Crossover 只使用相似度不高于 `0.80` 的组合，并按以下固定优先分数排序：

```text
Priority
= 70% × mean(parent composite scores)
+ 30% × (1 - pair similarity)
```

每个父代每代最多参与两次 Crossover，固定随机种子为 `random_seed + generation`。

## 验证 Refinement Agent 接口

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_refinement_mock
```

当前 Mock Client 只用于验证结构化接口：

- 输入父代 FormulaSpec、训练指标和固定任务；
- 输出诊断、单一修改类型、预期指标变化和新 FormulaSpec；
- 检查任务 ID、父子谱系、公式深度、lookback、方向和组合规则；
- Window 修改必须只改变一个窗口；
- Normalization 必须用 `rank` 或 `zscore` 包裹完整父代公式；
- 最终转换为标准 `CandidateProposal`。

Mock 子代不进入正式种群。后续真实 API Client 复用相同 Prompt、输出模型和确定性检查器。

## 验证 Crossover 与 Mutation 接口

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_crossover_mock
```

每个 Crossover 任务依次产生：

```text
两个定向父代
→ Crossover intermediate
→ 单步 Mutation
```

父代训练方向会显式写入公式，方向为 `-1` 时先增加 `negate`。Crossover 检查器要求中间因子完整使用两个定向父代公式，并限制公式深度、数据字段、lookback 和投资组合规则。Mutation 复用单一修改检查器。

Mock Crossover 使用两个定向父代横截面标准化后的等权和；该公式仅用于验证接口，不代表正式的交叉研究策略。

## 统一评价第 1 代候选

```bash
PYTHONPATH=factor-agent python -m evoalpha_workflow.run_candidate_evaluation
```

统一 Candidate Evaluator 对 12 个 Refinement、8 个 Crossover intermediate 和 8 个 Mutation 分别运行 2001 年训练期单因子评价。评分始终复用初始 157 个 Alpha158 建立的固定百分位参考，不根据子代重新拟合评分尺度。

每个候选保存 Mean Rank IC、ICIR、月度稳定性、跨周期一致性、综合分数，以及相对直接父代的指标变化。Mutation 同时保存相对 Crossover intermediate 和两个原始父代的变化。

## 运行真实 LLM 第 1 代

在仓库根目录的 `.env` 配置以下变量：

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=学校提供的 OpenAI-compatible endpoint
OPENAI_MODEL=学校接口支持的模型名
```

然后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=factor-agent \
python -m evoalpha_workflow.run_generation_one_real --max-attempts 5
```

运行器按候选保存检查点，可在 API 或校验异常后从已完成任务继续。模型只生成受限 DSL 公式和研究解释；`required_fields` 与 `lookback_days` 由程序根据公式自动推导。所有响应依次通过 Pydantic 结构验证、公式语义检查、训练期单因子评价和相似度过滤。

第 1 代真实运行结果位于 `results/evoalpha_generation_01_real/`。本次 28 个候选均成功执行，10 个通过相似度准入，最终 Top 30 保留 4 个新因子。本结果仅使用 2001 年训练期，尚未使用 2002 年验证/测试数据。

## 运行单因子样本外验证

冻结训练期公式和方向，在 2002 年运行预测指标与 Top 20 周频回测：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=factor-agent \
python -m evoalpha_workflow.run_single_factor_validation
```

默认验证 `G01-CROSS-004-MUTATION`，不进行参数调整或候选重选。结果写入 `results/evoalpha_validation_2002/`，包括 JSON 汇总和逐周 Parquet 明细。

## 使用 DeepSeek 运行三代真实流程

在仓库根目录的 `.env` 配置 `DEEPSEEK_API_KEY` 后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=factor-agent \
python -m evoalpha_workflow.run_three_generation_deepseek
```

默认协议固定为 2001–2010 训练、2011–2012 冻结验证、三代各 28 个候选。
Refinement 与 Mutation 使用 `deepseek-flash`，Crossover 使用高推理强度的
`deepseek-v4-pro`。运行器逐任务、逐阶段保存 checkpoint，恢复时复用已完成的
种子评价、Agent 输出、候选回测、相关性和种群结果。

真实结果位于 `results/evoalpha_deepseek_2001_2012_3gen/`。该目录中的 README
记录三代准入、验证 Top 10、指标与公式重复审计结论。

## 运行原始 Alpha158 长期基准

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=factor-agent \
python -m evoalpha_workflow.run_alpha158_benchmark
```

默认使用本地全部 2000–2026 数据，对 157 个可执行 Alpha158 输出逐年以及按年滚动的
5/10/20 年指标。每个因子只计算一次全历史日度 IC 序列和正反两个方向的周频持仓，
各窗口由这些基础序列聚合，避免重复扫描约 12,000 次完整面板。运行器按因子保存
checkpoint，并校验标签缓存对应的数据文件身份。

结果位于 `results/alpha158_full_history_benchmark/`，包含完整 Parquet/CSV、四类拆分表、
逐窗口汇总、逐因子稳定性汇总和运行说明。
