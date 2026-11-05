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
│
│   已实现确定性外层循环、三代协议、真实 Alpha158 Seed Evaluation
│
└── 后续模块
    ├── parent_selection.py
    ├── refinement.py
    ├── crossover.py
    ├── mutation.py
    ├── population.py
    └── record.py
```

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
