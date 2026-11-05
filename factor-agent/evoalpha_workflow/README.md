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

## 计划模块

```text
evoalpha_workflow/
├── config.py
├── state.py
├── seed_evaluation.py
├── parent_selection.py
├── refinement.py
├── crossover.py
├── mutation.py
├── population.py
├── record.py
└── workflow.py
```

当前已建立独立包和固定实验配置。后续模块逐步实现，并为每一步增加独立测试。
