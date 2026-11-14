# Two-Stage Research Pipeline

> 历史阶段文档：记录 Agent Workflow V1 的两阶段研究实现。当前假设深化方案见 [`../hypothesis_workflow_v2/design.md`](../hypothesis_workflow_v2/design.md)。

每个固定方向的 Specialist Research Pipeline 拆分为两个独立模块：

```text
Factor Knowledge Base + Web Sources + Experiment Memory
                         ↓
              Financial Intuition Agent
                         ↓
             Hypothesis Formulation Agent
                         ↓
          Checker → Implementation → Backtest
```

## Financial Intuition Agent

该模块负责形成金融机制，不直接生成公式或最终假设。输出包括：

- 核心机制与因果链；
- 适用的市场条件；
- 预期横截面模式；
- 2–4 个因子组件；
- 每个组件的角色：主信号、条件、确认或风险控制；
- 可能推翻该直觉的风险；
- 实际引用的网页资料与基础因子。

## Hypothesis Formulation Agent

该模块将金融直觉转换为一个五日横截面假设：

- 只使用当前 OHLCV 与市值数据；
- `required_concepts` 与 `factor_structure` 完全一致；
- 只能使用金融直觉已经提出的概念；
- 明确每个因子组件的角色和高低/变化关系；
- 不生成公式，公式仍由 Implementation Agent 负责。

选中的 `financial_intuition` 会写入 WorkflowState、Checkpoint 和 SQLite Research Record。这样可以分别诊断金融机制、假设表达和公式实现三个层次的问题。

每轮四个 Specialist 的候选通过确定性检查后，会分别进入 Implementation Agent、真实回测和 Evaluation Agent。Manager 在四组完整实验结果生成后，根据验收状态、Rank IC、净收益、换手率、回撤和经济逻辑选择本轮主因子。每个因子的回测产物按 `factor_id` 写入独立目录，避免候选之间覆盖。
