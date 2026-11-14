# Agentic Factor Research：可尝试改进方向

> 历史调研文档：保留 Sumus 与 EvoAlpha 的方法分析。当前采用的假设生成方案见 [`../hypothesis_workflow_v2/design.md`](../hypothesis_workflow_v2/design.md)。

本文档用于持续记录外部平台、论文和实验带来的可实施想法。所有内容先作为研究候选，不直接进入主流程；只有完成最小实验并证明有效后，才考虑正式集成。

> EvoAlpha 的正式项目改造决策已经收敛，后续实现以
> [`evoalpha_single_factor_design.md`](evoalpha_single_factor_design.md) 为准。
> 本文中的 EvoAlpha 内容保留为外部方法调研背景，其中“继续开放式生成新因子”、
> “8 个种子”和旧数据切分等建议不再代表当前实施方案。

## 候选方向总览

| 来源 | 核心启发 | 与当前系统的关系 | 建议优先级 |
|---|---|---|---|
| Sumus | 角色化 Agent、动态组队、多模型复核、人类共享工作空间 | 改善 Manager 调度与协作方式 | 中 |
| EvoAlpha | 以候选池、Mutation、Crossover 和反馈选择代替单轮自由生成 | 直接改善因子搜索与迭代机制 | 高 |

---

## 方向一：借鉴 Sumus 的动态 Agent 团队

### 平台定位

[Sumus](https://sumus.im/) 是一个多人和多个 AI Agent 共用同一上下文的协作平台。一个 Agent 可以接管任务、拆分子任务、调用其他 Agent、并行执行并汇总结果；不同 Agent 可以使用不同模型。官方资料还强调云端 Agent 负责研究规划、本地 Codex 或 Claude Code 负责实现。

### 值得借鉴的设计

1. **Manager 作为团队负责人**：Manager 不只在回测后选择动作，也可以根据当前失败类型动态调用 Literature Researcher、Factor Collision Analyst、Hypothesis Critic、Formula Reviewer 或 Backtest Diagnostician。
2. **角色与模型分离**：角色定义任务职责，模型是可替换执行器。同一角色可以使用低成本模型生成、高能力模型复核。
3. **独立交叉复核**：生成者与 Reviewer 使用不同上下文或模型，降低单一模型自我确认的风险。
4. **共享研究对象**：协作围绕结构化的 Research Brief、Financial Intuition、Hypothesis、FactorSpec、Backtest Evidence 和 Critique 展开，而不是依赖无限增长的群聊记录。
5. **人类参与接口**：让研究者能够查看 Agent 调用原因、淘汰依据和 Manager 决策，并在关键节点补充约束或纠正机制。

### 不应直接照搬的部分

- 多个 Agent 相互同意不能替代真实数据验证；
- 所有 Agent 共享完整上下文可能导致研究方向逐渐同质化；
- 自由群聊不利于实验复现和状态审计；
- 通用任务委派没有解决量化研究中的泄漏、样本隔离和统一回测问题。

### 对本项目的正确组合

```text
Sumus 式角色协作与动态调度
                +
现有 LangGraph、FactorSpec、确定性 Checker、真实回测与 SQLite Memory
```

当前不需要重写工作流。后续可以先让 Manager 根据失败类型选择 Reviewer Tool，验证动态组队是否比固定调用全部 Agent 更高效。

参考资料：[Sumus About](https://sumus.im/about)、[Quick Start](https://sumus.im/docs)、[Multi-model](https://sumus.im/features/multi-model)。

---

## 方向二：EvoAlpha 的 LLM 引导进化搜索

### 文献说明

用户提到的论文为 **“EvoAlpha: An LLM-Enhanced Evolutionary Framework for Formulaic Alpha Mining”**，收录于 ICASSP 2026，DOI 为 `10.1109/ICASSP55912.2026.11463591`。目前可检索到会议与书目信息，但完整正文没有公开获取。

同时存在一篇公开全文、方法高度相关的论文 **“EvoAlpha: Evolutionary Alpha Factor Discovery with Large Language Models”**。以下详细方法和实验数字主要依据这篇公开版本，因此不能直接视为 ICASSP 论文的逐字复述。

公开版本：[OpenReview PDF](https://openreview.net/pdf?id=ALpLmURYWy)。

### 它试图解决什么问题

一次性让 LLM 自由生成因子存在三个问题：

1. 每轮候选之间缺少继承关系，成功经验不能稳定保留；
2. LLM 容易生成非法、重复或只有表面经济意义的公式；
3. 系统没有清晰机制平衡局部改进与新方向探索。

EvoAlpha 将因子挖掘重新定义为一个进化搜索问题：

```text
初始因子池
    ↓
选择表现较好的 Parent Factors
    ↓
LLM Mutation / LLM Crossover
    ↓
语法、执行与重复检查
    ↓
计算 IC、Rank IC、ICIR 等指标
    ↓
保留 Top-K，进入下一代
    ↺
```

核心变化是：**LLM 不再直接负责寻找最终答案，而是作为具有金融语义的进化算子。**

### 因子的表示方式

每个因子被表示为计算树：

- 叶节点是 `open`、`high`、`low`、`close`、`volume` 等数据字段；
- 内部节点是收益、滚动统计、排名、四则运算等算子；
- 完整表达式是可执行的 Qlib 风格公式。

这与我们的 FactorSpec JSON AST 基本对应。因此 EvoAlpha 的 Mutation 和 Crossover 可以直接在结构化 AST 层实现，而不需要允许 Agent 生成任意 Python 代码。

### Mutation：局部改进

Mutation 从一个较好的 Parent Factor 出发，对局部结构做修改，例如：

- 将 10 日窗口改成 20 日；
- 替换标准化方式；
- 增加波动率或成交量稳定项；
- 替换计算树中的一个节点；
- 改变平滑、排名或风险调整结构。

它承担的是 exploitation：围绕已经出现正向信号的结构继续改进。

### Crossover：组合不同结构

Crossover 从两个或多个 Parent Factors 中提取可复用子结构，再组合成新因子，例如：

```text
Parent A：价格动量核心
Parent B：成交量或波动率调整项
Child：风险调整后的动量表达式
```

它承担的是 exploration：把不同因子家族的结构组合起来，探索新的区域。

论文允许两种 Agent 设计：

- Mutation Agent 与 Crossover Agent 分离；
- 使用统一 Agent，根据候选池和当前状态自行选择操作。

### Prompt 中包含什么

EvoAlpha 不是只给 LLM 一句“生成有效因子”，而是提供三类结构化信息：

1. 当前轮次、允许字段、算子和金融约束；
2. Top Parent Factors 的名称、表达式和匿名化表现摘要；
3. 本轮执行 Mutation 或 Crossover 的具体规则。

输出采用固定 JSON，包括候选名称、表达式和修改理由。这种约束的目的不是限制研究思想，而是让 LLM 在一个可执行搜索空间内操作。

### 候选过滤

生成结果进入多级过滤：

1. JSON 与表达式能否解析；
2. 字段和算子是否合法；
3. 括号、窗口和除数稳定项是否合法；
4. 是否与已有表达式完全重复；
5. 是否能在时限内执行；
6. 缺失值或 NaN 比例是否过高；
7. 候选是否覆盖不同因子角色，而不是集中在近似公式。

这与我们当前的 Pydantic、DSL Compiler 和执行检查非常兼容。我们还可以进一步加入相关性去重，而不只检查表达式完全相同。

### 公开版本的实验设置

#### 静态因子搜索

- 初始种子来自 Qlib Alpha158；
- 实验使用其中 38 个 Price 与 Rolling 因子；
- 候选池规模为 30；
- 进化 15 轮；
- Crossover 比例 0.7，Mutation 比例 0.3；
- CSI300 样本内区间为 2023–2024，样本外区间为 2024–2025；
- 评价指标包括 IC、Rank IC、ICIR 及其 Rank 版本。

论文报告候选池平均 IC、Rank IC 和 ICIR 随轮次上升，样本外 IC 中位数提高且分布逐渐收窄。展示的 Top Factor IC 从早期约 `0.026` 提升到第 15 轮约 `0.043`。

#### 稀疏投资组合实验

- 市场包括 US50、HSI45 与 CSI300；
- 选择 Top 10 或 Top 15 资产形成稀疏组合；
- 比较 1/N、Min-CVaR、Max-Sharpe、LGBM、XGBoost、mSSRM-PGA 和 ASMCVaR；
- 使用 GPT-4.1 与 DeepSeek-V3 执行 LLM 引导搜索；
- 评价累计财富、Sharpe 和最大回撤。

论文报告 LLM 方案在三个市场的大部分设置中取得更高累计财富和风险调整表现。例如 US50、`m=10` 时，DeepSeek-V3 的累计财富为 `25.1`，对比 ASMCVaR 的 `10.3`。

### 真正的创新点

1. **Population，而不是单候选链条**：系统同时维护多个候选，降低一次失败让整条研究路径失效的风险。
2. **Lineage**：新因子有明确 Parent、Mutation/Crossover 操作和历史指标，可以追踪有效结构来自哪里。
3. **LLM 是语义进化算子**：传统 Genetic Programming 随机替换节点，LLM 可以结合金融语义决定如何修改。
4. **反馈进入下一代搜索**：评价结果不是仅用于最终接受或拒绝，而是直接改变下一轮候选生成。
5. **探索与利用平衡**：Mutation 改进已有结构，Crossover 探索不同结构组合。

### 需要谨慎看待的地方

1. **论文结果不能直接迁移到美股 S&P 500。** 数据、市场、时期、标签与交易设置均不同。
2. **进化搜索容易过拟合样本内指标。** 15 轮持续选择 Top IC，本质上进行了大量多重试验，需要独立验证集、锁定测试集和多重检验控制。
3. **样本区间较短。** 2023–2025 很难覆盖完整市场周期，公开版本中的样本内与样本外年份还存在边界重叠表述，需要进一步核对原始数据切分实现。
4. **单目标 Top-K 会导致 IC 投机。** 只按 IC 选择可能产生高换手、不稳定、复杂或与已有因子高度相关的表达式。
5. **公式可能逐代膨胀。** Crossover 会让计算树越来越复杂，需要 Complexity Penalty 和深度限制。
6. **重复不能只按字符串判断。** 不同公式可能在数值上高度相关，需要表达式规范化和因子相关性过滤。
7. **LLM 和 Prompt 敏感性较强。** 不同模型、温度、提示示例和随机种子可能产生明显不同的搜索轨迹。
8. **成本较高。** 大候选池、多轮 LLM 调用和真实回测会显著增加 API 与计算成本。
9. **经济解释可能退化。** 进化过程可能逐渐只追求指标，让最初的金融假设失去语义一致性。

### 与我们当前流程的关键差异

当前系统：

```text
四个 Research Agents 各生成一个全新假设
    → 四个因子回测
    → Manager 选择相对最优者
    → 下一轮通常重新生成
```

EvoAlpha：

```text
维护候选因子池
    → 选择表现与多样性较好的 Parents
    → Mutation / Crossover 产生下一代
    → 评价并更新候选池
    → 多代进化
```

我们目前已经有 Research Agents、FactorSpec、DSL、真实回测、Evaluation 和 Memory，缺少的主要是：

- Candidate Population；
- Parent Selection；
- Mutation/Crossover 操作；
- Lineage Record；
- 面向多代搜索的 Fitness 与 Diversity 机制。

### 最适合我们的组合方案

不建议用 EvoAlpha 完全替换当前 Research Agent。更合理的是建立两层搜索：

```text
Research Agents
    ↓
提出不同经济机制与 Seed Factor Families
    ↓
EvoAlpha-style Evolution Engine
    ↓
在每个机制内部 Mutation，并在互补机制之间谨慎 Crossover
    ↓
统一 Evaluation + Manager
```

Research Agent 负责“研究什么”，Evolution Engine 负责“如何在可执行公式空间中高效搜索”。这样既保留金融直觉，也避免每一轮都从零开始。

### 建议的最小实验

先做一个独立实验分支，不立即改变主工作流：

1. 从现有四个方向各选 2 个合法因子，共 8 个 Seed Factors；
2. 维护 8–12 个候选的小型 Population；
3. 运行 3 个 Generation；
4. 每代生成 4 个 Mutation 和 2 个 Crossover；
5. 全部候选继续使用当前 FactorSpec、Polars 回测和固定交易口径；
6. Parent Selection 同时考虑：
   - Rank IC；
   - 分时期稳定性；
   - 成本后收益；
   - 换手率；
   - 与现有候选的相关性；
   - 公式复杂度；
7. 记录每个 Child 的 Parent IDs、操作类型、修改理由和指标变化；
8. 只使用训练期和研究验证期进化，保持 2020–2026 测试集锁定。

建议使用多目标 Fitness，而不是单一 Rank IC：

```text
fitness = predictive_power
        + temporal_stability
        + novelty
        + tradability
        - turnover_penalty
        - complexity_penalty
```

### 当前判断

EvoAlpha 对本项目的价值高于继续无限扩展 Research Prompt。它提供的不是另一个生成提示词，而是一套让成功结构能够继承、失败结构能够被淘汰、不同结构能够组合的搜索机制。

建议优先完成“小候选池 + 三代进化 + 多目标评价”的离线实验，再决定是否将 Evolution Engine 接入外层 Manager。
