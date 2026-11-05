# EvoAlpha 单因子进化改造设计

## 文档目的

本文档记录当前确定的 EvoAlpha 式单因子进化方案、已经采用的简化决策、尚未实现的工程事项和后续升级方向。它是后续实现的设计基线；现有代码仍以原 Manager 工作流和单因子 `factor_pipeline` 为准。

## 核心原则

本项目始终以**单因子直接评价**为主要研究口径：

```text
一个公式因子
→ 逐日横截面因子值
→ 未来收益 Rank IC / ICIR
→ 单因子 Top-K 组合诊断
```

`backtest/run_backtest_alpha158.py` 中的 LightGBM 是历史库级模型基线，用于检查整套 Alpha158 特征是否能够进入 Qlib 模型，不用于：

- 选择 EvoAlpha 初始 Top 30；
- 评价单个父代或子代；
- 计算父子指标增量；
- 更新进化种群。

新的 EvoAlpha 流程必须逐个计算和评价因子，不能用 LightGBM 特征重要性替代单因子表现。

## 保留的现有结构

现有系统不重写，继续保留：

- 外层 Manager 与 LangGraph 状态机；
- `Hypothesis Checker`；
- `FactorSpec` 与受限 JSON AST；
- Polars DSL Compiler；
- 历史时点 S&P 500 成分股数据接口；
- 单因子回测与 Evaluation；
- SQLite Research Record、Checkpoint 和 Memory。

Manager 不自主决定采用哪种研究路线。每轮任务由固定配置预先指定，Manager 只负责调用对应 Tool、维护状态、处理失败和保存记录。

## 候选来源

停止将“完全从零生成新因子”作为正式候选来源。只保留：

1. **单父代改进（Refinement）**：读取一个已有因子、指标和弱点，执行一次主要修改；
2. **双父代交叉（Crossover + Mutation）**：组合两个父代，保留交叉中间因子，再执行一次变异得到最终子代。

现有 Research Agent 的客户端、Context 构建、结构化输出和 Memory 读取能力继续复用，其任务改为基于父代证据进行诊断和改进。第一版使用通用 Refinement Agent，不建立动量、成交量等分类 Skill。

## 初始种子池

### 数据范围

初始来源同时包括 Alpha101 与 Alpha158，不加入此前 Agent 生成和回测过的候选因子。

当前日频面板提供：

```text
open, high, low, close, volume, mktcap
```

第一版排除：

- 需要真实 VWAP 的公式；
- 需要历史时点行业分类的公式；
- 任何依赖当前数据无法安全重建的字段。

真实 VWAP 需要日内价格与对应成交量，或成交金额除以成交股数。当前 OHLCV 不能精确恢复。第一版不使用 Typical Price 静默冒充 VWAP。

当前预计可进入实现阶段的候选为：

```text
Alpha101：53 个
Alpha158：157 个（排除 VWAP0）
合计：210 个
```

### 当前代码缺口

`agentic_workflow/factor_library.py` 目前主要是 Research Agent 的知识库，不是可执行公式导入器。

当前 DSL 仅覆盖基础字段、常量、滞后、收益、均值、标准差、横截面排名/标准化、四则运算、比较和条件。完整导入 Alpha101/158 仍需：

- Canonical Factor AST；
- Alpha101 公式 Parser；
- Alpha158/Qlib 表达式 Parser；
- `cs_rank` 与 `ts_rank` 的明确区分；
- rolling sum/min/max/rank/argmin/argmax/quantile；
- correlation、covariance、product、delta、decay linear；
- abs、sign、log、signed power、scale；
- slope、R-square、regression residual；
- 逻辑与 elementwise min/max；
- 与 Qlib Alpha158 的数值一致性验证。

## 数据切分与预测周期

第一阶段使用数据中最早的两年作为评价区间，并额外读取此前一年作为因子计算 warm-up：

```text
Warm-up：2000-01-03 ～ 2000-12-31（只计算历史窗口，不评价）
训练：2001-01-01 ～ 2001-12-31
验证：2002-01-01 ～ 2002-06-30
测试：2002-07-01 ～ 2002-12-31
```

规则：

- 因子计算可以读取训练开始前的历史数据作为 warm-up；
- 评价日期必须位于对应区间；
- 未来收益终点必须完整落在对应区间内；
- 测试集只在最终候选确定后运行，不参与公式修改或种群更新。

预测周期：

- 20 日未来收益为主要目标；
- 5 日和 10 日只用于跨周期一致性诊断；
- 因子方向只根据训练集 20 日 Mean Rank IC 确定；
- Mean Rank IC 为负时允许统一乘以 `-1`；
- 验证集和测试集冻结训练期确定的方向，禁止重新翻转。

## 初始因子评价

### 硬性门槛

候选必须满足：

```text
有效 Rank IC 日期比例 ≥ 80%
有效日股票覆盖率中位数 ≥ 80%
横截面因子值不是常数
FactorSpec、DSL 和执行过程无错误
```

### 综合评分

综合评分只强调预测能力：

```text
45%  Mean Rank IC 20d
35%  Rank ICIR 20d
15%  月度稳定性
 5%  5d / 10d / 20d 方向一致性
```

定义：

- 月度稳定性：翻转方向后，月度 Mean Rank IC 为正的月份比例；
- 跨周期一致性：5 日和 10 日 Mean Rank IC 都为正记 `1.0`，仅一个为正记 `0.5`，都不为正记 `0.0`；
- 初始筛选使用标准日频 `ICIR = Mean Rank IC / Std Rank IC`；
- Newey–West 调整只用于验证与最终报告，第一版不参与 Top 30；
- 收益、回撤、换手率继续记录，但不计入初始 Top 30 综合分数。

不同指标先根据初始 Alpha101/158 训练集结果转换为百分位，再加权。初始因子分布作为固定评分尺子，后续父代和子代不重新定义百分位映射。

### Top 30

Top 30 只根据训练集综合分数选择。此步骤不进行相关性去重，以保留“预测能力优先”的简单基线并贴近原始 EvoAlpha 的初始化逻辑。

验证集不参与初始 Top 30 选择。

## 固定进化协议

```yaml
population_size: 30
generations: 3
refinement_offspring_per_generation: 12
crossover_tasks_per_generation: 8
```

每个 Crossover 任务保留并回测：

1. 交叉中间因子；
2. 交叉后变异的最终子代。

如果中间因子表现优于最终子代，中间因子可以独立进入候选池。因此每代最多新增：

```text
12 个 Refinement 子代
+ 8 个交叉中间因子
+ 8 个交叉后变异因子
= 28 个候选
```

当前 30 个父代始终参与下一代竞争，不额外设置精英保留。

## 父代选择

### Refinement

- 按综合分数加权抽取 12 个不同父代；
- 每个父代每代最多改进一次；
- 抽样使用固定随机种子，保证复现。

### Crossover

- 两个父代都来自当前种群；
- 排除相似度高于 `0.80` 的组合；
- 在剩余组合中优先选择综合分数较高且相似度较低的组合；
- 同一父代每代最多参与两次交叉；
- Agent 必须解释两个父代的互补机制，机械拼接由 Checker 拒绝。

## 相关性与多样性

因子相似度定义为：

```text
Similarity(A, B)
= median_t(|Spearman(A_t, B_t)|)
```

即每天计算共同股票上的横截面 Spearman 相关系数，取绝对值，再对训练期有效交易日取中位数。

使用位置：

1. **不用于初始 Top 30 去重**；
2. **用于双父代配对**，避免两个高度重复的父代交叉；
3. **用于 Crossover 子代过滤**，子代与初始种子和历史子代的最大相似度高于 `0.90` 时拒绝；
4. **Refinement 子代不因高父子相关性直接拒绝**。微小修改本来就可能保持高相关；只有高相关且指标没有实质改善时才淘汰。

## Refinement 约束

每个 Refinement 子代只允许一个主要改动，例如：

- 修改一个窗口；
- 替换一个算子；
- 增加或删除一次标准化；
- 增加一个确认条件；
- 增加一次风险或波动率调整。

要求：

- 保留父代核心结构；
- 说明修改原因；
- 声明预期改善指标；
- 输出父子公式差异；
- 公式复杂度最多增加两层。

## Population Update

每代将当前 30 个父代与最多 28 个新候选合并：

```text
合法性与覆盖率检查
→ Crossover 多样性检查
→ 使用固定初始百分位尺子计算综合分数
→ 按综合分数保留 Top 30
```

所有父子增量必须单独记录：

```text
Delta Mean Rank IC
Delta Rank ICIR
Delta Monthly Stability
Delta Cross-Horizon Consistency
Parent–Child Similarity
```

交叉任务还需分别记录：

- 交叉中间因子相对两个父代的变化；
- 最终变异子代相对中间因子的变化；
- 最终子代相对表现更好父代的变化。

## 验证与测试

三代训练完成后：

```text
训练期最终 Population 30
→ 冻结公式和方向
→ 验证集重新评价
→ 选择验证表现 Top 10
→ 测试集只运行一次
```

测试结果只用于最终报告，不再影响候选选择。

## Memory 与谱系

后续需要为每个候选增加：

```text
generation
proposal_type
parent_factor_ids
intermediate_factor_id
change_intent
change_description
expected_metric_effects
actual_metric_deltas
factor_similarity
admission_decision
rejection_reason
```

## 已采用的简化决策

以下事项属于有意识的第一版简化，不应被误认为最终方案：

1. 使用最早两年评价数据并增加一年 warm-up，而不是完整市场周期；
2. 使用固定三代、固定候选数量，不让 Manager 动态分配预算；
3. 停止开放式新因子生成，只做单父代改进和双父代交叉；
4. 不建立动量、反转、成交量等因子分类；
5. 不在初始 Top 30 前进行相关性去重；
6. 初始评分以预测能力为主，收益、回撤和换手率只记录；
7. 初始 ICIR 不做 Newey–West 调整；
8. Refinement 第一版使用一个通用 Agent，不拆分类 Skill；
9. 使用受限 DSL，不执行 Agent 自由生成的 Python 代码；
10. 排除真实 VWAP 和历史行业数据要求，不使用未声明的代理字段；
11. 使用固定阈值 `0.80` 和 `0.90`，后续再做敏感性分析；
12. 使用固定初始因子分布作为百分位评分尺子。

## 待实现事项

### P0：单因子评价协议

- [x] 新增 5/10/20 日未来收益标签核心函数；
- [x] 保证标签终点不跨单个评价区间边界；
- [x] 逐日计算横截面 Rank IC；
- [x] 计算 ICIR、月度稳定性和跨周期一致性；
- [x] 实现训练期方向翻转，并支持向验证/测试传入冻结方向；
- [x] 实现可保存和复用的固定参考分布综合评分；
- [ ] 将单因子评价核心接入 Alpha158/101 批量运行入口；
- [ ] 输出逐因子成绩表，而不是 LightGBM 模型成绩。

### P1：统一 DSL 与公式导入

- [x] 扩展 Alpha158 所需的 Canonical AST 与缺失算子；
- [x] 保留 `rank` 为横截面排名，并新增 `ts_rank`；
- [x] 实现 Alpha158 Parser；
- [ ] 与 Qlib Alpha158 做数值一致性测试；
- [ ] 实现 Alpha101 Parser；
- [x] 生成 Alpha158 supported/excluded/reason 清单；
- [x] 确认 Alpha158 可执行候选为 157 个；
- [ ] 确认 Alpha101 最终可执行候选数量。

### P2：初始 Population

- [ ] 批量评价可执行 Alpha101/158；
- [ ] 应用覆盖率和常数检查；
- [ ] 生成固定百分位评分参考；
- [ ] 选择训练集 Top 30；
- [ ] 保存初始种群快照。

### P3：进化 Agent

- [ ] 将 Research Agent 改为父代诊断与 Refinement 输出；
- [ ] 新增 Crossover Agent；
- [ ] 保存交叉中间因子；
- [ ] 新增交叉后 Mutation；
- [ ] 扩展 Hypothesis Checker 检查修改一致性和交叉合理性；
- [ ] 限制公式深度与单次修改范围。

### P4：相关性、种群与谱系

- [ ] 实现日度横截面 Spearman 相似度；
- [ ] 实现父代配对阈值；
- [ ] 实现 Crossover 子代多样性过滤；
- [ ] 实现固定种群更新；
- [ ] 扩展 SQLite schema 保存 lineage 和 metric deltas；
- [ ] 实现每代审计报告。

### P5：验证与最终测试

- [ ] 冻结训练期公式与方向；
- [ ] 在验证集选择 Top 10；
- [ ] 增加 Newey–West 稳健性报告；
- [ ] 测试集只运行一次；
- [ ] 汇总父代、交叉中间因子和最终子代的增量表现。

## 后续升级候选

- 引入真实 VWAP 或明确标注的 VWAP Proxy 变体；
- 引入历史时点 GICS 后实现行业中性化；
- 扩展到更长训练窗口和多市场阶段；
- 对相关性阈值和进化预算做敏感性分析；
- 引入多重检验控制和更严格的样本外协议；
- 将收益、换手率、回撤和复杂度升级为多目标 Fitness；
- 根据实验结果再考虑因子分类与类别配额；
- 根据实验结果再考虑分类 Refinement Skill；
- 比较随机/轮盘赌配对与低相关父代配对；
- 评估 Sumus 式 Reviewer 或动态团队，但不改变确定性实验协议；
- 增加 Agent Skill 的版本、模型、Prompt 和随机种子审计。

## 实现验收标准

第一阶段完成的最低标准：

1. 每个种子因子都能追踪到 Alpha101/158 原始表达式；
2. 每个被排除因子都有机器可读原因；
3. 每个 Top 30 因子都有独立单因子指标；
4. 每个子代都有父代、操作类型和公式差异；
5. 每个交叉任务都保存中间因子和最终变异因子；
6. 所有种群更新都可由固定配置和记录完整复现；
7. 验证集与测试集从未进入 Agent Prompt 或训练期种群更新；
8. LightGBM 不参与初始筛选、父子比较和种群更新。
