# 第一版简化决策与后续完善清单

## 用途

本文档集中记录当前为了先跑通单因子进化流程而采用的简化方案，以及需要逐步补齐的事项。每次完成一项后更新状态、测试证据和相关文件。

## 分支与修改方式

旧工作流开发分支：

```text
feature/intuition-hypothesis-pipeline
```

EvoAlpha 独立开发分支：

```text
feature/evoalpha-workflow
```

两条工作流采用并列结构：

- 保留 Manager、LangGraph 工作流、Memory、数据接口和回测模块；
- 旧 `agentic_workflow/` 保持原有研究流程；
- 新 `evoalpha_workflow/` 独立承载种子评价、父代选择、Refinement、Crossover、Mutation 和种群更新；
- `factor_pipeline/` 作为共享底层，提供 FormulaSpec、DSL Compiler、数据清洗、单因子评价和回测；
- 后续由独立入口运行 EvoAlpha，需要时再通过明确 Adapter 接入旧 Manager；
- 每个阶段通过 Git commit 或后续 feature branch 保存可回退节点。

## 文档同步规则

每次代码或配置修改完成后必须同步 Markdown：

1. 在本文档的“修改记录”写明日期、修改内容、测试结果和待完成事项；
2. 架构、接口或运行方式变化时，同步更新对应专题文档；
3. 简化方案、排除项和临时假设必须进入待完善清单；
4. 测试失败或尚未验证的内容必须明确标注，不能写成已完成；
5. Git 提交前核对代码、测试与文档状态一致；
6. 修改记录按日期分组，同一天只保留一个 `### YYYY-MM-DD` 标题，各次修改使用四级小节。

## 修改记录

### 2026-11-06

#### 完成最佳第 1 代因子的 2002 年样本外验证

- 冻结 `G01-CROSS-004-MUTATION` 的公式、40 日窗口和训练期正向方向；
- 完整使用 2002 年作为样本外区间，不调用 LLM、不调整参数、不重新选择候选；
- 20 日 Rank IC 为 `0.056238`，ICIR 为 `0.547935`，5/10/20 日方向保持一致；
- 相对 2001 年训练期，20 日 Rank IC 下降 `0.008460`，ICIR 下降 `0.007384`；
- 月度稳定性由 `90.91%` 下降至 `75.00%`，固定训练参考下综合分数由 `0.983654` 下降至 `0.942788`；
- Top 20 周频组合毛收益为 `+2.70%`，10 bps 成本后净收益为 `-1.69%`；
- 年化波动率为 `31.07%`，Sharpe 为 `0.0969`，最大回撤为 `-34.73%`，平均单边换手率为 `83.94%`；
- 结果表明样本外横截面预测信号仍然存在，但高换手和较大回撤使第一版组合无法形成正净收益；
- 输出保存于 `factor-agent/results/evoalpha_validation_2002/`，本次仅验证一个训练期最佳入选因子，尚未执行最终种群 Top 30 的统一 Validation Top 10 选择。
- EvoAlpha、Formula DSL、数据管道和单因子评价相关测试扩展至 56 项并全部通过。

#### 完成真实 LLM 第 1 代端到端运行

- 接入学校 OpenAI-compatible API，并使用环境变量管理 endpoint、模型名和密钥；
- 真实执行 12 个 Refinement、8 个 Crossover intermediate 和 8 个 Mutation；
- 增加 JSON response mode、候选级检查点和最多 5 次确定性纠错重试；
- 将 `required_fields` 与 `lookback_days` 改为由程序从公式和股票池过滤条件自动推导；
- 为 Mutation 增加五类变异的精确公式结构规则，避免变异类型与公式不一致；
- 28 个候选均通过 FormulaSpec 编译和 2001 年训练期硬性评价门槛；
- 相似度准入接纳 10 个候选，其中 8 个 Refinement、1 个 Crossover、1 个 Mutation；
- 新 Top 30 最终保留 4 个子代，最佳入选子代为 `G01-CROSS-004-MUTATION`，综合分数 `0.9836538462`；
- 结果保存于 `factor-agent/results/evoalpha_generation_01_real/`，2002 年验证/测试数据仍保持隔离。
- EvoAlpha、Formula DSL、数据管道和单因子评价相关测试共 53 项通过；当前系统 Python 未安装旧工作流所需的 `langgraph`，旧工作流全套测试需在 `environment-manager.yml` 定义的 Python 3.11 环境执行。

#### 完成相似度准入与 Top 30 种群更新

- Refinement 与直接父代相似度高于 `0.90` 时，只有综合分数提升才可准入；
- Crossover intermediate 和 Mutation 与当前种群及已接纳子代的最高相似度不得超过 `0.90`；
- 使用通过准入的子代与原种群合并，并按固定综合分数选出下一代 Top 30；
- 输出每个候选的准入结论、原因、最相似因子及相似度，保证种群更新可审计。

#### 完成统一 Candidate Evaluator 与 Mock 子代训练期评价

- 统一接收 Refinement、Crossover intermediate 和 Mutation 的 CandidateProposal；
- 使用与种子阶段相同的 2001 年训练期、数据清洗和 5/10/20 日 Rank IC 协议；
- 所有候选复用初始 157 个 Alpha158 的固定百分位评分参考；
- 保存候选相对直接父代的 Mean Rank IC、ICIR、稳定性、一致性和综合分数增量；
- Mutation 同时记录相对中间因子和两个原始父代的指标变化；
- 真实运行 28 个 Mock 候选，全部通过硬性门槛，10 个优于至少一个直接父代；
- 细分结果为 Refinement `2/12`、Crossover `5/8`、Mutation `3/8`；
- 最高分为 `G01-CROSS-002-INTERMEDIATE`，固定综合分数达到上限 `1.0`；
- Mock 结果只验证评价链路，不代表真实 Agent 的表现。

#### 完成 Crossover 与 Mutation 结构化接口及 Mock 验证

- Crossover 输入固定双父代任务、两个父代 FormulaSpec、训练指标、训练方向和相似度；
- 方向为负的父代先使用 `negate` 定向，再参与公式交叉；
- Crossover 输出包含互补性分析、融合逻辑、预期指标影响和中间 FactorSpec；
- 检查器要求中间公式同时保留两个定向父代，并校验公式深度、lookback 和回测规则；
- Mutation 对中间因子只执行一次可审计修改，并复用 Refinement 单一修改检查器；
- 使用 Mock Client 生成 8 个中间因子和 8 个 Mutation 子代，全部通过结构与谱系校验；
- 本次 8 个 Mock Mutation 均为单窗口修改，不进入正式种群。

### 2026-11-05

#### 完成 Refinement Agent 结构化接口与 Mock 验证

- 定义 Refinement Prompt、结构化输出模型和可替换 Model Client 接口；
- 输入包含固定任务、父代 FormulaSpec、训练指标、允许算子和修改约束；
- 输出包含父代诊断、单一修改类型、修改说明、保留逻辑、预期指标影响和子代 FormulaSpec；
- 新增确定性检查器，验证任务身份、谱系、公式深度、lookback、方向和回测规则；
- Window 修改只允许改变一个窗口，Normalization 必须完整包裹父代公式；
- 使用 Mock Client 为第 1 代生成 12 个合法子代，其中 11 个窗口修改、1 个标准化修改；
- Mock 子代只验证接口，不进入正式种群，真实学校 API 尚未接入。

#### 完成第 1 代确定性父代选择

- Refinement 按训练期综合分数加权且不放回抽取 12 个不同父代；
- Crossover 仅从相似度 `≤0.80` 的组合中选择 8 对父代；
- Crossover 优先分数固定为 `70% × 父代平均分 + 30% × (1 - 相似度)`；
- 每个父代每代最多参与两次 Crossover；
- 固定第 1 代随机种子为 `43`，相同输入可完整复现；
- 已生成 `generation_01_parent_plan.json`，尚未调用 Agent 生成子代。

#### 完成初始 Top 30 相关性分析

- 使用 2001 年训练期清洗和市值中性化后的最终因子值；
- 对每个交易日计算 Top 30 两两横截面 Spearman；
- 使用日度绝对相关系数的中位数作为配对相似度；
- 435 个组合中有 390 个满足 `≤0.80` 配对阈值，45 个被排除；
- 相关性分布中位数为 `0.413430`；
- 最低相似组合为 `CORR60 × CNTP5`，最高相似组合为 `SUMN30 × SUMP30`；
- 已输出完整矩阵、逐对记录和机器可读摘要。

#### 接入真实 Alpha158 Seed Evaluator

- 使用现有 `panel.parquet` 与历史时点成分股区间；
- 固定读取 2000 年 warm-up，并仅使用 2001 年训练期选择种子；
- 逐个编译、清洗并评价 157 个可执行 Alpha158；
- 统一生成 5/10/20 日 Rank IC、ICIR、稳定性、方向和综合分数；
- 输出候选 JSON、指标 Parquet、固定评分参考和初始 Top 30；
- 2002 年数据继续隔离给 Validation 与最终 Test。
- 真实运行完成：157 个因子全部执行成功并通过硬性覆盖率门槛；
- 初始 Top 30 已保存，训练期最高分因子为 `ALPHA158-MIN60`，综合分数为 `0.9796474359`；
- 训练期方向统计为 67 个保持原方向、90 个统一翻转；
- 结果报告位于 `factor-agent/results/evoalpha_seed_2001/README.md`。

#### 完成 EvoAlpha 确定性主循环骨架

- 使用纯 Python 建立独立外层循环，不引入新的 LangGraph 状态机；
- 新增可序列化状态、阶段、候选、谱系、种群快照和事件模型；
- 固定执行 Seed Evaluation、Top 30、三代进化、验证 Top 10 和最终测试一次；
- 每代强制执行 12 个 Refinement、8 个 Crossover 中间因子和 8 个 Mutation；
- 通过 Tool Protocol 保留后续接入 Agent 和真实评价器的替换接口；
- 当前使用 Fake Tools 验证完整流程，真实 Alpha158 Seed Evaluator 尚待接入。

#### 建立 EvoAlpha 双重分支隔离

- 创建 Git 分支 `feature/evoalpha-workflow`；
- 新建独立 Python 包 `evoalpha_workflow/`；
- 撤销 Alpha158 扩展对旧 `agentic_workflow/factor_library.py` 的直接依赖；
- 旧 Agent Workflow 保留原入口和原内部模块；
- EvoAlpha 与旧工作流只共享确定性的 `factor_pipeline` 基础设施；
- 旧完整架构先保存为 `feature/intuition-hypothesis-pipeline` 的本地基线 commit；
- EvoAlpha 独立包随后保存为 `feature/evoalpha-workflow` 的本地 commit；
- 当前阶段只使用本地 Git 历史，不推送 GitHub。

#### Alpha158 可执行公式扩展

- 扩展 FormulaSpec 与 Polars Compiler，加入 Alpha158 所需滚动、相关性、回归和逐元素算子；
- 新增 Alpha158 统一公式目录、受限 Parser 和机器可读支持清单；
- 157 个非 VWAP 因子已通过合成数据解析与执行测试；
- VWAP0 因缺少真实 VWAP 数据继续排除；
- 相关 DSL、Alpha158、单因子评价和数据管道测试共 31 项通过；
- Qlib 逐列数值一致性验证和真实数据批量评价仍待完成。

## 当前已确定的实现口径

- 单因子直接评价，不使用 LightGBM 选择父代或子代；
- 使用最早两年作为训练/验证/测试评价数据，额外读取最早一年前的数据作为 warm-up；
- 预测周期以未来 20 日为主，5 日和 10 日用于一致性诊断；
- Alpha158 第一版执行 157 个公式，排除 VWAP0；
- 初始股票池使用历史时点 S&P 500 成分股；
- 因子清洗和评价采用统一确定性代码；
- Agent 只能输出受限 FormulaSpec AST，不能执行自由 Python；
- 初始 Top 30 以预测能力综合分数排序，不先做相关性去重；
- 研究方式固定为单父代 Refinement 与双父代 Crossover + Mutation；
- Manager 负责流程调用、失败处理和记录，实验协议由配置固定。

## 已完成

- [x] Polars 懒加载数据接口与点时成分股过滤；
- [x] 手写因子、因子清洗、中性化与 Top 20 等权回测链路；
- [x] 5/10/20 日 Rank IC、ICIR、月度稳定性和跨周期一致性评价；
- [x] 训练期方向确定与验证/测试方向冻结；
- [x] 固定初始分布的百分位综合评分；
- [x] Alpha158 统一公式目录；
- [x] Alpha158 Parser 与 FormulaSpec 转换；
- [x] Alpha158 所需滚动、相关性、回归和逐元素算子；
- [x] 157 个非 VWAP Alpha158 公式解析与合成数据执行测试；
- [x] Alpha158 supported/excluded/reason 机器可读清单。

## 待完善：数据

- [ ] 引入真实 VWAP 数据后恢复 Alpha158-VWAP0；
- [ ] 引入历史时点行业分类后实现行业中性化和行业类 Alpha101；
- [ ] 明确退市、停牌、价格异常、成交量为零和公司行动处理规则；
- [ ] 增加真实交易日历与数据完整性审计报告；
- [ ] 将两年快速实验扩展到更长市场周期并覆盖不同市场状态。

## 待完善：公式与数值一致性

- [ ] 在同一小样本上逐列比较本地 157 个 Alpha158 与 Qlib 输出；
- [ ] 为 slope、R-square、residual、rolling rank、IdxMax/IdxMin 增加固定数值基准；
- [ ] 统一 NaN、无穷值、零分母和最小窗口的清洗规则；
- [ ] 实现 Alpha101 Parser；
- [ ] 扩展 covariance、product、delta、decay linear、signed power、scale、逻辑组合等 Alpha101 算子；
- [ ] 为 Alpha101 生成 supported/excluded/reason 清单并确认最终数量。

## 待完善：批量单因子评价

- [x] 将 157 个 Alpha158 接入真实 2001 年训练数据的批量运行入口；
- [x] 输出训练期逐因子指标、执行失败信息和初始 Top 30；
- [x] 在真实 2001 年训练期完成 157 个 Alpha158 的批量运行；
- [ ] 输出 validation/test 指标表和失败清单；
- [ ] 增加覆盖率、常数因子和异常值硬性门槛；
- [ ] 保存训练期方向、固定评分参考和初始 Top 30 快照；
- [ ] 对最终报告增加 Newey–West 稳健标准误；
- [ ] 完整记录收益、换手率和回撤，继续保持其不参与第一版初选评分。

## 待完善：进化流程

- [x] 建立独立纯 Python EvoAlpha 外层主循环和状态模型；
- [x] 固定三代候选数量、种群规模、验证 Top 10 和测试一次协议；
- [x] 为各阶段定义可替换 Tool Protocol；
- [x] 实现初始 Top 30 训练期相似度矩阵和 `0.80` 父代配对过滤；
- [x] 实现固定随机种子的 Refinement 与 Crossover 父代选择器；
- [x] 完成 Refinement Agent Prompt、结构化输出、确定性检查器和 Mock 验证；
- [x] 完成 Crossover/Mutation Prompt、结构化输出、方向处理、检查器和 Mock 验证；
- [x] 完成统一 Candidate Evaluator、固定评分和父子指标增量记录；
- [x] 将 Research Agent 改为父代诊断与 Refinement Agent；
- [x] 新增双父代 Crossover Agent 和固定 Mutation 步骤；
- [x] 保存交叉中间因子并分别回测；
- [x] 实现父代配对相关性、Crossover 子代重复过滤和种群更新；
- [x] 使用真实学校 API 完成第 1 代 28 个候选的生成、回测与种群更新；
- [ ] 扩展 SQLite 保存 generation、parents、变异类型、父子指标增量和相似度；
- [ ] 输出每代种群审计报告；
- [ ] 完成训练、验证 Top 10、最终测试一次的冻结流程。

## 后续实验选项

- [ ] 比较固定 Prompt、Skill 化 Refinement 和 Reviewer 机制；
- [ ] 比较单父代微调、双父代交叉及交叉后变异的增量贡献；
- [ ] 比较不同相关性阈值和种群预算；
- [ ] 比较纯预测能力评分与加入换手、回撤、复杂度后的多目标评分；
- [ ] 评估 Sumus 式协作团队结构，但保持实验协议与评价代码确定性；
- [ ] 增加多重检验控制，降低大规模因子搜索的数据窥探风险。
