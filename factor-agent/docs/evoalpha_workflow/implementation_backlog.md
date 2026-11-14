# EvoAlpha 实现记录与后续清单

> 历史实验阶段文档：记录 EvoAlpha 的实现与实验进度，不作为当前 Hypothesis Workflow V2 的待办清单。

## 用途

本文档按日期记录 EvoAlpha 分支的实现和实验变化，并在文档末尾维护唯一一份当前“已完成 / 待完善”清单。历史日志保留当时状态，末尾清单代表截至 2026-11-08 的实际进度。

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

### 2026-11-07

#### 完成 DeepSeek 三代真实 EvoAlpha 流程

- 新增可恢复的 `run_three_generation_deepseek.py`，固定执行种子评价、三代进化和冻结验证；
- 训练期为 2001–2010，2000 年作为 warm-up，验证期为 2011–2012；
- 每代真实生成并回测 12 个 Refinement、8 个 Crossover intermediate 和 8 个 Mutation；
- Refinement/Mutation 使用 `deepseek-flash`，Crossover 使用 `deepseek-v4-pro` 与 high reasoning effort；
- 三代各 28 个候选均通过可执行性硬检验，分别有 11、9、9 个新子代进入下一代或最终种群；
- 最终 30 个候选全部完成冻结验证，最佳验证因子为 `G02-CROSS-006-INTERMEDIATE`；
- 最佳因子的 5/10/20 日 Mean Rank IC 分别为 0.012651、0.018401、0.032981，20 日 ICIR 为 0.348140；
- 运行结果、checkpoint、相关性矩阵、Top 10 和审计说明保存在 `results/evoalpha_deepseek_2001_2012_3gen/`。

#### 强化 Agent 输出与确定性边界

- 扩展 OpenAI-compatible Client，支持 DeepSeek 官方 endpoint、模型路由、reasoning effort、token 上限和请求计数；
- 将 task、generation、parent/child ID 和 FactorSpec ID 改为程序控制，避免模型返回错误身份；
- 将完整 DSL 算子元数与参数语义写入 Refinement、Crossover 和 Mutation Prompt；
- Window 类型变更由程序从建议中提取一个窗口，并投影到完整父公式，确定性丢弃额外结构变化；
- `required_fields` 与 `lookback_days` 继续由公式结构自动推导。

#### 增加公式指纹硬去重

- 公式级审计发现本次最终 30 个唯一 ID 只对应 26 条唯一公式；
- 根因是旧逻辑允许高相关但得分提升的 Refinement 准入，未先识别完全相同公式；
- `PopulationUpdater` 现先比较规范化公式指纹，完全相同的候选以 `exact_formula_duplicate` 拒绝；
- 本次真实结果保留原始状态，便于审计问题、修复和后续无重复重跑之间的差异。

#### 完成原始因子库长期基准检验

- 对最初输入 Agent 的原始因子库运行独立单因子基准，不引入 LightGBM；
- 使用 2000-01-03 至 2026-09-30 的全部本地历史时点成分股数据；
- 完成 157 个 Alpha158、27 个年度窗口、23 个五年窗口、18 个十年窗口和 8 个二十年窗口；
- 按一个自然年步长滚动，共输出 11,932 行完整结果；
- 主表包含 5/10/20 日 Rank IC、20 日 ICIR、稳定性、覆盖率、收益、波动、Sharpe、换手与回撤；
- 每个窗口内根据 20 日 Rank IC 重新确定方向，明确将其定位为历史诊断而非无前视交易回测；
- 采用全历史日度 IC 与双方向周频持仓预计算，避免为 11,932 个窗口重复执行完整因子管线；
- 增加逐因子 checkpoint、数据身份校验、Parquet/CSV 拆分表、窗口汇总和因子稳定性汇总；
- 157 个公式全部执行成功，仅 `ROC60` 的 2000 年窗口因 60 日历史不足未通过硬门槛；
- 完整结果保存在 `results/alpha158_full_history_benchmark/`。

#### 验证

- 新增 DeepSeek Client 参数透传测试；
- 新增复杂 Window 建议投影为单窗口变更测试；
- 新增公式完全相同候选的硬拒绝测试；
- 三代产物已核对 checkpoint 数量、候选数量、种群规模、唯一 ID 和数据切分。
- 长期基准窗口生成、方向聚合、覆盖率和收益聚合测试通过；
- `KMID` 的 2001 年指标与原 Seed Evaluator 逐项一致；
- 长期基准已核对 157 个 checkpoint、11,932 行、唯一因子—窗口键和有限数值。

### 2026-11-08

#### 整理阶段报告并新增 EvoAlpha 周报

- 将原基础 Agent 阶段报告迁移到 `factor-agent/weekly reports/`，并明确标注为 2026-11-06 历史快照；
- 新增 Week 2 Markdown 与浅色 HTML 报告，统一说明基础 Agent、EvoAlpha 独立分支、实验协议、三代搜索、长期因子基准和最终测试；
- 报告明确区分预测能力与投资组合表现：最佳新因子在训练、验证和测试均提高 20 日 IC并降低换手，但样本外收益、Sharpe 和回撤未全面改善；
- 新增报告索引，并从项目 README 建立入口，专题技术文档继续保留在 `docs/`。

#### 完成一次性冻结最终测试

- 冻结最佳新因子 `G02-CROSS-006-INTERMEDIATE`、原因子 `ALPHA158-MIN5`、训练期方向和回测参数；
- 使用未参与训练和验证的 2013-01-01 至 2025-12-31 作为最终测试期；
- 不执行参数调整、候选重选、公式更新或方向重估；
- 新因子相对 `MIN5` 的 5/10/20 日 Rank IC 分别提高 0.001288、0.002575、0.005199；
- 20 日 ICIR 提高 0.043341，平均周换手率下降 19.81 个百分点；
- 年化净收益下降 2.32 个百分点，Sharpe 下降 0.104，最大回撤恶化 3.01 个百分点；
- 新因子在训练、验证和测试三个区间均提高 20 日 IC，但投资组合收益提升只出现在训练集；
- 结果说明支持“中期预测增量具有一定样本外稳定性”，不支持“多头投资表现全面优于原因子”；
- 新增 `run_final_test.py`，结果目录存在时拒绝重复运行；
- 完整结果保存在 `results/evoalpha_final_test_2013_2025/`。

## 当前运行口径

- 旧 `agentic_workflow/` 与新 `evoalpha_workflow/` 并列保留，共享 `factor_pipeline/` 的确定性数据、公式和回测能力；
- EvoAlpha 使用纯 Python 固定流程编排，不由外层 Manager 自由改变实验协议；
- 使用历史时点 S&P 500 成分股、日频 OHLCV 与市值数据；
- 初始种子为 157 个可执行 Alpha158，缺少真实 VWAP 的 `VWAP0` 明确排除；
- 训练期为 2001–2010，2000 年仅用于 warm-up；验证期为 2011–2012；最终测试期为 2013–2025；
- 主要优化 20 日横截面 Rank IC，5 日和 10 日指标用于跨周期诊断；
- 初始 Top 30 依据固定预测能力评分选择；相关性用于双父代配对和子代准入；
- 每代固定生成 12 个 Refinement、8 个 Crossover intermediate 和 8 个 Mutation，已完成三代；
- Agent 只能输出受限 FormulaSpec AST；公式、字段、窗口、复杂度、覆盖率、相似度和重复检查由确定性代码执行；
- 单因子评价不使用 LightGBM；组合诊断采用周频 Top 20 等权、跌出 Top 30 退出和 10 bps 成本。

## 已完成

### 数据、公式与回测基础

- [x] Polars 懒加载、历史时点成分股过滤和市值中性化；
- [x] 5/10/20 日 Rank IC、20 日 ICIR、月度稳定性、覆盖率及跨周期一致性；
- [x] 周频 Top 20 等权回测、持仓缓冲、换手率、成本、收益、Sharpe 与最大回撤；
- [x] FactorSpec、受限 JSON AST、Polars Compiler 与 Alpha158 Parser；
- [x] 157 个非 VWAP Alpha158 公式的解析、执行、支持清单和批量单因子评价；
- [x] 覆盖率、常数因子、异常执行、训练方向及验证/测试方向冻结检查。

### EvoAlpha 搜索与实验

- [x] 独立 EvoAlpha 分支、可恢复主循环、候选 checkpoint 和固定三代协议；
- [x] Top 30 种子选择、父代相似度矩阵、Refinement、Crossover、Mutation 和种群更新；
- [x] 父子谱系、指标增量、相似度准入、公式复杂度约束与规范化公式指纹去重；
- [x] DeepSeek 三代真实搜索：每代 28 个候选，共完成 84 次候选回测；
- [x] 2011–2012 冻结验证、验证 Top 10 和最佳候选选择；
- [x] 2013–2025 一次性冻结最终测试及原因子 `MIN5` 对照；
- [x] 157 个 Alpha158 的年度及 5/10/20 年滚动基准，共 11,932 条结果；
- [x] Week 1 与 Week 2 阶段报告及实验结果索引。

## 待完善

### P0：提高有效子代产出率

- [ ] 将“相对直接父代的实际提升”纳入准入门槛，避免仅凭全种群综合排名保留子代；
- [ ] 将 Fitness 从 IC 主导评分升级为多目标或 Pareto 选择，同时考虑 IC、ICIR、稳定性、收益、换手率、回撤和公式复杂度；
- [ ] 分别统计 Refinement、Crossover intermediate 和 Mutation 的生成成功率、准入率、入选率及样本外提升率；
- [ ] 使用已修复的公式指纹去重逻辑重新运行完整三代，生成无完全重复公式的新种群；
- [ ] 保持 2013–2025 已用测试集封存，为后续方法版本预先指定新的验证协议，避免根据测试结果反复优化。

### P1：稳健性与统计检验

- [ ] 在训练区间内部增加滚动验证或 walk-forward 选择，减少单一训练区间的偶然性；
- [ ] 增加 Newey–West 稳健统计、多重检验控制和候选搜索预算审计；
- [ ] 对相关性阈值、种群规模、代数和父代选择策略进行敏感性实验；
- [ ] 比较纯预测目标与多目标 Fitness 对样本外 IC 和组合表现的影响。

### P2：数据与公式覆盖

- [ ] 用 Qlib 小样本逐列核对本地 Alpha158，重点覆盖 rolling rank、IdxMax/IdxMin、slope、R-square 和 residual；
- [ ] 接入真实 VWAP 后恢复 `VWAP0`；接入历史时点行业分类后增加行业中性化；
- [ ] 完善退市、停牌、缺失开盘价、公司行动、成交量为零和交易日历处理；
- [ ] 根据研究需要再实现 Alpha101 Parser 与缺失算子，不把扩大因子数量作为当前 P0。

### P3：记录与研究能力

- [ ] 将 generation、parents、proposal type、指标增量、相似度和准入原因写入统一 SQLite lineage schema；
- [ ] 生成每代标准化审计表，集中展示唯一公式数、失败原因、父子增量和最终去向；
- [ ] 在固定实验协议下比较 Prompt、Skill 化 Refinement 和独立 Reviewer；
- [ ] 将 Sumus 式协作或动态 Agent 团队保留为后续实验，不改变当前确定性主循环。
