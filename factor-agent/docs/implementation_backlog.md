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
5. Git 提交前核对代码、测试与文档状态一致。

## 修改记录

### 2026-11-05：完成 EvoAlpha 确定性主循环骨架

- 使用纯 Python 建立独立外层循环，不引入新的 LangGraph 状态机；
- 新增可序列化状态、阶段、候选、谱系、种群快照和事件模型；
- 固定执行 Seed Evaluation、Top 30、三代进化、验证 Top 10 和最终测试一次；
- 每代强制执行 12 个 Refinement、8 个 Crossover 中间因子和 8 个 Mutation；
- 通过 Tool Protocol 保留后续接入 Agent 和真实评价器的替换接口；
- 当前使用 Fake Tools 验证完整流程，真实 Alpha158 Seed Evaluator 尚待接入。

### 2026-11-05：建立 EvoAlpha 双重分支隔离

- 创建 Git 分支 `feature/evoalpha-workflow`；
- 新建独立 Python 包 `evoalpha_workflow/`；
- 撤销 Alpha158 扩展对旧 `agentic_workflow/factor_library.py` 的直接依赖；
- 旧 Agent Workflow 保留原入口和原内部模块；
- EvoAlpha 与旧工作流只共享确定性的 `factor_pipeline` 基础设施。
- 旧完整架构先保存为 `feature/intuition-hypothesis-pipeline` 的本地基线 commit；
- EvoAlpha 独立包随后保存为 `feature/evoalpha-workflow` 的本地 commit；
- 当前阶段只使用本地 Git 历史，不推送 GitHub。

### 2026-11-05：Alpha158 可执行公式扩展

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

- [ ] 将 157 个 Alpha158 接入真实最早两年数据的批量运行入口；
- [ ] 输出逐因子 train/validation/test 指标表和失败清单；
- [ ] 增加覆盖率、常数因子和异常值硬性门槛；
- [ ] 保存训练期方向、固定评分参考和初始 Top 30 快照；
- [ ] 对最终报告增加 Newey–West 稳健标准误；
- [ ] 完整记录收益、换手率和回撤，继续保持其不参与第一版初选评分。

## 待完善：进化流程

- [x] 建立独立纯 Python EvoAlpha 外层主循环和状态模型；
- [x] 固定三代候选数量、种群规模、验证 Top 10 和测试一次协议；
- [x] 为各阶段定义可替换 Tool Protocol；
- [ ] 将 Research Agent 改为父代诊断与 Refinement Agent；
- [ ] 新增双父代 Crossover Agent 和固定 Mutation 步骤；
- [ ] 保存交叉中间因子并分别回测；
- [ ] 实现父代配对相关性、Crossover 子代重复过滤和种群更新；
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
