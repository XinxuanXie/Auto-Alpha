# Hypothesis Workflow V2 实现记录

## 2026-11-14

### V2.1 已完成

- 将 Hypothesis Development 改为批判性审查：每个 Idea 必须同时给出支持、反驳、替代解释，并选择 ACCEPT、REVISE 或同方向 REPLACE。
- Hypothesis Expansion 改为每方向 1–4 个机制不同的假设，允许只保留一个高质量假设；每个假设新增核心机制、最强反驳、证伪标准和同批差异。
- 新增 Measurement Plan，将金融机制拆成可观测代理、字段、局限、Baseline 逻辑和 Full Mechanism 逻辑。
- FactorSpec 改为每假设 2–5 个实验，必须包含 Baseline 与 Full Mechanism，可选 Ablation/Robustness；移除仅靠周期变化扩充候选的设计。
- Experiment 新增 `measurement_plan_id`、`variant_type` 和 `represented_proxy_ids`，建立 Hypothesis → Measurement Plan → Formula 的可追踪谱系。
- Preflight 对首次和修正公式统一检查代理编号、字段覆盖和 Full Mechanism 完整性。
- 假设与实验修正均接收 Checker 的真实结构化问题；实验修正同时接收 Measurement Plan 和同假设已通过公式。
- `RealV2Tools` 改为延迟导入，Mock 和合同测试无需加载真实 LLM/数据依赖。
- 新增 Measurement Plan 代理编号及 Experiment 代理编号唯一性测试。
- 19 个 V2 专项测试通过；完整 Mock 工作流完成 20 个事件，包含 Measurement Plan、两类一次修正以及训练/验证/测试。

### 版本边界

- V2.0 正式 Run `70778337-b7c6-4a41-a250-dc62a0748441` 保留为历史结果：14 个假设、84 个实验、10 个训练通过、0 个验证通过。
- V2.1 增加必填数据契约，旧 Run 不使用新代码恢复；下一次真实实验创建新 `run_id`。
- V1 与 EvoAlpha 代码和结果均保持独立。

### 下一步

- 运行一次 `real-smoke`，人工检查四份 Critical Review、Hypothesis、Measurement Plan 和 Baseline/Full 公式是否保持金融语义。
- 冒烟产物通过人工审查后，再启动新的 `real-full` 运行。

## 2026-11-13（V2.0 历史实现）

### 已完成

- CLI 新增 `real-full`：每方向 2–5 个假设，每个假设最多 10 个实验。
- 接入 LangGraph SQLite checkpoint，支持使用原 `run_id` 和 `--resume` 恢复。
- FactorSpec 按假设写入独立 cache，后续假设失败时复用已完成 API 结果。
- FactorSpec 改为逐候选校验，同一批中的非法公式不再使合法候选全部失效。
- 明确 `residual`、`slope`、`rsquare` 是单序列滚动趋势算子，并规范化一元算子输入。
- Preflight 先裁剪 2008–2010 和 64 个真实股票，再编译执行公式，避免每个候选重复扫描完整面板。
- 新增 `file_guide.md`，记录所有 V2 文件职责、调用关系和 checkpoint 恢复方法。

### 正式运行进度

Run：`70778337-b7c6-4a41-a250-dc62a0748441`

- 4 个 Research Idea 已完成；
- 4 份 Insight Analysis 已完成；
- 14 个 Developed Hypothesis 全部通过检查；
- 84 个 FactorExperimentSpec 已生成；
- 首次 Preflight：80 个通过，4 个进入一次修正；
- 后续训练、验证和测试结果以该 Run 的最终 Artifact 为准。

## 2026-11-12（V2.0 初始实现）

### 已完成

- 保留 V1 与 EvoAlpha 目录，新增独立 `hypothesis_workflow_v2/` 分支实现。
- 建立 Pydantic contracts、轻量 State、原子 Artifact Repository 与固定 LangGraph Manager。
- 接入四路 DeepSeek Research、学校 OpenAI-compatible 两阶段 Hypothesis Development、DeepSeek FactorSpec。
- 接入确定性 Hypothesis Checker 与 FactorSpec Preflight；失败项各允许修正一次。
- Preflight 同时检查字段、DSL/Pydantic 结构、样本容量、完全重复，并在 2008–2010 样本上实际执行公式，确认能产生有限值。
- 接入 2001–2010 训练、2011–2012 验证、2013–2015 测试的数据分段。
- 实现逐日横截面 Rank IC、训练期方向确定与冻结、Newey–West、BH-FDR、Top 20 日频重叠 cohort、10 bps 成本和组合指标。
- CLI 支持 `mock` 与 `real-smoke`；真实冒烟限制为 4 个方向 × 1 个假设 × 最多 2 个实验。
- 将 Manager 从 `build_mock_manager` 重命名为通用 `build_manager`。

### 运行中发现并修复

1. 学校 OpenAI 网关偶发返回 HTTP 成功但空正文：LLM 适配层将空正文纳入应用级重试。
2. 推理型学校模型在 2,048 token 上限下可能耗尽预算而不输出正文：Development 阶段统一使用 8,192 token 上限，并记录 `finish_reason`。
3. DeepSeek 可能生成语法上接近但不满足 `FormulaNode` 的 JSON，例如给 `return` 错误设置 `min_periods`：结构化输出先经 Pydantic 校验，失败时把精确错误反馈给同一模型并完整重生成一次；Prompt 同时明确算子参数规则。

### 真实冒烟结果

Run：`2625e3be-bf57-4b29-95d0-28c738e7c383`

- 4/4 Research 分支成功；
- 4/4 假设通过 Hypothesis Checker；
- 8/8 FactorSpec 通过 Preflight；
- 8/8 训练回测完成；
- 最高 Mean Rank IC：`0.01682`；
- 最高项训练成本后累计净收益：`0.7122`；
- 训练门槛通过：0；
- 验证与测试未触发，符合固定筛选协议。

### 后续事项

- 将 Artifact 索引和跨轮经验统一写入 SQLite Memory。
- 在正式规模运行前评估 LLM 并发、速率限制和成本预算。
- 分析本轮 8 个候选低于 IC 门槛的原因，再决定优化 Research/Development Prompt、概念可实现性约束或 FactorSpec 变体覆盖。
- 长期保留集 2016–2025 继续保持封存。
