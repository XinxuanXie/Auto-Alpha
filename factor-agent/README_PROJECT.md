# factor-agent

阶段性研究报告统一保存在 [`weekly reports/`](weekly%20reports/README.md)。其中 Week 1 记录基础 Agent 工作流，Week 2 汇总 EvoAlpha 三代真实实验和冻结测试结果。

项目：美股因子回测。

## 数据

- 位置：仓库根目录的 `factor-data/`（`factor-agent/` 的同级目录）
- 内容：panel.parquet、qlib_data/

## Alpha158 库级模型基线（2026-10-29 确定，已冻结）

该基线把 Alpha158 作为一组特征输入 LightGBM，用于确认数据、Qlib 特征和模型链路能够运行。它不是单因子筛选或 EvoAlpha 种群评价口径，不参与初始 Top 30、父子增量比较和种群更新。

- 特征集：Alpha158；可用列：157 列；删除列：VWAP0；数据中无 $vwap
- 模型：Qlib 基准 LightGBM 参数：learning_rate 0.0421，lambda_l1 205.6999，lambda_l2 580.9768
- 分段：训练 2000-2014；验证 2015-2019；测试 2020-2026；测试段仅在里程碑时开启
- 成绩单：results/alpha158_baseline_result.md
- 验证段指标：IC 0.004265；Rank IC 0.004246；不计成本超额收益 +2.42%；计成本超额收益 -2.43%

## 复现

在 factor-agent/ 目录下执行：

MLFLOW_ALLOW_FILE_STORE=true /opt/conda/envs/factor-agent/bin/python backtest/run_backtest_alpha158.py --feature-set alpha158 --learning-rate 0.0421 --l1 205.6999 --l2 580.9768 --tag alpha158_baseline

结果写入 results/alpha158_baseline_result.md。

## 独立因子数据与回测管线

`factor_pipeline` 不依赖 Qlib，使用 Polars 懒加载处理 `panel.parquet`，用于验证后续 Agent 生成因子的统一数据与回测接口。

当前固定口径：

- 历史时点 S&P 500 成分股；
- 5 日收益与 20 日异常成交量反转因子；
- 横截面 1%/99% 缩尾、标准化和市值中性化；
- 每周最后一个交易日收盘生成信号；
- 下一交易日开盘买入 Top 20，等权持有至下一次调仓；
- 单边换手成本 10 bps；
- 默认仅运行 2015–2019 验证段。

在 `factor-agent/` 目录执行：

```bash
conda run -n autoalpha-agent python -m factor_pipeline
```

默认结果写入 `.runtime/backtests/`。详细设计和字段说明见 `docs/shared/data_backtest_pipeline.md`，当前验证结果见 `results/abnormal_volume_reversal_validation.md`。

运行任意合法 JSON FactorSpec：

```bash
conda run -n autoalpha-agent python -m factor_pipeline \
  --factor-spec factor_pipeline/example_factor_spec.json
```

FactorSpec、JSON AST 和支持算子见 `docs/shared/factor_dsl.md`。

## EvoAlpha 单因子进化改造

独立的 `evoalpha_workflow/` 已实现单父代 Refinement、双父代 Crossover + Mutation、真实 LLM 生成、训练期逐因子评价、相关性准入和 Top 30 种群更新。当前初始种子为 157 个可由现有数据安全执行的 Alpha158；Alpha101 仍属于后续扩展。所有选择和种群更新均依据逐因子预测指标，不使用 LightGBM。

第 1 代真实运行已完成 12 个 Refinement、8 个 Crossover intermediate 和 8 个 Mutation，结果位于 `results/evoalpha_generation_01_real/`。数据切分、20 日预测目标、综合评分、相关性规则、固定三代协议和后续清单见 `docs/evoalpha_workflow/evoalpha_single_factor_design.md`。

## 外层 Manager Agent

外层 Manager 使用 OpenAI-compatible Chat Completions API。它在四个候选全部完成实现、回测和评价后读取完整实验依据，返回合法的下一动作、理由、后续指令和置信度；随后 Research Record 持久化完整决策。LangGraph 负责动作白名单、最大迭代次数、Checkpoint 和失败路由。

首次使用：

```bash
cp .env.example .env
```

在 `.env` 中填写本地 Key：

```text
OPENAI_API_KEY=你的本地Key
OPENAI_BASE_URL=学校提供的 OpenAI-compatible endpoint
OPENAI_MODEL=gpt-5.6-luna
```

运行：

```bash
cd factor-agent
conda run -n autoalpha-agent python -m agentic_workflow \
  --manager openai \
  --research-goal "发现具有稳定横截面预测能力的美股因子"
```

`.env` 已被 Git 忽略。离线测试使用 `--manager rule`。当前 Hypothesis Research、Implementation、Backtest、Evaluation、Manager 与 SQLite Record 均已接入真实实现；Mock Tools 仅用于单元测试。第一版详细说明见 `docs/agent_workflow_v1/outer_manager_agent.md`。

四个 Research Agents 每轮执行无需 API Key 的实时检索：优先查询 Crossref 公共学术 API，失败时回退到 `ddgs` 网页元搜索。候选假设保存引用资料的标题、URL 与相关性说明；网页来源用于启发研究，最终假设仍受本地数据字段、历史失败记录和确定性 Checker 约束。

Research Agents 同时检索本地 Factor Knowledge Base。当前目录包含 Alpha101、Qlib Alpha158 和经典学术因子共 279 条，并记录公式、经济逻辑、数据要求和当前实现状态。详细说明见 `docs/shared/factor_knowledge_base.md`。

每个 Specialist 采用两阶段研究流程：`Financial Intuition Agent → Hypothesis Formulation Agent`。第一阶段组合 2–4 个因子组件形成金融机制，第二阶段将其约束为当前数据可验证的五日横截面假设。四个方向均通过 Checker 后分别进入 Implementation、真实回测和 Evaluation，再由 Manager 比较完整结果。第一版详细说明见 `docs/agent_workflow_v1/two_stage_research.md`。

## Hypothesis Workflow V2.1

当前主设计方向固定四个 Research Agent，并由共享 Development Agent 执行支持、反驳、替代解释和 ACCEPT/REVISE/REPLACE。每个方向产生 1–4 个机制不同的假设；Measurement Plan 先把机制映射到可观测代理，再生成 Baseline、Full Mechanism 与 Ablation/Robustness 实验。确定性 Preflight 同时检查 DSL 可执行性和 Hypothesis → Measurement Plan → Formula 谱系。Manager 仅负责固定状态转换、一次假设修正、一次实验修正、回测筛选和记录。

V2.0 历史正式 Run 生成 14 个假设和 84 个实验，训练通过 10 个，验证通过 0 个。该结果促成 V2.1 的批判性开发与 Measurement Plan 改造。旧 Run 只读保留；V2.1 新实验使用新 `run_id`。运行方式见 `hypothesis_workflow_v2/README.md`，总体设计见 `docs/hypothesis_workflow_v2/design.md`，实现记录见 `docs/hypothesis_workflow_v2/implementation_log.md`。

## ALPHA20

ALPHA20：不使用。

## 评判方法

- 主要评判工具：真实单因子回测与 Evaluation Agent；现有开放式 Research Workflow 的硬性门槛为 Rank IC > 0.02 且成本后净收益 > 0，同时记录 ICIR、Sharpe、回撤和换手率。EvoAlpha 改造使用 `docs/evoalpha_workflow/evoalpha_single_factor_design.md` 定义的预测能力综合评分。
- LightGBM 回测：仅用于历史库级里程碑检查，不参与单因子进化

## 目录

factor-agent/
- ENV.md
- README_PROJECT.md
- factor_pipeline/
- backtest/run_backtest_alpha158.py
- data_tools/convert_to_qlib.py
- data_tools/verify_qlib.py
- docs/README.md
- docs/shared/
- docs/agent_workflow_v1/
- docs/evoalpha_workflow/
- docs/hypothesis_workflow_v2/
- results/abnormal_volume_reversal_validation.md
- results/alpha158_baseline_result.md

同级目录：factor-data/、qlib-src/、rd-agent-log-backup/、mvp-agent-v1-archive-2026-10-29.zip
