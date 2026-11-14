# Manager Workflow 开发笔记

> 历史阶段文档：本文按开发顺序保留 Mock 到真实 Agent 的演进过程，不代表当前主设计。V1 实现以 `outer_manager_agent.md` 和 `two_stage_research.md` 为准；当前 V2 设计见 [`../hypothesis_workflow_v2/design.md`](../hypothesis_workflow_v2/design.md)。

本文档记录 Agentic Factor Research 系统中 Manager Workflow 的设计与实现过程。

目标是让后续阅读代码时能够回答三个问题：

1. 当前模块解决什么问题？
2. 为什么采用这种结构？
3. 下一步如何继续扩展？

## 总体目标

第一阶段先建立稳定的 Manager 主循环，再逐步将普通 Tool 替换为内部 Agent。

```text
Manager Workflow
    ↓
Hypothesis Tool
    ↓
Implementation Tool
    ↓
Backtest Tool
    ↓
Evaluation Tool
    ↓
Manager 决定下一步动作
    ↓
Research Record
    ↺
```

外层 Manager Agent 负责研究决策，LangGraph 负责状态管理、流程调度和硬约束。每个具体步骤通过统一 Tool 接口调用，因此 Tool 内部既可以是 Mock、普通 Python 函数，也可以在后续替换为 LLM Agent。

## 正式开发环境

Manager Workflow 使用独立的 Miniforge Conda 环境：

```text
环境名称：autoalpha-agent
Python：3.11.18
LangGraph：1.2.18
环境位置：~/miniforge3/envs/autoalpha-agent
```

这个环境与系统 Python、python.org Python、pyenv、旧 Miniconda 环境彼此独立。

### 激活环境

```bash
conda activate autoalpha-agent
```

激活后确认解释器：

```bash
which python
python --version
```

预期结果：

```text
/Users/xiexinxuan/miniforge3/envs/autoalpha-agent/bin/python
Python 3.11.18
```

### 退出环境

```bash
conda deactivate
```

### 在新电脑重建环境

仓库根目录的 `environment-manager.yml` 是跨平台的 Manager 开发环境声明：

```bash
conda env create -f environment-manager.yml
```

依赖发生变化后同步现有环境：

```bash
conda env update -n autoalpha-agent -f environment-manager.yml
```

仓库原有的 `environment.yml` 是完整 Linux/Qlib 基线环境；本机开发 Manager 时优先使用 `environment-manager.yml`。

## Step 1：定义共享状态

### 本步骤目标

在编写主循环之前，先规定所有节点共同读写的数据结构。

如果没有统一状态，不同模块会分别使用不同字段和返回格式，后续将普通 Tool 替换为 Agent 时就需要修改整个主循环。

### 新增文件

```text
factor-agent/
├── agentic_workflow/
│   ├── __init__.py
│   └── state.py
└── tests/
    └── test_state.py
```

### `WorkflowStage`

`WorkflowStage` 表示当前实验执行到了哪个阶段。

```text
INIT
PROPOSE_HYPOTHESIS
CHECK_HYPOTHESIS
IMPLEMENT_FACTOR
VALIDATE_IMPLEMENTATION
RUN_BACKTEST
EVALUATE_RESULT
DECIDE_NEXT_ACTION
RECORD_EXPERIMENT
APPLY_MANAGER_ACTION
COMPLETED
FAILED
```

每个阶段只能通过 Manager 定义的合法路径进入下一阶段。

### `ManagerAction`

`ManagerAction` 表示 Manager 在完成一轮评估后可以选择的下一步动作。

```text
NEW_HYPOTHESIS
REVISE_HYPOTHESIS
RETRY_IMPLEMENTATION
ADJUST_FACTOR
ACCEPT_FACTOR
REJECT_FACTOR
STOP_RESEARCH
```

使用枚举可以避免代码中出现含义不明确的自由文本。

### `WorkflowState`

`WorkflowState` 是整个工作流共享的状态对象，包含：

| 字段 | 作用 |
|---|---|
| `run_id` | 本次研究运行的唯一编号 |
| `research_goal` | 外层 Manager 接收的研究目标 |
| `stage` | 当前流程阶段 |
| `iteration` | 当前研究轮次 |
| `max_iterations` | 最大允许轮次 |
| `hypothesis` | Research Agent 输出的研究假设 |
| `factor_spec` | 结构化因子定义 |
| `implementation` | 因子公式、代码和测试结果 |
| `backtest_result` | 横截面回测结果 |
| `evaluation` | 规则评估器输出 |
| `manager_decision` | 外层 Manager 的动作、理由、指令、模型和 Token 用量 |
| `next_action` | Manager 选择的下一步动作 |
| `retry_counts` | 各阶段的重试次数 |
| `events` | 本轮流程产生的事件记录 |
| `error` | 当前错误信息 |

### `create_initial_state()`

这个函数用于创建一份干净的初始状态：

- 自动生成唯一 `run_id`。
- 将阶段设为 `INIT`。
- 将研究轮次设为 `0`。
- 清空假设、因子、回测和错误信息。
- 检查最大迭代次数至少为 `1`。

### 测试

测试文件验证：

1. 初始状态包含正确的默认值。
2. `max_iterations=0` 时会明确报错。

运行命令：

```bash
cd factor-agent
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

当前结果：

```text
Ran 2 tests
OK
```

## Step 2：统一 Tool 接口与 Mock Tools

### 本步骤目标

为所有可替换模块定义相同的调用方式，并使用确定性的 Mock Tool 验证各阶段的数据能够连接起来。

第一版 Tool 是普通 Python 类。后续接入 Agent 时，只替换 Tool 内部实现，Manager 的调用方式保持不变。

### 新增文件

```text
factor-agent/
├── agentic_workflow/
│   └── tools/
│       ├── __init__.py
│       ├── base.py
│       └── mock.py
└── tests/
    └── test_tools.py
```

### `WorkflowTool`

所有 Tool 都遵守同一个接口：

```text
输入：WorkflowState
输出：ToolResult
```

Tool 只完成一个边界明确的操作，不修改流程阶段，也不决定下一个节点。

### `ToolResult`

`ToolResult` 统一表示 Tool 的执行结果：

| 字段 | 作用 |
|---|---|
| `success` | 本次调用是否成功 |
| `payload` | 成功时返回的结构化数据 |
| `error` | 失败时返回的明确错误信息 |

统一返回结构可以让 Manager 对普通函数、Mock 和 Agent 使用同一套成功与失败处理逻辑。

### Mock Tools

当前实现五个确定性 Tool：

| Tool | 当前行为 | 后续替换目标 |
|---|---|---|
| `MockHypothesisTool` | 返回固定量价假设 | Research Agent |
| `MockImplementationTool` | 返回 FactorSpec 和 DSL 公式 | Factor Implementation Agent |
| `MockBacktestTool` | 返回固定回测指标 | 真实横截面回测引擎 |
| `MockEvaluationTool` | 根据固定阈值返回 PASS 或 FAIL | 规则评估器 |
| `MockRecordTool` | 返回结构化实验记录 | SQLite Repository |

### 职责边界

```text
Tool：处理当前任务并返回结果
Manager：更新共享状态并决定流程跳转
```

Tool 不允许自行改变 `WorkflowStage`。这保证流程控制只有一个来源，后续调试时可以准确判断是谁执行了路由。

### 当前 Mock 数据流

```text
MockHypothesisTool
    ↓ hypothesis
MockImplementationTool
    ↓ factor_spec + implementation
MockBacktestTool
    ↓ backtest_result
MockEvaluationTool
    ↓ evaluation + next_action
MockRecordTool
    ↓ record
```

### 测试

新增测试验证：

1. 五个 Mock Tool 的输出可以依次写入同一个 `WorkflowState`。
2. Mock Evaluation 最终产生 `ACCEPT_FACTOR` 动作。
3. 缺少 hypothesis 时，Implementation Tool 返回明确失败结果。

当前结果：

```text
Ran 4 tests
OK
```

## Step 3：LangGraph Manager 主循环

### 本步骤目标

使用 LangGraph 将前两步的共享状态与 Mock Tools 连接成可以自动执行的确定性状态图。

LangGraph 只负责流程编排。研究假设、因子实现、回测和评估仍然通过独立 Tool 完成。

### 新增与修改文件

```text
factor-agent/
├── agentic_workflow/
│   ├── __main__.py
│   ├── manager.py
│   ├── state.py
│   └── tools/
│       ├── __init__.py
│       └── mock.py
└── tests/
    └── test_manager.py
```

项目环境增加：

```text
langgraph==1.2.18
```

### `ManagerTools`

`ManagerTools` 是 Manager 使用的依赖集合：

```text
hypothesis
hypothesis_checker
implementation
backtest
evaluation
record
```

`ManagerTools.mock()` 创建一套完整的 Mock 实现。未来接入真实 Agent 时，只替换对应字段。

### 主循环状态图

```text
START
  ↓
INIT
  ↓
PROPOSE_HYPOTHESIS
  ↓
CHECK_HYPOTHESIS
  ↓
IMPLEMENT_FACTOR
  ↓
VALIDATE_IMPLEMENTATION
  ↓
RUN_BACKTEST
  ↓
EVALUATE_RESULT
  ↓
DECIDE_NEXT_ACTION
  ↓
RECORD_EXPERIMENT
  ↓
APPLY_MANAGER_ACTION
  ├─ ACCEPT / REJECT / STOP → COMPLETED
  ├─ NEW / REVISE          → 下一轮 INIT
  ├─ RETRY / ADJUST        → IMPLEMENT_FACTOR
  └─ 无合法动作             → FAILED
```

### Manager 与 Tool 的边界

每个 Tool 节点执行以下步骤：

1. 读取当前 `WorkflowState`。
2. 调用对应 Tool。
3. 将成功结果合并进状态。
4. 添加一条 Event。
5. 由 Manager 设置下一个 Stage。

Tool 失败时，Manager 统一写入 `error`，添加 `tool_failed` 事件，并路由到 `FAILED`。

### Event 轨迹

完整 Mock 流程产生九条事件：

```text
1. iteration_started
2. hypothesis tool_succeeded
3. hypothesis_checker tool_succeeded
4. implementation tool_succeeded
5. validation_succeeded
6. backtest tool_succeeded
7. evaluation tool_succeeded
8. record tool_succeeded
9. workflow_completed
```

Event 轨迹用于后续调试、SQLite 持久化和流程可视化。

### 运行 Mock 主循环

在 `factor-agent` 目录执行：

```bash
python -m agentic_workflow
```

程序会打印最终 `WorkflowState`，其中包含假设、FactorSpec、公式、回测指标、评估结果和完整事件轨迹。

### 测试

新增测试验证：

1. 一轮 Mock 研究可以到达 `COMPLETED`。
2. 最终动作是 `ACCEPT_FACTOR`。
3. Tool 失败时主循环进入 `FAILED`，并保存明确错误。

当前结果：

```text
Ran 6 tests
OK
```

## Step 4：SQLite Research Record

### 本步骤目标

将内存中的 `MockRecordTool` 替换为真实 SQLite 持久化模块，使实验记录在 Python 进程结束后仍然存在。

### 新增文件

```text
factor-agent/
├── agentic_workflow/
│   ├── storage/
│   │   ├── __init__.py
│   │   └── sqlite.py
│   └── tools/
│       └── sqlite_record.py
└── tests/
    └── test_sqlite_record.py
```

### 数据库位置

运行 Manager 后自动创建：

```text
factor-agent/.runtime/research.db
```

`.runtime/` 已加入 `.gitignore`，本地实验数据不会提交到 GitHub。

### `SQLiteResearchRepository`

Repository 负责：

1. 创建数据库目录。
2. 初始化 `research_records` 表。
3. 按尝试编号写入实验记录。
4. 按 `record_id` 查询记录。
5. 统计记录数量。

数据库表保存：

| 字段 | 作用 |
|---|---|
| `record_id` | `run_id:iteration:attempt` 组成的唯一记录编号 |
| `run_id` | 一次 Manager 运行的编号 |
| `iteration` | 当前假设研究轮次 |
| `attempt` | 当前假设下的实现与回测尝试编号 |
| `max_attempts_per_hypothesis` | 每个假设允许的最大实现与回测次数，默认 3 |
| `stage` | 写入记录时的流程阶段 |
| `hypothesis_id` | 对应假设编号 |
| `evaluation_status` | PASS、WEAK 或失败类型 |
| `next_action` | Manager 下一步动作 |
| `state_json` | 当前完整 WorkflowState 快照 |
| `created_at` | 首次创建时间 |
| `updated_at` | 最近更新时间 |

同一个 `run_id + iteration + attempt` 重复写入时执行幂等更新；不同 attempt 分别保存，保留因子调整前后的完整实验轨迹。

达到 `max_attempts_per_hypothesis` 后，Manager 不再允许选择 `adjust_factor` 或 `retry_implementation`，必须修订/更换假设、拒绝因子或终止研究。

执行 `adjust_factor` 或 `retry_implementation` 时，工作流将旧 FactorSpec、实现信息、回测结果、Evaluation 诊断和 Manager 决策保存到 `previous_attempt`。Implementation Agent 随后进入 revision 模式，基于完整前后文修改旧因子；开始新假设时清空该快照。

Research Agent 的 Memory Retrieval 从最近 100 条记录中评分并返回最多 10 条。评分依据包括研究概念重合、预测方向、失败规则、同一 run、历史验收结果、Rank IC、净收益与新近程度；每条结果附带 `retrieval.score` 和 `retrieval.match_reasons`，不调用额外 Agent 或 embedding 服务。

Research 阶段由 Momentum、Reversal、Liquidity/Volatility 和 Price-Volume 四个 Specialist Agents 组成。每个 Specialist 只检索本方向相关 Memory，并且只能选择本方向允许的研究概念。四个候选先经过确定性 Hypothesis Checker，Manager Candidate Selector 再从合格候选中选择一个进入 Implementation；全部候选与选择依据保存到 WorkflowState。

### `SQLiteRecordTool`

`SQLiteRecordTool` 实现统一的 `WorkflowTool` 接口：

```text
WorkflowState
    ↓
SQLiteRecordTool
    ↓
SQLiteResearchRepository.save()
    ↓
ToolResult(record)
```

因此 Manager 不需要了解 SQL，也不需要修改路由逻辑。

### 当前运行方式

`python -m agentic_workflow` 现在采用：

```text
Mock Hypothesis
Mock Checker
Mock Implementation
Mock Backtest
Mock Evaluation
Real SQLite Record
```

### 测试

新增测试验证：

1. 完整 Mock 流程能够写入 SQLite。
2. 重新连接数据库后能够读取记录。
3. `state_json` 中包含回测指标。
4. 同一实验重复保存不会产生重复记录。

当前结果：

```text
Ran 8 tests
OK
```

真实本地运行已经生成一条持久化记录，状态为 `PASS`，下一动作是 `accept_factor`。

## Step 5：LangGraph SQLite Checkpoint

### 本步骤目标

保存状态图每个节点执行后的状态，使程序中断、终端关闭或 Python 进程重启后能够继续执行。

新增依赖：

```text
langgraph-checkpoint-sqlite==3.1.2
```

### 新增文件

```text
factor-agent/
├── agentic_workflow/
│   └── checkpoint.py
└── tests/
    └── test_checkpoint.py
```

### 两个 SQLite 数据库的区别

```text
.runtime/research.db
保存业务实验记录：假设、因子、回测和评估结果

.runtime/checkpoints.db
保存工作流执行状态：当前节点、节点状态和恢复位置
```

Research Record 回答“实验结果是什么”；checkpoint 回答“流程执行到哪里”。

### `thread_id`

每次新运行使用 `run_id` 作为 LangGraph 的 `thread_id`：

```text
run_id = 1555fd88-...
thread_id = 1555fd88-...
```

所有属于同一次运行的 checkpoint 都通过这个编号关联。恢复时必须提供相同的 `thread_id`。

### `open_sqlite_checkpointer()`

这个上下文管理器负责：

1. 创建 `.runtime/checkpoints.db`。
2. 打开 SQLite 连接。
3. 创建 LangGraph `SqliteSaver`。
4. 运行结束后关闭连接。

反序列化采用严格白名单，只允许项目中的：

```text
ManagerAction
WorkflowStage
```

### Manager 注入方式

`build_manager()` 现在接受：

```text
checkpointer
interrupt_after
```

业务节点和 Tool 不依赖 checkpoint。持久化能力由 Manager 编译时注入。

### 新运行

```bash
python -m agentic_workflow
```

输出中的 `run_id` 就是后续恢复使用的 `thread_id`。

### 恢复运行

```bash
python -m agentic_workflow --resume <THREAD_ID>
```

LangGraph 会读取该 thread 最新 checkpoint，并从待执行节点继续。

### 恢复测试

测试执行以下过程：

```text
启动流程
→ 在 implement_factor 后暂停
→ 关闭 SQLite 连接
→ 创建新的 Manager 图实例
→ 使用相同 thread_id 调用 invoke(None)
→ 从 validate_implementation 继续
→ 完成回测、评估和记录
```

测试确认恢复后：

- `run_id` 保持不变。
- 已完成的 Implementation 不会重新生成。
- 工作流最终进入 `COMPLETED`。
- SQLite Research Record 只写入一次。

当前结果：

```text
Ran 10 tests
OK
```

真实本地运行生成 12 条 checkpoint 记录，并成功通过同一个 `thread_id` 重新加载。

## Step 6：规则型 Hypothesis Checker

### 本步骤目标

将 `MockHypothesisCheckerTool` 替换为真实的确定性规则检查器，并使用 `research.db` 拦截完全相同的历史假设。

### 新增文件

```text
factor-agent/
├── agentic_workflow/
│   ├── hypothesis.py
│   └── tools/
│       └── hypothesis_checker.py
└── tests/
    └── test_hypothesis_checker.py
```

### 当前检查规则

规则型 Checker 依次检查：

1. 必要字段是否存在且类型正确。
2. 预测方向是否为 `positive` 或 `negative`。
3. 预测周期是否为固定的未来五个交易日。
4. `required_concepts` 是否来自 OHLCV 与市值支持的概念集合。
5. 输入概念是否包含 `future_return`、`future_price` 等泄漏标签。
6. `research.db` 是否已经存在完全相同的规范化假设。

### 精确重复签名

假设先转换成规范化结构：

```text
title
claim
economic_rationale
prediction_direction
prediction_horizon_days
排序后的 required_concepts
```

文本会统一空格和大小写，再生成 SHA-256 签名。`hypothesis_id` 不参与签名，因此仅编号不同、研究内容完全相同的假设仍会被识别为重复。

当前只检查精确重复，不判断语义近似。

### 数据库迁移

`research_records` 增加：

```text
hypothesis_signature
```

启动 Repository 时会：

1. 检查旧表是否缺少该列。
2. 自动执行 `ALTER TABLE`。
3. 从旧记录的 `state_json` 回填签名。
4. 创建签名索引。

因此已有 `research.db` 不需要删除重建。

### 拒绝后的 Manager 路由

假设被规则拒绝属于正常研究结果，不属于程序错误：

```text
CHECK_HYPOTHESIS
    ↓ rejected
RECORD_EXPERIMENT
    ↓
NEW_HYPOTHESIS
    ↓
下一轮或达到迭代上限后停止
```

拒绝结果会以 `REJECTED_HYPOTHESIS` 写入 Research Record。

### 当前真实运行结果

固定 Mock 假设已存在于历史记录，因此连续三轮被判定为：

```text
exact_duplicate_hypothesis
```

Manager 最终保留第三轮拒绝结果，并以 `stop_research` 正常结束。

### 测试

当前测试覆盖：

- 接受新的合法 OHLCV 假设。
- 拒绝精确历史重复。
- 拒绝未来信息泄漏概念。
- 拒绝当前数据不支持的概念。
- 将拒绝结果写入 Research Record。
- 自动迁移旧数据库并回填签名。

当前结果：

```text
Ran 15 tests
OK
```

## MVP 简化决策与后续改进路线图

当前版本优先验证接口、状态和闭环，每个模块都先采用最小可解释实现。以下改进项作为后续版本路线图统一保留。

### 1. 开发环境

当前版本：

- macOS 使用精简 `environment-manager.yml`。
- Linux/Qlib 基线继续使用原 `environment.yml`。
- 依赖使用固定版本。

后续改进：

- 拆分 core、agent、backtest、dev 依赖组。
- 增加跨平台 lock file。
- 增加 GitHub Actions 测试 Python 3.11。
- 增加 Codespaces 或 Docker 的统一运行环境。

### 2. WorkflowState

当前版本：

- 使用一个 `TypedDict` 保存整轮状态。
- 数据结构允许使用普通字典快速迭代。

后续改进：

- 使用 Pydantic 定义 Hypothesis、FactorSpec、BacktestResult 等独立模型。
- 为状态增加 schema version 和迁移机制。
- 大型代码、日志和回测文件改为 artifact 引用，不直接放入状态。
- 增加研究预算、Token、耗时和资源使用字段。

### 3. Tool 接口

当前版本：

- 同步 `run(state) -> ToolResult`。
- 统一 success、payload 和 error。

后续改进：

- 增加泛型输入输出类型。
- 支持 async Tool、超时、取消和重试策略。
- 增加 Tool 版本、执行耗时和资源计量。
- 对代码执行类 Tool 增加独立沙箱与权限策略。

### 4. Manager 路由

当前版本：

- 固定 LangGraph 状态图。
- 单分支串行执行。
- 根据枚举动作路由。

后续改进：

- 将重试次数、停止规则和预算放入配置文件。
- 增加人工审批节点与暂停点。
- 支持多个研究分支和并发调度。
- 增加优先级、研究预算和 branch pruning。
- 在少数语义分支接入 LLM Manager Policy，流程边界继续由状态机控制。

### 5. Event 与可观测性

当前版本：

- Event 保存在 `WorkflowState.events`。
- 记录节点、Tool、结果和跳转。

后续改进：

- 将 Event 拆成独立数据库表。
- 增加开始时间、结束时间、耗时和错误堆栈。
- 增加 Prompt、模型版本、Token 和成本追踪。
- 增加可视化运行时间线与失败诊断页面。

### 6. Research Record

当前版本：

- 一轮实验对应 `research_records` 一行。
- 常用字段独立存储，完整状态保存为 JSON。

后续改进：

- 拆分 runs、hypotheses、factor_specs、implementations、backtests、evaluations、events。
- 增加数据库 migration 版本管理。
- 增加事务、外键、索引和并发写入策略。
- 将大型 artifact 保存到文件或对象存储。
- 增加备份、归档和数据保留策略。

### 7. Checkpoint

当前版本：

- 使用本地 SQLite。
- 通过 `thread_id` 恢复最新 checkpoint。
- 采用严格反序列化白名单。

后续改进：

- 增加列出、查看和删除历史 thread 的 CLI。
- 增加 checkpoint 保留与压缩策略。
- 处理 WorkflowState schema 升级后的恢复兼容。
- 多进程或服务部署时迁移至 PostgreSQL checkpointer。
- 对包含敏感 Prompt 或数据的 checkpoint 加密。

### 8. Hypothesis Checker

当前版本：

- 固定 OHLCV 与市值概念白名单。
- 基于标签检查潜在泄漏。
- 只拦截完全相同的规范化假设。

后续改进：

- 使用 embedding 或向量数据库检测语义近似假设。
- 根据历史实验证据识别被反复否定的研究方向。
- 检测假设之间的包含、矛盾和轻微改写关系。
- 根据真实数据目录动态判断字段与时间覆盖。
- 在 FactorSpec 或公式 AST 层执行严格未来数据泄漏检查。
- 引入 LLM Reviewer 处理规则无法判断的语义问题。

### 9. Research Agent

当前版本：

- 四个 Specialist 分别研究动量、反转、流动性—波动率和量价关系。
- 每个 Specialist 拆为 Financial Intuition Agent 与 Hypothesis Formulation Agent。
- Agent 同时读取实时资料、279 条因子知识库和相关历史实验。
- 四个候选全部通过 Checker 后分别进入真实实现、回测和评价。

后续改进：

- 增加信息覆盖、因子碰撞和增量信息分析。
- 增加语义 Reviewer，检查叙事与可执行假设是否一致。
- 增加 Prompt 版本管理和固定评测集。
- 探索 EvoAlpha 式候选池、Mutation 与 Crossover。

### 10. Factor Implementation

当前版本：

- 已定义正式 Pydantic FactorSpec。
- 已实现只允许 OHLCV、市值和基础安全算子的 JSON AST。
- 已实现确定性的 Polars 编译器与字段、窗口、方向校验。
- Implementation Agent 已根据每个候选假设生成正式 FactorSpec，并接受最多四次确定性错误反馈与纠错。

后续改进：

- 实现真实因子单元测试与覆盖率检查。
- 在沙箱中执行代码并限制资源。
- 扩展分组内比较、条件残差化和复杂交互表达。

### 11. Backtest

当前版本：

- 已实现独立于 Qlib 的 Polars 懒加载数据接口。
- 使用历史时点 S&P 500 股票池。
- 已实现手写异常放量反转因子、缩尾、标准化和市值中性化。
- 已实现周频 Top 20 等权、下一交易日开盘执行和 10 bps 成本。
- 已封装 `FactorBacktestTool` 并接入 Manager CLI。
- 每轮保存 `summary.json` 与 `periods.parquet` Artifact。

后续改进：

- 增加 SPY 与横截面等权基准及超额收益。
- 接入历史时点行业分类后增加行业中性化。
- 完善退市收益、缺失开盘价和停牌处理。
- 严格处理调仓时点、复权、停牌、缺失值和退市股票。
- 从固定时间切分扩展到 walk-forward 验证。

### 12. Evaluation

当前版本：

- 使用 Rank IC 与成本后净收益执行不可修改的硬性验收。
- Evaluation Agent 根据 ICIR、收益、换手、回撤和实现一致性解释失败原因并提出修订建议。

后续改进：

- 将全部阈值写入版本化配置。
- 增加 ICIR、年度稳定性、分位数组合、回撤和覆盖率。
- 增加多重检验与数据挖掘偏差控制。
- 将最终测试集保持锁定，仅在里程碑运行。
- 后期接入 Analysis Agent，固定规则继续作为底层证据。

### 13. Factor Library 与 Memory

当前版本：

- `research.db` 保存实验记录。
- 已建立包含 Alpha101、Qlib Alpha158 和经典学术因子的 279 条知识库。
- Research Agent 按概念、失败类型、历史表现与研究方向检索相关实验。
- Checkpoint、事件日志和独立回测 Artifact 已接入。

后续改进：

- 只将通过验证的因子写入 Factor Library。
- 保存公式、方向、适用时期、表现和失败条件。
- 为 Research Agent 提供结构化历史检索。
- 实验规模扩大后增加语义检索和经验总结。

## 下一步

基础 Agent 与真实四路回测已经接入。下一阶段重点是因子碰撞与增量信息评价、条件结构表达、候选谱系和 EvoAlpha 式多代搜索。
