# 外层 Manager Agent

## 定位

外层 Manager Agent 接收研究目标，读取一轮实验的完整结构化依据，并决定下一研究动作。它负责研究策略与迭代方向；LangGraph 负责流程执行与硬约束；内部 Agent 和 Tool 负责具体任务。

```text
Research Goal
    ↓
Hypothesis Tool / Research Agent
    ↓
Implementation Tool / Agent
    ↓
Backtest Tool
    ↓
Evaluation Tool / Agent
    ↓
OpenAI Manager Agent
    ├─ accept_factor
    ├─ adjust_factor
    ├─ new_hypothesis
    ├─ revise_hypothesis
    ├─ reject_factor
    └─ stop_research
    ↓
Research Record
    ↓
Apply Manager Action
```

## 当前实现

`OpenAIManagerPolicy` 使用 OpenAI-compatible Chat Completions API 和 Pydantic JSON 校验。模型只能返回：

```text
action
rationale
instructions
confidence
```

每次决策同时记录：

```text
policy
model
response_id
token usage
```

这些字段写入 `WorkflowState.manager_decision`，便于后续审计成本、复现决策，并将 `instructions` 传给下一轮内部 Agent。

## 安全边界

LLM 负责选择研究动作，代码负责执行以下约束：

- 动作必须属于当前状态的允许集合。
- 达到最大迭代次数后停止新一轮研究。
- Hypothesis Checker 继续执行字段、泄漏和重复检查。
- Tool 失败统一进入失败状态并保存事件。
- API Key 仅从环境变量或本地 `.env` 读取。
- `.env`、Checkpoint 和 Research DB 均不进入 Git。

这种结构保留 Agent 的语义判断能力，同时保证工作流边界可测试、可恢复和可审计。

## 本地配置

在仓库根目录执行：

```bash
cp .env.example .env
```

编辑 `.env`：

```text
OPENAI_API_KEY=你的本地Key
OPENAI_MODEL=gpt-5.6-luna
```

不要将 Key 写入 Python 文件、测试、日志或 Git Commit。

## 运行

使用 OpenAI Manager：

```bash
cd factor-agent
conda run -n autoalpha-agent python -m agentic_workflow \
  --manager openai \
  --model gpt-5.6-luna \
  --max-iterations 3 \
  --research-goal "发现具有稳定横截面预测能力的美股因子"
```

使用固定规则进行离线回归测试：

```bash
conda run -n autoalpha-agent python -m agentic_workflow --manager rule
```

恢复 OpenAI Manager 任务：

```bash
conda run -n autoalpha-agent python -m agentic_workflow \
  --manager openai \
  --resume THREAD_ID
```

## 低成本调试策略

当前默认模型为 `gpt-5.6-luna`。模型名由 `OPENAI_MODEL` 或 `--model` 控制，因此可以直接更换模型，无需修改工作流代码。

调试阶段采用以下控制：

- 每轮在四个候选全部完成回测后调用外层 Manager。
- 只发送结构化实验摘要，不发送完整 Event 和大型 Artifact。
- 最大迭代次数默认为 3。
- API Token 用量保存在 `manager_decision.usage`。
- 单元测试使用 Fake Client，不产生 API 成本。

## 验证状态

- 69 项自动化测试通过。
- OpenAI-compatible 学校 API 真实四路工作流验证通过。
- 四个候选能够分别生成 FactorSpec、保存独立回测 Artifact 并进入 Evaluation。
- JSON 结构校验、动作白名单、SQLite Research Record 和 LangGraph Checkpoint 均已验证。

## 内部模块状态

| 模块 | 当前实现 | 后续替换 |
|---|---|---|
| 外层 Manager | OpenAI LLM，根据四路完整实验决定下一动作 | 动态分配研究预算与 Reviewer |
| Hypothesis | 四个两阶段 Research Pipelines | 增加信息覆盖与因子碰撞分析 |
| Hypothesis Checker | 规则 + SQLite | 增加语义近似与重复机制检测 |
| Implementation | LLM 生成 FactorSpec，确定性校验与纠错 | 扩展分组、残差化和交互表达 |
| Backtest | 真实 FactorSpec + Polars Tool | 增加超额收益、退市与行业处理 |
| Evaluation | 固定门槛 + LLM 诊断 | 增加稳定性、多重检验与增量评价 |
| Record | SQLite + Checkpoint + Artifact | 增加候选谱系与进化记忆 |

下一步重点是提高候选的增量信息与反馈利用效率，而不是继续接入基础 Agent。
