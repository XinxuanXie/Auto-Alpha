# Hypothesis Workflow V2

本模块是独立于 `agentic_workflow/` 和 `evoalpha_workflow/` 的假设优先研究框架。

## 当前状态

V2.1 Mock 与真实工具链均已可运行，覆盖：

```text
四路 Research
→ Critical Development Review
→ 1–4 个机制不同的 Hypothesis
→ Hypothesis Check 与一次补生成
→ Measurement Plan
→ 每假设 2–5 个 Baseline / Full / Ablation Experiment
→ Preflight 与一次修正
→ Train Filter
→ Validation Filter
→ Test Report
→ Memory Summary
→ Run Record
```

真实模式已经接入 DeepSeek Research、学校 OpenAI-compatible 批判性 Hypothesis Development、DeepSeek Measurement Plan 与 FactorSpec、确定性检查、真实 Polars 数据接口和滚动重叠 cohort 回测。Mock 仅用于快速回归测试。

## 运行完整 Mock 流程

在 `factor-agent/` 目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
~/miniforge3/envs/autoalpha-agent/bin/python -m hypothesis_workflow_v2
```

默认 Artifact 写入：

```text
.runtime/hypothesis_workflow_v2_mock/<run_id>/
```

指定输出目录：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
~/miniforge3/envs/autoalpha-agent/bin/python -m hypothesis_workflow_v2 \
  --output-dir /tmp/hypothesis-v2-mock
```

## 运行真实冒烟流程

真实冒烟固定为 4 个方向、每方向 1 个假设、每个假设 2 个实验：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
~/miniforge3/envs/autoalpha-agent/bin/python -m hypothesis_workflow_v2 \
  --mode real-smoke --env-file ../.env
```

默认结果写入：

```text
.runtime/hypothesis_workflow_v2_real_smoke/<run_id>/
```

真实运行使用训练集 2001–2010、验证集 2011–2012、测试集 2013–2015。只有训练门槛通过项才进入验证，只有验证再次通过项才进入测试。

## 运行正式完整流程

完整模式由每个方向生成 1–4 个假设，每个假设生成 2–5 个实验：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
~/miniforge3/envs/autoalpha-agent/bin/python -m hypothesis_workflow_v2 \
  --mode real-full --env-file ../.env
```

默认 Artifact 和 checkpoint 写入：

```text
.runtime/hypothesis_workflow_v2_real_full/<run_id>/
.runtime/hypothesis_workflow_v2_real_full/checkpoints.sqlite
```

恢复中断运行：

```bash
python -m hypothesis_workflow_v2 \
  --mode real-full \
  --env-file ../.env \
  --run-id <原来的-run-id> \
  --resume
```

V2.0 Run `70778337-b7c6-4a41-a250-dc62a0748441` 缺少 V2.1 Measurement Plan 和新增必填字段，只用于历史复盘。V2.1 运行必须创建新 `run_id`，随后才能使用上面的命令恢复。

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
~/miniforge3/envs/autoalpha-agent/bin/python -m unittest discover \
  -s tests -p 'test_hypothesis_v2_*.py' -v
```

## 模块

| 文件 | 作用 |
|---|---|
| `contracts.py` | Pydantic 数据契约 |
| `state.py` | 轻量 LangGraph State 与 Stage |
| `artifacts.py` | JSON/Parquet Artifact 持久化与校验 |
| `tools.py` | V2 Tool Protocol 与完整 Mock Tools |
| `manager.py` | 固定 Manager 状态图 |
| `real_tools.py` | 真实 LLM、Checker 与工具适配 |
| `real_backtest.py` | 真实 Rank IC 与滚动重叠 cohort 回测 |
| `__main__.py` | Mock / real-smoke / real-full CLI 与 checkpoint 入口 |

逐文件说明见 `docs/hypothesis_workflow_v2/file_guide.md`。

## 历史真实结果

Run `2625e3be-bf57-4b29-95d0-28c738e7c383`：

- 4 个 Research Idea、4 个 Developed Hypothesis；
- 8 个 FactorSpec 全部通过真实 Preflight；
- 8 个训练回测全部完成；
- 最高训练 Mean Rank IC 为 `0.01682`，低于 `0.02` 门槛；
- 0 个候选进入验证和测试，流程按筛选协议正常结束。

完整实现记录见 `docs/hypothesis_workflow_v2/implementation_log.md`。
