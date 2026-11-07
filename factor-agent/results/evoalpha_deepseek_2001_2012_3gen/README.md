# DeepSeek EvoAlpha 三代真实实验

## 实验协议

- 训练期：2001-01-01 至 2010-12-31；2000 年仅用于 warm-up；
- 冻结验证期：2011-01-01 至 2012-12-31；
- 初始种子：157 个可执行 Alpha158，训练期评分后选 Top 30；
- 每代：12 个 Refinement、8 个 Crossover intermediate、8 个 Mutation，共 28 个候选；
- Refinement/Mutation：`deepseek-flash`；Crossover：`deepseek-v4-pro`，高推理强度；
- 单因子直接评价，不使用 LightGBM；方向、公式和评分参考在验证期冻结。

## 三代结果

| 代数 | 评价候选 | 通过硬检验 | 相关性准入 | 进入下一种群的新子代 | 训练期最佳 |
|---|---:|---:|---:|---:|---|
| 1 | 28 | 28 | 14 | 11 | `G01-CROSS-003-INTERMEDIATE` |
| 2 | 28 | 28 | 13 | 9 | `G02-CROSS-001-INTERMEDIATE` |
| 3 | 28 | 28 | 10 | 9 | `G02-CROSS-001-INTERMEDIATE` |

最终种群共有 30 个候选：5 个原始种子、8 个第一代子代、8 个第二代子代和 9 个第三代子代。

## 冻结验证结果

验证期最佳因子为 `G02-CROSS-006-INTERMEDIATE`，由 `ALPHA158-IMXD30` 与
`ALPHA158-MIN5` 定向标准化后相加得到。

| 指标 | 数值 |
|---|---:|
| 5 日 Mean Rank IC | 0.012651 |
| 10 日 Mean Rank IC | 0.018401 |
| 20 日 Mean Rank IC | 0.032981 |
| 20 日 Rank ICIR | 0.348140 |
| 月度稳定性 | 0.782609 |
| 验证综合分数 | 1.000000 |

完整 Top 10 及公式位于 `validation_2011_2012/validation_top10.json`。

## 审计结论与限制

- 三代 checkpoint 均完整包含 12/8/8 个 Agent 输出；每代候选评价为 28 条；每代种群均为 30 个唯一 ID；
- 本次多次从 checkpoint 恢复，`final_summary.json` 中的 API 请求数只代表最后一次进程调用，不能作为全程累计请求数；
- 公式级审计发现最终 30 个 ID 对应 26 条唯一公式。根因是旧种群更新器只按相关性与得分准入，没有先做公式指纹硬去重；
- 代码现已增加 `exact_formula_duplicate` 硬拒绝规则。当前目录保留本次真实运行原貌，作为问题发现与修复的可审计记录；
- 2011–2012 是用于最终候选排序的验证集，并非尚未触碰的最终测试集。正式结果仍需在独立测试期只运行一次。

## 主要产物

- `protocol.json`：固定数据切分、模型与每代预算；
- `seed/`：157 个种子评价、固定评分参考和初始 Top 30；
- `generation_01/` 至 `generation_03/`：任务计划、Agent checkpoint、候选评价、完整相关性与种群更新；
- `validation_2011_2012/`：冻结验证的 30 个候选与 Top 10；
- `final_summary.json`：全流程机器可读摘要。
