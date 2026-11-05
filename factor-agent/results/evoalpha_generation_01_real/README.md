# EvoAlpha 第 1 代真实 LLM 运行结果

## 运行范围

- 训练与评价区间：2001 年；
- 数据：历史时点 S&P 500 成分股及本地 OHLCV 面板；
- 初始种群：Alpha158 训练期 Top 30；
- 候选：12 个 Refinement、8 个 Crossover intermediate、8 个 Mutation；
- 评价方式：固定 5/10/20 日 Rank IC 协议，20 日预测能力为主；
- 评分基准：初始 157 个 Alpha158 的固定训练期百分位参考；
- Validation/Test：2002 年数据本次未使用。

## 结果摘要

| 项目 | 结果 |
|---|---:|
| 真实 LLM 候选 | 28 |
| 成功编译与回测 | 28 |
| 通过相似度准入 | 10 |
| Refinement 准入 | 8 / 12 |
| Crossover 准入 | 1 / 8 |
| Mutation 准入 | 1 / 8 |
| 最终 Top 30 中的新因子 | 4 |
| 最佳入选新因子 | `G01-CROSS-004-MUTATION` |
| 最佳入选新因子综合分数 | 0.9836538462 |

候选训练期原始最高分为 `G01-CROSS-002-INTERMEDIATE`（1.0），但其与已有种群的相似度超过准入阈值，因此未进入下一代。该差异体现了“预测能力排序 + 多样性准入”的双阶段设计。

最终 Top 30 中保留的四个新因子为：

1. `G01-CROSS-004-MUTATION`；
2. `G01-CROSS-007-INTERMEDIATE`；
3. `G01-REF-003-CHILD`；
4. `G01-REF-008-CHILD`。

## 主要产物

- `generation_01_real_agent_proposals.json`：真实模型输出的完整结构化提案；
- `generation_01_candidate_evaluations.json/.parquet`：28 个候选的训练期指标和父子增量；
- `generation_01_pool_pair_similarity.json/.parquet`：58 个因子的两两相似度；
- `generation_01_population_update.json`：逐候选准入决策和新种群；
- `generation_01_population.json`：第 1 代更新后的 Top 30；
- `generation_01_summary.json`：机器可读摘要；
- `generation_01_agent_checkpoint.json`：可恢复的生成检查点。

## 解读边界

这次运行证明真实 Agent 生成、确定性校验、单因子回测、相关性控制和种群更新已经形成完整闭环。训练期高分仍可能来自样本内选择，下一步应冻结第 1 代种群并在 2002 年验证区间评估泛化表现，避免用训练期结果直接宣称有效性。
