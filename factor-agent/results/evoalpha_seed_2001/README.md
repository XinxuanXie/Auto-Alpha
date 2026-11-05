# Alpha158 Seed Evaluation — 2001 Training Period

## 运行范围

- Warm-up：2000-01-03 至 2000-12-31；
- Training：2001-01-01 至 2001-12-31；
- Universe：历史时点 S&P 500 成分股；
- Seeds：157 个可执行 Alpha158，排除需要真实 VWAP 的 VWAP0；
- Primary target：未来 20 日收益；
- Diagnostics：未来 5 日和 10 日收益；
- 初始种群规模：Top 30。

## 运行结果

```text
Evaluated:             157
Passed hard filters:   157
Rejected or failed:      0
Selected population:    30
Best factor:          ALPHA158-MIN60
Best composite score: 0.9796474359
```

全部通过硬性门槛表示因子具有足够的有效日期、股票覆盖率和有限 Rank IC/ICIR。它不表示 157 个因子都具有同等预测能力；初始种群仍按固定训练期综合分数选择 Top 30。

训练期方向：

```text
保持原方向： 67
统一翻转：   90
```

## Top 10

| Rank | Factor | Composite | Mean Rank IC 20d | Rank ICIR 20d | Monthly Stability | Direction |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ALPHA158-MIN60 | 0.979647 | 0.067318 | 0.380137 | 0.8182 | 1 |
| 2 | ALPHA158-CNTP30 | 0.954327 | 0.057231 | 0.378832 | 0.8182 | -1 |
| 3 | ALPHA158-CORR60 | 0.953045 | 0.056678 | 0.614502 | 0.8182 | -1 |
| 4 | ALPHA158-CORR30 | 0.935096 | 0.051042 | 0.629251 | 0.8182 | -1 |
| 5 | ALPHA158-BETA60 | 0.911859 | 0.065703 | 0.331143 | 0.7273 | -1 |
| 6 | ALPHA158-ROC30 | 0.889103 | 0.073841 | 0.364625 | 0.6364 | 1 |
| 7 | ALPHA158-MA60 | 0.887500 | 0.075274 | 0.360797 | 0.6364 | 1 |
| 8 | ALPHA158-QTLD60 | 0.883974 | 0.068145 | 0.363616 | 0.6364 | 1 |
| 9 | ALPHA158-CNTD30 | 0.875321 | 0.052218 | 0.338330 | 0.7273 | -1 |
| 10 | ALPHA158-RANK60 | 0.857692 | 0.055276 | 0.299083 | 0.7273 | -1 |

## 输出文件

- `alpha158_seed_metrics.parquet`：紧凑逐因子指标表；
- `alpha158_seed_candidates.json`：157 个候选、FactorSpec、方向和完整指标；
- `alpha158_score_reference.json`：后续父代和子代复用的固定百分位评分尺子；
- `alpha158_top30.json`：初始种群完整记录。

## 尚未执行

- Qlib 逐列数值一致性比较；
- 2002 年上半年 Validation；
- 2002 年下半年最终 Test；
- 父代相关性分析、Refinement、Crossover、Mutation 和种群更新。
