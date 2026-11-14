# 数据、因子清洗与回测pipeline

## 目的

该管线先建立确定性的研究基础设施。后续 Implementation Agent 只需生成符合接口的原始因子，Manager 即可复用统一的数据约束、因子清洗和回测逻辑。

## 数据流

```text
panel.parquet
  → 字段与文件校验
  → OHLCV/市值合法性过滤
  → 信号日历史时点成分股过滤
  → 个股历史滚动计算
  → 最低价格/流动性/市值股票池筛选
  → 横截面因子清洗
  → 市值中性化
  → 周频 Top 20 等权回测，可选跌出 Top 30 退出缓冲
  → JSON 指标 + Parquet 周期明细
```

## 数据接口

`PanelDataSource` 提供三个核心接口：

- `validate()`：检查文件、必要字段、字段类型和数据规模。
- `panel_history()`：以 Polars LazyFrame 返回指定区间及预热期内的合法股票历史。
- `apply_membership(frame)`：仅在需要形成可交易股票池时应用历史成分区间。

滚动因子和横截面分位数在历史时点成分股股票池内计算。执行和平仓价格继续从完整股票历史读取，避免指数退出日截断持仓价格。

## 手写因子

经济假设：异常放量伴随的短期价格变动可能包含情绪过度反应，后续存在反转。

原始因子：

```text
recent_return       = close / lag(close, 5) - 1
volume_abnormality  = volume / rolling_mean(volume, 20)
raw_factor          = -recent_return × cs_zscore(volume_abnormality)
```

数值越高，预测未来收益越高。

该因子目前通过结构化 FactorSpec 和通用 Polars 编译器执行。其他合法 FactorSpec 使用同一数据清洗和回测路径，接口说明见 `factor_dsl.md`。

## 因子清洗与中性化

每个交易日独立执行：

1. 删除无穷值、无效市值及因子缺失值。
2. 将原始因子缩尾至横截面 1% 和 99% 分位数。
3. 对缩尾因子做横截面标准化。
4. 使用带截距的一元横截面回归消除 `log(mktcap)` 暴露。
5. 对回归残差再次做横截面标准化，得到最终 `factor`。

市值中性化使用解析 OLS 残差，能够在 Polars 表达式中完成，无需将完整面板转换为 Pandas。当前不使用行业中性化，因为仓库内 `securities.parquet` 的 GICS 是最新分类，不是历史时点分类。

## 回测时点

```text
每周最后交易日收盘：生成信号并排序
下一交易日开盘：买入 Top 20
下一周调仓日开盘：未跌出 Top 30 的旧持仓继续保留，再从高分股票补足 Top 20
```

该约定确保收盘后形成的信号不会在同一收盘价成交。系统为所有 FactorSpec 固定注入 Top 20/Exit 30 持仓缓冲规则。股票等权配置，换手率按新旧权重差绝对值的一半计算；首次建仓换手率为 1。净收益为组合毛收益减去 `换手率 × 10 bps`。

## 输出

默认输出：

- `.runtime/backtests/abnormal_volume_reversal.json`：配置、数据摘要和汇总指标。
- `.runtime/backtests/abnormal_volume_reversal.periods.parquet`：每个持有周期的日期、持仓数、换手率、成本和收益。

运行命令：

```bash
cd factor-agent
conda run -n autoalpha-agent python -m factor_pipeline
```

可以覆盖时间、持仓数、成本和输出位置：

```bash
conda run -n autoalpha-agent python -m factor_pipeline \
  --start 2015-01-01 \
  --end 2019-12-31 \
  --top-k 20 \
  --cost-bps 10 \
  --output .runtime/backtests/manual_factor.json
```

## 当前边界

- 当前回测验证单因子、周频、纯多头、等权策略。
- 当前缺少点时行业分类，因此仅执行市值中性化。
- 缺少下一调仓日开盘价的持仓不进入当期收益，并在 `missing_price_positions` 中单独报告。
- 当前指标为绝对收益，尚未接入 SPY 超额收益和退市收益专门处理。
- 默认验证段为 2015–2019；2020–2026 测试段继续保持锁定。

## Agent 接口方向

后续 Agent 不负责改写数据和交易时点规则。Implementation Agent 输出原始因子列后，统一进入：

```text
raw factor
  → clean_and_size_neutralize
  → run_weekly_topk_backtest
  → Evaluation Tool
  → Manager Agent
  → Research Record
```

这样可以保证不同 Agent 和不同因子使用同一评价口径。
