# FactorSpec 与结构化因子 DSL

## 目标

FactorSpec 是 Implementation Agent 与确定性回测系统之间的唯一因子接口。Agent 输出结构化 JSON，系统负责校验、编译、清洗和回测。Agent 不直接执行 Python 代码，也不修改回测引擎。

```text
Hypothesis
  → Implementation Agent
  → FactorSpec JSON
  → Pydantic Validator
  → Polars Compiler
  → raw_factor
  → 清洗与市值中性化
  → 统一回测
```

## FactorSpec 字段

| 字段 | 含义 |
|---|---|
| `factor_id` | 因子唯一标识 |
| `name` | 因子名称 |
| `description` | 因子经济与计算描述 |
| `hypothesis_id` | 来源假设，可为空 |
| `formula` | 结构化 JSON AST |
| `required_fields` | 公式实际引用的数据字段 |
| `lookback_days` | 计算所需最大历史窗口 |
| `higher_is_better` | 原公式值越高是否表示预期收益越高 |
| `rebalance_frequency` | 当前固定为 `weekly` |
| `universe_filters` | 可选的最低价格、流动性分位数和市值分位数筛选 |
| `portfolio_rules` | 系统固定注入的 Top 20 入选、跌出 Top 30 退出规则 |

`required_fields` 必须与公式引用字段完全一致；`lookback_days` 必须覆盖公式推导出的历史长度。

## 数据字段

第一版允许：

```text
open, high, low, close, volume, mktcap
```

## 支持算子

| 算子 | 参数 | 语义 |
|---|---|---|
| `field` | `field` | 读取数据列 |
| `constant` | `value` | 数值常量 |
| `lag` | 一个输入、`window` | 个股时间序列滞后 |
| `return` | 一个输入、`window` | `x / lag(x, window) - 1` |
| `mean` | 一个输入、`window` | 个股滚动均值 |
| `std` | 一个输入、`window` | 个股滚动标准差 |
| `sum` | 一个输入、`window` | 个股滚动求和 |
| `rolling_max` / `rolling_min` | 一个输入、`window` | 个股滚动最大值/最小值 |
| `quantile` | 一个输入、`window`、`quantile` | 个股滚动分位数 |
| `ts_rank` | 一个输入、`window` | 当前值在个股滚动窗口内的百分位排名 |
| `idx_max` / `idx_min` | 一个输入、`window` | 滚动窗口最大值/最小值的一基位置 |
| `correlation` | 两个输入、`window` | 个股滚动相关系数 |
| `slope` / `rsquare` / `residual` | 一个输入、`window` | 对时间序号的滚动线性回归统计量 |
| `zscore` | 一个输入 | 当日横截面标准化 |
| `rank` | 一个输入 | 当日横截面百分位排名 |
| `abs` / `log` | 一个输入 | 绝对值/自然对数 |
| `add` | 两个 `args` | 加法 |
| `subtract` | 两个 `args` | 减法 |
| `multiply` | 两个 `args` | 乘法 |
| `divide` | 两个 `args` | 安全除法，分母接近零时返回缺失 |
| `negate` | 一个输入 | 取负 |
| `greater_than` | 两个 `args` | 大于条件 |
| `less_than` | 两个 `args` | 小于条件 |
| `elementwise_max` / `elementwise_min` | 两个 `args` | 逐元素最大值/最小值 |
| `where` | 三个 `args` | 条件、成立值、不成立值 |

时间序列滚动算子按 `permno` 沿时间计算；`zscore` 和 `rank` 按 `date` 做横截面计算。`min_periods` 可显式记录滚动窗口的最小有效样本数，`std.ddof` 可区分总体标准差与样本标准差。Alpha158 导入器使用 Qlib 对应参数，现有手写因子保持原有默认值。

## 示例

```json
{
  "factor_id": "F-MOMENTUM-20",
  "name": "二十日反转",
  "description": "使用二十日收益的相反数预测未来收益。",
  "hypothesis_id": "H-MOMENTUM-20",
  "formula": {
    "op": "negate",
    "args": [
      {"op": "return", "field": "close", "window": 20}
    ]
  },
  "required_fields": ["close"],
  "lookback_days": 20,
  "higher_is_better": true,
  "rebalance_frequency": "weekly",
  "universe_filters": {
    "min_price": 5.0,
    "min_liquidity_percentile": 0.2,
    "liquidity_lookback_days": 20,
    "min_market_cap_percentile": 0.1
  },
  "portfolio_rules": {"top_k": 20, "exit_rank": 30}
}
```

完整异常放量反转示例位于：

```text
factor_pipeline/example_factor_spec.json
```

## 运行

```bash
cd factor-agent
conda run -n autoalpha-agent python -m factor_pipeline \
  --factor-spec factor_pipeline/example_factor_spec.json
```

## 校验规则

- 拒绝未知字段和未知算子。
- 拒绝额外 JSON 属性。
- 拒绝不合法的参数数量和窗口。
- 拒绝不一致的 `required_fields`。
- 拒绝不足的 `lookback_days`。
- 拒绝 `NaN` 和无穷常量。
- 编译器只生成 Polars 表达式，不执行任意代码。
- `higher_is_better=false` 时，编译器统一反转原始因子方向。

## 当前边界

- 当前仅支持日频 OHLCV 与市值。
- 当前只支持周频回测。
- 当前 AST 已覆盖 157 个无需真实 VWAP 的 Alpha158 公式；VWAP0 明确排除。
- Alpha101 仍待独立 Parser 与完整算子扩展，不包含行业数据和基本面数据。
- Manager 已直接读取 FactorSpec 并调用真实 Polars 回测 Tool。
