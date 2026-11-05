# Factor Knowledge Base

Research Agent 的本地因子知识库包含三类资料：

- WorldQuant Alpha101：101 条原论文公式；
- Microsoft Qlib Alpha158：158 条由官方生成规则构建的特征；
- 经典学术因子：20 条动量、反转、低波动、流动性、价值、质量等代表性定义。

每条记录统一包含：

- `factor_id`、名称与因子家族；
- 公式或概念定义；
- 经济逻辑；
- 所需数据字段；
- 对应的研究概念；
- 当前实现状态；
- 权威来源链接。

实现状态分为：

- `available_now`：当前 OHLCV、市值数据和 DSL 可以表达；
- `requires_dsl_extension`：数据具备，但需要扩展 DSL 算子；
- `requires_additional_data`：需要 VWAP、行业分类或财务报表等额外数据。

每个 Specialist Research Agent 每轮根据自身允许的概念，从 Alpha101、Alpha158 和学术因子中均衡检索相关样例。检索结果随 iteration 轮换，避免每轮看到固定样例。Agent 可以参考不可实现因子的经济机制，但最终假设必须符合当前数据能力。

候选假设使用 `factor_references` 保存实际参考的因子 ID、名称、家族、实现状态和改造方式。Agent 只能引用当前上下文中真实存在的因子 ID。

来源：

- Alpha101: https://arxiv.org/abs/1601.00991
- Alpha158: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
