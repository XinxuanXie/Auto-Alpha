# 数据说明（教授 S&P 500 数据 · 处理后）

处理日期：2026-10-28。来源：教授 Google Drive 文件夹（CRSP 口径）。处理标准：复权价口径、(permno, date) 唯一、缺失不填补、退市股保留历史、面板只含当日可知信息（不含未来收益）。核对结果见 checks.md。

## 文件

| 文件 | 内容 |
|---|---|
| panel.parquet | 主面板表（长表），一行 = 一只股票一天 |
| membership_intervals.parquet / .csv | 时点股票池：每只股票进入/退出 S&P 500 的区间 |
| securities.parquet / .csv | 证券信息：permno → ticker 映射 + GICS 行业（取最新分类）+ 是否在教授股票池名单内 |
| spy_benchmark.parquet / .csv | 基准：SPY 日频（收盘价、含分红收益 ret、不含分红收益 retx）+ 市场收益列（sprtrn、vwretd、ewretd） |
| checks.md | 处理核对记录（行数、对账、抽查） |
| recon_in_files_not_universe.csv | 对账明细：有数据文件但不在股票池名单的 76 个 permno |

## panel.parquet 字段

- date：交易日；permno：CRSP 永久代码（int32）
- open / high / low / close：复权开/高/低/收（Adj_*，含拆股与分红调整）
- volume：成交量（股）；mktcap：市值（千美元，CRSP 口径）

规模：5,260,971 行，1,195 只股票，2000-01-03 .. 2026-09-30。

## 使用口径

- 某日的可用股票池 = 当日落在 membership 区间内，且当日面板中有数据；
- 因子计算：按 permno 分组沿时间滚动 / 按 date 做横截面，均基于本表；
- 未来收益（标签）在回测时由 close 临时计算，不写入本表。
