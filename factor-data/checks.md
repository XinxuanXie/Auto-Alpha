# 数据处理核对记录

- input files: 1195
- panel rows (concat): 5,260,971; duplicate (permno,date) dropped: 0
- rows dropped for missing/non-positive close: 0; files with TICKER mismatch: 0
- panel rows final: 5,260,971; stocks: 1195
- date range: 2000-01-03 .. 2026-09-30
- negative volume rows: 0; NaN cells in OHLCV/mktcap (kept as missing): 5601
- rows per stock: min 3, median 4931, max 6726
- stocks whose data ends before the dataset end (delisted/merged/etc., history kept): 525
- membership intervals: 2098 rows, 1968 stocks
- securities rows: 1195; with GICS: 1193; in universe list: 1119
- reconciliation: files 1195 vs universe list 1119; in files but not in universe list: 76; in universe list but no file: 0
- SPY rows: 6726; 2000-01-03 .. 2026-09-30
- spot check s93436: panel rows 4089 vs raw 4089 -> True; close values identical -> True
