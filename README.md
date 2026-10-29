# AutoAlpha-Agent-Harness

Agentic factor mining on S&P 500 (CRSP data).

- factor-agent/: code, environment notes, baseline scorecard
- factor-data/: small tables in git; large files in Releases (tag data-v1)

Setup:
1. conda env create -f environment.yml (versions also in factor-agent/ENV.md)
2. From Releases data-v1 download: panel.parquet -> factor-data/ ; qlib_data.zip -> unzip into factor-data/ ; reference_ranks.parquet -> factor-agent/library/
3. Project spec and backtest: see factor-agent/README_PROJECT.md
