# AutoAlpha-Agent-Harness

Agentic factor mining on S&P 500 (CRSP data).

- factor-agent/: code, environment notes, baseline scorecard
- factor-data/: small tables in git; large files in Releases (tag data-v1)
- factor-agent/evoalpha_workflow/: independent single-factor evolutionary workflow

Setup:
1. Manager、Agent 与独立回测开发：`conda env create -f environment-manager.yml`
2. Linux/Qlib 完整基线：`conda env create -f environment.yml`
3. From Releases data-v1 download: panel.parquet -> factor-data/ ; qlib_data.zip -> unzip into factor-data/ ; reference_ranks.parquet -> factor-agent/library/
4. Project spec, Manager setup and backtest: see factor-agent/README_PROJECT.md
5. EvoAlpha architecture and real generation runner: see factor-agent/evoalpha_workflow/README.md
