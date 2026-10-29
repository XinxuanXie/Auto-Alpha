# factor-agent

项目：美股因子回测。

## 数据

- 位置：/workspaces/codespaces-blank/factor-data/（factor-agent 的同级目录）
- 内容：panel.parquet、qlib_data/

## 基线（2026-10-29 确定，已冻结）

- 特征集：Alpha158；可用列：157 列；删除列：VWAP0；数据中无 $vwap
- 模型：Qlib 基准 LightGBM 参数：learning_rate 0.0421，lambda_l1 205.6999，lambda_l2 580.9768
- 分段：训练 2000-2014；验证 2015-2019；测试 2020-2026；测试段仅在里程碑时开启
- 成绩单：results/alpha158_baseline_result.md
- 验证段指标：IC 0.004265；Rank IC 0.004246；不计成本超额收益 +2.42%；计成本超额收益 -2.43%

## 复现

在 factor-agent/ 目录下执行：

MLFLOW_ALLOW_FILE_STORE=true /opt/conda/envs/factor-agent/bin/python backtest/run_backtest_alpha158.py --feature-set alpha158 --learning-rate 0.0421 --l1 205.6999 --l2 580.9768 --tag alpha158_baseline

结果写入 results/alpha158_baseline_result.md。

## ALPHA20

ALPHA20：不使用。

## 评判方法

- 主要评判工具：因子级检验工具（待建设）；指标：因子 IC、ICIR、五分位收益差、换手率
- LightGBM 回测：用于库级里程碑检查

## 目录

factor-agent/
- ENV.md
- README_PROJECT.md
- backtest/run_backtest_alpha158.py
- data_tools/convert_to_qlib.py
- data_tools/verify_qlib.py
- results/alpha158_baseline_result.md

同级目录：factor-data/、qlib-src/、rd-agent-log-backup/、mvp-agent-v1-archive-2026-10-29.zip
