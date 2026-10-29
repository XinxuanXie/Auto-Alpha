# factor-agent 环境说明


## 统一环境
- 环境类型：conda 环境
- 环境名：factor-agent
- Python：3.11.18
- 环境路径：/opt/conda/envs/factor-agent
- 激活命令：conda activate factor-agent
- 创建命令：/opt/conda/bin/conda create -y -n factor-agent python=3.11
- 安装命令：/opt/conda/envs/factor-agent/bin/pip install pyqlib pandas pyarrow lightgbm scipy scikit-learn

## 版本
- qlib 0.9.7
- pandas 3.1.0
- pyarrow 25.0.1
- lightgbm 4.7.0
- scipy 1.17.1
- scikit-learn 1.9.1
- numpy 2.4.6

## 面板数据验证
- 文件：/workspaces/codespaces-blank/factor-data/panel.parquet
- 形状：5260971 行 x 8 列
- 列：date, permno, open, high, low, close, volume, mktcap
- date 最小值：2000-01-03 00:00:00
- date 最大值：2026-09-30 00:00:00
- permno 唯一数：1195
- date 唯一数：6726
