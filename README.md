# V3 最终版：透明塑料饮料瓶残留液体识别

本仓库根目录即 V3 版本。`final/` 放最终方案和输出成果，`V2.5/` 放历史归档。下载 GitHub 后，直接按下面的命令即可查看结果、重建最终交付包，或者重跑 V2 训练流程。

## 最终方案

- 最终模型：`EfficientNetV2-S 480x240 hybrid two-stage`
- Stage A：`empty` vs `non-empty`
- Stage B：`low` / `medium` / `high`
- 最终成绩：
  - End-to-end 4-class accuracy: `0.8593`
  - Q1 (`empty` vs `not`) accuracy: `0.9407`
  - Q2 (`low` / `medium` / `high`) accuracy: `0.8256`
  - Macro F1: `0.819`
  - Weighted F1: `0.858`

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `final/` | 最终交付包：模型、指标、图表、数据切分、说明文档 |
| `V2.5/` | 历史 V2.5 归档：旧代码、脚本、实验记录、日志、指标、文档 |
| `scripts/` | 当前可执行脚本 |
| `configs/` | 配置文件 |
| `data/` | 原始图像、标注、重标注、数据切分 |
| `models/` | 当前训练与复现用权重 |
| `outputs/` | 当前实验输出和 canonical 结果 |
| `run_all.bat` | Windows 一键生成最终交付包 |

## 环境安装

建议使用 Python 3.9+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果你更习惯 Conda，也可以先创建环境再安装依赖。

## 快速开始

### 1. 直接查看最终结果

最重要的最终成果已经放在 `final/` 里：

- `final/outputs/metrics/final_test_report.txt`
- `final/outputs/metrics/final_test_predictions.csv`
- `final/outputs/metrics/final_classification_metrics.csv`
- `final/outputs/figures/final_confusion_matrix.png`
- `final/outputs/figures/final_pipeline_flow.png`
- `final/outputs/figures/final_model_architecture.png`
- `final/outputs/figures/final_dataset_distribution.png`
- `final/outputs/figures/final_dataset_samples.png`

### 2. 一键生成最终交付包

- Windows：双击 `run_all.bat`
- 或直接运行：

```bash
python scripts/17_eval_final_hybrid.py
```

这个脚本会重新生成 `final/` 下的最终模型包、报告、预测结果和图表，并同步 canonical 输出到 `outputs/metrics/` 和 `outputs/figures/`。

它使用的最终权重来源是：

- `models/efficientnet_v2_s_480x240/two_stage/stage_a_empty_vs_not.pth`
- `outputs/metrics/efficientnet_480x240_multi_seed/seed_42/models/stage_b_low_med_high.pth`

### 3. 如果要重跑 V2 训练和评估

```bash
python scripts/run_all_v2.py
```

常用参数：

- `--skip-train`：只重切分和重建数据集
- `--skip-eval`：训练完就停
- `--use-final-split`：复用 `outputs/metrics/final_split_list.csv` 和 `outputs/metrics/final_split_stats.csv`

## 数据约定

- 原始图片放在 `data/raw_images/`
- LabelMe JSON 放在 `data/annotations_json/`
- 正式标签是：`bottle_empty`、`bottle_low`、`bottle_medium`、`bottle_high`
- 历史重标注记录在 `data/relabel.csv`
- 最终交付包还会保存切分与重标注副本到 `final/data/`

## 最终成果清单

- `final/models/efficientnet_v2_s_480x240_hybrid/stage_a_empty_vs_not_seed2026.pth`
- `final/models/efficientnet_v2_s_480x240_hybrid/stage_b_low_med_high_seed42.pth`
- `final/outputs/metrics/final_test_report.txt`
- `final/outputs/metrics/final_test_predictions.csv`
- `final/outputs/metrics/final_classification_metrics.csv`
- `final/outputs/figures/final_confusion_matrix.png`
- `final/outputs/figures/final_pipeline_flow.png`
- `final/outputs/figures/final_model_architecture.png`
- `final/outputs/figures/final_dataset_distribution.png`
- `final/outputs/figures/final_dataset_samples.png`
- `final/data/final_split_list.csv`
- `final/data/final_split_stats.csv`
- `final/data/final_split_list_v2.csv`
- `final/data/final_split_stats_v2.csv`
- `final/data/relabel.csv`
- `final/docs/experiment_archive_note.txt`

## 历史归档说明

`V2.5/` 只保留 V2.5 版本的历史代码、脚本、实验结果和文档；V1 和 V1.5 没有纳入这个归档。这样 GitHub 根目录就可以直接被理解为 V3 发布版。

如果你只是想看最终结果，直接打开 `final/` 即可；如果你想重跑训练，先看 `scripts/run_all_v2.py`；如果你想重建最终交付包，直接运行 `scripts/17_eval_final_hybrid.py`。
