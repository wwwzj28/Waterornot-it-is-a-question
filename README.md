# Water or Not: Transparent Bottle Residual Liquid Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Task-Computer%20Vision-green)](#)
[![Release](https://img.shields.io/badge/Version-V3_Final-orange)](#)

本项目面向数学建模竞赛中的机器视觉识别任务，目标是从透明塑料饮料瓶图像中自动判断瓶内残留液体水平。项目围绕真实拍摄图像、LabelMe 标注、深度学习分类模型和实验可视化结果构建，最终形成一套可复现、可展示、可用于论文撰写的 V3 方案。

仓库当前根目录即最终 V3 版本；`final/` 保存最终模型与输出成果，`V2.5/` 保存历史实验归档。

## 项目任务

给定透明塑料饮料瓶图像，识别瓶内残留液体属于以下四个等级之一：

| 类别 | 含义 | 说明 |
| --- | --- | --- |
| `empty` | 空瓶 | 无明显残留液体 |
| `low` | 少量残留 | 液面较低、残留量少 |
| `medium` | 中等残留 | 液面处于中间范围 |
| `high` | 较多残留 | 液面较高、残留量多 |

该任务的主要难点包括：

- 透明瓶体边缘和液面边界不明显；
- 光照、背景、瓶身反光会影响视觉特征；
- `low` 类样本较少，容易与 `empty` 或 `medium` 混淆；
- 单阶段四分类模型容易同时承担“是否有液体”和“液体高度细分”两个不同判别任务。

## 最终方案概览

V3 最终采用 EfficientNetV2-S 作为视觉骨干，并使用两阶段分类策略：

```text
输入瓶体图像
   |
   v
Stage A: empty vs non-empty
   |
   +-- empty -> 输出 empty
   |
   +-- non-empty
           |
           v
Stage B: low / medium / high
           |
           v
输出最终四分类结果
```

最终模型配置：

| 模块 | 模型 | 输入尺寸 | 任务 | 选用种子 |
| --- | --- | --- | --- | --- |
| Stage A | EfficientNetV2-S | 480 x 240 | `empty` / `non-empty` | 2026 |
| Stage B | EfficientNetV2-S | 480 x 240 | `low` / `medium` / `high` | 42 |

该设计将“是否为空瓶”和“非空瓶液位细分”拆开处理，更符合问题结构，也便于在论文中解释模型设计动机。

## 最终结果

最终测试集结果保存在 `final/outputs/metrics/final_test_report.txt`。

| 指标 | 数值 |
| --- | ---: |
| End-to-end 4-class Accuracy | 0.8593 |
| Q1 Empty vs Non-empty Accuracy | 0.9407 |
| Q2 Low/Medium/High Accuracy | 0.8256 |
| Macro F1 | 0.819 |
| Weighted F1 | 0.858 |
| Test Samples | 135 |

各类别结果：

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| empty | 0.918 | 0.918 | 0.918 | 49 |
| low | 0.750 | 0.643 | 0.692 | 14 |
| medium | 0.773 | 0.810 | 0.791 | 21 |
| high | 0.865 | 0.882 | 0.874 | 51 |

论文和答辩中建议重点展示：

- `final/outputs/figures/final_pipeline_flow.png`：最终流程图；
- `final/outputs/figures/final_model_architecture.png`：两阶段模型结构图；
- `final/outputs/figures/final_confusion_matrix.png`：最终混淆矩阵；
- `final/outputs/figures/final_dataset_distribution.png`：数据集分布；
- `final/outputs/figures/final_dataset_samples.png`：各类别样例。

## 项目结构

```text
.
|-- README.md                         # 项目首页说明
|-- requirements.txt                  # Python 依赖
|-- run_all.bat                       # Windows 一键重建 final/ 输出
|-- configs/                          # 配置文件
|-- scripts/                          # 当前可执行脚本
|   |-- run_all_v2.py                  # V2/V3 训练评估流程入口
|   |-- 15_train_two_stage.py          # 两阶段模型训练
|   |-- 16_eval_two_stage.py           # 两阶段模型评估
|   `-- 17_eval_final_hybrid.py        # 生成 V3 最终交付包
|-- final/                            # V3 最终模型、指标、图表和数据切分
|   |-- models/
|   |-- outputs/
|   |   |-- metrics/
|   |   `-- figures/
|   |-- data/
|   `-- docs/
`-- V2.5/                             # 历史 V2.5 代码、文档和实验记录归档
```

说明：为了保持 GitHub 仓库清晰，当前上传版本不包含完整训练数据、大量中间模型和全部实验输出。最终成果已集中保存在 `final/`，历史实验摘要保存在 `V2.5/`。

## 环境配置

建议环境：

- Python 3.9 或以上；
- PyTorch + torchvision；
- 如需重新训练，推荐使用支持 CUDA 的 GPU 环境。

安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 Conda：

```bash
conda create -n bottle-v3 python=3.10 -y
conda activate bottle-v3
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 快速开始

### 1. 查看最终结果

无需重新训练，直接查看：

```text
final/outputs/metrics/final_test_report.txt
final/outputs/metrics/final_classification_metrics.csv
final/outputs/metrics/final_test_predictions.csv
final/outputs/figures/final_confusion_matrix.png
```

### 2. 重建最终交付包

Windows 用户可双击：

```text
run_all.bat
```

或者在命令行运行：

```bash
python scripts/17_eval_final_hybrid.py
```

该脚本会重新生成：

- `final/models/efficientnet_v2_s_480x240_hybrid/`
- `final/outputs/metrics/final_test_report.txt`
- `final/outputs/metrics/final_test_predictions.csv`
- `final/outputs/metrics/final_classification_metrics.csv`
- `final/outputs/figures/final_confusion_matrix.png`
- `final/outputs/figures/final_pipeline_flow.png`
- `final/outputs/figures/final_model_architecture.png`
- `final/outputs/figures/final_dataset_distribution.png`
- `final/outputs/figures/final_dataset_samples.png`

### 3. 重新运行训练和评估流程

如本地拥有完整数据和训练资产，可运行：

```bash
python scripts/run_all_v2.py
```

常用参数：

```bash
python scripts/run_all_v2.py --use-final-split
python scripts/run_all_v2.py --skip-train
python scripts/run_all_v2.py --skip-eval
```

其中：

- `--use-final-split`：复用固定最终数据划分；
- `--skip-train`：只执行数据切分和重标注应用；
- `--skip-eval`：训练后不继续评估。

## 数据与标注规范

原始完整数据在本地实验环境中组织如下：

```text
data/
|-- raw_images/              # 原始瓶体图像
|-- annotations_json/        # LabelMe JSON 标注
|-- relabel.csv              # 重标注记录
|-- resnet_cls_v2/           # 两阶段分类数据集
`-- yolo_detect/             # 早期检测数据集
```

LabelMe 标注类别：

| LabelMe 标签 | 映射类别 |
| --- | --- |
| `bottle_empty` | `empty` |
| `bottle_low` | `low` |
| `bottle_medium` | `medium` |
| `bottle_high` | `high` |

最终交付包中保留了固定切分和重标注副本：

- `final/data/final_split_list.csv`
- `final/data/final_split_stats.csv`
- `final/data/final_split_list_v2.csv`
- `final/data/final_split_stats_v2.csv`
- `final/data/relabel.csv`

## 方法演进

项目经历了从单阶段 baseline 到两阶段模型的迭代：

| 版本 | 主要方法 | 说明 |
| --- | --- | --- |
| V1 | YOLO + ResNet50 单阶段四分类 | 初始 baseline，作为论文对比基础 |
| V1.5 | 修复路径、BOM、检测阈值和训练流程 | 提升检测稳定性 |
| V2/V2.5 | 两阶段分类框架 | 将空瓶判别和液位细分拆开 |
| V3 | EfficientNetV2-S 480x240 hybrid two-stage | 当前最终提交方案 |

`V2.5/` 中保留了历史代码、脚本、实验日志和结果摘要，便于论文中做版本对比。V1/V1.5 材料未放入本次 GitHub 根目录，以保持 V3 发布包简洁；本地仍可保留其原始实验材料作为论文证据。

## 最终成果清单

| 文件 | 用途 |
| --- | --- |
| `final/docs/final_manifest.txt` | 最终方案摘要 |
| `final/docs/experiment_archive_note.txt` | 实验归档说明 |
| `final/models/efficientnet_v2_s_480x240_hybrid/stage_a_empty_vs_not_seed2026.pth` | Stage A 最终权重 |
| `final/models/efficientnet_v2_s_480x240_hybrid/stage_b_low_med_high_seed42.pth` | Stage B 最终权重 |
| `final/outputs/metrics/final_test_report.txt` | 最终测试报告 |
| `final/outputs/metrics/final_classification_metrics.csv` | 分类指标表 |
| `final/outputs/metrics/final_test_predictions.csv` | 测试集逐样本预测 |
| `final/outputs/figures/final_confusion_matrix.png` | 最终混淆矩阵 |
| `final/outputs/figures/final_pipeline_flow.png` | 视觉流程图 |
| `final/outputs/figures/final_model_architecture.png` | 模型结构图 |
| `final/outputs/figures/final_dataset_distribution.png` | 数据分布图 |
| `final/outputs/figures/final_dataset_samples.png` | 数据样例图 |

## 适合论文撰写的分析角度

- 将任务拆分为“是否有残留液体”和“残留液体等级识别”；
- 用两阶段结构解释模型精度提升；
- 结合混淆矩阵分析 `low` 类的主要误差来源；
- 用数据分布图说明类别不均衡对结果的影响；
- 用 V1、V2.5、V3 的结果对比体现建模迭代过程。

## 备注

本仓库用于数学建模和机器视觉实验展示。若需要完全复现实验训练过程，请准备完整原始图片、LabelMe 标注文件以及相应训练数据目录；若仅需要查看最终模型与结果，直接使用 `final/` 即可。
