# 透明塑料饮料瓶残留液体识别项目

## 1. 项目简介

本项目用于识别透明塑料饮料瓶内的残留液体含量。项目采用“两阶段模型”方案：

1. **YOLO11 检测模型**：自动定位图片中的瓶体区域；
2. **ResNet50 分类模型**：对裁剪后的瓶体图像进行液体含量四分类。

最终识别类别包括：

```text
empty   # 空瓶 / 无明显残留液体
low     # 少量残留液体
medium  # 中等残留液体
high    # 较多残留液体
```

整体流程如下：

```text
原始图片
   ↓
LabelMe 标注检查
   ↓
数据集划分 train / val / test
   ↓
LabelMe JSON 转 YOLO 标签
   ↓
YOLO11 训练瓶体检测模型
   ↓
使用人工框或 YOLO 框裁剪瓶体
   ↓
ResNet50 训练液体含量分类模型
   ↓
检测评估 + 分类评估
   ↓
YOLO → 裁剪 → ResNet50 端到端预测
   ↓
生成结果图表与论文可视化材料
```

---

## 2. 项目特点

- 使用 **YOLO11** 自动定位瓶体，减少手动裁剪工作量；
- 使用 **ResNet50** 进行液体含量四分类，结构稳定、易复现；
- 支持 LabelMe JSON 标注格式；
- 支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 等常见图片格式；
- 所有路径统一由 `scripts/_paths.py` 管理；
- 训练、评估、预测和可视化流程均脚本化；
- 提供一键总控脚本，方便完整复现实验流程。

---

## 3. 项目目录结构

推荐项目目录如下：

```text
bottle_project/
│
├── data/
│   ├── raw_images/                 # 原始图片，支持 .jpg/.jpeg/.png/.bmp
│   ├── annotations_json/           # LabelMe JSON 标注文件
│   │
│   ├── yolo_detect/                # YOLO 检测数据集
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── resnet_cls/                 # ResNet50 分类数据集
│       ├── train/
│       │   ├── empty/
│       │   ├── low/
│       │   ├── medium/
│       │   └── high/
│       ├── val/
│       │   ├── empty/
│       │   ├── low/
│       │   ├── medium/
│       │   └── high/
│       └── test/
│           ├── empty/
│           ├── low/
│           ├── medium/
│           └── high/
│
├── configs/
│   ├── bottle_detect.yaml          # YOLO11 数据配置文件
│   └── train_config.yaml           # 通用训练配置文件
│
├── scripts/
│   ├── _paths.py                   # 项目路径统一管理
│   ├── 00_smoke_test.py            # 环境和目录快速测试
│   ├── 01_check_json.py            # 检查 LabelMe JSON 标注格式
│   ├── 02_split_dataset.py         # 划分 train/val/test 数据集
│   ├── 03_json_to_yolo.py          # JSON 标注转 YOLO txt 标签
│   ├── 04_train_yolo11.py          # 训练 YOLO11 检测模型
│   ├── 05_crop_bottle_by_label.py  # 使用人工标注框裁剪瓶体
│   ├── 06_crop_bottle_by_yolo.py   # 使用 YOLO 检测框裁剪瓶体
│   ├── 07_train_resnet50.py        # 训练 ResNet50 分类模型
│   ├── 08_eval_detection.py        # 评估 YOLO 检测效果
│   ├── 09_eval_classification.py   # 评估 ResNet50 分类效果
│   ├── 10_predict_pipeline.py      # 端到端预测流程
│   ├── 11_visualize_results.py     # 生成论文图表和可视化结果
│   └── 12_run_all.py               # 一键执行总控脚本
│
├── models/
│   ├── yolo/
│   │   └── yolo11_best.pt          # YOLO11 最佳权重
│   └── resnet/
│       └── resnet50_best.pth       # ResNet50 最佳权重
│
├── outputs/
│   ├── crops/                      # 裁剪后的瓶体图片
│   ├── figures/                    # 可视化图表
│   ├── metrics/                    # 评估指标和 CSV 结果
│   └── predictions/                # 端到端预测结果
│
├── runs/                           # 训练日志和中间输出
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明文档
├── AI_usage_statement.md           # AI 使用说明
├── run_all.bat                     # Windows 一键运行入口
└── run_all.sh                      # macOS/Linux 一键运行入口
```

---

## 4. 环境配置

### 4.1 创建 Python 环境

建议使用 Python 3.9 或以上版本。

使用 Conda：

```bash
conda create -n bottle_project python=3.10 -y
conda activate bottle_project
```

或者使用 venv：

```bash
python -m venv .venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

---

### 4.2 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

如果使用 GPU 训练，请根据自己的 CUDA 版本安装对应的 PyTorch。可先安装基础依赖，再单独确认 `torch` 是否正确识别 GPU。

检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出为 `True`，说明 PyTorch 可以使用 GPU。

---

## 5. 数据准备

### 5.1 原始图片

将所有原始图片放入：

```text
data/raw_images/
```

支持的图片格式：

```text
.jpg
.jpeg
.png
.bmp
```

建议图片命名不要包含中文、空格或特殊符号。例如：

```text
bottle_0001.jpg
bottle_0002.jpg
bottle_0003.jpg
```

---

### 5.2 LabelMe 标注文件

将 LabelMe 生成的 JSON 文件放入：

```text
data/annotations_json/
```

要求：

1. JSON 文件必须与原始图片同名；
2. 例如图片为 `bottle_0001.jpg`，标注文件应为 `bottle_0001.json`；
3. 每个 JSON 中至少包含一个矩形框；
4. 每个矩形框应标注为正式标签之一。

---

## 6. LabelMe 标注规范

每个 JSON 文件应包含以下关键字段：

```text
shapes
imagePath
imageWidth
imageHeight
```

其中：

- `shapes`：标注列表；
- `label`：瓶体及液体含量标签；
- `points`：矩形框坐标；
- `imagePath`：对应图片文件名；
- `imageWidth` / `imageHeight`：原始图片宽高。

---

### 6.1 正式标签

本项目使用以下四种 LabelMe 标签：

```text
bottle_empty
bottle_low
bottle_medium
bottle_high
```

对应含义如下：

| LabelMe 标签 | 分类类别 | 含义 |
|---|---|---|
| `bottle_empty` | `empty` | 空瓶或无明显残留液体 |
| `bottle_low` | `low` | 少量残留液体 |
| `bottle_medium` | `medium` | 中等残留液体 |
| `bottle_high` | `high` | 较多残留液体 |

---

### 6.2 矩形框格式

LabelMe 中矩形框的 `points` 应为两个点：

```python
[[xmin, ymin], [xmax, ymax]]
```

示例：

```json
{
  "label": "bottle_low",
  "points": [
    [120.5, 80.0],
    [420.0, 680.5]
  ],
  "shape_type": "rectangle"
}
```

---

### 6.3 分类标签映射

检测阶段只检测瓶体，类别统一为：

```text
bottle
```

分类阶段再将 LabelMe 标签映射为四类：

```python
CLASS_MAP = {
    "bottle_empty": "empty",
    "bottle_low": "low",
    "bottle_medium": "medium",
    "bottle_high": "high"
}
```

---

## 7. 脚本说明

| 脚本 | 功能 |
|---|---|
| `00_smoke_test.py` | 快速检查项目环境、目录和基础依赖是否正常 |
| `01_check_json.py` | 检查 LabelMe JSON 是否规范，包括 `shapes`、`label`、`points`、bbox 范围和图片尺寸 |
| `02_split_dataset.py` | 按类别分层划分 train/val/test，并生成 `data/split_list.csv` |
| `03_json_to_yolo.py` | 将 LabelMe JSON 转换为 YOLO txt 标签，并复制图片到 `data/yolo_detect/images/` |
| `04_train_yolo11.py` | 训练 YOLO11 检测模型，训练后复制最佳权重到 `models/yolo/yolo11_best.pt` |
| `05_crop_bottle_by_label.py` | 使用人工标注框裁剪瓶体，生成 ResNet50 分类训练数据 |
| `06_crop_bottle_by_yolo.py` | 使用训练好的 YOLO11 模型裁剪瓶体，用于端到端测试 |
| `07_train_resnet50.py` | 训练 ResNet50 四分类模型，保存最佳权重到 `models/resnet/resnet50_best.pth` |
| `08_eval_detection.py` | 评价 YOLO11 检测效果，输出 IoU、Precision、Recall 和 CSV 结果 |
| `09_eval_classification.py` | 评价 ResNet50 分类效果，输出 Accuracy、F1、混淆矩阵和 CSV 结果 |
| `10_predict_pipeline.py` | 完整端到端预测：YOLO 检测 → 裁剪瓶体 → ResNet50 分类 → 输出预测和可视化 |
| `11_visualize_results.py` | 生成论文图表，包括数据集样例、YOLO 检测结果、裁剪图、训练曲线、混淆矩阵等 |
| `12_run_all.py` | 一键执行完整实验流程 |

---

## 8. 推荐运行顺序

完整实验推荐按以下顺序执行：

```text
01_check_json.py
→ 02_split_dataset.py
→ 03_json_to_yolo.py
→ 05_crop_bottle_by_label.py
→ 04_train_yolo11.py
→ 07_train_resnet50.py
→ 08_eval_detection.py
→ 09_eval_classification.py
→ 06_crop_bottle_by_yolo.py
→ 10_predict_pipeline.py
→ 11_visualize_results.py
```

该顺序的原因是：

1. 先检查标注文件，避免错误标注影响后续训练；
2. 再划分数据集，保证检测和分类阶段使用一致的数据划分；
3. 将 JSON 转为 YOLO 格式，供 YOLO11 训练；
4. 使用人工标注框裁剪瓶体，得到质量更高的 ResNet50 分类训练数据；
5. 分别训练检测模型和分类模型；
6. 分别评估检测与分类性能；
7. 使用 YOLO 检测框进行端到端裁剪和预测；
8. 最后生成论文或汇报所需图表。

---

## 9. 一键运行

如果已经添加 `12_run_all.py`，可以直接在项目根目录执行：

```bash
python scripts/12_run_all.py --mode all
```

该命令会依次执行：

```text
数据检查
数据划分
YOLO 标签转换
人工框裁剪
YOLO11 训练
ResNet50 训练
YOLO 检测评估
ResNet50 分类评估
YOLO 框裁剪
端到端预测
论文图表生成
```

---

### 9.1 分阶段运行

只进行数据准备：

```bash
python scripts/12_run_all.py --mode prepare
```

只训练模型：

```bash
python scripts/12_run_all.py --mode train
```

只进行模型评估：

```bash
python scripts/12_run_all.py --mode eval
```

只进行端到端预测：

```bash
python scripts/12_run_all.py --mode predict
```

只生成可视化图表：

```bash
python scripts/12_run_all.py --mode visualize
```

---

### 9.2 预览流程但不执行

```bash
python scripts/12_run_all.py --mode all --dry-run
```

该命令只会打印即将执行的步骤，不会真正运行训练或预测。

---

### 9.3 跳过部分阶段

完整流程中跳过训练阶段：

```bash
python scripts/12_run_all.py --mode all --skip-train
```

跳过评估阶段：

```bash
python scripts/12_run_all.py --mode all --skip-eval
```

跳过端到端预测阶段：

```bash
python scripts/12_run_all.py --mode all --skip-predict
```

跳过可视化阶段：

```bash
python scripts/12_run_all.py --mode all --skip-visualize
```

---

### 9.4 Windows 双击运行

如果项目根目录下存在 `run_all.bat`，可以直接双击运行完整流程。

`run_all.bat` 内容示例：

```bat
@echo off
chcp 65001 > nul
python scripts\12_run_all.py --mode all
pause
```

---

### 9.5 macOS / Linux 一键运行

如果项目根目录下存在 `run_all.sh`，先赋予执行权限：

```bash
chmod +x run_all.sh
```

然后执行：

```bash
./run_all.sh
```

`run_all.sh` 内容示例：

```bash
#!/usr/bin/env bash
set -e
python3 scripts/12_run_all.py --mode all
```

---

## 10. 单独运行各阶段脚本

如果不使用一键脚本，也可以逐步运行。

### 10.1 环境快速检查

```bash
python scripts/00_smoke_test.py
```

---

### 10.2 检查 JSON 标注

```bash
python scripts/01_check_json.py
```

该脚本主要检查：

- JSON 是否能正常读取；
- 是否存在 `shapes`；
- 标签是否属于正式标签；
- 标注框 `points` 是否为两个点；
- bbox 是否越界；
- 是否存在 `imageWidth` 和 `imageHeight`；
- JSON 文件是否能匹配到对应原图。

---

### 10.3 划分数据集

```bash
python scripts/02_split_dataset.py
```

输出：

```text
data/split_list.csv
```

`split_list.csv` 通常包含：

```text
image_name,json_name,label,split
```

其中 `split` 为：

```text
train
val
test
```

---

### 10.4 转换 YOLO 标签

```bash
python scripts/03_json_to_yolo.py
```

输出目录：

```text
data/yolo_detect/images/train/
data/yolo_detect/images/val/
data/yolo_detect/images/test/

data/yolo_detect/labels/train/
data/yolo_detect/labels/val/
data/yolo_detect/labels/test/
```

YOLO 标签格式为：

```text
class_id x_center y_center width height
```

由于检测阶段只检测瓶体，所以 `class_id` 通常为：

```text
0
```

---

### 10.5 使用人工框裁剪瓶体

```bash
python scripts/05_crop_bottle_by_label.py
```

输出目录：

```text
data/resnet_cls/train/empty/
data/resnet_cls/train/low/
data/resnet_cls/train/medium/
data/resnet_cls/train/high/

data/resnet_cls/val/empty/
data/resnet_cls/val/low/
data/resnet_cls/val/medium/
data/resnet_cls/val/high/

data/resnet_cls/test/empty/
data/resnet_cls/test/low/
data/resnet_cls/test/medium/
data/resnet_cls/test/high/
```

---

### 10.6 训练 YOLO11

```bash
python scripts/04_train_yolo11.py
```

训练完成后，最佳权重保存为：

```text
models/yolo/yolo11_best.pt
```

训练日志通常保存在：

```text
runs/
```

---

### 10.7 训练 ResNet50

```bash
python scripts/07_train_resnet50.py
```

训练完成后，最佳权重保存为：

```text
models/resnet/resnet50_best.pth
```

---

### 10.8 评估 YOLO11 检测模型

```bash
python scripts/08_eval_detection.py
```

输出内容通常包括：

```text
IoU
Precision
Recall
检测结果 CSV
```

结果保存位置：

```text
outputs/metrics/
```

---

### 10.9 评估 ResNet50 分类模型

```bash
python scripts/09_eval_classification.py
```

输出内容通常包括：

```text
Accuracy
F1-score
Confusion Matrix
分类结果 CSV
```

结果保存位置：

```text
outputs/metrics/
outputs/figures/
```

---

### 10.10 使用 YOLO 框裁剪瓶体

```bash
python scripts/06_crop_bottle_by_yolo.py
```

输出位置：

```text
outputs/crops/
```

---

### 10.11 端到端预测

```bash
python scripts/10_predict_pipeline.py
```

该脚本执行：

```text
输入原始图片
→ YOLO11 检测瓶体
→ 选择置信度最高的瓶体框
→ 裁剪瓶体区域
→ ResNet50 判断 empty / low / medium / high
→ 保存预测结果和可视化图片
```

输出位置：

```text
outputs/predictions/
outputs/figures/
```

---

### 10.12 生成可视化结果

```bash
python scripts/11_visualize_results.py
```

可生成：

- 数据集样例图；
- YOLO 检测可视化；
- 瓶体裁剪效果图；
- ResNet50 训练曲线；
- 分类混淆矩阵；
- 端到端预测展示图。

输出目录：

```text
outputs/figures/
```

---

## 11. 配置文件说明

### 11.1 `configs/bottle_detect.yaml`

YOLO11 检测配置文件示例：

```yaml
path: data/yolo_detect
train: images/train
val: images/val
test: images/test

names:
  0: bottle
```

说明：

- `path`：YOLO 数据集根目录；
- `train`：训练图片路径；
- `val`：验证图片路径；
- `test`：测试图片路径；
- `names`：检测类别名。

本项目检测阶段只有一个类别：

```text
bottle
```

---

### 11.2 `configs/train_config.yaml`

训练配置文件可根据实际脚本实现进行设置，常见字段包括：

```yaml
seed: 42

split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

yolo:
  model: yolo11n.pt
  epochs: 100
  imgsz: 640
  batch: 16
  device: 0

resnet:
  epochs: 50
  batch_size: 32
  lr: 0.001
  image_size: 224
  num_classes: 4
```

如果没有单独配置文件，也可以在各脚本中直接修改对应参数。

---

## 12. 输出结果说明

### 12.1 模型权重

YOLO11 最佳权重：

```text
models/yolo/yolo11_best.pt
```

ResNet50 最佳权重：

```text
models/resnet/resnet50_best.pth
```

---

### 12.2 中间裁剪结果

人工框裁剪结果：

```text
data/resnet_cls/
```

YOLO 框裁剪结果：

```text
outputs/crops/
```

---

### 12.3 评估指标

评估指标保存位置：

```text
outputs/metrics/
```

可能包含：

```text
detection_metrics.csv
classification_metrics.csv
classification_report.csv
confusion_matrix.csv
predictions.csv
```

---

### 12.4 可视化图表

图表保存位置：

```text
outputs/figures/
```

可能包含：

```text
dataset_samples.png
yolo_detection_examples.png
crop_examples.png
resnet_training_curve.png
confusion_matrix.png
pipeline_predictions.png
```

---

## 13. 方法说明

### 13.1 为什么采用两阶段模型

由于透明塑料瓶与背景之间对比度可能较低，且液体残留区域容易受到反光、瓶身变形、拍摄角度等因素影响，直接对整张图片分类可能会受到背景干扰。

因此本项目采用两阶段方法：

1. 先用 YOLO11 检测瓶体区域；
2. 再对瓶体区域进行裁剪；
3. 最后用 ResNet50 判断液体残留程度。

这样可以让分类模型更关注瓶体内部特征，提高识别稳定性。

---

### 13.2 YOLO11 检测阶段

YOLO11 负责定位瓶体区域。虽然 LabelMe 标注中包含四种类别标签，但在检测阶段会统一为一个类别：

```text
bottle
```

这样做的原因是检测模型只需要学习“瓶子在哪里”，不需要判断瓶内液体多少。

---

### 13.3 ResNet50 分类阶段

ResNet50 输入的是瓶体裁剪图，输出四个类别之一：

```text
empty
low
medium
high
```

分类阶段更关注：

- 瓶底残留液体；
- 液面高度；
- 透明瓶体反光；
- 液体颜色和阴影；
- 瓶体内部区域纹理。

---

## 14. 实验建议

### 14.1 数据采集建议

为了提升模型泛化能力，建议数据集中包含：

- 不同形状的透明塑料瓶；
- 不同背景；
- 不同光照条件；
- 不同拍摄角度；
- 不同液体颜色；
- 不同残留液体高度；
- 空瓶、少量、中等、较多四类样本数量尽量均衡。

---

### 14.2 标注建议

标注瓶体框时建议：

- 尽量框住完整瓶体；
- 不要包含太多背景；
- 保持同一批数据的标注标准一致；
- 如果图片中有多个瓶子，应根据项目任务要求决定是否全部标注；
- 如果瓶子严重遮挡或无法判断液体含量，建议剔除或单独记录。

---

### 14.3 训练建议

- 数据量较少时，可优先使用预训练模型；
- YOLO11 可先使用较小模型，例如 `yolo11n.pt`，再根据效果尝试更大模型；
- ResNet50 可使用 ImageNet 预训练权重；
- 如果分类准确率较低，应优先检查裁剪质量和类别标注是否一致；
- 如果检测框不准，应检查 YOLO 标签转换是否正确；
- 如果某一类识别效果较差，应检查该类样本数量和拍摄条件是否充足。

---

## 15. 常见问题

### Q1：为什么检测阶段只有一个类别 `bottle`？

因为检测阶段只负责定位瓶体，不负责判断液体含量。液体含量由后续 ResNet50 分类模型完成。

---

### Q2：LabelMe 中为什么还要标 `bottle_empty`、`bottle_low` 等四类？

因为这些标签同时用于：

1. 生成 YOLO 检测框；
2. 生成 ResNet50 分类数据集；
3. 根据标签将裁剪图放入 `empty/low/medium/high` 文件夹。

---

### Q3：JSON 文件和图片文件必须同名吗？

建议必须同名。例如：

```text
data/raw_images/bottle_0001.jpg
data/annotations_json/bottle_0001.json
```

这样可以避免脚本匹配错误。

---

### Q4：图片是 `.jpeg` 后缀可以吗？

可以。脚本设计中应兼容：

```text
.jpg
.jpeg
.png
.bmp
```

---

### Q5：为什么要先用人工框裁剪训练 ResNet50？

人工标注框通常比模型预测框更准确。先使用人工框生成分类训练数据，可以让 ResNet50 学到更稳定的瓶体特征。

---

### Q6：为什么后面还要用 YOLO 框裁剪？

真实端到端预测时没有人工框，因此需要使用 YOLO11 自动检测框来裁剪瓶体，以验证完整系统在实际使用中的效果。

---

### Q7：如果 YOLO 检测出多个框怎么办？

默认选择置信度最高的检测框作为瓶体区域。

如果图片中存在多个瓶子，可以根据任务需求修改策略，例如：

- 保留所有检测框；
- 只保留最大框；
- 根据中心位置选择目标瓶；
- 对每个瓶子分别裁剪和分类。

---

### Q8：分类效果不好怎么办？

可以从以下方面排查：

1. 检查 `data/resnet_cls/` 中裁剪图片是否正确；
2. 检查四类标签是否有混淆；
3. 检查类别样本数量是否均衡；
4. 增加数据增强；
5. 尝试更长训练轮数；
6. 调整学习率；
7. 尝试其他分类模型，例如 EfficientNet、ConvNeXt、Swin Transformer。

---

### Q9：检测效果不好怎么办？

可以从以下方面排查：

1. 检查 YOLO txt 标签是否正确；
2. 检查 bbox 是否越界；
3. 检查训练集和验证集是否划分合理；
4. 增加不同背景和光照下的样本；
5. 增加训练轮数；
6. 尝试更大的 YOLO11 模型；
7. 检查是否存在漏标或错标。

---

### Q10：运行一键脚本失败怎么办？

先观察终端中显示的失败步骤。例如：

```text
[ERROR] 03_json_to_yolo.py 执行失败
```

然后单独运行该脚本：

```bash
python scripts/03_json_to_yolo.py
```

根据报错信息检查输入文件和路径。常见原因包括：

- 原始图片路径不正确；
- JSON 文件缺失；
- JSON 标签不属于正式标签；
- 图片和 JSON 不同名；
- 没有安装依赖；
- 模型权重文件不存在。

---

## 16. 复现实验流程

从零开始复现实验时，推荐步骤如下：

### 第一步：准备数据

```text
data/raw_images/
data/annotations_json/
```

确保原图和 JSON 标注文件已经放入对应目录。

---

### 第二步：安装依赖

```bash
pip install -r requirements.txt
```

---

### 第三步：检查环境

```bash
python scripts/00_smoke_test.py
```

---

### 第四步：运行完整流程

```bash
python scripts/12_run_all.py --mode all
```

---

### 第五步：查看输出结果

模型权重：

```text
models/yolo/yolo11_best.pt
models/resnet/resnet50_best.pth
```

评估指标：

```text
outputs/metrics/
```

可视化图表：

```text
outputs/figures/
```

端到端预测结果：

```text
outputs/predictions/
```

---

## 17. 项目结果汇报建议

论文或汇报中可以按照以下结构介绍本项目：

1. **研究背景**：透明塑料瓶残留液体识别在回收、分拣或质检中的意义；
2. **数据集构建**：图片采集、LabelMe 标注、类别设置；
3. **方法设计**：YOLO11 检测 + ResNet50 分类的两阶段框架；
4. **实验设置**：训练集/验证集/测试集划分、训练参数、评价指标；
5. **检测结果**：YOLO11 的 IoU、Precision、Recall；
6. **分类结果**：ResNet50 的 Accuracy、F1-score、混淆矩阵；
7. **端到端结果**：从原图输入到最终液体含量预测的完整效果；
8. **误差分析**：反光、遮挡、液面不明显、类别边界模糊等情况；
9. **改进方向**：增加数据、多模型对比、引入注意力机制或更强分类网络。

---

## 18. 注意事项

1. 所有脚本建议从项目根目录运行；
2. 所有路径建议通过 `scripts/_paths.py` 统一管理；
3. LabelMe JSON 必须与原图同名；
4. 检测阶段只识别 `bottle`；
5. 分类阶段再区分 `empty`、`low`、`medium`、`high`；
6. 一键脚本默认某一步失败后停止；
7. 修改目录结构后，需要同步修改 `_paths.py`；
8. 训练前请确认 GPU、CUDA 和 PyTorch 版本兼容；
9. 如果数据量较少，建议重点检查数据增强和类别均衡；
10. 实验结果应以测试集指标为准，不应只看训练集准确率。

---

## 19. 许可证

本项目主要用于课程设计、实验研究或学习交流。若用于实际生产环境，请根据数据来源、模型权重和第三方依赖库许可要求进行合规检查。

---

## 20. 致谢

本项目使用了 YOLO11、PyTorch、TorchVision、OpenCV、scikit-learn、matplotlib 等开源工具。感谢相关开源社区提供的模型、框架和工具支持。
