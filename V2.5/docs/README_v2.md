# V2 升级说明（写给接手数据的队友）

> 这份文档假设你已经收到我打包的 V2 工程。要做的就两件事：
> 1. 把在路上的 ~200 张新图（含 LabelMe JSON）丢进对应目录
> 2. 跑一行命令重训重评

---

## 0. V2 相对 V1 的关键变化

- **数据集**：经过一轮人工重标，68 张俯拍/严重遮挡样本被剔为 `invalid`，medium 阈值放宽（35–70%），统一在 `data/relabel.csv` 里。
- **建模方案**：从单一 4 类改成**两阶段**：
  - Stage A：`empty` vs `非empty`（二分类，用全部有效样本）
  - Stage B：`low` / `medium` / `high`（三分类，仅用非 empty 样本）
  - 推理时先过 A，empty 直接输出；非 empty 再过 B
- **输入尺寸**：从 `224×224` 改成竖长形 `384×192`(H×W)，垂直方向像素翻倍，水线特征不再被压扁。
- **训练策略**：两阶段微调（warmup 训 fc 头 → 全网解冻 + cosine LR），class_weight 抑制不平衡。
- **数据流**：`02_split_dataset.py` 和 `14_apply_relabel.py` 都会感知 `data/relabel.csv`，invalid 自动排除，重标过的样本以 relabel 为准。

V1 baseline test acc：0.5882。
V2 截至当前数据（250 张）的 test acc：**端到端 4 类 0.8158 / Q1 (empty) 0.9474 / Q2 (low/med/high) 0.8214**。

---

## 1. 你需要做的事（按顺序）

### 1.1 放新数据

```
data/raw_images/         ← 新拍的 ~200 张 .jpg/.jpeg/.png 直接放进来，文件名延续即可
data/annotations_json/   ← 对应的 LabelMe .json 也放进来，stem 和图片同名
```

LabelMe 标注的 `label` 字段必须是 `bottle_empty` / `bottle_low` / `bottle_medium` / `bottle_high` 之一（也接受不带 `bottle_` 前缀）。

### 1.2 不需要跑 GUI 重标

老数据的判定保存在 `data/relabel.csv` 里，新数据没有这个 CSV 记录，脚本会**自动回落到 LabelMe JSON 的原始 label**。所以你只要 LabelMe 标对了就行，不需要打开 `13_relabel_gui.py`。

> 例外：如果你发现某些**老样本**的判定有问题想改，或者新样本里有**应该排除的俯拍/严重遮挡**，再开 GUI：
> ```bash
> python scripts/13_relabel_gui.py --filter unlabeled   # 只过没标过的
> python scripts/13_relabel_gui.py                      # 全量过一遍
> ```
> GUI 里按键：1=empty 2=low 3=medium 4=high 0=invalid s=skip ←/→ 翻页
> 每次设标自动写盘到 `data/relabel.csv`。

### 1.3 一键重跑

```bash
python scripts/run_all_v2.py
```

这个脚本会按顺序跑：

1. `02_split_dataset.py`：基于全量数据重新切 70/15/15，invalid 排除，relabel 覆盖。
2. `14_apply_relabel.py`：按新 split 重新裁剪输出 `data/resnet_cls_v2/{train,val,test}/{empty,low,medium,high}/`。
3. `15_train_two_stage.py`：训 Stage A + Stage B（GPU 大约 7 分钟）。
4. `16_eval_two_stage.py`：在 test 集出 4 类 / Q1 / Q2 三组指标。

每一步的 stdout 都写到 `outputs/logs/run_all_v2/*.log` 留档。

调试用开关：
```bash
python scripts/run_all_v2.py --skip-train   # 只重切+重建数据集，看新分布
python scripts/run_all_v2.py --skip-eval    # 跑到训练为止
```

### 1.4 看结果

- 终端最后 30 行：4 类 acc + Q1 + Q2 + 混淆矩阵
- 详细：`outputs/metrics/two_stage_test_report.txt`、`outputs/metrics/two_stage_test_predictions.csv`
- 混淆矩阵图：`outputs/figures/two_stage_confusion_matrix.png`
- 模型权重：`models/resnet/two_stage/stage_a_empty_vs_not.pth`、`stage_b_low_med_high.pth`

---

## 2. 脚本一览（V2 新增/改造）

| 文件 | 状态 | 作用 |
|---|---|---|
| `02_split_dataset.py` | **改造** | 现在会读 `data/relabel.csv`，invalid 排除，new_label 覆盖 LabelMe 原始 label。`--ignore-relabel` 可关 |
| `13_relabel_gui.py` | 新增 | Tkinter GUI，并排显示原图+裁剪图，键盘打标到 relabel.csv。**新数据不需要跑** |
| `14_apply_relabel.py` | **改造** | 按 split + relabel + JSON 回落，重新裁出 `data/resnet_cls_v2/` |
| `15_train_two_stage.py` | 新增 | 两阶段训练：Stage A 二分类 + Stage B 三分类，warmup + cosine |
| `16_eval_two_stage.py` | 新增 | 端到端评估，给 4 类 / Q1 / Q2 三组指标 |
| `run_all_v2.py` | 新增 | 一键串：split → apply → train → eval |
| `diag_waterline_cv.py` | 新增 | 零样本 OpenCV 水线检测基线（**结果 23.5% 不如随机，已弃**，仅留作消融） |

V1 时代的 `07_train_resnet50.py` / `09_eval_classification.py` / `10_predict_pipeline.py` 仍然能跑（输入尺寸已改成 384×192），但**主提交方案是两阶段**，不要再用 `07/09/10`。

---

## 3. 关键路径速查

```
data/
├── raw_images/                    ← 你放新图的地方
├── annotations_json/              ← 你放新 LabelMe JSON 的地方
├── relabel.csv                    ← 老数据 GUI 判定（不要手动改）
├── resnet_cls/                    ← V1 残留，可以删
└── resnet_cls_v2/                 ← V2 数据集（apply 自动重建）
    └── {train,val,test}/{empty,low,medium,high}/

models/
├── resnet/two_stage/              ← V2 主提交模型
│   ├── stage_a_empty_vs_not.pth
│   └── stage_b_low_med_high.pth
├── resnet/resnet50_best.pth       ← V1 残留
└── yolo/yolo11_best.pt            ← 推理 pipeline 还在用

outputs/
├── metrics/
│   ├── split_list.csv             ← 02 输出，含每张图的 split
│   ├── split_stats.csv            ← 02 输出，分布统计
│   ├── split_stats_v2.csv         ← 14 输出，按 v2 实际落盘统计
│   ├── two_stage_test_report.txt  ← 16 输出，最重要的成绩单
│   └── two_stage_test_predictions.csv
├── figures/two_stage_confusion_matrix.png
└── logs/run_all_v2/               ← run_all_v2.py 各步日志
```

---

## 4. 期望结果与潜在瓶颈

加完新数据后预期：
- **low 类**：当前 train 只有 15 张是真正瓶颈，Stage B 的 low recall 是 0%。新数据如果让 low 训练量到 50+，Stage B 应该能到 70–80%。
- **empty**：Stage A 已经 0.95，再涨边际很小。
- **总体**：端到端 4 类 acc 预期从 0.82 → **0.88–0.92**。

如果跑完远低于预期：
- 看 `outputs/metrics/split_stats.csv`，确认四类的 train/val/test 分布是否合理（low 不要再 < 30）
- 看 `two_stage_test_predictions.csv`，挑出错样人工看一遍，多半能看到分布偏移或标注不一致

---

## 5. 我没来得及做的实验（队友可以试）

按预期收益从高到低：

1. **把 V2 的 384×192 输入再翻一倍到 480×240**，水线分辨率再 +1 倍。改 `15_train_two_stage.py:18` 和 `16_eval_two_stage.py:21` 的 `INPUT_H/INPUT_W`。代价：训练时间 +50%。
2. **轻量几何增强**：在 `15_train_two_stage.py:46` 的 `train_tf` 里加 `transforms.RandomAffine(degrees=5, translate=(0.05,0.05))` 和 `transforms.RandomResizedCrop((INPUT_H, INPUT_W), scale=(0.85, 1.0))`。**别加** RandomVerticalFlip / 大角度旋转 / Cutout / 强 ColorJitter，会破坏水线语义。
3. **WeightedRandomSampler**：在两阶段 `build_loaders` 里给 train 加按类倒数采样权重，配合现在的 weighted loss。预期对 Stage B 的 low 类有用。
4. **ConvNeXt-Tiny / EfficientNetV2-S 替换 backbone**：比 ResNet50 强，参数量差不多。

---

## 6. 兜底 / 排错

- `02 split` 报 "no valid label" → 检查那张图的 JSON 是不是用了别的 label（比如 `bottle_full` 这种没列入 CLASS_MAP 的）
- `14 apply` 报 "Skipped: {'no_bbox': N}" → JSON 里 shape 的 `points` 不是矩形两点对，让标注人重画
- 训练 val acc 一直在 0.4–0.5 上下震荡 → 多半是新加入的数据 split 有泄漏（同一个瓶子同时出现在 train 和 test），目前 `02_split` 不做按瓶 ID 去重，**如果新数据是同一个瓶子拍的多张要小心**
- Windows 下 `13_relabel_gui.py` 中文字体异常 → 那是 Tkinter 默认字体问题，不影响标注

---

## 7. 联系方式

V1 → V2 改造的全过程对话和实验记录在我本地的 Claude Code 会话里，需要原始诊断证据可以找我要。
