"""两阶段端到端评估：在 v2 test 集上模拟比赛推理流程，给出 4 类等价混淆矩阵。

推理流程：
  1. Stage A 判 empty vs 非 empty
  2. 若非 empty → Stage B 判 low/medium/high
  3. 综合得到 4 类预测：empty / low / medium / high

输出：
  outputs/metrics/two_stage_test_predictions.csv
  outputs/metrics/two_stage_test_report.txt
  outputs/figures/two_stage_confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms
from torchvision.models import resnet50

from _paths import CLS_DIR_V2, FIGURES_DIR, METRICS_DIR, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_H, INPUT_W = 384, 192

STAGE_A_PATH = MODEL_DIR / "resnet" / "two_stage" / "stage_a_empty_vs_not.pth"
STAGE_B_PATH = MODEL_DIR / "resnet" / "two_stage" / "stage_b_low_med_high.pth"

PRED_CSV = METRICS_DIR / "two_stage_test_predictions.csv"
REPORT_TXT = METRICS_DIR / "two_stage_test_report.txt"
CM_PNG = FIGURES_DIR / "two_stage_confusion_matrix.png"

# 注意：Stage B 三分类的 idx 顺序必须与训练时 label_to_idx 一致（low=0, medium=1, high=2）
STAGE_B_IDX2LBL = {0: "low", 1: "medium", 2: "high"}
ALL_LABELS = ["empty", "low", "medium", "high"]

eval_tf = transforms.Compose([
    transforms.Resize((INPUT_H, INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path: Path, num_classes: int) -> nn.Module:
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def predict_image(img_path: Path, model_a, model_b):
    img = Image.open(img_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_a = model_a(x)
        prob_a = torch.softmax(out_a, dim=1)[0]
        idx_a = int(prob_a.argmax())
        if idx_a == 0:
            return "empty", prob_a[0].item(), None, None
        out_b = model_b(x)
        prob_b = torch.softmax(out_b, dim=1)[0]
        idx_b = int(prob_b.argmax())
        return STAGE_B_IDX2LBL[idx_b], prob_a[1].item(), STAGE_B_IDX2LBL[idx_b], prob_b[idx_b].item()


def main():
    if not STAGE_A_PATH.exists() or not STAGE_B_PATH.exists():
        raise FileNotFoundError("missing trained models, run 15_train_two_stage.py first")

    model_a = load_model(STAGE_A_PATH, 2)
    model_b = load_model(STAGE_B_PATH, 3)

    test_root = CLS_DIR_V2 / "test"
    rows = []
    for cls_dir in sorted(test_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            pred_label, p_nonempty, b_label, p_b = predict_image(p, model_a, model_b)
            rows.append({
                "image": p.name,
                "true_label": true_label,
                "pred_label": pred_label,
                "p_nonempty": round(p_nonempty, 4),
                "stage_b_pred": b_label or "",
                "p_stage_b": round(p_b, 4) if p_b is not None else "",
            })

    df = pd.DataFrame(rows)
    PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved per-image predictions: {PRED_CSV}")
    print()

    # 1) 端到端 4 类
    e2e_acc = accuracy_score(df["true_label"], df["pred_label"])
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=ALL_LABELS)
    print(f"=== End-to-end 4-class test acc: {e2e_acc:.4f} ({(df.true_label == df.pred_label).sum()}/{len(df)}) ===")
    print()
    print("Confusion (rows=true, cols=pred):")
    print("              " + "  ".join(f"{l:>7}" for l in ALL_LABELS))
    for l, row in zip(ALL_LABELS, cm):
        print(f"{l:>12}", "  ".join(f"{v:>7d}" for v in row))
    print()

    # 2) Stage A 二分类等价（empty vs 非empty）
    df_q1 = df.copy()
    df_q1["true_q1"] = (df_q1["true_label"] != "empty").astype(int)
    df_q1["pred_q1"] = (df_q1["pred_label"] != "empty").astype(int)
    q1_acc = (df_q1["true_q1"] == df_q1["pred_q1"]).mean()
    print(f"=== Q1 (empty vs not) test acc: {q1_acc:.4f} ===")
    print(confusion_matrix(df_q1["true_q1"], df_q1["pred_q1"], labels=[0, 1]))
    print()

    # 3) Q2 子集（仅在真实非 empty 的样本上看 low/medium/high 准确率）
    df_q2 = df[df["true_label"] != "empty"]
    if len(df_q2) > 0:
        # 注意：Q2 acc 反映"全链路上有多少真非 empty 被正确细分"
        q2_correct = (df_q2["pred_label"] == df_q2["true_label"]).sum()
        q2_acc = q2_correct / len(df_q2)
        print(f"=== Q2 (low/medium/high on true-non-empty subset) test acc: {q2_acc:.4f} ({q2_correct}/{len(df_q2)}) ===")
        cm2 = confusion_matrix(df_q2["true_label"], df_q2["pred_label"],
                               labels=["low", "medium", "high", "empty"])
        print("Confusion incl. 'empty' (Stage A 把它判 empty 时):")
        print("              " + "  ".join(f"{l:>7}" for l in ["low", "medium", "high", "empty"]))
        for l, row in zip(["low", "medium", "high"], cm2[:3]):
            print(f"{l:>12}", "  ".join(f"{v:>7d}" for v in row))
        print()

    print("=== Per-class report (4-class) ===")
    rep = classification_report(df["true_label"], df["pred_label"],
                                labels=ALL_LABELS, zero_division=0, digits=3)
    print(rep)

    # 写报告
    with REPORT_TXT.open("w", encoding="utf-8-sig") as f:
        f.write(f"End-to-end 4-class test acc: {e2e_acc:.4f}\n")
        f.write(f"Q1 (empty vs not) test acc:  {q1_acc:.4f}\n")
        if len(df_q2) > 0:
            f.write(f"Q2 (low/med/high on true-nonempty) test acc: {q2_acc:.4f}\n")
        f.write("\n")
        f.write(rep)
    print(f"\nSaved report: {REPORT_TXT}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ALL_LABELS, yticklabels=ALL_LABELS, cbar=False)
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.title(f"Two-stage end-to-end (test acc={e2e_acc:.3f})")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=120)
    plt.close()
    print(f"Saved confusion matrix: {CM_PNG}")


if __name__ == "__main__":
    main()
