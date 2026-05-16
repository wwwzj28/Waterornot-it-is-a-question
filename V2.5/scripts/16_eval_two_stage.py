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

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, resnet50

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet50", "efficientnet_v2_s"], default="resnet50")
    parser.add_argument("--input-h", type=int, default=384, help="Input image height")
    parser.add_argument("--input-w", type=int, default=192, help="Input image width")
    parser.add_argument("--run-name", default=None, help="Optional output suffix for experiments")
    return parser.parse_args()


def backbone_output_name(backbone: str, run_name: str | None = None) -> str:
    name = "resnet" if backbone == "resnet50" else backbone
    return f"{name}_{run_name}" if run_name else name


def output_paths(backbone: str, run_name: str | None = None):
    if backbone == "resnet50" and not run_name:
        return (
            METRICS_DIR / "two_stage_test_predictions.csv",
            METRICS_DIR / "two_stage_test_report.txt",
            FIGURES_DIR / "two_stage_confusion_matrix.png",
        )
    suffix = backbone_output_name(backbone, run_name)
    return (
        METRICS_DIR / f"two_stage_test_predictions_{suffix}.csv",
        METRICS_DIR / f"two_stage_test_report_{suffix}.txt",
        FIGURES_DIR / f"two_stage_confusion_matrix_{suffix}.png",
    )

eval_tf = transforms.Compose([
    transforms.Resize((INPUT_H, INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path: Path, num_classes: int, backbone: str) -> nn.Module:
    if backbone == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def predict_image(img_path: Path, model_a, model_b, eval_tf):
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


def confusion_counts(true_labels, pred_labels, labels):
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label in idx and pred_label in idx:
            matrix[idx[true_label]][idx[pred_label]] += 1
    return matrix


def render_classification_report(true_labels, pred_labels, labels, digits=3):
    rows = []
    total = len(true_labels)
    correct_total = sum(t == p for t, p in zip(true_labels, pred_labels))
    supports = []
    precisions = []
    recalls = []
    f1s = []
    weights = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true_labels, pred_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, pred_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, pred_labels))
        support = sum(t == label for t in true_labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((label, precision, recall, f1, support))
        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weights.append(support)
    macro = (
        sum(precisions) / len(labels),
        sum(recalls) / len(labels),
        sum(f1s) / len(labels),
    )
    weighted = (
        sum(p * w for p, w in zip(precisions, weights)) / total if total else 0.0,
        sum(r * w for r, w in zip(recalls, weights)) / total if total else 0.0,
        sum(f * w for f, w in zip(f1s, weights)) / total if total else 0.0,
    )
    fmt = f"{{:>12}} {{:>10.{digits}f}} {{:>9.{digits}f}} {{:>9.{digits}f}} {{:>9}}"
    lines = ["              precision    recall  f1-score   support", ""]
    for label, precision, recall, f1, support in rows:
        lines.append(fmt.format(label, precision, recall, f1, support))
    lines.extend([
        "",
        f"    accuracy                          {correct_total / total if total else 0.0:>{9}.{digits}f} {total:>9}",
        fmt.format("macro avg", macro[0], macro[1], macro[2], total),
        fmt.format("weighted avg", weighted[0], weighted[1], weighted[2], total),
    ])
    return "\n".join(lines) + "\n"


def save_confusion_matrix_png(matrix, labels, title, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    model_dir = MODEL_DIR / backbone_output_name(args.backbone, args.run_name) / "two_stage"
    stage_a_path = model_dir / "stage_a_empty_vs_not.pth"
    stage_b_path = model_dir / "stage_b_low_med_high.pth"
    pred_csv, report_txt, cm_png = output_paths(args.backbone, args.run_name)

    eval_tf = transforms.Compose([
        transforms.Resize((args.input_h, args.input_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not stage_a_path.exists() or not stage_b_path.exists():
        raise FileNotFoundError("missing trained models, run 15_train_two_stage.py first")

    model_a = load_model(stage_a_path, 2, args.backbone)
    model_b = load_model(stage_b_path, 3, args.backbone)

    test_root = CLS_DIR_V2 / "test"
    rows = []
    for cls_dir in sorted(test_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            pred_label, p_nonempty, b_label, p_b = predict_image(p, model_a, model_b, eval_tf)
            rows.append({
                "image": p.name,
                "true_label": true_label,
                "pred_label": pred_label,
                "p_nonempty": round(p_nonempty, 4),
                "stage_b_pred": b_label or "",
                "p_stage_b": round(p_b, 4) if p_b is not None else "",
            })

    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with pred_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_label", "pred_label", "p_nonempty", "stage_b_pred", "p_stage_b"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-image predictions: {pred_csv}")
    print()

    true_labels = [row["true_label"] for row in rows]
    pred_labels = [row["pred_label"] for row in rows]

    # 1) 端到端 4 类
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    e2e_acc = correct / len(rows) if rows else 0.0
    cm = confusion_counts(true_labels, pred_labels, ALL_LABELS)
    print(f"=== End-to-end 4-class test acc: {e2e_acc:.4f} ({correct}/{len(rows)}) ===")
    print()
    print("Confusion (rows=true, cols=pred):")
    print("              " + "  ".join(f"{l:>7}" for l in ALL_LABELS))
    for l, row in zip(ALL_LABELS, cm):
        print(f"{l:>12}", "  ".join(f"{v:>7d}" for v in row))
    print()

    # 2) Stage A 二分类等价（empty vs 非empty）
    true_q1 = [0 if label == "empty" else 1 for label in true_labels]
    pred_q1 = [0 if label == "empty" else 1 for label in pred_labels]
    q1_correct = sum(t == p for t, p in zip(true_q1, pred_q1))
    q1_acc = q1_correct / len(true_q1) if true_q1 else 0.0
    cm_q1 = confusion_counts(true_q1, pred_q1, [0, 1])
    print(f"=== Q1 (empty vs not) test acc: {q1_acc:.4f} ===")
    print(cm_q1)
    print()

    # 3) Q2 子集（仅在真实非 empty 的样本上看 low/medium/high 准确率）
    q2_rows = [row for row in rows if row["true_label"] != "empty"]
    q2_acc = 0.0
    if q2_rows:
        q2_true = [row["true_label"] for row in q2_rows]
        q2_pred = [row["pred_label"] for row in q2_rows]
        q2_correct = sum(t == p for t, p in zip(q2_true, q2_pred))
        q2_acc = q2_correct / len(q2_rows)
        print(f"=== Q2 (low/medium/high on true-non-empty subset) test acc: {q2_acc:.4f} ({q2_correct}/{len(q2_rows)}) ===")
        cm2 = confusion_counts(q2_true, q2_pred, ["low", "medium", "high", "empty"])
        print("Confusion incl. 'empty' (Stage A 把它判 empty 时):")
        print("              " + "  ".join(f"{l:>7}" for l in ["low", "medium", "high", "empty"]))
        for l, row in zip(["low", "medium", "high"], cm2[:3]):
            print(f"{l:>12}", "  ".join(f"{v:>7d}" for v in row))
        print()

    print("=== Per-class report (4-class) ===")
    rep = render_classification_report(true_labels, pred_labels, ALL_LABELS, digits=3)
    print(rep)

    # 写报告
    with report_txt.open("w", encoding="utf-8-sig") as f:
        f.write(f"End-to-end 4-class test acc: {e2e_acc:.4f}\n")
        f.write(f"Q1 (empty vs not) test acc:  {q1_acc:.4f}\n")
        if q2_rows:
            f.write(f"Q2 (low/med/high on true-nonempty) test acc: {q2_acc:.4f}\n")
        f.write("\n")
        f.write(rep)
    print(f"\nSaved report: {report_txt}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_png(cm, ALL_LABELS, f"Two-stage end-to-end (test acc={e2e_acc:.3f})", cm_png)
    print(f"Saved confusion matrix: {cm_png}")


if __name__ == "__main__":
    main()
