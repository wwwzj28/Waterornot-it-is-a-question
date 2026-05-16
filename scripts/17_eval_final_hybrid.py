"""Evaluate final hybrid two-stage model.

Final scheme:
- Stage A: EfficientNetV2-S 480x240 seed 2026
- Stage B: EfficientNetV2-S 480x240 seed 42
"""

from pathlib import Path
import csv
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from _paths import CLS_DIR_V2, FIGURES_DIR, METRICS_DIR, MODEL_DIR, RAW_IMAGES_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_H, INPUT_W = 480, 240
ALL_LABELS = ["empty", "low", "medium", "high"]
STAGE_B_IDX2LBL = {0: "low", 1: "medium", 2: "high"}

STAGE_A_SRC = MODEL_DIR / "efficientnet_v2_s_480x240" / "two_stage" / "stage_a_empty_vs_not.pth"
STAGE_B_SRC = METRICS_DIR / "efficientnet_480x240_multi_seed" / "seed_42" / "models" / "stage_b_low_med_high.pth"

FINAL_ROOT = Path("final")
FINAL_MODELS_DIR = FINAL_ROOT / "models" / "efficientnet_v2_s_480x240_hybrid"
FINAL_OUTPUTS_DIR = FINAL_ROOT / "outputs"
FINAL_METRICS_DIR = FINAL_OUTPUTS_DIR / "metrics"
FINAL_FIGURES_DIR = FINAL_OUTPUTS_DIR / "figures"
FINAL_DOCS_DIR = FINAL_ROOT / "docs"
FINAL_DATA_DIR = FINAL_ROOT / "data"

PRED_CSV = FINAL_METRICS_DIR / "final_test_predictions.csv"
REPORT_TXT = FINAL_METRICS_DIR / "final_test_report.txt"
CM_PNG = FINAL_FIGURES_DIR / "final_confusion_matrix.png"

EVAL_TF = transforms.Compose([
    transforms.Resize((INPUT_H, INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path: Path, num_classes: int):
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def predict_image(img_path: Path, model_a, model_b):
    img = Image.open(img_path).convert("RGB")
    x = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob_a = torch.softmax(model_a(x), dim=1)[0]
        prob_b = torch.softmax(model_b(x), dim=1)[0]
        p_empty = float(prob_a[0])
        p_nonempty = float(prob_a[1])
        if p_empty >= p_nonempty:
            return {
                "pred_label": "empty",
                "p_empty": p_empty,
                "p_nonempty": p_nonempty,
                "stage_b_pred": "",
                "p_low": float(prob_b[0]),
                "p_medium": float(prob_b[1]),
                "p_high": float(prob_b[2]),
                "p_stage_b": "",
            }
        idx_b = int(prob_b.argmax())
        label_b = STAGE_B_IDX2LBL[idx_b]
        return {
            "pred_label": label_b,
            "p_empty": p_empty,
            "p_nonempty": p_nonempty,
            "stage_b_pred": label_b,
            "p_low": float(prob_b[0]),
            "p_medium": float(prob_b[1]),
            "p_high": float(prob_b[2]),
            "p_stage_b": float(prob_b[idx_b]),
        }


def confusion_counts(true_labels, pred_labels, labels):
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[idx[true_label]][idx[pred_label]] += 1
    return matrix


def classification_metrics(true_labels, pred_labels, labels):
    total = len(true_labels)
    correct_total = sum(t == p for t, p in zip(true_labels, pred_labels))
    rows = []
    precisions, recalls, f1s, supports = [], [], [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true_labels, pred_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, pred_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, pred_labels))
        support = sum(t == label for t in true_labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    macro = {
        "precision": sum(precisions) / len(labels),
        "recall": sum(recalls) / len(labels),
        "f1": sum(f1s) / len(labels),
    }
    weighted = {
        "precision": sum(p * s for p, s in zip(precisions, supports)) / total,
        "recall": sum(r * s for r, s in zip(recalls, supports)) / total,
        "f1": sum(f * s for f, s in zip(f1s, supports)) / total,
    }
    return correct_total / total, rows, macro, weighted


def report_text(e2e_acc, q1_acc, q2_acc, metric_rows, macro, weighted, n_test):
    lines = [
        "Final model: EfficientNetV2-S 480x240 hybrid two-stage",
        "Stage A: seed 2026, empty vs non-empty",
        "Stage B: seed 42, low / medium / high",
        f"End-to-end 4-class test acc: {e2e_acc:.4f}",
        f"Q1 (empty vs not) test acc:  {q1_acc:.4f}",
        f"Q2 (low/med/high on true-nonempty) test acc: {q2_acc:.4f}",
        "",
        "              precision    recall  f1-score   support",
        "",
    ]
    for row in metric_rows:
        lines.append(
            f"{row['label']:>12} {row['precision']:>10.3f} {row['recall']:>9.3f} {row['f1']:>9.3f} {row['support']:>9}"
        )
    lines.extend([
        "",
        f"    accuracy                          {e2e_acc:>9.3f} {n_test:>9}",
        f"   macro avg {macro['precision']:>10.3f} {macro['recall']:>9.3f} {macro['f1']:>9.3f} {n_test:>9}",
        f"weighted avg {weighted['precision']:>10.3f} {weighted['recall']:>9.3f} {weighted['f1']:>9.3f} {n_test:>9}",
        "",
    ])
    return "\n".join(lines)


def save_confusion_matrix(matrix, labels, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Final Hybrid EfficientNetV2-S Confusion Matrix")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metrics_csv(metric_rows, macro, weighted, e2e_acc, q1_acc, q2_acc):
    path = FINAL_METRICS_DIR / "final_classification_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["label", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({
                "label": row["label"],
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1"], 4),
                "support": row["support"],
            })
        writer.writerow({"label": "macro avg", "precision": round(macro["precision"], 4), "recall": round(macro["recall"], 4), "f1": round(macro["f1"], 4), "support": ""})
        writer.writerow({"label": "weighted avg", "precision": round(weighted["precision"], 4), "recall": round(weighted["recall"], 4), "f1": round(weighted["f1"], 4), "support": ""})
        writer.writerow({"label": "end_to_end_acc", "precision": round(e2e_acc, 4), "recall": "", "f1": "", "support": ""})
        writer.writerow({"label": "q1_acc", "precision": round(q1_acc, 4), "recall": "", "f1": "", "support": ""})
        writer.writerow({"label": "q2_acc", "precision": round(q2_acc, 4), "recall": "", "f1": "", "support": ""})


def save_pipeline_diagram():
    path = FINAL_FIGURES_DIR / "final_pipeline_flow.png"
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    boxes = [
        ("Raw image", 0.05),
        ("YOLO bottle crop", 0.25),
        ("Stage A\nEfficientNetV2-S\nempty / non-empty", 0.48),
        ("Stage B\nEfficientNetV2-S\nlow / medium / high", 0.72),
        ("Final class\nempty / low / medium / high", 0.91),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", fc="#eef6ff", ec="#2563eb", lw=1.5))
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1] - 0.09, 0.55), xytext=(boxes[i][1] + 0.09, 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f2937"))
    ax.text(0.60, 0.15, "If Stage A predicts empty, output empty; otherwise Stage B predicts liquid level.",
            ha="center", va="center", fontsize=10, color="#374151")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_diagram():
    path = FINAL_FIGURES_DIR / "final_model_architecture.png"
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.92, "Final Two-Stage Hybrid Model", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.2, 0.62, "Stage A\nEfficientNetV2-S\nInput 480x240\nSeed 2026\nOutput: empty / non-empty",
            ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.5", fc="#ecfdf5", ec="#059669", lw=1.5))
    ax.text(0.8, 0.62, "Stage B\nEfficientNetV2-S\nInput 480x240\nSeed 42\nOutput: low / medium / high",
            ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.5", fc="#fff7ed", ec="#ea580c", lw=1.5))
    ax.text(0.5, 0.30, "Decision rule", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.15, "Stage A = empty -> final empty\nStage A = non-empty -> Stage B result", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#334155", lw=1.2))
    ax.annotate("", xy=(0.37, 0.62), xytext=(0.63, 0.62), arrowprops=dict(arrowstyle="->", lw=1.4, color="#64748b"))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_dataset_bar_chart():
    stats_path = METRICS_DIR / "final_split_stats_v2.csv"
    if not stats_path.exists():
        stats_path = METRICS_DIR / "split_stats_v2.csv"
    rows = []
    with stats_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] in {"train", "val", "test"}:
                rows.append(row)
    labels = ["empty", "low", "medium", "high"]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.24
    offsets = {"train": -width, "val": 0, "test": width}
    colors = {"train": "#2563eb", "val": "#f59e0b", "test": "#10b981"}
    for row in rows:
        split = row["split"]
        vals = [int(row[label]) for label in labels]
        ax.bar([i + offsets[split] for i in x], vals, width=width, label=split, color=colors[split])
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Number of samples")
    ax.set_title("Final Dataset Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "final_dataset_distribution.png", dpi=180)
    plt.close(fig)


def save_sample_grid():
    path = FINAL_FIGURES_DIR / "final_dataset_samples.png"
    labels = ALL_LABELS
    images = []
    for label in labels:
        d = CLS_DIR_V2 / "test" / label
        sample = next((p for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}), None)
        if sample:
            images.append((label, sample))
    thumb_w, thumb_h = 180, 300
    label_h = 42
    canvas = Image.new("RGB", (thumb_w * len(images), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, path_img) in enumerate(images):
        img = Image.open(path_img).convert("RGB")
        img = ImageOps.contain(img, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = i * thumb_w + (thumb_w - img.width) // 2
        y = (thumb_h - img.height) // 2
        canvas.paste(img, (x, y))
        draw.text((i * thumb_w + 10, thumb_h + 10), f"{label}: {path_img.name}", fill=(0, 0, 0), font=font)
    canvas.save(path)


def copy_final_artifacts():
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst_name in [
        (STAGE_A_SRC, "stage_a_empty_vs_not_seed2026.pth"),
        (STAGE_B_SRC, "stage_b_low_med_high_seed42.pth"),
    ]:
        shutil.copy2(src, FINAL_MODELS_DIR / dst_name)
    for src in [
        METRICS_DIR / "final_split_list.csv",
        METRICS_DIR / "final_split_stats.csv",
        METRICS_DIR / "final_split_list_v2.csv",
        METRICS_DIR / "final_split_stats_v2.csv",
    ]:
        if src.exists():
            shutil.copy2(src, FINAL_DATA_DIR / src.name)
    relabel = Path("data") / "relabel.csv"
    if relabel.exists():
        shutil.copy2(relabel, FINAL_DATA_DIR / "relabel.csv")


def archive_experiments_note():
    note = FINAL_DOCS_DIR / "experiment_archive_note.txt"
    note.write_text(
        "Non-final experiments are preserved in outputs/metrics/, outputs/figures/, and models/.\n"
        "The final paper-ready artifacts are collected under final/.\n"
        "Selected final method: EfficientNetV2-S 480x240 hybrid two-stage model.\n"
        "Stage A uses seed 2026; Stage B uses seed 42.\n",
        encoding="utf-8",
    )


def main():
    for d in [FINAL_MODELS_DIR, FINAL_METRICS_DIR, FINAL_FIGURES_DIR, FINAL_DOCS_DIR, FINAL_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not STAGE_A_SRC.exists() or not STAGE_B_SRC.exists():
        raise FileNotFoundError("missing final hybrid model weights")

    model_a = load_model(STAGE_A_SRC, 2)
    model_b = load_model(STAGE_B_SRC, 3)

    rows = []
    for cls_dir in sorted((CLS_DIR_V2 / "test").iterdir()):
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            pred = predict_image(img_path, model_a, model_b)
            rows.append({
                "image": img_path.name,
                "true_label": true_label,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in pred.items()},
            })

    with PRED_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["image", "true_label", "pred_label", "p_empty", "p_nonempty", "stage_b_pred", "p_low", "p_medium", "p_high", "p_stage_b"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    true_labels = [row["true_label"] for row in rows]
    pred_labels = [row["pred_label"] for row in rows]
    e2e_acc, metric_rows, macro, weighted = classification_metrics(true_labels, pred_labels, ALL_LABELS)

    true_q1 = [0 if label == "empty" else 1 for label in true_labels]
    pred_q1 = [0 if label == "empty" else 1 for label in pred_labels]
    q1_acc = sum(t == p for t, p in zip(true_q1, pred_q1)) / len(true_q1)
    q2_pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if t != "empty"]
    q2_acc = sum(t == p for t, p in q2_pairs) / len(q2_pairs)

    report = report_text(e2e_acc, q1_acc, q2_acc, metric_rows, macro, weighted, len(rows))
    REPORT_TXT.write_text(report, encoding="utf-8-sig")

    cm = confusion_counts(true_labels, pred_labels, ALL_LABELS)
    save_confusion_matrix(cm, ALL_LABELS, CM_PNG)
    save_metrics_csv(metric_rows, macro, weighted, e2e_acc, q1_acc, q2_acc)
    save_pipeline_diagram()
    save_model_diagram()
    save_dataset_bar_chart()
    save_sample_grid()
    copy_final_artifacts()
    archive_experiments_note()

    # Promote final artifacts to canonical output paths as well.
    shutil.copy2(REPORT_TXT, METRICS_DIR / "two_stage_test_report.txt")
    shutil.copy2(PRED_CSV, METRICS_DIR / "two_stage_test_predictions.csv")
    shutil.copy2(CM_PNG, FIGURES_DIR / "two_stage_confusion_matrix.png")

    print(report)
    print(f"Final artifacts saved to: {FINAL_ROOT}")


if __name__ == "__main__":
    main()
