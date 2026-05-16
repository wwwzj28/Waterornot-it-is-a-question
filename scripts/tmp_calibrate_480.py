from pathlib import Path
import csv
import itertools

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = ROOT / "models" / "efficientnet_v2_s_480x240" / "two_stage"
DATA_DIR = ROOT / "data" / "resnet_cls_v2"
OUT_DIR = ROOT / "outputs" / "metrics" / "efficientnet_480_bias_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LABELS = ["empty", "low", "medium", "high"]
STAGE_B = {0: "low", 1: "medium", 2: "high"}
TF = transforms.Compose([
    transforms.Resize((480, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path: Path, num_classes: int):
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def collect_probs(split: str, model_a, model_b):
    rows = []
    with torch.no_grad():
        for cls_dir in sorted((DATA_DIR / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            true_label = cls_dir.name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                x = TF(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
                pa = torch.softmax(model_a(x), dim=1)[0].detach().cpu().tolist()
                pb = torch.softmax(model_b(x), dim=1)[0].detach().cpu().tolist()
                rows.append({
                    "image": img_path.name,
                    "true": true_label,
                    "pa_empty": pa[0],
                    "pa_nonempty": pa[1],
                    "pb_low": pb[0],
                    "pb_medium": pb[1],
                    "pb_high": pb[2],
                })
    return rows


def predict(row, params):
    empty_score = row["pa_empty"] * params["empty_bias"]
    nonempty_score = row["pa_nonempty"] * params["nonempty_bias"]
    if empty_score >= nonempty_score:
        return "empty"
    scores = {
        "low": row["pb_low"] * params["low_bias"],
        "medium": row["pb_medium"] * params["medium_bias"],
        "high": row["pb_high"] * params["high_bias"],
    }
    return max(scores, key=scores.get)


def calc_metrics(rows, params):
    trues = [r["true"] for r in rows]
    preds = [predict(r, params) for r in rows]
    total = len(rows)
    acc = sum(t == p for t, p in zip(trues, preds)) / total
    q1 = sum((t != "empty") == (p != "empty") for t, p in zip(trues, preds)) / total
    non = [(t, p) for t, p in zip(trues, preds) if t != "empty"]
    q2 = sum(t == p for t, p in non) / len(non)
    out = {"acc": acc, "q1": q1, "q2": q2}
    f1s = []
    weighted_f1 = 0.0
    for label in ALL_LABELS:
        tp = sum(t == label and p == label for t, p in zip(trues, preds))
        fp = sum(t != label and p == label for t, p in zip(trues, preds))
        fn = sum(t == label and p != label for t, p in zip(trues, preds))
        support = sum(t == label for t in trues)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"{label}_precision"] = precision
        out[f"{label}_recall"] = recall
        out[f"{label}_f1"] = f1
        out[f"{label}_support"] = support
        f1s.append(f1)
        weighted_f1 += f1 * support
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["weighted_f1"] = weighted_f1 / total
    return out, preds


def main():
    model_a = load_model(MODEL_DIR / "stage_a_empty_vs_not.pth", 2)
    model_b = load_model(MODEL_DIR / "stage_b_low_med_high.pth", 3)
    val_rows = collect_probs("val", model_a, model_b)
    test_rows = collect_probs("test", model_a, model_b)

    baseline = {
        "empty_bias": 1.0,
        "nonempty_bias": 1.0,
        "low_bias": 1.0,
        "medium_bias": 1.0,
        "high_bias": 1.0,
    }

    candidates = []
    for empty_bias, nonempty_bias, low_bias, medium_bias, high_bias in itertools.product(
        [0.9, 1.0, 1.05],
        [0.95, 1.0, 1.05],
        [1.0, 1.15, 1.3, 1.5, 1.8, 2.2],
        [0.85, 1.0, 1.1],
        [0.9, 1.0, 1.1],
    ):
        params = {
            "empty_bias": empty_bias,
            "nonempty_bias": nonempty_bias,
            "low_bias": low_bias,
            "medium_bias": medium_bias,
            "high_bias": high_bias,
        }
        metrics, _ = calc_metrics(val_rows, params)
        candidates.append(((metrics["acc"], metrics["macro_f1"], metrics["q2"], metrics["low_f1"]), params, metrics))

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_params, best_val = candidates[0]

    baseline_val, _ = calc_metrics(val_rows, baseline)
    baseline_test, baseline_preds = calc_metrics(test_rows, baseline)
    calibrated_test, calibrated_preds = calc_metrics(test_rows, best_params)

    summary_path = OUT_DIR / "summary.csv"
    fields = [
        "split", "method", "acc", "q1", "q2", "macro_f1", "weighted_f1",
        "empty_recall", "low_precision", "low_recall", "low_f1", "medium_recall", "high_recall", "params",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, method, metrics, params in [
            ("val", "baseline", baseline_val, baseline),
            ("val", "calibrated_best_on_val", best_val, best_params),
            ("test", "baseline", baseline_test, baseline),
            ("test", "calibrated_best_on_val", calibrated_test, best_params),
        ]:
            writer.writerow({
                "split": split,
                "method": method,
                "params": params,
                **{k: round(metrics[k], 4) for k in fields if k in metrics},
            })

    pred_path = OUT_DIR / "test_predictions_calibrated.csv"
    with pred_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_label", "baseline_pred", "calibrated_pred"])
        writer.writeheader()
        for row, base_pred, cal_pred in zip(test_rows, baseline_preds, calibrated_preds):
            writer.writerow({
                "image": row["image"],
                "true_label": row["true"],
                "baseline_pred": base_pred,
                "calibrated_pred": cal_pred,
            })

    def small(metrics):
        keys = ["acc", "q1", "q2", "macro_f1", "weighted_f1", "low_recall", "low_f1"]
        return {k: round(metrics[k], 4) for k in keys}

    print("best_params", best_params)
    print("baseline_val", small(baseline_val))
    print("best_val", small(best_val))
    print("baseline_test", small(baseline_test))
    print("calibrated_test", small(calibrated_test))
    print("summary", summary_path)


if __name__ == "__main__":
    main()
