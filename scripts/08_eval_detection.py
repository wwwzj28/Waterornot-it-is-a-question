"""评价 YOLO11 检测效果。"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

from _paths import RAW_IMAGES_DIR, JSON_DIR, METRICS_DIR, YOLO_DIR

# 置信度阈值：CLI > 环境变量 > 默认 0.25
def _resolve_conf():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--conf", type=float, default=None)
    args, _ = p.parse_known_args()
    if args.conf is not None:
        return args.conf
    env = os.environ.get("YOLO_CONF_THRESH")
    if env:
        return float(env)
    return 0.25

CONF_THRESH = _resolve_conf()

# 测试集文件夹
TEST_SPLIT = "test"

# 模型路径
YOLO_MODEL_PATH = Path("models/yolo/yolo11_best.pt")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = METRICS_DIR / "yolo_detection_results.csv"

VALID_LABELS = {"bottle_empty", "bottle_low", "bottle_medium", "bottle_high"}
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_bbox_from_labelme(json_data):
    """
    取第一个合法 shape 的 bbox
    返回 (xmin, ymin, xmax, ymax)
    """
    shapes = json_data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label", "").lower()
        if label in VALID_LABELS:
            points = shape.get("points")
            if points and len(points) == 2:
                xmin, ymin = points[0]
                xmax, ymax = points[1]
                return [float(xmin), float(ymin), float(xmax), float(ymax)]
    return None


def compute_iou(box1, box2):
    """
    计算两个框的 IoU
    box = [xmin, ymin, xmax, ymax]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    unionArea = box1Area + box2Area - interArea
    if unionArea == 0:
        return 0
    iou = interArea / unionArea
    return iou


def main():
    if not YOLO_MODEL_PATH.exists():
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(YOLO_MODEL_PATH))

    test_image_dir = RAW_IMAGES_DIR  # 可以根据 split_list.csv 或文件夹结构筛选测试集
    test_images = [
        p for p in test_image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]

    records = []

    TP = 0
    FP = 0
    FN = 0
    iou_list = []

    for img_path in sorted(test_images):
        # 读取标注框
        json_path = JSON_DIR / f"{img_path.stem}.json"
        label_bbox = None
        if json_path.exists():
            data_json = load_json(json_path)
            label_bbox = extract_bbox_from_labelme(data_json)

        # YOLO11 检测
        results = model.predict(str(img_path), imgsz=640, conf=CONF_THRESH, verbose=False)
        pred_bbox = None
        conf_score = 0
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            conf_score = float(best_box.conf[0])
            x1, y1, x2, y2 = map(float, best_box.xyxy[0])
            pred_bbox = [x1, y1, x2, y2]

        iou = 0
        if label_bbox is not None and pred_bbox is not None:
            iou = compute_iou(label_bbox, pred_bbox)
            iou_list.append(iou)

            if iou >= 0.5:
                TP += 1
            else:
                FP += 1
                FN += 1
        elif label_bbox is not None and pred_bbox is None:
            FN += 1
        elif label_bbox is None and pred_bbox is not None:
            FP += 1

        records.append({
            "image": img_path.name,
            "label_bbox": label_bbox,
            "pred_bbox": pred_bbox,
            "conf_score": conf_score,
            "iou": iou,
        })

    # 保存 CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"YOLO detection results saved to {OUTPUT_CSV}")

    # 计算总体指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0

    print("\n========== YOLO11 Detection Evaluation ==========")
    print(f"Total images          : {len(test_images)}")
    print(f"True Positives (TP)   : {TP}")
    print(f"False Positives (FP)  : {FP}")
    print(f"False Negatives (FN)  : {FN}")
    print(f"Precision             : {precision:.4f}")
    print(f"Recall                : {recall:.4f}")
    print(f"Mean IoU              : {mean_iou:.4f}")
    print("===============================================\n")


if __name__ == "__main__":
    main()