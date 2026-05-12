"""根据 YOLO11 检测框裁剪瓶体图，用于端到端测试。

划分严格沿用 02_split_dataset.py 生成的 outputs/metrics/split_list.csv。
"""

# scripts/06_crop_bottle_by_yolo.py
import argparse
import csv
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from _paths import RAW_IMAGES_DIR, CLS_DIR, CROPS_DIR, PRED_DIR, METRICS_DIR

# padding 比例
PADDING_RATIO = 0.05

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

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# 输出 CSV 文件
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PRED_DIR / "yolo_boxes.csv"

# 裁剪保存路径
CROPS_YOLO_DIR = CROPS_DIR / "yolo_crops"

# 读取 02 步骤生成的 split_list.csv
SPLIT_LIST_CSV = METRICS_DIR / "split_list.csv"
def _load_split_dict():
    d = {}
    if not SPLIT_LIST_CSV.exists():
        print(f"Warning: {SPLIT_LIST_CSV} not found. All images will default to 'train'.")
        return d
    with open(SPLIT_LIST_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d[row["image"]] = row["split"]
    return d

SPLIT_DICT = _load_split_dict()

def crop_with_padding(img, bbox, padding_ratio=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    return img[y1_pad:y2_pad, x1_pad:x2_pad]

def main():
    # 加载 YOLO11 模型
    model_path = Path("models/yolo/yolo11_best.pt")
    if not model_path.exists():
        print(f"Error: YOLO model not found at {model_path}")
        return
    model = YOLO(str(model_path))

    # 打开 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "conf", "split"])

        # 遍历图片
        image_files = [
            p for p in RAW_IMAGES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
        ]
        total = 0
        stats = {"train": 0, "val": 0, "test": 0}

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Cannot read image {img_path.name}")
                continue

            results = model.predict(str(img_path), imgsz=640, conf=CONF_THRESH, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                print(f"Warning: No bottle detected in {img_path.name}")
                continue

            # 选择置信度最高的框
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            # 裁剪
            crop_img = crop_with_padding(img, [x1, y1, x2, y2], PADDING_RATIO)

            # 沿用 02 步骤的划分，避免数据泄露
            split = SPLIT_DICT.get(img_path.name, "train")

            save_dir = CROPS_YOLO_DIR / split
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / img_path.name
            cv2.imwrite(str(save_path), crop_img)

            # 保存 CSV
            writer.writerow([img_path.name, x1, y1, x2, y2, conf, split])

            total += 1
            stats[split] += 1

        print(f"Total cropped images: {total}")
        print(f"Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
        print(f"Detection boxes saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()