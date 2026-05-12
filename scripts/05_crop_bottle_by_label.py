"""根据人工标注框裁剪瓶体图，用于训练 ResNet50。"""

# scripts/05_crop_bottle_by_label.py
import json
from pathlib import Path
import cv2
import random
import shutil
import csv
from _paths import RAW_IMAGES_DIR, CLS_DIR, METRICS_DIR

# 分类映射
CLASS_MAP = {
    "bottle_empty": "empty",
    "bottle_low": "low",
    "bottle_medium": "medium",
    "bottle_high": "high",
    "empty": "empty",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# padding 比例
PADDING_RATIO = 0.05

# 读取划分 CSV
split_csv = METRICS_DIR / "split_list.csv"
split_dict = {}  # image_name -> split
if split_csv.exists():
    with open(split_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_dict[row["image"]] = row["split"]
else:
    print(f"Warning: {split_csv} not found. All images will default to 'train'.")

def crop_with_padding(img, bbox, padding_ratio=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1_pad = max(0, int(x1 - pad_x))
    y1_pad = max(0, int(y1 - pad_y))
    x2_pad = min(w, int(x2 + pad_x))
    y2_pad = min(h, int(y2 + pad_y))

    return img[y1_pad:y2_pad, x1_pad:x2_pad]

def main():
    image_files = [
        p for p in RAW_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]
    total = 0
    stats = {"train": 0, "val": 0, "test": 0}

    for img_path in sorted(image_files):
        json_path = RAW_IMAGES_DIR.parent / "annotations_json" / f"{img_path.stem}.json"
        if not json_path.exists():
            print(f"Warning: JSON not found for {img_path.name}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read image {img_path.name}")
            continue

        shapes = data_json.get("shapes", [])
        img_w = data_json.get("imageWidth", img.shape[1])
        img_h = data_json.get("imageHeight", img.shape[0])

        if not shapes:
            print(f"Warning: No shapes in {json_path.name}")
            continue

        for shape in shapes:
            label_raw = shape.get("label", "").lower()
            label = CLASS_MAP.get(label_raw)
            if label is None:
                print(f"Warning: Unknown label '{label_raw}' in {json_path.name}")
                continue

            points = shape.get("points")
            if not points or len(points) != 2:
                print(f"Warning: Invalid bbox in {json_path.name}")
                continue

            xmin, ymin = points[0]
            xmax, ymax = points[1]
            bbox = [xmin, ymin, xmax, ymax]

            crop_img = crop_with_padding(img, bbox, PADDING_RATIO)

            # 划分 train / val / test
            split = split_dict.get(img_path.name, "train")
            save_dir = CLS_DIR / split / label
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / img_path.name
            cv2.imwrite(str(save_path), crop_img)

            total += 1
            stats[split] += 1

    print(f"Total cropped images: {total}")
    print(f"Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")

if __name__ == "__main__":
    main()