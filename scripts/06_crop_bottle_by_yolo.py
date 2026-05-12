"""根据 YOLO11 检测框裁剪瓶体图，用于端到端测试。"""

# scripts/06_crop_bottle_by_yolo.py
import csv
import random
from pathlib import Path
import cv2
from ultralytics import YOLO
from _paths import RAW_IMAGES_DIR, CLS_DIR, CROPS_DIR, PRED_DIR

# 数据集划分比例
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

# padding 比例
PADDING_RATIO = 0.05

# 置信度阈值
CONF_THRESH = 0.5

# 输出 CSV 文件
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PRED_DIR / "yolo_boxes.csv"

# 裁剪保存路径
CROPS_YOLO_DIR = CROPS_DIR / "yolo_crops"

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
        image_files = list(RAW_IMAGES_DIR.glob("*.[jp][pn]g"))
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

            # 划分 train/val/test
            r = random.random()
            if r < SPLIT_RATIO["train"]:
                split = "train"
            elif r < SPLIT_RATIO["train"] + SPLIT_RATIO["val"]:
                split = "val"
            else:
                split = "test"

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