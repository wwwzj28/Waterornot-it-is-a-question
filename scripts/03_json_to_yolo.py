"""将 JSON 标注转换为 YOLO 检测格式。

YOLO 检测阶段只保留一个类别：bottle。
标签格式：
class_id x_center y_center width height

坐标均为 0-1 之间的归一化值。
"""

# scripts/03_json_to_yolo.py
import json
from pathlib import Path
import shutil
import csv
from _paths import RAW_IMAGES_DIR, JSON_DIR, YOLO_DIR, METRICS_DIR

# 分类 ID 映射
CLASS_ID = {"bottle": 0}

# 输出目录
YOLO_IMAGES_DIR = YOLO_DIR / "images"
YOLO_LABELS_DIR = YOLO_DIR / "labels"
YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# 读取划分 CSV
split_csv = METRICS_DIR / "split_list.csv"
split_dict = {}  # image_name -> split
with open(split_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        split_dict[row["image"]] = row["split"]

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h

def main():
    json_files = list(JSON_DIR.glob("*.json"))
    total_txt = 0
    for json_file in sorted(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data.get("imagePath")
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        shapes = data.get("shapes", [])

        split = split_dict.get(img_name, "train")
        img_dest_dir = YOLO_IMAGES_DIR / split
        label_dest_dir = YOLO_LABELS_DIR / split
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        label_dest_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片
        src_img_path = RAW_IMAGES_DIR / img_name
        dest_img_path = img_dest_dir / img_name
        if not dest_img_path.exists():
            shutil.copy(src_img_path, dest_img_path)

        # 写 TXT
        txt_path = label_dest_dir / f"{Path(img_name).stem}.txt"
        lines = []
        for shape in shapes:
            label = shape.get("label", "").lower()
            if not label.startswith("bottle"):
                continue  # YOLO阶段只识别bottle
            cls_id = CLASS_ID["bottle"]
            (xmin, ymin), (xmax, ymax) = shape["points"]
            x_center, y_center, w, h = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if lines:
            with open(txt_path, "w", encoding="utf-8") as ftxt:
                ftxt.write("\n".join(lines))
            total_txt += 1

    print(f"YOLO txt files created: {total_txt}")
    print(f"YOLO images saved in {YOLO_IMAGES_DIR}")

if __name__ == "__main__":
    main()