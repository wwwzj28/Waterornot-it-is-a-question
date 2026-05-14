"""将 JSON 标注转换为 YOLO 检测格式。

YOLO 检测阶段只保留一个类别：bottle。
标签格式：
class_id x_center y_center width height

坐标均为 0-1 之间的归一化值。
"""

# scripts/03_json_to_yolo.py
import base64
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
with open(split_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        split_dict[row["image"]] = row["split"]

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def decode_base64_image(image_data):
    if not image_data:
        return None
    if image_data.startswith("data:") and "," in image_data:
        image_data = image_data.split(",", 1)[1]
    return base64.b64decode(image_data)


def save_embedded_image(img_name, image_data):
    image_bytes = decode_base64_image(image_data)
    if image_bytes is None:
        return None
    target_path = RAW_IMAGES_DIR / img_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(image_bytes)
    return target_path


def find_image_file(img_name, image_data=None):
    if img_name:
        candidate = RAW_IMAGES_DIR / img_name
        if candidate.exists():
            return candidate

        stem = Path(img_name).stem
        matches = list(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        if matches:
            return matches[0]

        if stem.isdigit():
            padded = stem.zfill(3)
            for ext in ["jpg", "jpeg", "png", "webp"]:
                candidate = RAW_IMAGES_DIR / f"{padded}.{ext}"
                if candidate.exists():
                    return candidate

        if image_data:
            saved = save_embedded_image(img_name, image_data)
            if saved and saved.exists():
                return saved

    return None


def main():
    json_files = list(JSON_DIR.glob("*.json"))
    total_txt = 0
    for json_file in sorted(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data.get("imagePath")
        image_data = data.get("imageData")
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        shapes = data.get("shapes", [])

        if not img_name and image_data:
            img_name = f"{json_file.stem}.jpg"

        split = split_dict.get(img_name, "train")
        img_dest_dir = YOLO_IMAGES_DIR / split
        label_dest_dir = YOLO_LABELS_DIR / split
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        label_dest_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片或从 JSON imageData 保存
        src_img_path = find_image_file(img_name, image_data)
        if src_img_path is None:
            raise FileNotFoundError(
                f"Cannot locate image for '{img_name}' in {RAW_IMAGES_DIR}. "
                "如果 imageData 存在，脚本会尝试保存它；否则请检查 JSON 的 imagePath 是否与 raw_images 中的文件名匹配。"
            )
        dest_img_path = img_dest_dir / src_img_path.name
        if not dest_img_path.exists():
            shutil.copy(src_img_path, dest_img_path)

        # 写 TXT
        txt_path = label_dest_dir / f"{Path(src_img_path).stem}.txt"
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