"""完整预测流程：原图 -> YOLO11 定位 -> 裁剪 -> ResNet50 分类 -> 输出结果。"""
"""
预测单张或指定多张图片，在命令行输入：
python scripts/10_predict_pipeline.py --images data/raw_images/test1.jpeg data/raw_images/test2.jpeg
"""

import argparse
import csv
import os
from pathlib import Path
import cv2
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import pandas as pd
from _paths import RAW_IMAGES_DIR, MODEL_DIR, CROPS_DIR, FIGURES_DIR, PRED_DIR, METRICS_DIR

# ==== 配置 ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
CLASS_NAMES = ["empty", "low", "medium", "high"]

YOLO_MODEL_PATH = MODEL_DIR / "yolo/yolo11_best.pt"
RESNET_MODEL_PATH = MODEL_DIR / "resnet/resnet50_best.pth"

PRED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PRED_DIR / "pipeline_predictions.csv"

resnet_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ==== 命令行参数 ====
parser = argparse.ArgumentParser(description="YOLO11 + ResNet50 end-to-end pipeline")
parser.add_argument("--images", nargs="*", type=str, default=None,
                    help="Path(s) to input image(s). Overrides --split.")
parser.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test", "all"],
                    help="Which split from split_list.csv to evaluate. Default: test")
parser.add_argument("--conf", type=float, default=None,
                    help="YOLO confidence threshold. Default: env YOLO_CONF_THRESH or 0.25")
args = parser.parse_args()

if args.conf is not None:
    CONF_THRESH = args.conf
elif os.environ.get("YOLO_CONF_THRESH"):
    CONF_THRESH = float(os.environ["YOLO_CONF_THRESH"])
else:
    CONF_THRESH = 0.25


# ==== 加载模型 ====
yolo_model = YOLO(str(YOLO_MODEL_PATH))

resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()


# ==== 辅助函数 ====
def crop_with_padding(img, bbox, padding_ratio=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)
    return img[y1_pad:y2_pad, x1_pad:x2_pad]


def classify_crop(crop_img):
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    input_tensor = resnet_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = resnet_model(input_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, 1)
        return CLASS_NAMES[pred_idx.item()], conf.item()


# ==== 图片列表 ====
valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}

if args.images:
    image_files = [Path(p) for p in args.images]
else:
    split_list_csv = METRICS_DIR / "split_list.csv"
    if args.split == "all" or not split_list_csv.exists():
        if not split_list_csv.exists():
            print(f"Warning: {split_list_csv} not found, falling back to all raw images")
        image_files = sorted([p for p in RAW_IMAGES_DIR.iterdir()
                              if p.suffix.lower() in valid_suffixes])
    else:
        target_names = set()
        with open(split_list_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["split"] == args.split:
                    target_names.add(row["image"])
        image_files = sorted([p for p in RAW_IMAGES_DIR.iterdir()
                              if p.suffix.lower() in valid_suffixes and p.name in target_names])
        print(f"Running pipeline on {len(image_files)} images from split='{args.split}' (conf={CONF_THRESH})")

if not image_files:
    print("No images found to process.")
    exit()

# ==== 预测流程 ====
results_list = []

for img_path in image_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image {img_path}")
        continue

    # YOLO检测
    yolo_results = yolo_model.predict(str(img_path), imgsz=640, conf=CONF_THRESH, verbose=False)
    if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
        results_list.append({
            "image": img_path.name,
            "xmin": None,
            "ymin": None,
            "xmax": None,
            "ymax": None,
            "det_conf": None,
            "pred_class": None,
            "class_conf": None
        })
        continue

    best_box = max(yolo_results[0].boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    det_conf = float(best_box.conf[0])

    crop_img = crop_with_padding(img, [x1, y1, x2, y2])
    pred_class, class_conf = classify_crop(crop_img)

    results_list.append({
        "image": img_path.name,
        "xmin": x1,
        "ymin": y1,
        "xmax": x2,
        "ymax": y2,
        "det_conf": det_conf,
        "pred_class": pred_class,
        "class_conf": class_conf
    })

    # 可视化
    label_text = f"{pred_class}:{class_conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label_text, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    vis_path = FIGURES_DIR / img_path.name
    cv2.imwrite(str(vis_path), img)

# 保存 CSV
df = pd.DataFrame(results_list)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Pipeline predictions saved to {OUTPUT_CSV}")