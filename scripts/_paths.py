from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
JSON_DIR = DATA_DIR / "annotations_json"

YOLO_DIR = DATA_DIR / "yolo_detect"
YOLO_IMAGES_DIR = YOLO_DIR / "images"
YOLO_LABELS_DIR = YOLO_DIR / "labels"

CLS_DIR = DATA_DIR / "resnet_cls"

CONFIG_DIR = ROOT / "configs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
PRED_DIR = OUTPUT_DIR / "predictions"
CROPS_DIR = OUTPUT_DIR / "crops"
