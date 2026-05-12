"""生成论文图表：检测框、裁剪图、混淆矩阵、训练曲线。"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import random
import numpy as np

from _paths import METRICS_DIR, CROPS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# 1. 数据集样例图
# ===========================
RAW_IMAGES_DIR = Path("data/raw_images")
ANNOTATIONS_JSON_DIR = Path("data/annotations_json")
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(dir_path: Path, recursive: bool = False):
    if not dir_path.exists():
        return []
    iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()
    return [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]

def show_sample_images(num_samples=5):
    image_files = list_images(RAW_IMAGES_DIR)
    image_files = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(15, 3 * len(image_files)))

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取 JSON bbox
        import json
        json_path = ANNOTATIONS_JSON_DIR / f"{img_path.stem}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data_json = json.load(f)
            shapes = data_json.get("shapes", [])
            for shape in shapes:
                points = shape.get("points")
                label = shape.get("label", "")
                if points and len(points) == 2:
                    xmin, ymin = map(int, points[0])
                    xmax, ymax = map(int, points[1])
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(img, label, (xmin, max(0, ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        plt.subplot(num_samples,1,idx)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    save_path = FIGURES_DIR / "dataset_sample.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Dataset sample figure saved to {save_path}")

# ===========================
# 2. YOLO11 检测结果图
# ===========================
YOLO_PRED_DIR = Path("outputs/figures")  # 之前pipeline保存的可视化图像

def show_yolo_detection_samples(num_samples=5):
    yolo_images = list_images(YOLO_PRED_DIR)
    yolo_images = random.sample(yolo_images, min(num_samples, len(yolo_images)))

    plt.figure(figsize=(15, 3 * len(yolo_images)))

    for idx, img_path in enumerate(yolo_images, 1):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(num_samples,1,idx)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    save_path = FIGURES_DIR / "yolo_detection_samples.png"
    plt.savefig(save_path)
    plt.close()
    print(f"YOLO detection sample figure saved to {save_path}")

# ===========================
# 3. 裁剪图示例
# ===========================
CROPS_YOLO_DIR = CROPS_DIR / "yolo_crops"  # 06 生成的 YOLO 框裁剪图（按 split 分子目录）

def show_crop_samples(num_samples=5):
    crop_images = list_images(CROPS_YOLO_DIR, recursive=True)
    if not crop_images:
        print(f"No crop images found under {CROPS_YOLO_DIR}, skipping crop_samples figure.")
        return
    crop_images = random.sample(crop_images, min(num_samples, len(crop_images)))

    plt.figure(figsize=(15, 3 * len(crop_images)))
    for idx, img_path in enumerate(crop_images, 1):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(crop_images), 1, idx)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    save_path = FIGURES_DIR / "crop_samples.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Cropped bottle sample figure saved to {save_path}")

# ===========================
# 4. ResNet50训练曲线
# ===========================
LOSS_CSV = METRICS_DIR / "resnet_train_loss.csv"
ACC_CSV = METRICS_DIR / "resnet_train_acc.csv"

def plot_training_curves():
    if LOSS_CSV.exists():
        df_loss = pd.read_csv(LOSS_CSV)
        plt.figure()
        plt.plot(df_loss.index+1, df_loss["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ResNet50 Training Loss")
        plt.grid(True)
        plt.legend()
        save_path = FIGURES_DIR / "resnet_train_loss.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Training loss curve saved to {save_path}")

    if ACC_CSV.exists():
        df_acc = pd.read_csv(ACC_CSV)
        plt.figure()
        plt.plot(df_acc.index+1, df_acc["val_acc"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("ResNet50 Validation Accuracy")
        plt.grid(True)
        plt.legend()
        save_path = FIGURES_DIR / "resnet_val_accuracy.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Validation accuracy curve saved to {save_path}")

# ===========================
# 5. 混淆矩阵
# ===========================
CONFUSION_MATRIX_PNG = FIGURES_DIR / "resnet_confusion_matrix.png"
CLASSIFICATION_REPORT_CSV = METRICS_DIR / "resnet_classification_report.csv"

def plot_confusion_matrix():
    if not CLASSIFICATION_REPORT_CSV.exists():
        print("Classification report CSV not found. Skipping confusion matrix.")
        return
    # 假设之前09_eval_classification.py生成了混淆矩阵
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    pred_csv = METRICS_DIR / "resnet_test_predictions.csv"
    if not pred_csv.exists():
        print("Prediction CSV not found. Skipping confusion matrix.")
        return
    df = pd.read_csv(pred_csv)
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    cm = confusion_matrix(y_true, y_pred, labels=["empty","low","medium","high"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["empty","low","medium","high"],
                yticklabels=["empty","low","medium","high"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ResNet50 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG)
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PNG}")


# ===========================
# Run All
# ===========================
if __name__ == "__main__":
    show_sample_images()
    show_yolo_detection_samples()
    show_crop_samples()
    plot_training_curves()
    plot_confusion_matrix()
    print("All figures generated in", FIGURES_DIR)