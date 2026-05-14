"""评价 ResNet50 分类效果。"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from _paths import CLS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR

# ==== 配置 ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 与训练保持一致：竖长形 (H, W)
INPUT_H = 384
INPUT_W = 192
BATCH_SIZE = 16

NUM_CLASSES = 4

MODEL_PATH = MODEL_DIR / "resnet" / "resnet50_best.pth"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONFUSION_MATRIX_PNG = FIGURES_DIR / "resnet_confusion_matrix.png"
CLASSIFICATION_REPORT_CSV = METRICS_DIR / "resnet_classification_report.csv"
PREDICTIONS_CSV = METRICS_DIR / "resnet_test_predictions.csv"

# ==== 数据预处理 ====
data_transforms = transforms.Compose([
    transforms.Resize((INPUT_H, INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=str(CLS_DIR / "test"), transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 关键：用 ImageFolder 的实际索引顺序（字母序），训练时也是这套，必须对齐
CLASS_NAMES = test_dataset.classes
print("Class index order:", CLASS_NAMES)

# ==== 模型 ====
model = resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== 预测 ====
all_labels = []
all_preds = []
all_image_names = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ==== Accuracy & Classification Report ====
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

report_dict = classification_report(all_labels, all_preds,
                                    target_names=CLASS_NAMES,
                                    output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(CLASSIFICATION_REPORT_CSV, index=True)
print(f"Classification report saved to {CLASSIFICATION_REPORT_CSV}")

# ==== 混淆矩阵 ====
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet50 Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG)
plt.close()
print(f"Confusion matrix saved to {CONFUSION_MATRIX_PNG}")

# ==== 保存预测结果 ====
for idx, (path, _) in enumerate(test_dataset.samples):
    image_name = Path(path).name
    all_image_names.append(image_name)

pred_df = pd.DataFrame({
    "image": all_image_names,
    "true_label": [CLASS_NAMES[i] for i in all_labels],
    "pred_label": [CLASS_NAMES[i] for i in all_preds]
})
pred_df.to_csv(PREDICTIONS_CSV, index=False)
print(f"Test predictions saved to {PREDICTIONS_CSV}")