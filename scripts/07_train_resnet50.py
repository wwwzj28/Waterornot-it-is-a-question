"""训练 ResNet50 四分类模型：empty / low / medium / high。"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from pathlib import Path
from _paths import CLS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR

import time
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt

# ==== Configuration ====

# 训练参数
NUM_CLASSES = 4
INPUT_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7  # lr scheduler patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径准备
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "resnet50_best.pth"
TRAIN_LOSS_CSV = METRICS_DIR / "resnet_train_loss.csv"
TRAIN_ACC_CSV = METRICS_DIR / "resnet_train_acc.csv"


# ==== Dataset and DataLoader ====

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),  # ResNet50 ImageNet normalization
    ]),
    "val": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
}

data_dirs = {
    "train": CLS_DIR / "train",
    "val": CLS_DIR / "val",
    "test": CLS_DIR / "test"
}

# 创建 dataset
image_datasets = {
    x: datasets.ImageFolder(root=str(data_dirs[x]), transform=data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

print("Classes:", class_names)
print("Dataset sizes:", dataset_sizes)


# ==== Model Setup ====

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

# 替换最后的分类头
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# 先冻结 backbone 其余层
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                  patience=PATIENCE,
                                                  factor=0.1)


# ==== Training & Validation ====

def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    train_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # 每个 epoch 都有 train 和 val
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录
            if phase == "train":
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc.item())
                scheduler.step(epoch_loss)

            # 深度复制最好模型
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, BEST_MODEL_PATH)
                print(f"*** Best model updated (val acc: {best_val_acc:.4f}) ***")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_val_acc:.4f}")

    # 训练日志保存
    pd.DataFrame({"train_loss": train_loss_list}).to_csv(TRAIN_LOSS_CSV, index=False)
    pd.DataFrame({"val_acc": val_acc_list}).to_csv(TRAIN_ACC_CSV, index=False)

    # 加载最好模型权重
    model.load_state_dict(best_model_wts)
    return model


# ==== Run ====

best_model = train_model(model, criterion, optimizer, exp_lr_scheduler, NUM_EPOCHS)
print("\nFinal best model saved at:", BEST_MODEL_PATH)