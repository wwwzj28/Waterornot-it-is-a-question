"""训练 ResNet50 四分类模型：empty / low / medium / high。

两阶段微调：
- Stage 1 (warmup): 冻结 backbone，仅训 fc 头
- Stage 2 (finetune): 解冻全部参数，低 LR 配 cosine 退火
配合 class-weighted CrossEntropyLoss 抑制类别不平衡。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
from _paths import CLS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR

import time
import copy
import pandas as pd

# ==== Configuration ====

NUM_CLASSES = 4
# 输入尺寸：竖长形 (H, W)，瓶子绝大多数是竖放 + 部分细长瓶被压扁丢水线信息，
# 垂直方向给更多像素以保住水线特征
INPUT_H = 384
INPUT_W = 192
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4

# Stage 1: 仅训 fc 头预热
WARMUP_EPOCHS = 8
WARMUP_LR = 1e-4

# Stage 2: 全网低 LR 微调
FINETUNE_EPOCHS = 60
FINETUNE_LR = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径准备
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "resnet" / "resnet50_best.pth"
BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TRAIN_LOSS_CSV = METRICS_DIR / "resnet_train_loss.csv"
TRAIN_ACC_CSV = METRICS_DIR / "resnet_train_acc.csv"


# ==== Dataset and DataLoader ====

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.RandomHorizontalFlip(),
        # 弱化 ColorJitter：水位类问题对亮度变化敏感（可能模糊水线），仅保留轻微扰动
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),  # ResNet50 ImageNet normalization
    ]),
    "val": transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
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
                                   shuffle=True, num_workers=0)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

print("Classes:", class_names)
print("Dataset sizes:", dataset_sizes)


# ==== Model Setup ====

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features

# 替换最后的分类头
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# 类别权重：按训练集每类样本数倒数（归一化后）补偿不平衡
train_targets = np.array(image_datasets["train"].targets)
class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)
print("Class counts:", dict(zip(class_names, class_counts.tolist())))
print("Class weights:", dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()])))

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


def freeze_backbone(m):
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True


def unfreeze_all(m):
    for p in m.parameters():
        p.requires_grad = True


# ==== Training & Validation ====

def run_phase(model, criterion, optimizer, scheduler, num_epochs, stage_name,
              best_val_acc, best_model_wts, train_loss_list, val_acc_list):
    """跑一个训练阶段（warmup 或 finetune），返回更新后的最佳 val acc / 权重。"""
    for epoch in range(num_epochs):
        print(f"\n[{stage_name}] Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

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

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} (lr={current_lr:.2e})")

            if phase == "train":
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, BEST_MODEL_PATH)
                print(f"*** Best model updated (val acc: {best_val_acc:.4f}) ***")

        # 每个 epoch 末（train+val 跑完后）推进 scheduler
        if scheduler is not None:
            scheduler.step()

    return best_val_acc, best_model_wts


def train_model(model, criterion):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    train_loss_list = []
    val_acc_list = []

    # ---- Stage 1: warmup, 仅训 fc 头 ----
    print("\n========== Stage 1: warmup (fc only) ==========")
    freeze_backbone(model)
    optimizer = optim.AdamW(model.fc.parameters(), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)
    # warmup 阶段不退火，保持小步稳进
    best_val_acc, best_model_wts = run_phase(
        model, criterion, optimizer, scheduler=None,
        num_epochs=WARMUP_EPOCHS, stage_name="warmup",
        best_val_acc=best_val_acc, best_model_wts=best_model_wts,
        train_loss_list=train_loss_list, val_acc_list=val_acc_list,
    )

    # ---- Stage 2: 解冻全部参数 + cosine 退火 ----
    print("\n========== Stage 2: full finetune (cosine LR) ==========")
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    best_val_acc, best_model_wts = run_phase(
        model, criterion, optimizer, scheduler=scheduler,
        num_epochs=FINETUNE_EPOCHS, stage_name="finetune",
        best_val_acc=best_val_acc, best_model_wts=best_model_wts,
        train_loss_list=train_loss_list, val_acc_list=val_acc_list,
    )

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_val_acc:.4f}")

    pd.DataFrame({"train_loss": train_loss_list}).to_csv(TRAIN_LOSS_CSV, index=False)
    pd.DataFrame({"val_acc": val_acc_list}).to_csv(TRAIN_ACC_CSV, index=False)

    model.load_state_dict(best_model_wts)
    return model


# ==== Run ====

if __name__ == "__main__":
    best_model = train_model(model, criterion)
    print("\nFinal best model saved at:", BEST_MODEL_PATH)