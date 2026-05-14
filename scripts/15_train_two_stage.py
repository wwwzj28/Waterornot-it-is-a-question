"""两阶段建模训练脚本（Stage A: empty 二分类；Stage B: 非empty 三分类）。

数据来源：data/resnet_cls_v2/{train,val,test}/{empty,low,medium,high}/

Stage A 训练数据：把 v2 的 empty 当 0 类，{low,medium,high} 全部当 1 类（非empty）
Stage B 训练数据：仅 v2 的 {low,medium,high}，作为 0/1/2 三分类
                  注意：Stage B 把 v2 目录里的 empty 子目录排除在外即可
                  ImageFolder 不能选择性加载 → 自定义 Dataset

每个 stage 内部沿用 07_train_resnet50.py 的两阶段微调：
  - warmup: 冻结 backbone，仅训 fc 头
  - finetune: 全网解冻 + cosine LR
配 class_weight 抑制类别不均。

输出：
  models/resnet/two_stage/
    stage_a_empty_vs_not.pth
    stage_b_low_med_high.pth
  outputs/metrics/two_stage_train_log.csv
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from _paths import CLS_DIR_V2, METRICS_DIR, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_H, INPUT_W = 384, 192
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 8
WARMUP_LR = 1e-4
FINETUNE_EPOCHS = 60
FINETUNE_LR = 1e-5

OUT_DIR = MODEL_DIR / "resnet" / "two_stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV = METRICS_DIR / "two_stage_train_log.csv"


def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class FilteredImageDataset(Dataset):
    """从 CLS_DIR_V2/{split}/{label}/ 加载，按 label_to_idx 决定类映射；不在映射中的子目录会被忽略。"""

    def __init__(self, split: str, label_to_idx: dict, transform):
        self.samples = []
        self.transform = transform
        self.classes = sorted(label_to_idx.keys(), key=lambda k: label_to_idx[k])
        root = CLS_DIR_V2 / split
        for lbl, idx in label_to_idx.items():
            d = root / lbl
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, idx))
        self.targets = np.array([t for _, t in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y


def build_loaders(label_to_idx):
    train_tf, eval_tf = make_transforms()
    ds_train = FilteredImageDataset("train", label_to_idx, train_tf)
    ds_val = FilteredImageDataset("val", label_to_idx, eval_tf)
    return {
        "train": DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "val": DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }, {"train": len(ds_train), "val": len(ds_val)}, ds_train


def make_model(num_classes):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)


def freeze_backbone(m):
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True


def unfreeze_all(m):
    for p in m.parameters():
        p.requires_grad = True


def class_weight_tensor(targets, num_classes):
    counts = np.bincount(targets, minlength=num_classes)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float, device=DEVICE), counts


def run_phase(model, loaders, sizes, criterion, optimizer, scheduler,
              num_epochs, stage_tag, best_acc, best_wts, log_rows):
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train(phase == "train")
            running_loss = 0.0
            running_corr = 0
            for x, y in loaders[phase]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    out = model(x)
                    _, pred = out.max(1)
                    loss = criterion(out, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * x.size(0)
                running_corr += (pred == y).sum().item()
            ep_loss = running_loss / sizes[phase]
            ep_acc = running_corr / sizes[phase]
            lr_now = optimizer.param_groups[0]["lr"]
            log_rows.append({
                "stage": stage_tag, "epoch": epoch + 1, "phase": phase,
                "loss": round(ep_loss, 4), "acc": round(ep_acc, 4), "lr": lr_now,
            })
            if phase == "val" and ep_acc > best_acc:
                best_acc = ep_acc
                best_wts = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()
        last_train = log_rows[-2]
        last_val = log_rows[-1]
        print(f"[{stage_tag}] ep {epoch + 1:>2}/{num_epochs}  "
              f"train loss={last_train['loss']:.3f} acc={last_train['acc']:.3f}  "
              f"val acc={last_val['acc']:.3f}  best={best_acc:.3f}  lr={lr_now:.1e}")
    return best_acc, best_wts


def train_one_stage(stage_name: str, label_to_idx: dict, save_path: Path, log_rows: list):
    print(f"\n========== {stage_name} | label_map={label_to_idx} ==========")
    loaders, sizes, ds_train = build_loaders(label_to_idx)
    # 真实类别数 = 不同 idx 的数量（多个 label 可能映射到同一 idx，如 Stage A 把 low/medium/high 都映成 1）
    num_classes = max(label_to_idx.values()) + 1
    print(f"num_classes={num_classes} | train n={sizes['train']} | val n={sizes['val']}")

    weights_t, counts = class_weight_tensor(ds_train.targets, num_classes)
    print(f"class counts (idx -> n):  {dict(enumerate(counts.tolist()))}")
    print(f"class weights (idx -> w): {dict(enumerate([round(x, 3) for x in weights_t.tolist()]))}")

    model = make_model(num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights_t)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    # warmup
    print(f"\n-- {stage_name} | warmup (fc only) --")
    freeze_backbone(model)
    opt = optim.AdamW(model.fc.parameters(), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)
    best_acc, best_wts = run_phase(model, loaders, sizes, criterion, opt, None,
                                   WARMUP_EPOCHS, f"{stage_name}.warmup",
                                   best_acc, best_wts, log_rows)

    # full finetune + cosine
    print(f"\n-- {stage_name} | finetune (all params) --")
    unfreeze_all(model)
    opt = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    best_acc, best_wts = run_phase(model, loaders, sizes, criterion, opt, sch,
                                   FINETUNE_EPOCHS, f"{stage_name}.finetune",
                                   best_acc, best_wts, log_rows)

    torch.save(best_wts, save_path)
    print(f"\n[{stage_name}] best val acc = {best_acc:.4f}  ->  {save_path}")
    return best_acc


def main():
    log_rows = []
    t0 = time.time()

    # Stage A: empty(0) vs not_empty(1)
    # 把 low/medium/high 全部映射成 1（同一目录树共享）→ 自定义 Dataset 通过 label_to_idx 处理
    stage_a_map = {"empty": 0, "low": 1, "medium": 1, "high": 1}
    # 注意 FilteredImageDataset 用 label_to_idx 直接拿 target，多个 label 映射到同一 idx 没问题
    acc_a = train_one_stage("StageA",
                            label_to_idx=stage_a_map,
                            save_path=OUT_DIR / "stage_a_empty_vs_not.pth",
                            log_rows=log_rows)

    # Stage B: low(0) / medium(1) / high(2)，empty 子目录被自动忽略
    stage_b_map = {"low": 0, "medium": 1, "high": 2}
    acc_b = train_one_stage("StageB",
                            label_to_idx=stage_b_map,
                            save_path=OUT_DIR / "stage_b_low_med_high.pth",
                            log_rows=log_rows)

    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"\nTrain log -> {LOG_CSV}")
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print(f"Summary | Stage A best val acc = {acc_a:.4f} | Stage B best val acc = {acc_b:.4f}")


if __name__ == "__main__":
    main()
