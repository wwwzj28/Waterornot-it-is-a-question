"""把 data/relabel.csv（如果有）+ annotations_json/ 原始标签 作用到 data/resnet_cls_v2/。

行为：
  - 读 data/relabel.csv（image, new_label, note）：老数据走过 GUI 的判定
  - 读 data/annotations_json/{stem}.json：新数据没走过 GUI，回落到原始 LabelMe label
  - 优先级：relabel.csv 的 new_label > annotations_json 原始 label
  - 读 outputs/metrics/split_list.csv 沿用旧划分；没记录的样本默认进 train
  - 跳过 invalid / 无任何 label / 缺 json 的样本
  - 输出到 data/resnet_cls_v2/{train,val,test}/{empty,low,medium,high}/
  - 写 outputs/metrics/split_stats_v2.csv 与 split_list_v2.csv

不改动现有 data/resnet_cls/，可以直接对照旧版评估效果。
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from _paths import (
    CLS_DIR_V2,
    DATA_DIR,
    JSON_DIR,
    METRICS_DIR,
    RAW_IMAGES_DIR,
)

RELABEL_CSV = DATA_DIR / "relabel.csv"
SPLIT_CSV = METRICS_DIR / "split_list.csv"

CLASSES = ["empty", "low", "medium", "high"]
PADDING_RATIO = 0.05
VALID_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def load_relabel():
    """relabel.csv 不存在则返回空 dict，允许只用 JSON 跑全新数据。"""
    if not RELABEL_CSV.exists():
        return {}
    out = {}
    with RELABEL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = (row.get("new_label", "") or "").strip().lower()
    return out


def load_split():
    if not SPLIT_CSV.exists():
        return {}
    out = {}
    with SPLIT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = row["split"]
    return out


def load_json_label(json_path: Path) -> str:
    """从 annotations_json/{stem}.json 读 LabelMe 的原始 label，返回 empty/low/medium/high 或 ''。"""
    if not json_path.exists():
        return ""
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for s in d.get("shapes", []):
        lbl = (s.get("label", "") or "").lower().replace("bottle_", "")
        if lbl in {"empty", "low", "medium", "high"}:
            return lbl
    return ""


def load_bbox(json_path: Path):
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for s in d.get("shapes", []):
        pts = s.get("points")
        if pts and len(pts) == 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            return (
                int(min(x1, x2)),
                int(min(y1, y2)),
                int(max(x1, x2)),
                int(max(y1, y2)),
            )
    return None


def crop_with_padding(img, bbox, padding_ratio=PADDING_RATIO):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)
    return img[y1p:y2p, x1p:x2p]


def resolve_label(name: str, relabel: dict, json_path: Path):
    """返回 (final_label, source)：source 用于计数 / 日志。"""
    rl = relabel.get(name, "")
    if rl == "invalid":
        return "invalid", "relabel"
    if rl in CLASSES:
        return rl, "relabel"
    # 回落到 JSON 原始 label
    jl = load_json_label(json_path)
    if jl in CLASSES:
        return jl, "json_fallback"
    return "", "missing"


def main():
    relabel = load_relabel()
    split = load_split()

    if CLS_DIR_V2.exists():
        shutil.rmtree(CLS_DIR_V2)
    for sp in ["train", "val", "test"]:
        for c in CLASSES:
            (CLS_DIR_V2 / sp / c).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)        # (split, label) -> count
    src_stats = defaultdict(int)    # source -> count
    skipped = defaultdict(int)
    out_rows = []

    raw_imgs = sorted(p for p in RAW_IMAGES_DIR.iterdir() if p.suffix.lower() in VALID_SUFFIX)
    for p in raw_imgs:
        name = p.name
        json_path = JSON_DIR / f"{p.stem}.json"

        final_label, src = resolve_label(name, relabel, json_path)
        if final_label == "invalid":
            skipped["invalid"] += 1
            continue
        if final_label == "":
            skipped[f"missing_label({src})"] += 1
            continue

        sp = split.get(name, "train")  # 没记录默认进 train（新数据保护）
        if name not in split:
            src_stats["new_into_train"] += 1

        bbox = load_bbox(json_path)
        if bbox is None:
            skipped["no_bbox"] += 1
            continue

        img = cv2.imread(str(p))
        if img is None:
            skipped["read_fail"] += 1
            continue

        cropped = crop_with_padding(img, bbox, PADDING_RATIO)
        if cropped.size == 0:
            skipped["empty_crop"] += 1
            continue

        cv2.imwrite(str(CLS_DIR_V2 / sp / final_label / name), cropped)
        stats[(sp, final_label)] += 1
        src_stats[src] += 1
        out_rows.append({"image": name, "split": sp, "label": final_label, "source": src})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # split_stats_v2.csv
    stats_csv = METRICS_DIR / "split_stats_v2.csv"
    with stats_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split"] + CLASSES + ["total"])
        for sp in ["train", "val", "test"]:
            row = [sp]
            total = 0
            for c in CLASSES:
                cnt = stats[(sp, c)]
                row.append(cnt); total += cnt
            row.append(total); w.writerow(row)
        total_row = ["total"]
        for c in CLASSES:
            cnt = sum(stats[(sp, c)] for sp in ["train", "val", "test"])
            total_row.append(cnt)
        total_row.append(sum(stats.values()))
        w.writerow(total_row)

    # split_list_v2.csv (含 source)
    list_csv = METRICS_DIR / "split_list_v2.csv"
    with list_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "split", "label", "source"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"=== Wrote v2 dataset to {CLS_DIR_V2} ===")
    for sp in ["train", "val", "test"]:
        line = f"{sp:<6} | "
        line += "  ".join(f"{c}={stats[(sp, c)]:>3}" for c in CLASSES)
        line += f"  | total={sum(stats[(sp, c)] for c in CLASSES)}"
        print(line)
    print(f"\nTotal valid samples: {sum(stats.values())}")
    print(f"Label sources: {dict(src_stats)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"\nWrote: {stats_csv}")
    print(f"Wrote: {list_csv}")


if __name__ == "__main__":
    main()
