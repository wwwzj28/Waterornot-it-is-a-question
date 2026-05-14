"""划分 train / val / test。

建议：先按类别分层，再按 70/15/15 或 80/10/10 划分。
"""

# scripts/02_split_dataset.py
"""
自动划分 train / val / test 数据集。

适用数据格式：
1. 原始图片位于 data/raw_images/
2. LabelMe JSON 标注位于 data/annotations_json/
3. 图片和 JSON 文件同名，例如：
   data/raw_images/1000.jpeg
   data/annotations_json/1000.json

LabelMe JSON 示例：
{
    "shapes": [
        {
            "label": "bottle_empty",
            "points": [[xmin, ymin], [xmax, ymax]],
            "shape_type": "rectangle"
        }
    ],
    "imagePath": "1000.jpeg",
    "imageHeight": 259,
    "imageWidth": 194
}

输出：
1. outputs/metrics/split_list.csv
2. outputs/metrics/split_stats.csv
"""

import csv
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

from _paths import RAW_IMAGES_DIR, JSON_DIR, METRICS_DIR, DATA_DIR


# 允许的图片后缀
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# LabelMe 标注标签 -> 分类标签
CLASS_MAP = {
    "bottle_empty": "empty",
    "bottle_low": "low",
    "bottle_medium": "medium",
    "bottle_high": "high",
    "empty": "empty",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# 默认划分比例
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

RELABEL_CSV = DATA_DIR / "relabel.csv"


def load_relabel_overrides():
    """读 relabel.csv -> {image_name: new_label}; new_label 可以是 empty/low/medium/high/invalid。

    没有 relabel.csv 时返回空 dict（纯靠 LabelMe 原始 label 切分）。
    """
    if not RELABEL_CSV.exists():
        return {}
    out = {}
    with RELABEL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("new_label", "") or "").strip().lower()
            if v:
                out[row["image"]] = v
    return out


def get_image_files(image_dir: Path):
    """
    获取 raw_images 文件夹下的所有图片。
    支持 .jpg / .jpeg / .png / .bmp，且不区分大小写。
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]

    return sorted(image_files)


def load_json(json_path: Path):
    """
    读取 JSON 文件。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_label_from_labelme(data_json: dict, json_path: Path):
    """
    从 LabelMe JSON 中提取分类标签。

    本项目中每张图片理论上只有一个瓶子。
    如果 JSON 中有多个 shape，则默认使用第一个合法 shape 的 label。
    """
    shapes = data_json.get("shapes", [])

    if not shapes:
        raise ValueError(f"No shapes found in {json_path.name}")

    for shape in shapes:
        label_raw = shape.get("label", "").strip().lower()

        if label_raw in CLASS_MAP:
            return CLASS_MAP[label_raw], label_raw

    labels_in_file = [
        shape.get("label", "") for shape in shapes
    ]
    raise ValueError(
        f"No valid label found in {json_path.name}. "
        f"Labels in file: {labels_in_file}"
    )


def collect_records(use_relabel: bool = True):
    """
    收集全部有效样本。

    如果 use_relabel=True 且 data/relabel.csv 存在：
      - relabel.csv 里 new_label==invalid 的样本直接跳过
      - relabel.csv 里 new_label in {empty,low,medium,high} 的样本以 new_label 为准
      - 其余样本走 LabelMe 原始 label

    返回：
    records_by_class = {
        "empty": [
            {"image": "1000.jpeg", "json": "1000.json", "label": "empty", "raw_label": "bottle_empty"}
        ],
        ...
    }
    """
    overrides = load_relabel_overrides() if use_relabel else {}
    if overrides:
        n_invalid = sum(1 for v in overrides.values() if v == "invalid")
        n_override = sum(1 for v in overrides.values() if v in CLASS_MAP)
        print(f"[relabel] loaded {len(overrides)} entries  (invalid={n_invalid}, override={n_override})")

    image_files = get_image_files(RAW_IMAGES_DIR)

    records_by_class = defaultdict(list)
    error_records = []
    skipped_invalid = 0

    for img_path in image_files:
        json_path = JSON_DIR / f"{img_path.stem}.json"

        if not json_path.exists():
            error_records.append({
                "image": img_path.name,
                "json": json_path.name,
                "error": "missing json",
            })
            continue

        # invalid 优先：直接排除
        ov = overrides.get(img_path.name, "")
        if ov == "invalid":
            skipped_invalid += 1
            continue

        try:
            data_json = load_json(json_path)

            # 拿 raw_label 用于报告，label 优先取 overrides
            try:
                label_from_json, raw_label = extract_label_from_labelme(data_json, json_path)
            except Exception as e:
                label_from_json, raw_label = None, ""

            if ov in CLASS_MAP:
                label = CLASS_MAP[ov]
                if not raw_label:
                    raw_label = ov
            else:
                if label_from_json is None:
                    raise ValueError(f"no valid label in JSON and no relabel override for {img_path.name}")
                label = label_from_json

            # 可选检查：JSON 里的 imagePath 和当前图片文件名是否一致
            image_path_in_json = data_json.get("imagePath", "")
            if image_path_in_json and Path(image_path_in_json).name != img_path.name:
                error_records.append({
                    "image": img_path.name,
                    "json": json_path.name,
                    "error": f"imagePath mismatch: {image_path_in_json}",
                })

            record = {
                "image": img_path.name,
                "json": json_path.name,
                "label": label,
                "raw_label": raw_label,
            }

            records_by_class[label].append(record)

        except Exception as e:
            error_records.append({
                "image": img_path.name,
                "json": json_path.name,
                "error": str(e),
            })

    if skipped_invalid:
        print(f"[relabel] skipped {skipped_invalid} samples marked invalid")

    return records_by_class, error_records


def split_one_class(items, train_ratio, val_ratio):
    """
    对某一类别的数据进行 train / val / test 划分。

    对小样本类别做了简单保护：
    - 1 张：全部进 train
    - 2 张：1 train，1 test
    - 3 张：1 train，1 val，1 test
    - 4 张及以上：按比例划分，尽量保证每部分都有样本
    """
    n = len(items)

    if n == 0:
        return [], [], []

    if n == 1:
        return items, [], []

    if n == 2:
        return items[:1], [], items[1:]

    if n == 3:
        return items[:1], items[1:2], items[2:]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # 保证 train / val / test 至少有基本样本
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        n_val = n - n_train - n_test

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return train_items, val_items, test_items


def split_dataset(records_by_class, train_ratio, val_ratio, seed):
    """
    按类别分层划分数据集。
    """
    random.seed(seed)

    split_records = []

    for class_name in ["empty", "low", "medium", "high"]:
        items = records_by_class.get(class_name, [])

        # 固定随机种子后 shuffle，保证每次划分结果一致
        items = items.copy()
        random.shuffle(items)

        train_items, val_items, test_items = split_one_class(
            items,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        for item in train_items:
            item = item.copy()
            item["split"] = "train"
            split_records.append(item)

        for item in val_items:
            item = item.copy()
            item["split"] = "val"
            split_records.append(item)

        for item in test_items:
            item = item.copy()
            item["split"] = "test"
            split_records.append(item)

    return split_records


def save_split_list(split_records, output_path: Path):
    """
    保存 split_list.csv。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "json", "label", "raw_label", "split"]
        )
        writer.writeheader()
        writer.writerows(split_records)


def save_error_report(error_records, output_path: Path):
    """
    保存划分过程中发现的问题。
    """
    if not error_records:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "json", "error"]
        )
        writer.writeheader()
        writer.writerows(error_records)


def save_split_stats(split_records, output_path: Path):
    """
    保存每个类别在 train / val / test 中的数量统计。
    """
    stats = defaultdict(lambda: defaultdict(int))

    for record in split_records:
        label = record["label"]
        split = record["split"]
        stats[label][split] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "train", "val", "test", "total"])

        for label in ["empty", "low", "medium", "high"]:
            train_count = stats[label]["train"]
            val_count = stats[label]["val"]
            test_count = stats[label]["test"]
            total = train_count + val_count + test_count

            writer.writerow([
                label,
                train_count,
                val_count,
                test_count,
                total,
            ])


def print_summary(split_records, error_records):
    """
    在终端打印划分结果。
    """
    split_count = defaultdict(int)
    class_count = defaultdict(int)
    class_split_count = defaultdict(lambda: defaultdict(int))

    for record in split_records:
        split = record["split"]
        label = record["label"]

        split_count[split] += 1
        class_count[label] += 1
        class_split_count[label][split] += 1

    print("\n========== Dataset Split Summary ==========")
    print(f"Total valid images: {len(split_records)}")
    print(f"Errors / warnings : {len(error_records)}")

    print("\nOverall split:")
    print(f"  train: {split_count['train']}")
    print(f"  val  : {split_count['val']}")
    print(f"  test : {split_count['test']}")

    print("\nClass distribution:")
    for label in ["empty", "low", "medium", "high"]:
        print(
            f"  {label:<7} "
            f"total={class_count[label]:<4} "
            f"train={class_split_count[label]['train']:<4} "
            f"val={class_split_count[label]['val']:<4} "
            f"test={class_split_count[label]['test']:<4}"
        )

    print("===========================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split LabelMe dataset into train / val / test."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Ratio of training set. Default: 0.70",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Ratio of validation set. Default: 0.15",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Ratio of test set. Default: 0.15",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--ignore-relabel",
        action="store_true",
        help="Ignore data/relabel.csv even if it exists; split purely from LabelMe labels.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, "
            f"but got {ratio_sum}"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    split_list_path = METRICS_DIR / "split_list.csv"
    split_stats_path = METRICS_DIR / "split_stats.csv"
    error_report_path = METRICS_DIR / "split_errors.csv"

    records_by_class, error_records = collect_records(use_relabel=not args.ignore_relabel)

    split_records = split_dataset(
        records_by_class=records_by_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 为了输出稳定，按 split 和类别排序
    split_order = {"train": 0, "val": 1, "test": 2}
    label_order = {"empty": 0, "low": 1, "medium": 2, "high": 3}

    split_records = sorted(
        split_records,
        key=lambda r: (
            split_order.get(r["split"], 99),
            label_order.get(r["label"], 99),
            r["image"],
        )
    )

    save_split_list(split_records, split_list_path)
    save_split_stats(split_records, split_stats_path)
    save_error_report(error_records, error_report_path)

    print_summary(split_records, error_records)

    print(f"Split list saved to : {split_list_path}")
    print(f"Split stats saved to: {split_stats_path}")

    if error_records:
        print(f"Errors saved to     : {error_report_path}")


if __name__ == "__main__":
    main()