#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_run_all.py

透明塑料饮料瓶残留液体识别项目的一键总控脚本。

常用命令：
    python scripts/12_run_all.py --mode all
    python scripts/12_run_all.py --mode prepare
    python scripts/12_run_all.py --mode train
    python scripts/12_run_all.py --mode eval
    python scripts/12_run_all.py --mode predict
    python scripts/12_run_all.py --mode visualize

说明：
1. 本脚本会按项目既定流程顺序调用 01~11 号脚本。
2. 默认某一步失败后立即停止，避免后续结果不可信。
3. 如需只检查流程但不真正运行，可加 --dry-run。
4. 如需失败后继续执行后续步骤，可加 --continue-on-error。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    group: str
    required_before_run: Optional[List[str]] = None


STEPS: List[Step] = [
    Step(
        name="01 检查 LabelMe JSON 标注规范",
        script="01_check_json.py",
        group="prepare",
        required_before_run=["data/raw_images", "data/annotations_json"],
    ),
    Step(
        name="02 分层划分 train/val/test",
        script="02_split_dataset.py",
        group="prepare",
        required_before_run=["data/raw_images", "data/annotations_json"],
    ),
    Step(
        name="03 LabelMe JSON 转 YOLO 标签",
        script="03_json_to_yolo.py",
        group="prepare",
        required_before_run=["data/raw_images", "data/annotations_json", "outputs/metrics/split_list.csv"],
    ),
    Step(
        name="05 使用人工框裁剪瓶体，生成 ResNet 分类数据",
        script="05_crop_bottle_by_label.py",
        group="prepare",
        required_before_run=["data/raw_images", "data/annotations_json", "outputs/metrics/split_list.csv"],
    ),
    Step(
        name="04 训练 YOLO11 检测模型",
        script="04_train_yolo11.py",
        group="train",
        required_before_run=["configs/bottle_detect.yaml", "data/yolo_detect/images", "data/yolo_detect/labels"],
    ),
    Step(
        name="07 训练 ResNet50 四分类模型",
        script="07_train_resnet50.py",
        group="train",
        required_before_run=["data/resnet_cls"],
    ),
    Step(
        name="08 评价 YOLO11 检测效果",
        script="08_eval_detection.py",
        group="eval",
        required_before_run=["models/yolo/yolo11_best.pt", "data/yolo_detect/images", "data/yolo_detect/labels"],
    ),
    Step(
        name="09 评价 ResNet50 分类效果",
        script="09_eval_classification.py",
        group="eval",
        required_before_run=["models/resnet/resnet50_best.pth", "data/resnet_cls"],
    ),
    Step(
        name="06 使用 YOLO11 框裁剪瓶体，用于端到端测试",
        script="06_crop_bottle_by_yolo.py",
        group="predict",
        required_before_run=["models/yolo/yolo11_best.pt", "data/raw_images"],
    ),
    Step(
        name="10 完整端到端预测：YOLO → 裁剪 → ResNet50",
        script="10_predict_pipeline.py",
        group="predict",
        required_before_run=["models/yolo/yolo11_best.pt", "models/resnet/resnet50_best.pth", "data/raw_images"],
    ),
    Step(
        name="11 生成论文图表与可视化结果",
        script="11_visualize_results.py",
        group="visualize",
        required_before_run=["outputs", "models"],
    ),
]


MODE_TO_GROUPS = {
    "prepare": ["prepare"],
    "train": ["train"],
    "eval": ["eval"],
    "predict": ["predict"],
    "visualize": ["visualize"],
    "all": ["prepare", "train", "eval", "predict", "visualize"],
}


def find_project_root() -> Path:
    """
    自动判断项目根目录。
    - 如果本文件位于 bottle_project/scripts/ 下，则根目录为上一级。
    - 否则优先使用当前工作目录。
    """
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return Path.cwd().resolve()


def ensure_output_dirs(project_root: Path) -> None:
    """创建常用输出目录，避免后续脚本因目录不存在报错。"""
    dirs = [
        "models/yolo",
        "models/resnet",
        "outputs/crops",
        "outputs/figures",
        "outputs/metrics",
        "outputs/predictions",
        "runs",
    ]
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)


def check_script_exists(project_root: Path, step: Step) -> bool:
    script_path = project_root / "scripts" / step.script
    return script_path.is_file()


def check_required_paths(project_root: Path, step: Step) -> List[str]:
    missing: List[str] = []
    for rel in step.required_before_run or []:
        if not (project_root / rel).exists():
            missing.append(rel)
    return missing


def run_step(
    project_root: Path,
    step: Step,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> bool:
    script_path = project_root / "scripts" / step.script

    print("\n" + "=" * 80)
    print(f"[RUN] {step.name}")
    print(f"[SCRIPT] {script_path.relative_to(project_root)}")
    print("=" * 80)

    if not check_script_exists(project_root, step):
        print(f"[ERROR] 找不到脚本：{script_path}")
        return continue_on_error

    missing = check_required_paths(project_root, step)
    if missing:
        print("[WARN] 以下前置文件/目录不存在，当前步骤可能失败：")
        for item in missing:
            print(f"  - {item}")

    cmd = [sys.executable, str(script_path)]

    if dry_run:
        print("[DRY-RUN] 将执行命令：")
        print(" ".join(cmd))
        return True

    started = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - started

    if result.returncode == 0:
        print(f"[OK] {step.script} 执行完成，用时 {elapsed:.1f} 秒")
        return True

    print(f"[ERROR] {step.script} 执行失败，退出码：{result.returncode}，用时 {elapsed:.1f} 秒")
    if continue_on_error:
        print("[WARN] 已启用 --continue-on-error，将继续执行后续步骤。")
        return True

    return False


def select_steps(mode: str, skip_train: bool, skip_eval: bool, skip_predict: bool, skip_visualize: bool) -> List[Step]:
    groups = MODE_TO_GROUPS[mode]
    selected = [s for s in STEPS if s.group in groups]

    skip_groups = set()
    if skip_train:
        skip_groups.add("train")
    if skip_eval:
        skip_groups.add("eval")
    if skip_predict:
        skip_groups.add("predict")
    if skip_visualize:
        skip_groups.add("visualize")

    if skip_groups:
        selected = [s for s in selected if s.group not in skip_groups]

    return selected


def print_plan(project_root: Path, steps: Iterable[Step]) -> None:
    steps = list(steps)
    print("\n项目根目录：", project_root)
    print("即将执行步骤：")
    for idx, step in enumerate(steps, 1):
        print(f"  {idx:02d}. [{step.group}] {step.script} - {step.name}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="饮料瓶残留液体识别项目一键执行总控脚本")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "eval", "predict", "visualize", "all"],
        default="all",
        help="选择执行阶段。默认 all。",
    )
    parser.add_argument("--skip-train", action="store_true", help="跳过训练阶段")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评价阶段")
    parser.add_argument("--skip-predict", action="store_true", help="跳过端到端预测阶段")
    parser.add_argument("--skip-visualize", action="store_true", help="跳过可视化阶段")
    parser.add_argument("--dry-run", action="store_true", help="只打印执行计划，不真正运行")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一步失败后仍继续运行后续步骤；不建议正式实验使用",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    ensure_output_dirs(project_root)

    steps = select_steps(
        mode=args.mode,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        skip_predict=args.skip_predict,
        skip_visualize=args.skip_visualize,
    )

    if not steps:
        print("[ERROR] 没有可执行步骤，请检查 --mode 和 skip 参数。")
        return 2

    print_plan(project_root, steps)

    total_started = time.time()
    passed = 0

    for step in steps:
        ok = run_step(
            project_root=project_root,
            step=step,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if ok:
            passed += 1
        else:
            print("\n流程已停止。请先修复上方报错，再重新运行。")
            print("常用排查：")
            print("  1. 确认原图在 data/raw_images/")
            print("  2. 确认 LabelMe JSON 在 data/annotations_json/")
            print("  3. 确认 JSON 文件名与原图同名")
            print("  4. 确认 requirements.txt 依赖已安装")
            return 1

    total_elapsed = time.time() - total_started

    print("\n" + "=" * 80)
    print("[DONE] 一键流程执行结束")
    print(f"完成步骤：{passed}/{len(steps)}")
    print(f"总用时：{total_elapsed:.1f} 秒")
    print("主要输出位置：")
    print("  - models/yolo/yolo11_best.pt")
    print("  - models/resnet/resnet50_best.pth")
    print("  - outputs/crops/")
    print("  - outputs/metrics/")
    print("  - outputs/predictions/")
    print("  - outputs/figures/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
