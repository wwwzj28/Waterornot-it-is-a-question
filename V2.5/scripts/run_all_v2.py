"""一键全量重跑：02 split -> 14 apply -> 15 train two-stage -> 16 eval two-stage。

适合"队友新拍了一批样本进 raw_images/ + 跑过 LabelMe + 可选过 GUI relabel"之后，
直接重新切分、重新建数据集、重新训练、重新评估。

用法：
  python scripts/run_all_v2.py            # 完整全跑
  python scripts/run_all_v2.py --skip-train  # 只重切+重建数据集，不训练
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list, log_path: Path):
    print(f"\n========== {name} ==========")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.decode("utf-8", errors="replace")
    log_path.write_text(output, encoding="utf-8")
    # 把日志最后 30 行打到终端，留个手感
    tail = output.strip().splitlines()[-30:]
    print("\n".join(tail).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
    elapsed = time.time() - t0
    print(f"\n[{name}] exit={proc.returncode}  elapsed={elapsed:.1f}s  log={log_path}")
    if proc.returncode != 0:
        print(f"[{name}] FAILED, stopping.")
        sys.exit(proc.returncode)


def restore_final_split():
    metrics_dir = ROOT / "outputs" / "metrics"
    required = ["final_split_list.csv", "final_split_stats.csv"]
    missing = [name for name in required if not (metrics_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing final split files: {missing}")
    for src_name, dst_name in [
        ("final_split_list.csv", "split_list.csv"),
        ("final_split_stats.csv", "split_stats.csv"),
    ]:
        (metrics_dir / dst_name).write_bytes((metrics_dir / src_name).read_bytes())
    print("\n========== 02 split_dataset ==========")
    print("Using frozen final split from outputs/metrics/final_split_list.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="只跑 split+apply，不训练不评估")
    parser.add_argument("--skip-eval", action="store_true", help="跑 split+apply+train，不评估")
    parser.add_argument("--use-final-split", action="store_true", help="复用 outputs/metrics/final_split_list.csv，不重新随机划分")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子，默认 42")
    args = parser.parse_args()

    log_dir = ROOT / "outputs" / "logs" / "run_all_v2"
    py = sys.executable

    if args.use_final_split:
        restore_final_split()
    else:
        run_step("02 split_dataset", [py, "scripts/02_split_dataset.py"], log_dir / "02_split.log")
    run_step("14 apply_relabel", [py, "scripts/14_apply_relabel.py"], log_dir / "14_apply.log")

    if args.skip_train:
        print("\nSkipped train + eval (--skip-train).")
        return
    run_step("15 train_two_stage", [py, "scripts/15_train_two_stage.py", "--seed", str(args.seed)], log_dir / "15_train.log")

    if args.skip_eval:
        print("\nSkipped eval (--skip-eval).")
        return
    run_step("16 eval_two_stage", [py, "scripts/16_eval_two_stage.py"], log_dir / "16_eval.log")

    print("\nAll done.")


if __name__ == "__main__":
    main()
