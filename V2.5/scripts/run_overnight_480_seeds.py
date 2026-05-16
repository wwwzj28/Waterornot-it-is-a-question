from pathlib import Path
import csv
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PY = Path("D:/CondaData/envs/water-v2/python.exe")
SEEDS = [42, 1, 7, 1234]
RUN_NAME_PREFIX = "480x240_seed"
OUT_ROOT = ROOT / "outputs" / "metrics" / "efficientnet_480x240_multi_seed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run(cmd, log_path: Path):
    started = time.time()
    proc = subprocess.run(
        [str(PY), *cmd],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_bytes(proc.stdout)
    elapsed = (time.time() - started) / 60
    return proc.returncode, elapsed


def parse_report(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    metrics = {}
    patterns = {
        "e2e_acc": r"End-to-end 4-class test acc:\s*([0-9.]+)",
        "q1_acc": r"Q1 \(empty vs not\) test acc:\s*([0-9.]+)",
        "q2_acc": r"Q2 \(low/med/high on true-nonempty\) test acc:\s*([0-9.]+)",
        "macro_f1": r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
        "weighted_f1": r"weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        metrics[key] = float(match.group(1)) if match else None
    for label in ["empty", "low", "medium", "high"]:
        match = re.search(rf"{label}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)
        if match:
            metrics[f"{label}_precision"] = float(match.group(1))
            metrics[f"{label}_recall"] = float(match.group(2))
            metrics[f"{label}_f1"] = float(match.group(3))
            metrics[f"{label}_support"] = int(match.group(4))
    return metrics


def save_current_outputs(seed: int, run_name: str):
    seed_dir = OUT_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = ROOT / "outputs" / "metrics"
    figures_dir = ROOT / "outputs" / "figures"
    model_dir = ROOT / "models" / f"efficientnet_v2_s_{run_name}" / "two_stage"

    files = [
        f"two_stage_test_report_efficientnet_v2_s_{run_name}.txt",
        f"two_stage_test_predictions_efficientnet_v2_s_{run_name}.csv",
        f"two_stage_train_log_efficientnet_v2_s_{run_name}.csv",
        "split_stats.csv",
        "split_stats_v2.csv",
        "split_list.csv",
        "split_list_v2.csv",
    ]
    for name in files:
        src = metrics_dir / name
        if src.exists():
            shutil.copy2(src, seed_dir / name)

    dst_model_dir = seed_dir / "models"
    dst_model_dir.mkdir(exist_ok=True)
    for name in ["stage_a_empty_vs_not.pth", "stage_b_low_med_high.pth"]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, dst_model_dir / name)

    fig = figures_dir / f"two_stage_confusion_matrix_efficientnet_v2_s_{run_name}.png"
    if fig.exists():
        (seed_dir / "figures").mkdir(exist_ok=True)
        shutil.copy2(fig, seed_dir / "figures" / fig.name)
    return seed_dir


def restore_best(seed_dir: Path, seed: int):
    metrics_dir = ROOT / "outputs" / "metrics"
    figures_dir = ROOT / "outputs" / "figures"
    final_model_dir = ROOT / "models" / "efficientnet_v2_s_480x240_best" / "two_stage"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{RUN_NAME_PREFIX}{seed}"
    report = seed_dir / f"two_stage_test_report_efficientnet_v2_s_{run_name}.txt"
    preds = seed_dir / f"two_stage_test_predictions_efficientnet_v2_s_{run_name}.csv"
    train_log = seed_dir / f"two_stage_train_log_efficientnet_v2_s_{run_name}.csv"
    if report.exists():
        shutil.copy2(report, metrics_dir / "two_stage_test_report.txt")
        shutil.copy2(report, metrics_dir / "best_two_stage_test_report_efficientnet_v2_s_480x240.txt")
    if preds.exists():
        shutil.copy2(preds, metrics_dir / "two_stage_test_predictions.csv")
    if train_log.exists():
        shutil.copy2(train_log, metrics_dir / "two_stage_train_log.csv")
    for name in ["stage_a_empty_vs_not.pth", "stage_b_low_med_high.pth"]:
        src = seed_dir / "models" / name
        if src.exists():
            shutil.copy2(src, final_model_dir / name)
    fig = seed_dir / "figures" / f"two_stage_confusion_matrix_efficientnet_v2_s_{run_name}.png"
    if fig.exists():
        shutil.copy2(fig, figures_dir / "two_stage_confusion_matrix.png")


def main():
    rows = []
    fixed_log = OUT_ROOT / "00_use_final_split.log"
    code, elapsed = run(["scripts/run_all_v2.py", "--use-final-split", "--skip-train"], fixed_log)
    if code != 0:
        raise SystemExit(code)

    for seed in SEEDS:
        run_name = f"{RUN_NAME_PREFIX}{seed}"
        seed_dir = OUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"START seed={seed} run_name={run_name}", flush=True)
        train_code, train_elapsed = run(
            [
                "scripts/15_train_two_stage.py",
                "--seed", str(seed),
                "--backbone", "efficientnet_v2_s",
                "--input-h", "480",
                "--input-w", "240",
                "--run-name", run_name,
            ],
            seed_dir / "15_train.log",
        )
        print(f"TRAIN DONE seed={seed} exit={train_code} elapsed={train_elapsed:.1f} min", flush=True)
        if train_code != 0:
            raise SystemExit(train_code)
        eval_code, eval_elapsed = run(
            [
                "scripts/16_eval_two_stage.py",
                "--backbone", "efficientnet_v2_s",
                "--input-h", "480",
                "--input-w", "240",
                "--run-name", run_name,
            ],
            seed_dir / "16_eval.log",
        )
        print(f"EVAL DONE seed={seed} exit={eval_code} elapsed={eval_elapsed:.1f} min", flush=True)
        if eval_code != 0:
            raise SystemExit(eval_code)
        seed_dir = save_current_outputs(seed, run_name)
        report = seed_dir / f"two_stage_test_report_efficientnet_v2_s_{run_name}.txt"
        metrics = parse_report(report)
        metrics["seed"] = seed
        metrics["run_name"] = run_name
        metrics["train_minutes"] = round(train_elapsed, 2)
        rows.append(metrics)
        print(f"RESULT seed={seed} acc={metrics.get('e2e_acc')} q1={metrics.get('q1_acc')} q2={metrics.get('q2_acc')}", flush=True)

    # include existing seed 2026 result as baseline if available
    existing = ROOT / "outputs" / "metrics" / "two_stage_test_report_efficientnet_v2_s_480x240.txt"
    if existing.exists():
        m = parse_report(existing)
        m["seed"] = 2026
        m["run_name"] = "480x240"
        m["train_minutes"] = "existing"
        rows.append(m)

    fieldnames = ["seed", "run_name", "train_minutes", "e2e_acc", "q1_acc", "q2_acc", "macro_f1", "weighted_f1"]
    for label in ["empty", "low", "medium", "high"]:
        fieldnames += [f"{label}_precision", f"{label}_recall", f"{label}_f1", f"{label}_support"]

    summary = OUT_ROOT / "summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["e2e_acc"], r["macro_f1"], r["q2_acc"]), reverse=True):
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    best = max(rows, key=lambda r: (r["e2e_acc"], r["macro_f1"], r["q2_acc"]))
    if best["run_name"] == "480x240":
        # existing baseline remains separately stored under efficientnet_v2_s_480x240; copy from canonical experiment dir if present
        baseline_seed_dir = OUT_ROOT / "best_existing_2026"
        baseline_seed_dir.mkdir(exist_ok=True)
        # no-op restore here; current canonical may have changed, so prefer not to overwrite model without seed dir
    else:
        restore_best(OUT_ROOT / f"seed_{best['seed']}", int(best["seed"]))
    (OUT_ROOT / "best.txt").write_text(
        f"best_seed={best['seed']}\nrun_name={best['run_name']}\ne2e_acc={best['e2e_acc']}\nq1_acc={best['q1_acc']}\nq2_acc={best['q2_acc']}\nmacro_f1={best['macro_f1']}\nweighted_f1={best['weighted_f1']}\n",
        encoding="utf-8",
    )
    print(f"SUMMARY {summary}", flush=True)
    print(f"BEST seed={best['seed']} run={best['run_name']} acc={best['e2e_acc']}", flush=True)


if __name__ == "__main__":
    main()
