"""零样本液面检测基线（OpenCV）。

目的：验证"几何路线（找水线 -> 算高度比 -> 落类）"是否可行，
不依赖任何深度模型，不训练任何参数。

流程：
  1. 读 data/resnet_cls/test 下的瓶子裁剪图（已是 YOLO/标注裁出的瓶身）
  2. 灰度 + 高斯平滑
  3. Sobel 取横向边缘强度 (|d/dy|)，水线特征是水平方向延伸的强边
  4. 对每一行（y）求行强度 = 该行像素强度的 sum
  5. 在瓶身的中部 [10%, 95%] 区域内找峰值 => 水线 y
  6. ratio = 1 - y_peak / H  （从底部往上量的水位）
  7. 按阈值 5/35/70 落 4 类
  8. 与真实标签对照，输出 acc + 把每张图水线画出来存到 outputs/figures/waterline_debug/
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from _paths import CLS_DIR, FIGURES_DIR, METRICS_DIR

OUT_DIR = FIGURES_DIR / "waterline_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = METRICS_DIR / "waterline_cv_predictions.csv"

# 与你提议一致：5 / 35 / 70 % 阈值
T_EMPTY_LOW = 0.05
T_LOW_MED = 0.35
T_MED_HIGH = 0.70

# 在瓶身上下边沿的搜索区间（避开瓶盖/底反光误判为水线）
SEARCH_TOP = 0.05
SEARCH_BOT = 0.97


def ratio_to_class(r: float) -> str:
    if r < T_EMPTY_LOW:
        return "empty"
    if r < T_LOW_MED:
        return "low"
    if r < T_MED_HIGH:
        return "medium"
    return "high"


def detect_waterline(img_bgr: np.ndarray):
    """返回 (ratio, y_line, debug_strip)。"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel 横向边缘（水线 = 水平延伸的强边，所以取 dy）
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sob_y_abs = np.abs(sob_y)

    # 中心 60% 列做行积分，避开瓶身左右轮廓干扰
    x1, x2 = int(w * 0.20), int(w * 0.80)
    if x2 - x1 < 5:
        x1, x2 = 0, w
    row_energy = sob_y_abs[:, x1:x2].sum(axis=1)

    # 中位平滑去掉单行突刺
    k = max(3, h // 60 | 1)
    row_energy_s = cv2.GaussianBlur(row_energy.reshape(-1, 1).astype(np.float32),
                                    (1, k), 0).ravel()

    # 在 [SEARCH_TOP, SEARCH_BOT] 范围内找最大行
    y_lo = int(h * SEARCH_TOP)
    y_hi = int(h * SEARCH_BOT)
    region = row_energy_s[y_lo:y_hi]
    if region.size == 0:
        return 0.0, h - 1, None
    y_peak = y_lo + int(np.argmax(region))

    ratio = 1.0 - (y_peak / h)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return ratio, y_peak, row_energy_s


def draw_debug(img_bgr, y_peak, ratio, true_lbl, pred_lbl):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    color = (0, 255, 0) if true_lbl == pred_lbl else (0, 0, 255)
    cv2.line(vis, (0, y_peak), (w, y_peak), color, 2)
    text = f"r={ratio:.2f} T={true_lbl} P={pred_lbl}"
    cv2.putText(vis, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return vis


def main():
    rows = []
    for split in ["test"]:
        split_dir = CLS_DIR / split
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            true_lbl = cls_dir.name
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                ratio, y_peak, _ = detect_waterline(img)
                pred_lbl = ratio_to_class(ratio)
                rows.append({
                    "split": split,
                    "image": p.name,
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "ratio": round(ratio, 4),
                    "y_peak": y_peak,
                    "h": img.shape[0],
                    "w": img.shape[1],
                })
                vis = draw_debug(img, y_peak, ratio, true_lbl, pred_lbl)
                cv2.imwrite(str(OUT_DIR / f"{true_lbl}_{p.stem}.jpg"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved per-image predictions: {OUT_CSV}")
    print(f"Saved debug images to:       {OUT_DIR}")
    print()

    labels = ["empty", "low", "medium", "high"]
    acc = accuracy_score(df["true_label"], df["pred_label"])
    print(f"Test acc (CV zero-shot): {acc:.4f}  ({(df['true_label']==df['pred_label']).sum()}/{len(df)})")
    print()
    print("Per-class:")
    for l in labels:
        sub = df[df["true_label"] == l]
        if len(sub) == 0:
            continue
        ok = (sub["pred_label"] == l).sum()
        print(f"  {l:<7} {len(sub):>3}  correct={ok}  acc={ok/len(sub):.3f}")
    print()
    print("Confusion (rows=true, cols=pred):")
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    print("              " + "  ".join(f"{l:>7}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:>12}", "  ".join(f"{v:>7d}" for v in row))


if __name__ == "__main__":
    main()
