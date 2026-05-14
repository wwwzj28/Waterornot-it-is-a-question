"""重标注 GUI：并排展示原图与裁剪图，打上新标签。

输出到 data/relabel.csv：image, new_label, note（保留记录、不动现有数据）。
new_label 取值：empty / low / medium / high / invalid / skip。

键盘快捷键：
  1=empty   2=low   3=medium   4=high   0=invalid   s=skip
  ← / → 上/下一张   space=下一张   ctrl+s 手动存盘（每次点击也会自动存）

启动：python scripts/13_relabel_gui.py
可选 --filter only_lowmed 只看原标签是 low/medium 的样本。
"""

from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from _paths import RAW_IMAGES_DIR, JSON_DIR, METRICS_DIR

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RELABEL_CSV = DATA_DIR / "relabel.csv"

LABEL_KEYS = {
    "1": "empty",
    "2": "low",
    "3": "medium",
    "4": "high",
    "0": "invalid",
    "s": "skip",
}
LABEL_ORDER = ["empty", "low", "medium", "high", "invalid", "skip"]
LABEL_COLORS = {
    "empty": "#9ca3af",
    "low": "#3b82f6",
    "medium": "#f59e0b",
    "high": "#10b981",
    "invalid": "#ef4444",
    "skip": "#6b7280",
}

VALID_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def load_existing_labels(csv_path: Path) -> dict[str, dict]:
    """返回 image_name -> {new_label, note}，用于断点续标。"""
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["image"]] = {
                "new_label": row.get("new_label", ""),
                "note": row.get("note", ""),
            }
    return out


def save_all_labels(csv_path: Path, store: dict[str, dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "new_label", "note"])
        for name in sorted(store.keys()):
            row = store[name]
            w.writerow([name, row.get("new_label", ""), row.get("note", "")])


def load_raw_label(json_path: Path) -> str:
    if not json_path.exists():
        return ""
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for s in d.get("shapes", []):
        lbl = s.get("label", "").lower()
        for prefix in ("bottle_", ""):
            if lbl.startswith(prefix):
                short = lbl.replace("bottle_", "")
                if short in {"empty", "low", "medium", "high"}:
                    return short
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
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return None


class RelabelApp:
    def __init__(self, root: tk.Tk, image_files: list[Path], filter_name: str | None):
        self.root = root
        self.image_files = image_files
        self.filter_name = filter_name
        self.idx = 0

        self.store = load_existing_labels(RELABEL_CSV)

        self.root.title("Relabel GUI")
        self.root.geometry("1400x900")
        self.root.bind("<Key>", self._on_key)

        self._build_ui()
        self._jump_to_first_unlabeled()
        self._show()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.lbl_progress = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.lbl_progress.pack(side=tk.LEFT)

        self.lbl_filename = ttk.Label(top, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_filename.pack(side=tk.LEFT, padx=20)

        self.lbl_old = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.lbl_old.pack(side=tk.LEFT, padx=10)

        self.lbl_new = ttk.Label(top, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_new.pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="← Prev", command=self.prev).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Next →", command=self.next).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Save now", command=self.save).pack(side=tk.RIGHT, padx=4)

        # 中部：左原图右裁剪图
        center = ttk.Frame(self.root)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)

        self.left_label = ttk.Label(center, text="raw image", anchor="center")
        self.left_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.right_label = ttk.Label(center, text="cropped bottle", anchor="center")
        self.right_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # 底部：按钮 + 备注
        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        ttk.Label(bottom, text="note:").pack(side=tk.LEFT)
        self.note_var = tk.StringVar()
        self.note_entry = ttk.Entry(bottom, textvariable=self.note_var, width=40)
        self.note_entry.pack(side=tk.LEFT, padx=4)
        # 防止 Entry 抢了快捷键：点 Esc 可以跳出输入
        self.note_entry.bind("<Escape>", lambda e: self.root.focus_set())

        for lbl in LABEL_ORDER:
            color = LABEL_COLORS[lbl]
            key = next(k for k, v in LABEL_KEYS.items() if v == lbl)
            btn = tk.Button(
                bottom,
                text=f"{lbl} [{key}]",
                bg=color,
                fg="white",
                width=10,
                command=lambda l=lbl: self.set_label(l),
            )
            btn.pack(side=tk.LEFT, padx=3)

    def _jump_to_first_unlabeled(self):
        for i, p in enumerate(self.image_files):
            rec = self.store.get(p.name)
            if not rec or not rec.get("new_label"):
                self.idx = i
                return
        self.idx = 0

    def _on_key(self, event):
        # 如果焦点在 note Entry 里，别抢快捷键
        if self.root.focus_get() is self.note_entry:
            return
        k = event.keysym.lower()
        if k in ("left",):
            self.prev()
        elif k in ("right", "space"):
            self.next()
        elif event.char in LABEL_KEYS:
            self.set_label(LABEL_KEYS[event.char])
        elif (event.state & 0x4) and k == "s":  # ctrl+s
            self.save()

    def _show(self):
        if not self.image_files:
            self.lbl_filename.config(text="no images")
            return
        self.idx = max(0, min(self.idx, len(self.image_files) - 1))
        img_path = self.image_files[self.idx]
        json_path = JSON_DIR / f"{img_path.stem}.json"

        # raw 图 + bbox
        try:
            raw = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.lbl_filename.config(text=f"cannot open {img_path.name}: {e}")
            return
        bbox = load_bbox(json_path)
        raw_draw = raw.copy()
        if bbox:
            d = ImageDraw.Draw(raw_draw)
            d.rectangle(bbox, outline="#facc15", width=4)
        crop_img = raw.crop(bbox) if bbox else raw

        self._show_image(self.left_label, raw_draw, max_size=(640, 760))
        self._show_image(self.right_label, crop_img, max_size=(640, 760))

        old = load_raw_label(json_path)
        rec = self.store.get(img_path.name, {})
        new = rec.get("new_label", "")
        note = rec.get("note", "")

        labeled = sum(1 for v in self.store.values() if v.get("new_label"))
        self.lbl_progress.config(text=f"[{self.idx + 1}/{len(self.image_files)}]  labeled={labeled}")
        self.lbl_filename.config(text=img_path.name)
        self.lbl_old.config(text=f"old: {old or '?'}")
        new_color = LABEL_COLORS.get(new, "#000000")
        self.lbl_new.config(text=f"new: {new or '—'}", foreground=new_color)
        self.note_var.set(note)

    def _show_image(self, widget: ttk.Label, pil_img: Image.Image, max_size):
        w, h = pil_img.size
        mw, mh = max_size
        scale = min(mw / w, mh / h, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(pil_img)
        widget.configure(image=tkimg)
        widget.image = tkimg  # 防止 GC

    def set_label(self, lbl: str):
        if not self.image_files:
            return
        name = self.image_files[self.idx].name
        if lbl == "skip":
            # skip = 保留原标签不记录任何东西，直接下一张
            self.next()
            return
        rec = self.store.get(name, {})
        rec["new_label"] = lbl
        rec["note"] = self.note_var.get().strip()
        self.store[name] = rec
        save_all_labels(RELABEL_CSV, self.store)  # 每次设标都指盘
        self.next()

    def next(self):
        if self.idx < len(self.image_files) - 1:
            self.idx += 1
            self._show()
        else:
            messagebox.showinfo("done", "已是最后一张")

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def save(self):
        save_all_labels(RELABEL_CSV, self.store)
        messagebox.showinfo("saved", f"saved to {RELABEL_CSV}")


def collect_images(filter_name: str | None) -> list[Path]:
    files = sorted(p for p in RAW_IMAGES_DIR.iterdir() if p.suffix.lower() in VALID_SUFFIX)

    if filter_name == "only_lowmed":
        keep = []
        for p in files:
            old = load_raw_label(JSON_DIR / f"{p.stem}.json")
            if old in {"low", "medium"}:
                keep.append(p)
        return keep
    if filter_name == "unlabeled":
        store = load_existing_labels(RELABEL_CSV)
        return [p for p in files if not store.get(p.name, {}).get("new_label")]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=["all", "only_lowmed", "unlabeled"], default="all")
    args = parser.parse_args()

    files = collect_images(None if args.filter == "all" else args.filter)
    if not files:
        print("no images to label.")
        return

    root = tk.Tk()
    RelabelApp(root, files, args.filter)
    root.mainloop()


if __name__ == "__main__":
    main()
