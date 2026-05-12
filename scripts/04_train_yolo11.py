"""训练 YOLO11 瓶体检测模型。"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="configs/bottle_detect.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project="runs/detect",
        name="bottle_yolo11n",
    )

    # 训练完成后，自动复制 best.pt 到 models/yolo/
    best_pt_src = Path("runs/detect/bottle_yolo11n/weights/best.pt")
    best_pt_dst = Path("models/yolo/yolo11_best.pt")
    best_pt_dst.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    if best_pt_src.exists():
        shutil.copy(best_pt_src, best_pt_dst)
        print(f"best.pt copied to {best_pt_dst}")
    else:
        print(f"Warning: best.pt not found at {best_pt_src}")

if __name__ == "__main__":
    main()
