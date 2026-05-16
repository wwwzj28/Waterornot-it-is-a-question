"""训练 YOLO11 瓶体检测模型。"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="configs/bottle_detect.yaml",
        epochs=300,
        imgsz=640,
        batch=8,
        patience=80,
        project="runs/detect",
        name="bottle_yolo11n",
    )

    # 训练完成后，自动复制 best.pt 到 models/yolo/
    # 用 trainer.save_dir 拿实际保存路径，避免不同 Ultralytics 版本路径不一致
    save_dir = Path(model.trainer.save_dir)
    best_pt_src = save_dir / "weights" / "best.pt"
    best_pt_dst = Path("models/yolo/yolo11_best.pt")
    best_pt_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_pt_src.exists():
        shutil.copy(best_pt_src, best_pt_dst)
        print(f"best.pt copied to {best_pt_dst}")
    else:
        raise FileNotFoundError(f"YOLO training did not produce best.pt at {best_pt_src}")

if __name__ == "__main__":
    main()
