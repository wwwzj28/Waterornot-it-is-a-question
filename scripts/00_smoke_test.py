import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import ultralytics

from _paths import ROOT, RAW_IMAGES_DIR, JSON_DIR, YOLO_DIR, CLS_DIR


def main():
    print("Python:", sys.version)
    print("Project root:", ROOT)
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Ultralytics:", ultralytics.__version__)

    required_dirs = [
        RAW_IMAGES_DIR,
        JSON_DIR,
        YOLO_DIR,
        CLS_DIR,
    ]

    for d in required_dirs:
        print(f"{d}: {'OK' if d.exists() else 'MISSING'}")

    print("\nSmoke test finished.")


if __name__ == "__main__":
    main()
