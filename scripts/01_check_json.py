"""检查原始图片和 JSON 标注是否匹配。
已经根据你们 labelMe 的 JSON 格式补充解析函数。
"""

# scripts/01_check_json.py
import json
from pathlib import Path
import pandas as pd
from _paths import RAW_IMAGES_DIR, JSON_DIR, METRICS_DIR

# 有效标签
VALID_LABELS = {"bottle_empty", "bottle_low", "bottle_medium", "bottle_high"}

METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = METRICS_DIR / "json_check_report.csv"

def check_shape(shape, img_w, img_h):
    label = shape.get("label", "").lower()
    points = shape.get("points")
    if label not in VALID_LABELS:
        return False, f"Invalid label '{label}'"
    if not points or len(points) != 2:
        return False, "Invalid bbox points"
    xmin, ymin = points[0]
    xmax, ymax = points[1]
    if not (0 <= xmin <= img_w and 0 <= xmax <= img_w and 0 <= ymin <= img_h and 0 <= ymax <= img_h):
        return False, f"Coordinates out of image bounds [{xmin},{ymin},{xmax},{ymax}]"
    return True, ""

def main():
    image_files = list(RAW_IMAGES_DIR.glob("*.[jp][pn]g"))  # 支持 jpg/jpeg/png
    records = []

    for img_path in sorted(image_files):
        json_path = JSON_DIR / f"{img_path.stem}.json"
        status = "ok"
        messages = []

        if not json_path.exists():
            status = "error"
            messages.append("Missing JSON")
        else:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)

                img_w = data_json.get("imageWidth")
                img_h = data_json.get("imageHeight")
                shapes = data_json.get("shapes", [])

                if not shapes:
                    status = "error"
                    messages.append("No shapes in JSON")
                else:
                    for shape in shapes:
                        valid, msg = check_shape(shape, img_w, img_h)
                        if not valid:
                            status = "error"
                            messages.append(msg)
            except Exception as e:
                status = "error"
                messages.append(str(e))

        records.append({
            "image": img_path.name,
            "json": json_path.name,
            "status": status,
            "message": "; ".join(messages)
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(df["status"].value_counts())
    print(f"Saved JSON check report to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()