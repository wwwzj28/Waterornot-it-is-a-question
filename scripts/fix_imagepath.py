"""按 JSON 文件名 stem 在 raw_images/ 中找到实际图片，重写 imagePath。"""
import json
from _paths import RAW_IMAGES_DIR, JSON_DIR


def main():
    updated, unchanged, missing = 0, 0, []
    for json_file in sorted(JSON_DIR.glob("*.json")):
        stem = json_file.stem
        matches = sorted(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        if not matches:
            missing.append(json_file.name)
            continue
        actual_name = matches[0].name

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("imagePath") == actual_name:
            unchanged += 1
            continue

        data["imagePath"] = actual_name
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        updated += 1

    print(f"updated: {updated}")
    print(f"already correct: {unchanged}")
    if missing:
        print(f"no matching image for {len(missing)} JSON(s):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
