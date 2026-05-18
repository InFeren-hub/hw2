"""将 VisDrone 检测标注转换为 YOLO 格式（归一化中心坐标 + 宽高）。"""
import argparse
from pathlib import Path

from PIL import Image

# VisDrone 原始类别 ID 到 YOLO 类别 ID 的映射
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9,  # motor
}


def convert_split(raw_root: Path, yolo_root: Path, raw_split: str, yolo_split: str):
    image_dir = raw_root / raw_split / "images"
    ann_dir = raw_root / raw_split / "annotations"
    out_image_dir = yolo_root / "images" / yolo_split
    out_label_dir = yolo_root / "labels" / yolo_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob("*.jpg"):
        with Image.open(image_path) as image:
            width, height = image.size
        target_image = out_image_dir / image_path.name
        if not target_image.exists():
            target_image.write_bytes(image_path.read_bytes())

        ann_path = ann_dir / f"{image_path.stem}.txt"
        label_lines = []
        if ann_path.exists():
            with ann_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = [int(float(x)) for x in line.strip().split(",") if x.strip()]
                    if len(parts) < 8:
                        continue
                    x, y, w, h, score, category, truncation, occlusion = parts[:8]
                    if score == 0 or category not in VISDRONE_TO_YOLO or w <= 0 or h <= 0:
                        continue
                    cls = VISDRONE_TO_YOLO[category]
                    cx = (x + w / 2) / width   
                    cy = (y + h / 2) / height  
                    label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")

        (out_label_dir / f"{image_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
    print(f"Converted {raw_split} -> {yolo_split}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone detection annotations to YOLO format.")
    parser.add_argument("--raw-root", required=True, help="Directory containing VisDrone2019-DET-train/dev/test-dev folders.")
    parser.add_argument("--out-root", default="data/VisDrone")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    yolo_root = Path(args.out_root)
    mapping = [
        ("VisDrone2019-DET-train", "train"),
        ("VisDrone2019-DET-val", "val"),
        ("VisDrone2019-DET-test-dev", "test"),
    ]
    for raw_split, yolo_split in mapping:
        if (raw_root / raw_split).exists():
            convert_split(raw_root, yolo_root, raw_split, yolo_split)
        else:
            print(f"Skip missing split: {raw_root / raw_split}")


if __name__ == "__main__":
    main()
