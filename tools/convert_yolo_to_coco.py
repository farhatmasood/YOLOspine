"""
Convert YOLO polygon / bbox annotations to COCO JSON format.

Required by Detectron2 and RF-DETR baselines which expect COCO-style
``instances_{split}.json`` files.

Usage::

    python tools/convert_yolo_to_coco.py \\
        --dataset-dir /path/to/dataset_disorders \\
        --output-dir  /path/to/coco_output

Reads ``images/{split}`` and ``labels/{split}`` directories.  Label files
may contain either YOLO bounding boxes (``class cx cy w h``) or YOLO
polygons (``class x1 y1 x2 y2 ...``).  Both formats are auto-detected.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "DDD",
    "LDB",
    "Normal_IVD",
    "SS",
    "TDB",
    "Spondylolisthesis",
]


def _yolo_bbox_to_coco(parts, w, h):
    """Convert ``cx cy bw bh`` (normalised) to COCO ``[x, y, w, h]``."""
    cx, cy, bw, bh = [float(x) for x in parts]
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return [x, y, bw * w, bh * h]


def _yolo_poly_to_coco(parts, w, h):
    """Convert normalised polygon to COCO bbox + segmentation."""
    coords = [float(x) for x in parts]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs) * w, max(xs) * w
    y_min, y_max = min(ys) * h, max(ys) * h
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    seg = []
    for i, v in enumerate(coords):
        seg.append(v * w if i % 2 == 0 else v * h)

    # Shoelace area
    n = len(xs)
    area = abs(sum(xs[i] * w * ys[(i + 1) % n] * h -
                   xs[(i + 1) % n] * w * ys[i] * h for i in range(n))) / 2
    return bbox, [seg], area


def convert_split(dataset_dir: Path, output_dir: Path, split: str):
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    coco = {
        "info": {
            "description": f"Spinal Disorders - {split}",
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [{"id": 1, "name": "Research Use Only", "url": ""}],
        "categories": [
            {"id": i, "name": n, "supercategory": "spine"}
            for i, n in enumerate(CLASS_NAMES)
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(images_dir.glob("*.jpg"))
    print(f"{split.upper()}: {len(image_files)} images")

    out_img = output_dir / split / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    for img_id, img_path in enumerate(image_files, 1):
        im = Image.open(img_path)
        w, h = im.size
        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
        )
        shutil.copy2(img_path, out_img / img_path.name)

        lbl = labels_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cid = int(parts[0])
            rest = parts[1:]

            if len(rest) == 4:
                # Standard YOLO bbox
                bbox = _yolo_bbox_to_coco(rest, w, h)
                seg = []
                area = bbox[2] * bbox[3]
            else:
                # Polygon
                bbox, seg, area = _yolo_poly_to_coco(rest, w, h)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": bbox,
                "segmentation": seg,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    ann_file = output_dir / split / f"instances_{split}.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"  Annotations: {ann_id - 1}  ->  {ann_file}")


def main():
    p = argparse.ArgumentParser(description="YOLO -> COCO conversion")
    p.add_argument("--dataset-dir", required=True,
                   help="Root with images/{split} and labels/{split}")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for COCO dataset")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    ds = Path(args.dataset_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        convert_split(ds, out, split)

    print("Done.")


if __name__ == "__main__":
    main()
