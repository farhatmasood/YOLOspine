"""
Variable spatial IoU threshold analysis for overlapping pathology pairs.

Spinal pathologies co-occur at the same anatomical level (e.g. DDD + SS at
L4-L5). Fixed IoU thresholds either suppress valid overlapping detections
or introduce false positives. This script implements pathology-pair-specific
variable thresholds with anatomical compatibility scoring.

Usage::

    python analysis/threshold_analysis.py \\
        --dataset-dir /path/to/dataset_disorders \\
        --output results/threshold_analysis.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["DDD", "LDB", "Normal_IVD", "SS", "TDB", "Spondylolisthesis"]

# Variable IoU thresholds per pathology pair.
# Lower threshold = more overlap expected between these two pathologies.
PAIR_THRESHOLDS = {
    ("DDD", "SS"): 0.20,              # Very high spatial overlap
    ("DDD", "Spondylolisthesis"): 0.25,
    ("LDB", "SS"): 0.25,
    ("SS", "Spondylolisthesis"): 0.20,
    ("DDD", "LDB"): 0.30,
    ("LDB", "Spondylolisthesis"): 0.30,
}
DEFAULT_THRESHOLD = 0.50


# ------------------------------------------------------------------
# Anatomical compatibility
# ------------------------------------------------------------------

# Compatibility scores (0-1) assessing how clinically plausible it is
# for two pathologies to co-occur at the same level.
COMPATIBILITY = {
    ("DDD", "SS"):                1.0,
    ("DDD", "Spondylolisthesis"): 0.9,
    ("LDB", "SS"):                0.8,
    ("SS", "Spondylolisthesis"):  0.95,
    ("DDD", "LDB"):               0.7,
    ("LDB", "Spondylolisthesis"): 0.6,
    ("Normal_IVD", "DDD"):         0.0,
    ("Normal_IVD", "LDB"):         0.0,
    ("Normal_IVD", "SS"):          0.2,
    ("Normal_IVD", "Spondylolisthesis"): 0.1,
}


def _pair_key(a: str, b: str):
    return tuple(sorted([a, b]))


def get_threshold(cls_a: str, cls_b: str) -> float:
    key = _pair_key(cls_a, cls_b)
    return PAIR_THRESHOLDS.get(key, DEFAULT_THRESHOLD)


def get_compatibility(cls_a: str, cls_b: str) -> float:
    key = _pair_key(cls_a, cls_b)
    return COMPATIBILITY.get(key, 0.5)


# ------------------------------------------------------------------
# IoU and dataset analysis
# ------------------------------------------------------------------

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def load_yolo_boxes(label_file: Path, img_w: int, img_h: int):
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        if len(parts) == 5:
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
        else:
            coords = list(map(float, parts[1:]))
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append({"bbox": [x1, y1, x2, y2], "class_id": cid,
                       "class_name": CLASS_NAMES[cid]})
    return boxes


def analyse_dataset(dataset_dir: Path, split: str = "train"):
    """Collect pairwise IoU distributions from ground-truth annotations."""
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split

    pair_ious = defaultdict(list)
    total = 0

    for lbl in sorted(lbl_dir.glob("*.txt")):
        img_path = next(img_dir.glob(f"{lbl.stem}.*"), None)
        if img_path is None:
            continue
        im = Image.open(img_path)
        boxes = load_yolo_boxes(lbl, im.width, im.height)
        total += 1

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i]["class_name"] == boxes[j]["class_name"]:
                    continue
                iou = _iou(boxes[i]["bbox"], boxes[j]["bbox"])
                if iou > 0:
                    key = _pair_key(boxes[i]["class_name"], boxes[j]["class_name"])
                    pair_ious[key].append(iou)

    return pair_ious, total


def apply_combined_filter(pair_ious: dict, min_compat: float = 0.30):
    """
    For each pair, report how many overlaps pass the combined criterion:
    IoU >= variable_threshold  AND  anatomical_compatibility >= min_compat.
    """
    results = {}
    for pair, ious in pair_ious.items():
        a, b = pair
        thr = get_threshold(a, b)
        compat = get_compatibility(a, b)
        passes = [v for v in ious if v >= thr and compat >= min_compat]
        results[f"{a}-{b}"] = {
            "count": len(ious),
            "threshold": thr,
            "compatibility": compat,
            "passes_combined": len(passes),
            "retention_pct": len(passes) / len(ious) * 100 if ious else 0,
            "iou_mean": float(np.mean(ious)) if ious else 0,
            "iou_median": float(np.median(ious)) if ious else 0,
        }
    return results


def main():
    p = argparse.ArgumentParser(description="Variable threshold analysis")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="results/threshold_analysis.json")
    args = p.parse_args()

    ds = Path(args.dataset_dir)
    print(f"Analysing {ds / 'labels' / args.split} ...")
    pair_ious, total = analyse_dataset(ds, args.split)
    print(f"  {total} images,  {sum(len(v) for v in pair_ious.values())} overlap pairs")

    results = apply_combined_filter(pair_ious)

    print("\nPair                      Count  Thr   Compat  Pass   Retention")
    print("-" * 70)
    for pair, r in sorted(results.items()):
        print(f"{pair:25s} {r['count']:5d}  {r['threshold']:.2f}  "
              f"{r['compatibility']:.2f}    {r['passes_combined']:5d}  "
              f"{r['retention_pct']:6.1f}%")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"total_images": total, "pairs": results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
