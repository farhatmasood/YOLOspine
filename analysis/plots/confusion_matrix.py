"""
Generate confusion matrix visualisations from detection results.

Usage::

    python analysis/plots/confusion_matrix.py \\
        --predictions results/test_predictions.json \\
        --ground-truth /path/to/labels \\
        --output figures/confusion_matrix.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASS_NAMES = ["DDD", "LDB", "Normal_IVD", "SS", "TDB", "Spondylolisthesis"]
NUM_CLASSES = 6


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def build_matrix(gt_boxes: dict, pred_boxes: dict, iou_thresh: float = 0.5):
    """
    Build a (NUM_CLASSES+1) x (NUM_CLASSES+1) confusion matrix.

    Last row/col = background (missed / false positive).
    """
    mat = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    for stem, gts in gt_boxes.items():
        pds = pred_boxes.get(stem, [])
        used = set()
        for g in gts:
            best_iou, best_j, best_c = 0.0, -1, -1
            for j, p in enumerate(pds):
                if j in used:
                    continue
                v = _iou(g["bbox"], p["bbox"])
                if v > best_iou:
                    best_iou, best_j, best_c = v, j, p["class_id"]
            if best_iou >= iou_thresh and best_j >= 0:
                mat[g["class_id"], best_c] += 1
                used.add(best_j)
            else:
                mat[g["class_id"], NUM_CLASSES] += 1  # FN
        for j, p in enumerate(pds):
            if j not in used:
                mat[NUM_CLASSES, p["class_id"]] += 1  # FP

    return mat


def plot_matrix(mat: np.ndarray, output: str, title: str = "Confusion Matrix"):
    labels = CLASS_NAMES + ["BG"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


def main():
    p = argparse.ArgumentParser(description="Confusion matrix plot")
    p.add_argument("--predictions", required=True,
                   help="JSON file {stem: [{bbox, class_id, score}]}")
    p.add_argument("--ground-truth", required=True,
                   help="JSON file or YOLO label dir")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--output", default="figures/confusion_matrix.png")
    p.add_argument("--title", default="Detection Confusion Matrix")
    args = p.parse_args()

    with open(args.predictions) as f:
        pred_boxes = json.load(f)

    gt_path = Path(args.ground_truth)
    if gt_path.is_file():
        with open(gt_path) as f:
            gt_boxes = json.load(f)
    else:
        raise NotImplementedError(
            "Pass a JSON file with the same structure as predictions."
        )

    mat = build_matrix(gt_boxes, pred_boxes, args.iou)
    plot_matrix(mat, args.output, args.title)


if __name__ == "__main__":
    main()
