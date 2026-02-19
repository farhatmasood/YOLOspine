"""
Evaluate a trained Detectron2 model on the internal test set.

Computes:
  - Classification metrics (Accuracy, F1, Kappa, MCC) via box matching
  - Detection metrics (mAP@50, mAP@50-95)

Usage::

    python baselines/detectron2/evaluate.py \\
        --config runs/detectron2/cascade_rcnn_spine_config.yaml \\
        --weights runs/detectron2/model_final.pth \\
        --test-images /path/to/dataset_disorders/images/test \\
        --test-ann /path/to/dataset_disorders/annotations/instances_test.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

CLASS_NAMES = ["DDD", "LDB", "Normal_IVD", "SS", "TDB", "Spondylolisthesis"]
NUM_CLASSES = 6


# ------------------------------------------------------------------
# Ground-truth loading
# ------------------------------------------------------------------

def load_coco_annotations(ann_file: str) -> dict:
    """Return ``{image_stem: [{'bbox': [x1,y1,x2,y2], 'class_id': int}]}``."""
    with open(ann_file) as f:
        coco = json.load(f)

    id2stem = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        stem = id2stem.get(ann["image_id"])
        if stem is None:
            continue
        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]
        if not (0 <= cid < NUM_CLASSES):
            continue
        gt[stem].append({"bbox": [x, y, x + w, y + h], "class_id": cid})
    return dict(gt)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def load_model(config_file: str, weights_file: str, threshold: float = 0.25):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def run_inference(predictor, images_dir: Path, threshold: float = 0.25) -> dict:
    preds: dict = {}
    image_files = sorted(images_dir.glob("*.png"))
    for idx, p in enumerate(image_files):
        if idx % 20 == 0:
            print(f"  {idx}/{len(image_files)}")
        img = cv2.imread(str(p))
        out = predictor(img)["instances"].to("cpu")
        boxes = out.pred_boxes.tensor.numpy() if len(out) else np.empty((0, 4))
        scores = out.scores.numpy() if len(out) else np.empty(0)
        classes = out.pred_classes.numpy() if len(out) else np.empty(0, dtype=int)
        items = []
        for i in range(len(out)):
            s = float(scores[i])
            if s < threshold:
                continue
            cid = int(classes[i])
            if not (0 <= cid < NUM_CLASSES):
                continue
            items.append({"bbox": boxes[i].tolist(), "score": s, "class_id": cid})
        preds[p.stem] = items
    return preds


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(gt_boxes, predictions, iou_thresh=0.5):
    """Match predictions to GT; unmatched GT gets class -1."""
    gt_cls, pred_cls = [], []
    for stem, gts in gt_boxes.items():
        pds = predictions.get(stem, [])
        used = set()
        for g in gts:
            best_iou, best_idx, best_c = 0, -1, -1
            for j, p in enumerate(pds):
                if j in used:
                    continue
                iou = _iou(g["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_idx, best_c = iou, j, p["class_id"]
            gt_cls.append(g["class_id"])
            if best_iou >= iou_thresh:
                pred_cls.append(best_c)
                used.add(best_idx)
            else:
                pred_cls.append(-1)
    return gt_cls, pred_cls


def classification_metrics(gt_cls, pred_cls) -> dict:
    mask = [p != -1 for p in pred_cls]
    gv = [g for g, m in zip(gt_cls, mask) if m]
    pv = [p for p, m in zip(pred_cls, mask) if m]
    if not gv:
        return dict(accuracy=0, macro_f1=0, weighted_f1=0, kappa=0, mcc=0)
    return {
        "accuracy": accuracy_score(gv, pv),
        "macro_f1": f1_score(gv, pv, average="macro", zero_division=0),
        "weighted_f1": f1_score(gv, pv, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(gv, pv),
        "mcc": matthews_corrcoef(gv, pv),
    }


def _ap_for_class(gt_boxes, predictions, cid, iou_thresh):
    all_p = []
    n_gt = 0
    gt_matched = {}
    for stem, gts in gt_boxes.items():
        cls_gts = [g for g in gts if g["class_id"] == cid]
        n_gt += len(cls_gts)
        gt_matched[stem] = [False] * len(cls_gts)

    for stem, pds in predictions.items():
        for p in pds:
            if p["class_id"] == cid:
                all_p.append({"stem": stem, "score": p["score"], "bbox": p["bbox"]})
    if n_gt == 0:
        return 0.0

    all_p.sort(key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(all_p))
    fp = np.zeros(len(all_p))
    for i, det in enumerate(all_p):
        cls_gts = [g for g in gt_boxes.get(det["stem"], []) if g["class_id"] == cid]
        best_iou, best_j = 0, -1
        for j, g in enumerate(cls_gts):
            v = _iou(det["bbox"], g["bbox"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and not gt_matched[det["stem"]][best_j]:
            tp[i] = 1
            gt_matched[det["stem"]][best_j] = True
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / n_gt
    prec = cum_tp / (cum_tp + cum_fp)
    # 101-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    ap = sum(np.max(mpre[mrec >= t]) for t in np.linspace(0, 1, 101) if np.any(mrec >= t)) / 101
    return ap


def compute_map(gt_boxes, predictions):
    ap50, ap5095 = [], []
    for cid in range(NUM_CLASSES):
        ap50.append(_ap_for_class(gt_boxes, predictions, cid, 0.5))
        aps = [_ap_for_class(gt_boxes, predictions, cid, t) for t in np.arange(0.5, 0.96, 0.05)]
        ap5095.append(np.mean(aps))
    return {
        "ap50_per_class": ap50,
        "ap50_95_per_class": ap5095,
        "mAP50": float(np.mean(ap50)),
        "mAP50_95": float(np.mean(ap5095)),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate Detectron2 on test set")
    p.add_argument("--config", required=True, help="Saved YAML config")
    p.add_argument("--weights", required=True, help="model_final.pth")
    p.add_argument("--test-images", required=True, help="Images directory")
    p.add_argument("--test-ann", required=True, help="COCO JSON annotations")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--output", default=None, help="JSON file for results")
    args = p.parse_args()

    gt_boxes = load_coco_annotations(args.test_ann)
    print(f"Loaded {len(gt_boxes)} test images from annotations")

    predictor = load_model(args.config, args.weights, args.threshold)
    predictions = run_inference(predictor, Path(args.test_images), args.threshold)
    print(f"Predictions on {len(predictions)} images")

    gt_cls, pred_cls = match_preds_to_gt(gt_boxes, predictions)
    cls_m = classification_metrics(gt_cls, pred_cls)
    map_m = compute_map(gt_boxes, predictions)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in cls_m.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"  {'mAP@50':15s}: {map_m['mAP50']:.4f}")
    print(f"  {'mAP@50-95':15s}: {map_m['mAP50_95']:.4f}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n:25s}  AP50={map_m['ap50_per_class'][i]:.4f}  "
              f"AP50-95={map_m['ap50_95_per_class'][i]:.4f}")

    results = {"classification": cls_m, "detection": map_m}
    out_path = args.output or str(Path(args.weights).parent / "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
