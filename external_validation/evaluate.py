"""
Evaluate a trained model on an external dataset.

Computes per-class and aggregate detection metrics (mAP@50, mAP@50-95,
Precision, Recall) on an out-of-distribution dataset to assess
generalisability.

Usage::

    python external_validation/evaluate.py \\
        --model-type yolospine_v33 \\
        --weights checkpoints/best.pt \\
        --images /path/to/external/images \\
        --labels /path/to/external/labels \\
        --output results/external_eval.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

CLASS_NAMES = {0: "DDD", 1: "LDB", 2: "Normal_IVD",
               3: "SS", 4: "TDB", 5: "Spondylolisthesis"}
NUM_CLASSES = 6


# ------------------------------------------------------------------
# Model loading helpers
# ------------------------------------------------------------------

def _load_ultralytics(weights: str, threshold: float):
    from ultralytics import YOLO
    model = YOLO(weights)

    def predict(img_path, thr):
        results = model.predict(str(img_path), conf=thr, verbose=False)
        dets = []
        for box in results[0].boxes:
            dets.append({
                "bbox": box.xyxy[0].tolist(),
                "score": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })
        return dets
    return predict


def _load_yolospine(weights: str, version: str, device: str):
    from yolospine.models import YOLOspine, YOLOspineV2, YOLOspineV33
    from yolospine.utils.decode import decode_predictions

    factory = {"v1": YOLOspine, "v2": YOLOspineV2, "v33": YOLOspineV33}
    cls = factory[version]
    model = cls(num_classes=NUM_CLASSES)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def predict(img_path, thr):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = model(t)
        return decode_predictions(out, thr, NUM_CLASSES)
    return predict


def build_predict_fn(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mt = args.model_type.lower()
    if mt in ("yolo", "rtdetr", "ultralytics"):
        return _load_ultralytics(args.weights, args.threshold)
    version = {"yolospine_v1": "v1", "yolospine_v2": "v2",
               "yolospine_v33": "v33", "yolospine": "v33"}.get(mt)
    if version:
        return _load_yolospine(args.weights, version, device)
    raise ValueError(f"Unknown model type: {args.model_type}")


# ------------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------------

def load_yolo_labels(label_dir: Path, img_dir: Path):
    """Return ``{stem: [{'bbox': [x1,y1,x2,y2], 'class_id': int}]}``."""
    gt = {}
    for lbl in sorted(label_dir.glob("*.txt")):
        stem = lbl.stem
        img_path = next(img_dir.glob(f"{stem}.*"), None)
        if img_path is None:
            continue
        im = Image.open(img_path)
        w, h = im.size
        boxes = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if len(parts) == 5:
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
            else:
                coords = list(map(float, parts[1:]))
                xs = [coords[i] * w for i in range(0, len(coords), 2)]
                ys = [coords[i] * h for i in range(1, len(coords), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append({"bbox": [x1, y1, x2, y2], "class_id": cid})
        gt[stem] = boxes
    return gt


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _ap_for_class(gt_all, pred_all, cid, iou_thresh):
    all_p = []
    n_gt = 0
    gt_matched = {}
    for stem, gts in gt_all.items():
        cls_gts = [g for g in gts if g["class_id"] == cid]
        n_gt += len(cls_gts)
        gt_matched[stem] = [False] * len(cls_gts)
    for stem, pds in pred_all.items():
        for p in pds:
            if p["class_id"] == cid:
                all_p.append({"stem": stem, "score": p["score"], "bbox": p["bbox"]})
    if n_gt == 0:
        return 0.0
    all_p.sort(key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(all_p))
    fp = np.zeros(len(all_p))
    for i, det in enumerate(all_p):
        cls_gts = [g for g in gt_all.get(det["stem"], []) if g["class_id"] == cid]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(cls_gts):
            v = _iou(det["bbox"], g["bbox"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and not gt_matched[det["stem"]][best_j]:
            tp[i] = 1
            gt_matched[det["stem"]][best_j] = True
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    rec = cum_tp / n_gt
    prec = cum_tp / (cum_tp + cum_fp)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    ap = sum(np.max(mpre[mrec >= t])
             for t in np.linspace(0, 1, 101) if np.any(mrec >= t)) / 101
    return float(ap)


def compute_all_metrics(gt_all, pred_all):
    results = {}
    ap50_list, ap5095_list = [], []
    for cid in range(NUM_CLASSES):
        ap50 = _ap_for_class(gt_all, pred_all, cid, 0.5)
        aps = [_ap_for_class(gt_all, pred_all, cid, t) for t in np.arange(0.5, 0.96, 0.05)]
        ap5095 = float(np.mean(aps))
        results[CLASS_NAMES[cid]] = {"AP50": ap50, "AP50_95": ap5095}
        ap50_list.append(ap50)
        ap5095_list.append(ap5095)
    results["mAP50"] = float(np.mean(ap50_list))
    results["mAP50_95"] = float(np.mean(ap5095_list))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="External dataset evaluation")
    p.add_argument("--model-type", required=True,
                   choices=["yolospine_v1", "yolospine_v2", "yolospine_v33",
                            "yolospine", "yolo", "rtdetr", "ultralytics"])
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="External images directory")
    p.add_argument("--labels", required=True, help="YOLO-format labels directory")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--output", default="results/external_eval.json")
    args = p.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    print("Loading ground truth...")
    gt_all = load_yolo_labels(lbl_dir, img_dir)
    print(f"  {len(gt_all)} images with labels")

    print("Loading model...")
    predict_fn = build_predict_fn(args)

    print("Running inference...")
    pred_all = {}
    t0 = time.time()
    for stem in gt_all:
        img_path = next(img_dir.glob(f"{stem}.*"), None)
        if img_path is None:
            continue
        pred_all[stem] = predict_fn(img_path, args.threshold)
    elapsed = time.time() - t0
    print(f"  {len(pred_all)} images in {elapsed:.1f}s")

    print("Computing metrics...")
    metrics = compute_all_metrics(gt_all, pred_all)
    metrics["num_images"] = len(pred_all)
    metrics["inference_time_s"] = elapsed

    print("\n" + "=" * 50)
    print(f"  mAP@50   : {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50_95']:.4f}")
    for cid in range(NUM_CLASSES):
        n = CLASS_NAMES[cid]
        print(f"    {n:20s}  AP50={metrics[n]['AP50']:.4f}  "
              f"AP50-95={metrics[n]['AP50_95']:.4f}")
    print("=" * 50)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
