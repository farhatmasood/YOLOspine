"""
YOLOspine Evaluation Script
============================

Evaluate a trained YOLOspine checkpoint on a dataset split.

Computes:
    - mAP@0.5, Precision, Recall, F1
    - Per-class Average Precision
    - Optional prediction visualizations

Usage::

    python tools/evaluate.py --checkpoint runs/best_mAP.pth \\
        --data_dir dataset_disorders --split val

    python tools/evaluate.py --checkpoint runs/best_mAP.pth \\
        --model v33 --visualize --num_vis 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yolospine.data.dataset import (
    SpinalMRIDataset,
    collate_fn,
    get_transform,
    CLASS_NAMES,
)
from yolospine.utils.decode import decode_predictions
from yolospine.utils.metrics import compute_ap, compute_metrics
from yolospine.utils.visualization import plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOspine")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="dataset_disorders")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--model", type=str, default="v2",
                   choices=["v1", "v2", "v33"])
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--nms_thresh", type=float, default=0.45)
    p.add_argument("--iou_thresh", type=float, default=0.5,
                   help="IoU threshold for mAP computation")
    p.add_argument("--visualize", action="store_true",
                   help="Save prediction visualizations")
    p.add_argument("--num_vis", type=int, default=10,
                   help="Number of visualizations to save")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: runs/<model>_eval)")
    return p.parse_args()


def load_model(variant, checkpoint_path, num_classes, image_size, device):
    """Load model and weights from checkpoint."""
    if variant == "v1":
        from yolospine.models.yolospine import YOLOspine
        model = YOLOspine(num_classes=num_classes)
    elif variant == "v2":
        from yolospine.models.yolospine_v2 import build_model as _build_v2
        model = _build_v2(num_classes=num_classes, image_size=image_size)
    elif variant == "v33":
        from yolospine.models.yolospine_v33 import YOLOspineV33
        model = YOLOspineV33(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: epoch {epoch}")

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, args):
    """Run full evaluation."""
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True)
                   for k, v in batch["targets"].items()}

        with autocast(enabled=True):
            out = model(images)
            s1 = out[0] if isinstance(out, tuple) else out

        preds = decode_predictions(
            s1,
            num_classes=args.num_classes,
            image_size=args.image_size,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
        )
        all_preds.extend(preds)
        all_targets.append({
            "boxes": targets["boxes"],
            "labels": targets["labels"],
            "batch_size": images.shape[0],
        })

    # Re-index
    offset = 0
    boxes_list, labels_list = [], []
    for t in all_targets:
        b = t["boxes"]
        if b.numel() > 0:
            b = b.clone()
            b[:, 0] += offset
            boxes_list.append(b)
            labels_list.append(t["labels"])
        offset += t["batch_size"]

    if boxes_list:
        combined = {
            "boxes": torch.cat(boxes_list),
            "labels": torch.cat(labels_list),
        }
    else:
        combined = {
            "boxes": torch.empty((0, 5), device=device),
            "labels": torch.empty(0, dtype=torch.long, device=device),
        }

    metrics = compute_metrics(
        all_preds, combined,
        iou_threshold=args.iou_thresh,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    return metrics, all_preds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = args.output_dir or os.path.join(
        "runs", f"{args.model}_eval")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"YOLOspine {args.model.upper()} Evaluation")
    print("=" * 70)

    model = load_model(
        args.model, args.checkpoint, args.num_classes,
        args.image_size, device)

    dataset = SpinalMRIDataset(
        os.path.join(args.data_dir, "images", args.split),
        os.path.join(args.data_dir, "labels", args.split),
        CLASS_NAMES,
        transform=get_transform(train=False, img_size=args.image_size),
        img_size=args.image_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True)

    print(f"Split: {args.split} ({len(dataset)} images)")
    print(f"Confidence: {args.conf_thresh}, NMS: {args.nms_thresh}, "
          f"IoU: {args.iou_thresh}")

    metrics, preds = evaluate(model, loader, device, args)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  mAP@{args.iou_thresh:.2f}: {metrics['mAP']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Num GT:     {metrics['num_gt']}")

    if metrics["aps"]:
        print("\n  Per-class AP:")
        for i, ap in enumerate(metrics["aps"]):
            name = CLASS_NAMES.get(i, str(i))
            print(f"    {name:<20s}: {ap:.4f}")

    # Save results JSON
    results = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "conf_thresh": args.conf_thresh,
        "nms_thresh": args.nms_thresh,
        "iou_thresh": args.iou_thresh,
        "mAP": metrics["mAP"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "aps": metrics["aps"],
        "num_gt": metrics["num_gt"],
    }
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Visualizations
    if args.visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        img_dir = os.path.join(args.data_dir, "images", args.split)
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        n_vis = min(args.num_vis, len(img_files), len(preds))
        indices = np.random.choice(len(img_files), n_vis, replace=False)

        for idx in indices:
            img_path = os.path.join(img_dir, img_files[idx])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (args.image_size, args.image_size))

            boxes, scores, labels = preds[idx]
            save_path = os.path.join(vis_dir,
                                     f"{Path(img_files[idx]).stem}_pred.png")
            plot_predictions(
                img, boxes.cpu(), scores.cpu(), labels.cpu(),
                title=img_files[idx], save_path=save_path)

        print(f"Saved {n_vis} visualizations to {vis_dir}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
