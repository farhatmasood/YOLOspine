"""
Train segmentation baselines (U-Net++, Swin-UNet, TransUNet).

Segmentation masks are derived from YOLO bounding-box annotations by
filling each box region with its class index (+1 for background=0).
The resulting masks are evaluated using converted-box metrics.

Usage::

    python baselines/train_baseline.py --model unetplusplus \\
        --data_dir dataset_disorders --epochs 50

    python baselines/train_baseline.py --model swinunet \\
        --epochs 30 --batch_size 4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.ops as ops
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.segmentation_models import get_baseline_model
from yolospine.data.dataset import (
    CLASS_NAMES, SpinalMRIDataset, collate_fn, get_transform,
    create_dataloaders,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _masks_to_boxes_metrics(pred_mask, gt_mask, num_classes,
                            iou_thresh=0.5):
    """Convert class masks to boxes and compute precision/recall."""
    precisions, recalls = [], []
    B, H, W = pred_mask.shape

    for b in range(B):
        p = pred_mask[b].cpu().numpy()
        g = gt_mask[b].cpu().numpy()

        for cls in range(1, num_classes):
            gt_c, _ = cv2.findContours(
                (g == cls).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_boxes = [cv2.boundingRect(c) for c in gt_c]
            gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]

            pr_c, _ = cv2.findContours(
                (p == cls).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pr_boxes = [cv2.boundingRect(c) for c in pr_c]
            pr_boxes = [[x, y, x + w, y + h] for x, y, w, h in pr_boxes]

            if not gt_boxes and not pr_boxes:
                continue
            if not gt_boxes:
                precisions.append(0.0)
                continue
            if not pr_boxes:
                recalls.append(0.0)
                continue

            g_t = torch.tensor(gt_boxes, dtype=torch.float32)
            p_t = torch.tensor(pr_boxes, dtype=torch.float32)
            iou_matrix = ops.box_iou(p_t, g_t)

            tp, matched = 0, set()
            for pi in range(len(pr_boxes)):
                if iou_matrix.shape[1] > 0:
                    mi, mg = torch.max(iou_matrix[pi], dim=0)
                    if mi > iou_thresh and mg.item() not in matched:
                        tp += 1
                        matched.add(mg.item())

            fp = len(pr_boxes) - tp
            fn = len(gt_boxes) - len(matched)
            precisions.append(tp / (tp + fp + 1e-6))
            recalls.append(tp / (tp + fn + 1e-6))

    return (np.mean(precisions) if precisions else 0.0,
            np.mean(recalls) if recalls else 0.0)


def _mean_iou(pred, gt, num_classes):
    p = pred.reshape(-1)
    g = gt.reshape(-1)
    ious = []
    for c in range(num_classes):
        pc = (p == c)
        gc = (g == c)
        inter = (pc & gc).sum().item()
        union = (pc | gc).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.nanmean(ious)) if ious else 0.0


# ------------------------------------------------------------------
# Build segmentation targets from bounding boxes
# ------------------------------------------------------------------

def _build_seg_target(batch, num_seg_classes, H, W, device):
    """Fill a [B, H, W] long tensor with class+1 inside each box."""
    B = batch["image"].shape[0]
    masks = torch.zeros((B, H, W), dtype=torch.long, device=device)
    targets = batch["targets"]

    boxes_all = targets["boxes"].to(device)
    labels_all = targets["labels"].to(device)

    for i in range(B):
        m = boxes_all[:, 0] == i
        if not m.any():
            continue
        bxs = boxes_all[m, 1:]  # cx, cy, w, h normalised
        lbs = labels_all[m]
        for j in range(len(bxs)):
            cx, cy, bw, bh = bxs[j]
            cls = lbs[j].long()
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            masks[i, y1:y2, x1:x2] = cls + 1  # 0 = background
    return masks


# ------------------------------------------------------------------
# Train / validate
# ------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    num_seg_classes):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        B, _, H, W = images.shape
        masks = _build_seg_target(batch, num_seg_classes, H, W, device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, num_seg_classes):
    model.eval()
    total_loss = total_iou = total_p = total_r = 0.0
    n = 0
    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        B, _, H, W = images.shape
        masks = _build_seg_target(batch, num_seg_classes, H, W, device)

        out = model(images)
        total_loss += criterion(out, masks).item()

        preds = out.argmax(dim=1)
        total_iou += _mean_iou(preds, masks, num_seg_classes)
        p, r = _masks_to_boxes_metrics(preds, masks, num_seg_classes)
        total_p += p
        total_r += r
        n += 1

    n = max(n, 1)
    return (total_loss / len(loader), total_iou / n,
            total_p / n, total_r / n)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["unetplusplus", "swinunet", "transunet"])
    p.add_argument("--data_dir", default="dataset_disorders")
    p.add_argument("--pfirrmann_json", default=None)
    p.add_argument("--meyerding_json", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224 if args.model in ("swinunet", "transunet") else 384

    train_ld, val_ld, test_ld, cnames = create_dataloaders(
        args.data_dir,
        pfirrmann_json=args.pfirrmann_json,
        meyerding_json=args.meyerding_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
    )

    num_seg = len(cnames) + 1  # +background
    model = get_baseline_model(args.model, num_classes=num_seg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(f"runs/baselines/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        tl = train_one_epoch(
            model, train_ld, criterion, optimizer, device, num_seg)
        vl, viou, vp, vr = validate(
            model, val_ld, criterion, device, num_seg)
        logger.info(
            f"E{epoch + 1}/{args.epochs}  "
            f"TL={tl:.4f}  VL={vl:.4f}  "
            f"mIoU={viou:.4f}  P={vp:.4f}  R={vr:.4f}")
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), save_dir / "best.pt")

    # Test with best model
    model.load_state_dict(torch.load(save_dir / "best.pt",
                                     weights_only=True))
    tl, tiou, tp, tr = validate(
        model, test_ld, criterion, device, num_seg)
    logger.info(
        f"TEST  Loss={tl:.4f}  mIoU={tiou:.4f}  P={tp:.4f}  R={tr:.4f}")


if __name__ == "__main__":
    main()
