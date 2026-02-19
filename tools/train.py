"""
YOLOspine Training Script
=========================

Unified training pipeline for all YOLOspine model variants (V1, V2, V3.3).

Supports:
    - Two-stage DDE + MLDR training (V2)
    - Single-stage DFL training (V1 / V3.3)
    - Mixed-precision training (AMP)
    - OneCycleLR scheduling with warmup
    - Gradient accumulation and clipping
    - Checkpointing (best mAP, best loss, periodic)

Usage::

    python tools/train.py --data_dir path/to/dataset \\
        --model v2 --epochs 150 --batch_size 8

    python tools/train.py --data_dir path/to/dataset \\
        --model v33 --epochs 150
"""

import argparse
import gc
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the repo root is on sys.path
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
from yolospine.utils.metrics import compute_metrics


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOspine")
    p.add_argument("--data_dir", type=str, default="dataset_disorders",
                   help="Root dataset directory (images/labels splits)")
    p.add_argument("--pfirrmann_json", type=str, default=None,
                   help="Path to Pfirrmann grading JSON")
    p.add_argument("--meyerding_json", type=str, default=None,
                   help="Path to Meyerding grading JSON")
    p.add_argument("--model", type=str, default="v2",
                   choices=["v1", "v2", "v33"],
                   help="Model variant")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--conf_thresh", type=float, default=0.001)
    p.add_argument("--nms_thresh", type=float, default=0.65)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Enable mixed-precision training")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint for resuming training")
    return p.parse_args()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------

def build_model(variant, num_classes, image_size, device):
    """Instantiate the requested model variant."""
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
        raise ValueError(f"Unknown model variant: {variant}")
    return model.to(device)


def build_loss(variant, num_classes, image_size):
    """Build the loss criterion for the given model variant."""
    if variant == "v2":
        from yolospine.losses.two_stage_loss import YOLOspineLoss
        return YOLOspineLoss(
            num_classes=num_classes,
            image_size=image_size,
            s1_cls_weight=1.0,
            s1_box_weight=5.0,
            s1_obj_weight=1.0,
            s1_obj_focal_alpha=0.25,
            s1_obj_focal_gamma=2.0,
            s2_weight=0.5,
            s2_warmup_epochs=10,
        )
    else:
        # V1 / V33 use v8DetectionLoss (requires model instance)
        return None  # constructed after model is built


# ---------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------

def train_one_epoch_v2(model, loader, optimizer, scheduler, loss_fn,
                       scaler, device, epoch, args):
    """Train one epoch for V2 (two-stage) model."""
    model.train()
    loss_fn.set_epoch(epoch)

    total_loss = total_s1 = total_s2 = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"E{epoch + 1}/{args.epochs}", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True)
                   for k, v in batch["targets"].items()}

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            s1_out, s2_out, proposals = model(images)
            loss, ld = loss_fn(s1_out, s2_out, proposals, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_s1 += ld["s1_total"]
        total_s2 += ld["s2_total"]
        n += 1

        pbar.set_postfix(
            L=f"{loss.item():.3f}",
            L1=f"{ld['s1_total']:.2f}",
            L2=f"{ld['s2_total']:.2f}",
            LR=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if batch_idx % 50 == 0:
            clear_memory()

    return {"loss": total_loss / n, "s1_loss": total_s1 / n,
            "s2_loss": total_s2 / n}


def train_one_epoch_single(model, loader, optimizer, scheduler, loss_fn,
                           scaler, device, epoch, args):
    """Train one epoch for single-stage (V1/V33) model."""
    model.train()

    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"E{epoch + 1}/{args.epochs}", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        targets_raw = batch["targets"]
        batch_data = {"targets": {k: v.to(device, non_blocking=True)
                                  for k, v in targets_raw.items()}}

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            preds = model(images)
            if isinstance(preds, tuple):
                preds = preds[0]  # stage1 outputs
            loss = loss_fn(preds, batch_data)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n += 1

        pbar.set_postfix(
            L=f"{loss.item():.3f}",
            LR=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if batch_idx % 50 == 0:
            clear_memory()

    return {"loss": total_loss / n}


@torch.no_grad()
def validate(model, loader, device, args):
    """Run validation and compute detection metrics."""
    model.eval()
    all_predictions = []
    all_targets_list = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True)
                   for k, v in batch["targets"].items()}

        with autocast(enabled=args.amp):
            out = model(images)
            if isinstance(out, tuple):
                stage1_out = out[0]
            else:
                stage1_out = out

        predictions = decode_predictions(
            stage1_out,
            num_classes=args.num_classes,
            image_size=args.image_size,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
        )
        all_predictions.extend(predictions)
        all_targets_list.append({
            "boxes": targets["boxes"],
            "labels": targets["labels"],
            "batch_size": images.shape[0],
        })

    # Re-index batch indices
    offset = 0
    adj_boxes, adj_labels = [], []
    for t in all_targets_list:
        boxes = t["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0] += offset
            adj_boxes.append(boxes)
            adj_labels.append(t["labels"])
        offset += t["batch_size"]

    if adj_boxes:
        combined = {
            "boxes": torch.cat(adj_boxes),
            "labels": torch.cat(adj_labels),
        }
    else:
        combined = {
            "boxes": torch.empty((0, 5), device=device),
            "labels": torch.empty(0, dtype=torch.long, device=device),
        }

    metrics = compute_metrics(
        all_predictions, combined,
        iou_threshold=args.iou_thresh,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    return metrics


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolospine_{args.model}_{timestamp}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"YOLOspine {args.model.upper()} Training: {run_name}")
    print(f"Device: {device}")
    print("=" * 70)

    # Model
    model = build_model(args.model, args.num_classes, args.image_size, device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} (Trainable: {train_p:,})")

    # Loss
    if args.model == "v2":
        loss_fn = build_loss("v2", args.num_classes, args.image_size)
    else:
        from yolospine.losses.detection_loss import v8DetectionLoss
        loss_fn = v8DetectionLoss(model, num_classes=args.num_classes)

    # Data
    train_ds = SpinalMRIDataset(
        os.path.join(args.data_dir, "images", "train"),
        os.path.join(args.data_dir, "labels", "train"),
        CLASS_NAMES,
        transform=get_transform(train=True, img_size=args.image_size),
        pfirrmann_json=args.pfirrmann_json,
        meyerding_json=args.meyerding_json,
        img_size=args.image_size,
    )
    val_ds = SpinalMRIDataset(
        os.path.join(args.data_dir, "images", "val"),
        os.path.join(args.data_dir, "labels", "val"),
        CLASS_NAMES,
        transform=get_transform(train=False, img_size=args.image_size),
        pfirrmann_json=args.pfirrmann_json,
        meyerding_json=args.meyerding_json,
        img_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1000.0)

    scaler = GradScaler(enabled=args.amp)

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_mAP = 0.0
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        if args.model == "v2":
            train_m = train_one_epoch_v2(
                model, train_loader, optimizer, scheduler, loss_fn,
                scaler, device, epoch, args)
        else:
            train_m = train_one_epoch_single(
                model, train_loader, optimizer, scheduler, loss_fn,
                scaler, device, epoch, args)

        val_m = validate(model, val_loader, device, args)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch + 1}: "
            f"Train={train_m['loss']:.4f} | "
            f"mAP={val_m['mAP']:.4f} | "
            f"P={val_m['precision']:.4f} | "
            f"R={val_m['recall']:.4f} | "
            f"LR={lr_now:.2e}")

        # Checkpointing
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args),
        }

        if val_m["mAP"] > best_mAP:
            best_mAP = val_m["mAP"]
            ckpt["mAP"] = best_mAP
            torch.save(ckpt, os.path.join(save_dir, "best_mAP.pth"))
            print(f"  -> New best mAP: {best_mAP:.4f}")

        if train_m["loss"] < best_loss:
            best_loss = train_m["loss"]
            ckpt["loss"] = best_loss
            torch.save(ckpt, os.path.join(save_dir, "best_loss.pth"))

        if (epoch + 1) % args.save_freq == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))

        clear_memory()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best mAP: {best_mAP:.4f}")
    print(f"Checkpoints: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
