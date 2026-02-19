"""
YOLOspine Inference Script
===========================

Run inference on individual images or a directory. Supports all model
variants and produces annotated output images with bounding boxes.

Usage::

    python tools/inference.py --checkpoint runs/best_mAP.pth \\
        --input path/to/image.png --output results/

    python tools/inference.py --checkpoint runs/best_mAP.pth \\
        --input path/to/images_dir/ --model v33
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yolospine.data.dataset import CLASS_NAMES
from yolospine.utils.decode import decode_predictions
from yolospine.utils.visualization import draw_predictions, plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description="YOLOspine Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True,
                   help="Image file or directory")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory")
    p.add_argument("--model", type=str, default="v2",
                   choices=["v1", "v2", "v33"])
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--nms_thresh", type=float, default=0.45)
    return p.parse_args()


def load_model(variant, ckpt_path, num_classes, image_size, device):
    if variant == "v1":
        from yolospine.models.yolospine import YOLOspine
        model = YOLOspine(num_classes=num_classes)
    elif variant == "v2":
        from yolospine.models.yolospine_v2 import build_model as _bv2
        model = _bv2(num_classes=num_classes, image_size=image_size)
    elif variant == "v33":
        from yolospine.models.yolospine_v33 import YOLOspineV33
        model = YOLOspineV33(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def preprocess(image_path, image_size):
    """Load and preprocess a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return img_resized, tensor.unsqueeze(0)


@torch.no_grad()
def infer(model, tensor, device, args):
    tensor = tensor.to(device)
    with autocast(enabled=True):
        out = model(tensor)
        s1 = out[0] if isinstance(out, tuple) else out

    dets = decode_predictions(
        s1,
        num_classes=args.num_classes,
        image_size=args.image_size,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
    )
    return dets[0]  # single image


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    model = load_model(
        args.model, args.checkpoint, args.num_classes,
        args.image_size, device)

    # Collect input images
    inp = Path(args.input)
    if inp.is_file():
        image_paths = [inp]
    elif inp.is_dir():
        image_paths = sorted(
            p for p in inp.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Running inference on {len(image_paths)} image(s)...")

    for img_path in image_paths:
        img_rgb, tensor = preprocess(img_path, args.image_size)
        boxes, scores, labels = infer(model, tensor, device, args)

        n = len(boxes)
        print(f"  {img_path.name}: {n} detection(s)")

        out_path = os.path.join(args.output, f"{img_path.stem}_pred.png")
        annotated = draw_predictions(
            img_rgb, boxes.cpu().numpy(), scores.cpu().numpy(),
            labels.cpu().numpy(), CLASS_NAMES,
            score_thresh=args.conf_thresh)
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
