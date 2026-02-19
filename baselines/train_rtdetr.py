"""
RT-DETR fine-tuning via Ultralytics.

Usage::

    python baselines/train_rtdetr.py --data configs/data_disorders.yaml \\
        --epochs 50 --model rtdetr-l.pt

    python baselines/train_rtdetr.py --model rtdetr-x.pt --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR


def main():
    p = argparse.ArgumentParser(description="RT-DETR fine-tuning")
    p.add_argument("--model", default="rtdetr-l.pt",
                   help="Pretrained weight file (rtdetr-l.pt / rtdetr-x.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--data", default="configs/data_disorders.yaml",
                   help="YAML data config for Ultralytics")
    p.add_argument("--project", default="runs/YOLOspine-Benchmark")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    name = args.name or args.model.replace(".pt", "")
    print(f"Training {args.model} on {args.data}")

    model = RTDETR(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=name,
        exist_ok=True,
        patience=10,
    )
    print(f"Training complete. Results: {args.project}/{name}")


if __name__ == "__main__":
    main()
