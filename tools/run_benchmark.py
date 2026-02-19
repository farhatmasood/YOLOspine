"""
Run Ultralytics YOLO / RT-DETR benchmark across multiple architectures.

Trains and evaluates each model variant on the spinal disorder dataset
using identical hyper-parameters for fair comparison.

Usage::

    python tools/run_benchmark.py --data configs/data_disorders.yaml

    # Specific models only
    python tools/run_benchmark.py --data configs/data_disorders.yaml \\
        --models yolov8m yolov9m yolo11m
"""

import argparse
from pathlib import Path

from ultralytics import YOLO, RTDETR

# Models and their Ultralytics weight keys
MODELS = {
    "yolov8m": "yolov8m.pt",
    "yolov9m": "yolov9m.pt",
    "yolov10m": "yolov10m.pt",
    "yolo11m": "yolo11m.pt",
    "yolo12m": "yolo12m.pt",
    "yolo26m": "yolo26m.pt",
    "yolov8m-seg": "yolov8m-seg.pt",
    "yolo11m-seg": "yolo11m-seg.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}


def train_model(name: str, weights: str, args):
    print(f"\n{'=' * 70}")
    print(f"  Training: {name}")
    print(f"{'=' * 70}")

    if "rtdetr" in name:
        model = RTDETR(weights)
    else:
        model = YOLO(weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=name,
        exist_ok=True,
        patience=args.patience,
        device=args.device,
    )

    metrics = model.val(data=args.data, split="test")
    print(f"  {name} — mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="YOLO / RT-DETR benchmark")
    p.add_argument("--data", required=True, help="YAML data config")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of models to train (default: all)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/YOLOspine-Benchmark")
    args = p.parse_args()

    selected = args.models or list(MODELS.keys())
    for name in selected:
        if name not in MODELS:
            print(f"Unknown model: {name} — skipping")
            continue
        train_model(name, MODELS[name], args)

    print("\nBenchmark complete. Results in:", args.project)


if __name__ == "__main__":
    main()
