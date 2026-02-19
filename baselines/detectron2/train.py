"""
Detectron2 Cascade R-CNN training for lumbar spine disorder detection.

Cascade R-CNN is chosen for its progressive refinement of overlapping boxes,
which is critical since 100% of spinal MRI images contain multiple
co-occurring pathologies at the same anatomical location.

Usage::

    python baselines/detectron2/train.py \\
        --dataset-dir /path/to/dataset_disorders \\
        --output-dir runs/detectron2

Classes (6): DDD, LDB, Normal_IVD, SS, TDB, Spondylolisthesis
"""

import argparse
import json
import os
import logging
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger("detectron2")

CLASS_NAMES = [
    "Degenerative Disc Disease",
    "LDB",
    "Normal IVD",
    "Spinal Stenosis",
    "TDB",
    "Spondylolisthesis",
]


def register_spine_datasets(dataset_dir: Path):
    """Register train / val / test splits with Detectron2."""
    for split in ["train", "val", "test"]:
        name = f"spine_disorders_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        ann_file = dataset_dir / "annotations" / f"instances_{split}.json"
        img_dir = dataset_dir / "images" / split

        register_coco_instances(name, {}, str(ann_file), str(img_dir))
        MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES)
        logger.info("Registered %s  ann=%s  img=%s", name, ann_file, img_dir)


class SpineDisorderTrainer(DefaultTrainer):
    """Trainer with periodic COCO evaluation."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def build_config(args) -> object:
    """Cascade Mask R-CNN R50-FPN config tuned for overlapping pathologies."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    )
    # Disable mask head — dataset has bbox annotations only
    cfg.MODEL.MASK_ON = False

    cfg.DATASETS.TRAIN = ("spine_disorders_train",)
    cfg.DATASETS.TEST = ("spine_disorders_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Weights — either local file or Detectron2 model-zoo URL
    if args.weights and Path(args.weights).exists():
        cfg.MODEL.WEIGHTS = str(args.weights)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    # Higher NMS threshold to preserve overlapping detections
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    # Spine anatomy: tall narrow structures
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0]]

    # Multi-scale training
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 544
    cfg.INPUT.MAX_SIZE_TEST = 800

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.6), int(args.max_iter * 0.8))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.EVAL_PERIOD = 500

    cfg.OUTPUT_DIR = str(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def train(args):
    """Run training + final evaluation."""
    register_spine_datasets(Path(args.dataset_dir))
    cfg = build_config(args)

    print("=" * 70)
    print("DETECTRON2  Cascade R-CNN  —  Spine Disorder Detection")
    print("=" * 70)
    print(f"  ROI NMS threshold : {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}")
    print(f"  Score threshold   : {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    print(f"  Batch size        : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate     : {cfg.SOLVER.BASE_LR}")
    print(f"  Max iterations    : {cfg.SOLVER.MAX_ITER}")
    print(f"  Output            : {cfg.OUTPUT_DIR}")

    # Save config
    config_path = Path(cfg.OUTPUT_DIR) / "cascade_rcnn_spine_config.yaml"
    with open(config_path, "w") as f:
        f.write(cfg.dump())

    # Train
    trainer = SpineDisorderTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Final evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluator = COCOEvaluator(
        "spine_disorders_val",
        output_dir=str(Path(cfg.OUTPUT_DIR) / "results"),
    )
    val_loader = build_detection_test_loader(cfg, "spine_disorders_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)

    results_file = Path(cfg.OUTPUT_DIR) / "results" / "val_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete.  Results:", results_file)
    return results


def main():
    p = argparse.ArgumentParser(description="Train Cascade R-CNN baseline")
    p.add_argument("--dataset-dir", required=True,
                   help="Root of COCO-formatted dataset (images/ + annotations/)")
    p.add_argument("--output-dir", default="runs/detectron2")
    p.add_argument("--weights", default=None,
                   help="Local .pkl weights or omit to use model-zoo")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-iter", type=int, default=10000)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
