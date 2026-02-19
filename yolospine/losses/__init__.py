"""
Loss functions for YOLOspine detector variants.

- ``detection_loss``: Single-stage (V1) DFL-based loss with TaskAlignedAssigner.
- ``two_stage_loss``: Two-stage (V2) DDE + MLDR loss with CIoU and refinement.
"""

from .detection_loss import (
    bbox_iou,
    bbox2dist,
    dist2bbox,
    make_anchors,
    TaskAlignedAssigner,
    BboxLoss,
    v8DetectionLoss,
)
from .two_stage_loss import (
    FocalLoss,
    QualityFocalLoss,
    assign_targets_to_grid,
    compute_stage1_loss,
    compute_stage2_loss,
    YOLOspineLoss,
)

__all__ = [
    "bbox_iou",
    "bbox2dist",
    "dist2bbox",
    "make_anchors",
    "TaskAlignedAssigner",
    "BboxLoss",
    "v8DetectionLoss",
    "FocalLoss",
    "QualityFocalLoss",
    "assign_targets_to_grid",
    "compute_stage1_loss",
    "compute_stage2_loss",
    "YOLOspineLoss",
]
