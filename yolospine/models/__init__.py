"""
YOLOspine Model Architectures
==============================

Three model variants for spinal disorder detection:

- YOLOspine (V1): Single-stage YOLOv8-style detector with AreaAttention
  and Distribution Focal Loss head. Uses C2f backbone, PANet neck.

- YOLOspineV2: Two-stage detector (DDE + MLDR) with RoI-based refinement.
  Stage 1 produces multi-scale detections; Stage 2 refines via RoI pooling.

- YOLOspineV33: DenseC2f backbone variant with improved normalization
  and area attention at FPN boundaries.
"""

from .yolospine import YOLOspine
from .yolospine_v2 import YOLOspineV2, build_model
from .yolospine_v33 import YOLOspineV33

__all__ = [
    "YOLOspine",
    "YOLOspineV2",
    "YOLOspineV33",
    "build_model",
]
