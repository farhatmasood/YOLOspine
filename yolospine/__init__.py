"""
YOLOspine: Two-Stage YOLO-Based Spinal Disorder Detection
==========================================================

A multi-scale, two-stage object detection framework for automated
spinal disorder identification in lumbar MRI. Integrates Pfirrmann
disc degeneration grading and Meyerding spondylolisthesis classification.

Modules:
    models   - Network architectures (YOLOspine V1, V2, V3.3)
    data     - Dataset loaders and augmentation pipelines
    losses   - Detection and refinement loss functions
    utils    - Metrics, decoding, and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "YOLOspine Research Team"
