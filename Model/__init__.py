"""
YOLOspine Model Package

Two-stage YOLO-based architecture for spine segmentation with Pfirrmann grade prediction.

Academic References:
- YOLO architecture: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- Two-stage detection: Girshick, "Fast R-CNN"
- RoI Align: He et al., "Mask R-CNN"
- Pfirrmann grading: Pfirrmann et al., "Magnetic Resonance Classification of Lumbar IVD Degeneration"
"""

from .architecture import YOLOspine, C3Block, RELANBlock, AreaAttention
from .dataset import SpinalMRIDataset, create_dataloaders, load_pfirrmann_grades
from .loss import (
    assign_targets,
    compute_stage1_loss,
    compute_stage2_loss,
    compute_grade_loss,
    save_checkpoint,
    load_checkpoint
)

__version__ = '1.0.0'
__author__ = 'YOLOspine Research Team'

__all__ = [
    'YOLOspine',
    'C3Block',
    'RELANBlock',
    'AreaAttention',
    'SpinalMRIDataset',
    'create_dataloaders',
    'load_pfirrmann_grades',
    'assign_targets',
    'compute_stage1_loss',
    'compute_stage2_loss',
    'compute_grade_loss',
    'save_checkpoint',
    'load_checkpoint'
]
