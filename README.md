# YOLOspine: Two-Stage Spine Segmentation with Pfirrmann Grade Prediction

## Overview

YOLOspine is a two-stage deep learning architecture for automated spinal MRI analysis that combines:
1. **Spine structure detection and segmentation** (IVD, vertebral endplates, vertebral bodies)
2. **Pfirrmann grade classification** for intervertebral disc degeneration assessment

## Architecture

### Stage 1: Coarse Detection
- **Task**: Binary classification (IVD vs Posterior Elements) + bounding box regression
- **Features**: Multi-scale feature pyramids (P3, P4, P5)
- **Strides**: 8, 16, 32 pixels

### Stage 2: Fine-Grained Classification
- **Task**: 4-class vertebral body classification + refined regression + Pfirrmann grading
- **RoI Pooling**: 7×7 RoI Align for feature extraction
- **Grading**: 5-class Pfirrmann classification (Grades I-V)

### Anatomical Grade Mapping
The system uses **vertical position sorting** to map Pfirrmann grades to detected discs:
- Boxes sorted by Y-coordinate (top-to-bottom in image)
- Lower Y = Superior vertebrae (L1-L3)
- Higher Y = Inferior vertebrae (L3-L5)
- Mapping: D3 (superior), D4 (middle), D5 (inferior)

## Project Structure

```
YOLOspine-2May25/
├── Model/
│   ├── __init__.py          # Package initialization
│   ├── architecture.py      # YOLOspine model definition
│   ├── dataset.py           # Data loading and augmentation
│   └── loss.py              # Loss functions and utilities
├── train.py                 # Main training script
├── PfirrmannGrade.csv       # Ground truth Pfirrmann grades
├── data.yaml                # Dataset configuration
├── train/                   # Training images and labels
│   ├── images/
│   └── labels/
├── val/                     # Validation set
│   ├── images/
│   └── labels/
├── test/                    # Test set
│   ├── images/
│   └── labels/
└── checkpoints/             # Model checkpoints
```

## Installation

```bash
pip install torch torchvision
pip install opencv-python pandas
pip install albumentations pyyaml
```

## Usage

### Training

```bash
python train.py --batch_size 8 --epochs 100 --lr 0.001
```

### Training with Custom Parameters

```bash
python train.py \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.0005 \
    --data_dir "D:\path\to\your\data"
```

## Configuration

Key parameters in `train.py`:

```python
class Config:
    IMG_SIZE = (384, 384)        # Input image size
    NUM_CLASSES = 6              # Spine structure classes
    NUM_GRADES = 5               # Pfirrmann grades (I-V)
    BATCH_SIZE = 8               # Training batch size
    NUM_EPOCHS = 100             # Training epochs
    LEARNING_RATE = 1e-3         # Initial learning rate
    LAMBDA_REG = 1.0             # Regression loss weight
    LAMBDA_GRADE = 0.5           # Grade loss weight
    IOU_THRESHOLD = 0.2          # Box matching threshold
```

## Dataset Format

### Images
- Format: PNG or JPG
- Size: 384×384 pixels (resized automatically)
- Location: `{train|val|test}/images/`

### Labels
- Format: YOLO format (normalized coordinates)
- Structure: `class_id cx cy w h` per line
- Location: `{train|val|test}/labels/`

### Pfirrmann Grades
CSV format with columns:
- `Patient_ID`: Numeric patient identifier
- `D3`: Grade for L3-L4 disc (1-5)
- `D4`: Grade for L4-L5 disc (1-5)
- `D5`: Grade for L5-S1 disc (1-5)

## Class Definitions

| Class ID | Structure | Description |
|----------|-----------|-------------|
| 0 | IVD | Intervertebral Disc |
| 1 | PE | Posterior Elements |
| 2 | VB_Type1 | Vertebral Body Type 1 |
| 3 | VB_Type2 | Vertebral Body Type 2 |
| 4 | VB_Type3 | Vertebral Body Type 3 |
| 5 | VB_Type4 | Vertebral Body Type 4 |

## Model Outputs

### Stage 1
- Classification scores: [B, 2, H, W] (IVD vs PE probabilities)
- Regression outputs: [B, 5, H, W] (x, y, w, h, objectness)

### Stage 2
- Classification scores: [N, 4] (VB type probabilities)
- Regression outputs: [N, 5] (refined box coordinates + objectness)
- Grade predictions: [N, 5] (Pfirrmann grade logits I-V)

## Loss Functions

1. **Stage 1 Loss**: Binary cross-entropy (classification) + Smooth L1 (regression)
2. **Stage 2 Loss**: Binary cross-entropy (classification) + Smooth L1 (regression)
3. **Grade Loss**: Cross-entropy (5-class Pfirrmann grading)

Total Loss: `L = L_stage1 + L_stage2 + λ_grade * L_grade`

## Academic Foundation

This implementation is based on the following principles:

1. **Two-Stage Detection**: Follows the cascade architecture principle where coarse detection precedes fine-grained classification (R-CNN family)

2. **Feature Pyramid Networks**: Multi-scale feature extraction for handling anatomical structures of varying sizes

3. **RoI Align**: Precise spatial feature extraction avoiding quantization errors of RoI Pooling

4. **Anatomical Prior**: Leverages spatial relationships (superior-inferior ordering) for grade mapping, consistent with clinical assessment methodology

5. **Multi-Task Learning**: Joint optimization of detection, segmentation, and clinical grading tasks

## Checkpoints

The system saves three types of checkpoints:

1. **latest.pth**: Most recent epoch (for resuming training)
2. **best.pth**: Best validation loss (for inference)
3. **checkpoint_epoch_N.pth**: Periodic snapshots (every 5 epochs)

## Performance Monitoring

Training logs include:
- Stage 1 loss (detection + coarse classification)
- Stage 2 loss (fine classification + refined regression)
- Grade loss (Pfirrmann classification)
- Validation metrics per epoch
- Learning rate schedule updates

## Citation

If you use this code, please cite:

```
YOLOspine: Two-Stage Deep Learning for Automated Spinal Segmentation 
and Pfirrmann Grade Classification
[Your Publication Details]
```

## License

Academic and research use only.

## Contact

For questions or issues, please open an issue in the repository.
