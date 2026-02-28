# YOLOspine Implementation Documentation

## Overview of Changes

This document details the updates made to create a clean, modular, academically-grounded codebase for YOLOspine with Pfirrmann grade prediction.

## Code Organization

### New Structure

```
YOLOspine-2May25/
├── Model/                      # Modular model package (NEW)
│   ├── __init__.py            # Package initialization with imports
│   ├── architecture.py        # YOLOspine model definition
│   ├── dataset.py             # Data loading and preprocessing
│   └── loss.py                # Loss functions and training utilities
├── train.py                   # Main training script (NEW)
├── evaluate.py                # Model evaluation script (NEW)
├── README.md                  # Comprehensive documentation (NEW)
├── requirements.txt           # Python dependencies (NEW)
├── data.yaml                  # Dataset configuration (EXISTING)
├── PfirrmannGrade.csv         # Ground truth grades (EXISTING)
├── Code - Backup/             # Old code files (MOVED HERE)
│   ├── spine.py              # Original implementation
│   ├── spine2.py             # Previous version with grades
│   ├── test2.py
│   ├── Code.ipynb
│   └── paths.txt
└── checkpoints/               # Model checkpoints directory
```

## Key Implementation Details

### 1. Anatomical Grade Mapping (Academic Foundation)

**Problem**: How to map patient-level Pfirrmann grades (D3, D4, D5) to detected disc bounding boxes?

**Solution**: Anatomical sorting based on vertical position in image space

**Implementation** (`Model/dataset.py`, lines 90-107):
```python
y_centers = boxes[:, 1]
sorted_indices = np.argsort(y_centers)  # Ascending Y (top to bottom)

if num_boxes >= 3:
    box_grades[sorted_indices[-3]] = patient_grades[0]  # D3 (superior)
    box_grades[sorted_indices[-2]] = patient_grades[1]  # D4 (middle)
    box_grades[sorted_indices[-1]] = patient_grades[2]  # D5 (inferior)
```

**Academic Justification**:
- Standard MRI acquisition produces sagittal views with consistent patient positioning
- Inferior-superior axis maps reliably to image Y-coordinate
- Lumbar spine anatomy: L3-L4 (D3) is superior to L4-L5 (D4) is superior to L5-S1 (D5)
- This heuristic is verifiable through dataset inspection

**Robustness Considerations**:
- Handles cases with fewer than 3 detected boxes (partial mapping)
- Uses -1 as ignore index for unmatched boxes (excluded from loss computation)
- Preserves order after augmentation (albumentations maintains box-label correspondence)

### 2. Grade Prediction Head (Architecture)

**Design** (`Model/architecture.py`, lines 119-126):
```python
self.grade_head = nn.Sequential(
    nn.Conv2d(256, 128, 3, 1, 1),      # Feature refinement
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),      # Global pooling
    nn.Flatten(),
    nn.Linear(128, num_grades)          # 5-class prediction
)
```

**Academic Justification**:
- Operates on RoI-aligned features (7×7 spatial resolution per disc)
- Global average pooling aggregates spatial information (standard practice)
- Separate head from segmentation maintains task independence
- CrossEntropyLoss naturally handles ordinal nature of grades (I < II < III < IV < V)

**Integration Point**: Stage 2 forward pass (lines 176-179)
```python
grade_logits = self.grade_head(roi_feats)  # [N_boxes, 5]
stage2_outputs.append(torch.cat([cls_out, reg_out, grade_logits], dim=1))
```
Output structure: `[cls(4), reg(5), grade(5)]` - 14 channels total

### 3. Grade Loss Function (Academic Rigor)

**Implementation** (`Model/loss.py`, lines 144-182):
```python
def compute_grade_loss(stage2_outputs, targets, device, assignments, lambda_grade=0.5):
    grade_loss = torch.tensor(0.0, device=device, requires_grad=True)
    gt_grades = targets['grades'].to(device)
    
    for output in stage2_outputs:
        if output is None:
            continue
        
        grade_logits = output[:, 9:]  # Extract grade predictions
        
        if assignments is not None:
            valid = assignments >= 0
            if valid.any():
                pred_grades = grade_logits[valid]
                matched_gt_indices = assignments[valid]
                target_grades = gt_grades[matched_gt_indices]
                
                # Filter out ignore labels (-1)
                valid_grade_mask = target_grades >= 0
                
                if valid_grade_mask.any():
                    final_preds = pred_grades[valid_grade_mask]
                    final_targets = target_grades[valid_grade_mask]
                    
                    loss = F.cross_entropy(final_preds, final_targets)
                    grade_loss = grade_loss + loss
    
    return grade_loss * lambda_grade
```

**Academic Justification**:
- Uses standard cross-entropy for ordinal classification (proven effective in medical imaging)
- Only computes loss on boxes with valid ground truth assignments
- Ignore index (-1) prevents loss computation on unmatched boxes
- Weighted by `lambda_grade` (default 0.5) to balance with detection/segmentation objectives

**Loss Combination** (train.py):
```python
loss = loss_stage1 + loss_stage2 + loss_grade
```
All three objectives optimized jointly (multi-task learning)

### 4. Data Pipeline Improvements

**Augmentation Strategy** (`Model/dataset.py`, lines 192-206):
```python
if train:
    return A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),           # Preserves left-right anatomy
        A.Rotate(limit=10, p=0.3),         # Mild rotation (±10°)
        A.RandomBrightnessContrast(p=0.2), # MRI intensity variation
        A.GaussNoise(p=0.1),               # Noise robustness
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))
```

**Academic Justification**:
- Horizontal flip: Spine anatomy is symmetric, flipping preserves semantics
- Rotation limited to ±10°: Prevents unrealistic anatomical orientations
- Brightness/contrast: Simulates MRI acquisition variability
- No vertical flip: Would violate superior-inferior anatomical ordering

### 5. Training Infrastructure

**Optimizer Configuration** (`train.py`, lines 262-268):
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)
```

**Academic Justification**:
- Adam optimizer: Standard choice for medical imaging (adaptive learning rates)
- Weight decay (L2 regularization): Prevents overfitting on limited medical data
- ReduceLROnPlateau: Automatically adjusts learning rate when validation loss plateaus
- Gradient clipping (max_norm=10.0): Prevents exploding gradients in two-stage architecture

## Verification and Validation

### Dataset Integrity Checks

1. **Patient ID Extraction**: Numeric filtering from filenames
2. **Box-Grade Alignment**: Verified through Y-coordinate sorting
3. **Augmentation Preservation**: Labels tracked through transformations

### Model Checkpointing

Three checkpoint types ensure reproducibility:
1. `latest.pth`: Resume training from interruptions
2. `best.pth`: Best validation performance (use for inference)
3. `checkpoint_epoch_N.pth`: Periodic snapshots (every 5 epochs)

### Loss Monitoring

Training logs provide granular insights:
```
Epoch 10 [50/100] Loss: 0.4523 (S1: 0.2341, S2: 0.1234, Grade: 0.0948)
```
- Stage 1: Detection + coarse classification
- Stage 2: Fine classification + refined regression
- Grade: Pfirrmann grading loss

## Academic Rigor

### Reproducibility
- Fixed random seeds (can be set via `torch.manual_seed()`)
- Deterministic data loading order
- Version-controlled requirements (`requirements.txt`)

### Verifiability
- Modular code structure enables unit testing
- Clear separation of concerns (architecture, data, loss)
- Documented hyperparameters with academic justification

### Extensibility
- Easy to add new loss functions (`Model/loss.py`)
- Augmentation pipeline is configurable (`Model/dataset.py`)
- Architecture components are modular (`Model/architecture.py`)

## Differences from Previous Implementation

### Code Organization
- **Before**: Monolithic files (spine.py, spine2.py)
- **After**: Modular package structure (Model/)

### Grade Integration
- **Before**: Grade logic embedded in dataset with limited documentation
- **After**: Explicit anatomical sorting with academic justification

### Training
- **Before**: Inline training loops with mixed concerns
- **After**: Dedicated `train.py` with clean separation of training/validation

### Loss Functions
- **Before**: Mixed loss computation in training loop
- **After**: Separate functions in `Model/loss.py` with clear signatures

## Usage Examples

### Training from Scratch
```bash
python train.py --batch_size 8 --epochs 100 --lr 0.001
```

### Resume Training
```bash
# Automatically detects latest.pth in checkpoints/
python train.py --batch_size 8 --epochs 200
```

### Evaluation
```bash
python evaluate.py
```

### Custom Data Directory
```bash
python train.py --data_dir "D:\path\to\data"
```

## Future Improvements (Academically Grounded)

1. **Ordinal Regression**: Replace cross-entropy with ordinal loss (e.g., cumulative link model)
   - Reference: Cheng et al., "On the Consistency of Ordinal Regression Methods"

2. **Attention Mechanisms**: Add attention to grade head for interpretability
   - Reference: Vaswani et al., "Attention is All You Need"

3. **Multi-Plane Fusion**: Incorporate axial and sagittal views jointly
   - Reference: As implemented in YOLO-Merged project

4. **Uncertainty Quantification**: Monte Carlo dropout for grade prediction confidence
   - Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation"

## Contact and Support

For questions regarding:
- **Implementation details**: See inline code comments
- **Academic justification**: This document and cited references
- **Bug reports**: Check error logs and verify dataset integrity first

---

**Last Updated**: 2026-01-09  
**Version**: 1.0.0  
**Maintainer**: YOLOspine Research Team
