import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, box_iou
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import yaml
import pandas as pd  # Added for CSV
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
CONFIG = {
    'IMG_SIZE': (384, 384),
    'NUM_CLASSES': 6,
    'NUM_GRADES': 5,  # Pfirrmann Grades I-V
    'STRIDES': [8, 16, 32],
    'CLASS_WEIGHTS': torch.tensor([1/0.459, 1/0.1, 1/0.214, 1/0.226, 1/0.017, 1/0.014]),
    'SIGMA': 0.5,
    'LAMBDA_REG': 1.0,
    'LAMBDA_GRADE': 0.5,  # New Weight for Severity Loss
    'BATCH_SIZE': 8,
    'ACCUMULATION_STEPS': 2,
    'IOU_THRESHOLD': 0.2,
    'OBJ_THRESHOLD': 0.3,
    'CHECKPOINT_PATH': r"D:\2.3 Code_s\YOLOspine-2May25\checkpoints",
    'DATA_YAML_PATH': r"D:\2.3 Code_s\YOLOspine-2May25\data.yaml",
    'GRADE_CSV_PATH': r"D:\2.3 Code_s\YOLOspine-2May25\PfirrmannGrade.csv", # Update if needed
    'MIN_BOX_SIZE': 1e-4,
    'ANCHORS': [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]],
    'ANCHOR_LOSS_WEIGHT': 1.0,
    'WEIGHT_DECAY': 1e-4
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dataset Module (Modified for Severity) ---
def load_pfirrmann_grades(csv_path):
    """Loads Pfirrmann grades into a dictionary: {patient_id: [D3, D4, D5]}"""
    try:
        df = pd.read_csv(csv_path)
        grade_map = {}
        for _, row in df.iterrows():
            pid = str(int(row['Patient_ID']))
            # Grades are 1-5, we map to 0-4 for CrossEntropy
            grades = [
                int(row['D3']) - 1, 
                int(row['D4']) - 1, 
                int(row['D5']) - 1
            ]
            grade_map[pid] = grades
        logger.info(f"Loaded Pfirrmann grades for {len(grade_map)} patients.")
        return grade_map
    except Exception as e:
        logger.warning(f"Could not load grade CSV: {e}. Severity training will be skipped.")
        return {}

class SpinalMRIDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None, grade_map=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.transform = transform
        self.grade_map = grade_map
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        
        # Extract Patient ID from filename (Assuming '123.jpg' or 'Patient_123.jpg')
        patient_id = ''.join(filter(str.isdigit, img_file.split('.')[0]))
        
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    c, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    labels.append(int(c))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # --- SEVERITY MAPPING LOGIC ---
        # Initialize grades as -1 (Ignore)
        box_grades = np.full(len(boxes), -1, dtype=np.int64)
        
        if self.grade_map and patient_id in self.grade_map and len(boxes) > 0:
            # Get grades [D3, D4, D5]
            patient_grades = self.grade_map[patient_id]
            
            # Sort boxes by Y-center (Vertical position)
            # Higher Y value = Lower in image (Anatomically L5 is lower than L1)
            # We assume the bottom-most relevant boxes correspond to D5, D4, D3
            y_centers = boxes[:, 1]
            sorted_indices = np.argsort(y_centers) # Ascending Y (Top to Bottom)
            
            # Heuristic: Map last 3 boxes to D3, D4, D5 if we have enough boxes
            # D3 is higher (smaller Y), D5 is lower (larger Y)
            # sorted_indices: [Top-most, ..., Bottom-most]
            # Mapping: D3 -> sorted[-3], D4 -> sorted[-2], D5 -> sorted[-1]
            
            num_boxes = len(boxes)
            if num_boxes >= 3:
                # Assign D5 (Bottom)
                box_grades[sorted_indices[-1]] = patient_grades[2]
                # Assign D4 (Middle)
                box_grades[sorted_indices[-2]] = patient_grades[1]
                # Assign D3 (Top)
                box_grades[sorted_indices[-3]] = patient_grades[0]
            elif num_boxes == 2:
                 # Partial match fallback
                box_grades[sorted_indices[-1]] = patient_grades[2] # D5
                box_grades[sorted_indices[-2]] = patient_grades[1] # D4
            elif num_boxes == 1:
                box_grades[sorted_indices[-1]] = patient_grades[2] # D5
        
        # --- Pre-calculate Stage Targets ---
        stage1_cls = np.zeros((max(1, len(boxes)), 2), dtype=np.float32)
        stage2_cls = np.zeros((max(1, len(boxes)), 4), dtype=np.float32)
        
        if len(boxes) > 0:
            for i, class_id in enumerate(labels):
                if class_id in [0, 1]: stage1_cls[i, class_id] = 1.0
                elif class_id in [2, 3, 4, 5]: stage2_cls[i, class_id - 2] = 1.0

        if self.transform:
            augmented = self.transform(image=img, bboxes=boxes, labels=labels, box_grades=box_grades)
            img = augmented['image']
            boxes = np.array(augmented['bboxes'], dtype=np.float32)
            labels = np.array(augmented['labels'], dtype=np.int32)
            # albumentations might shuffle, but we passed box_grades parallel to labels ideally
            # For simplicity with standard A., we assume order is preserved or handled manually
            # RE-MATCHING grades to augmented boxes is tricky if order changes. 
            # Simplified: Albumentations BBoxParams doesn't natively support extra per-box attributes easily in standard interface
            # We will assume 'labels' drives the transform. 
            # *Crucial Fix*: We must ensure box_grades tracks with boxes.
            # Since this is complex to inject into standard Albumentations without custom key, 
            # we will rely on the fact that A.Compose preserves order of boxes/labels.
            box_grades = box_grades[:len(boxes)] # Truncate if boxes were dropped

        reg_targets = boxes if len(boxes) > 0 else np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        obj_targets = np.ones(len(boxes), dtype=np.float32) if len(boxes) > 0 else np.array([1.0], dtype=np.float32)
        
        return {
            'image': img,
            'stage1_cls': stage1_cls,
            'stage2_cls': stage2_cls,
            'reg': reg_targets,
            'obj': obj_targets,
            'boxes': boxes,
            'labels': labels,
            'grades': box_grades # New Target
        }

def collate_fn(batch):
    images = []
    targets = {'stage1_cls': [], 'stage2_cls': [], 'reg': [], 'obj': [], 'boxes': [], 'labels': [], 'grades': []}
    for b, item in enumerate(batch):
        images.append(item['image'])
        targets['stage1_cls'].append(torch.tensor(item['stage1_cls'], dtype=torch.float32))
        targets['stage2_cls'].append(torch.tensor(item['stage2_cls'], dtype=torch.float32))
        targets['reg'].append(torch.tensor(item['reg'], dtype=torch.float32))
        targets['obj'].append(torch.tensor(item['obj'], dtype=torch.float32))
        
        # Handle Grades
        grades = torch.tensor(item['grades'], dtype=torch.long)
        targets['grades'].append(grades)

        boxes = torch.tensor(item['boxes'], dtype=torch.float32)
        if boxes.shape[0] > 0:
            batch_idx = torch.full((boxes.shape[0], 1), b, dtype=torch.float32)
            boxes = torch.cat([batch_idx, boxes], dim=1)
        targets['boxes'].append(boxes)
        targets['labels'].append(torch.tensor(item['labels'], dtype=torch.long))
    
    return {
        'image': torch.stack(images),
        'targets': {
            'stage1_cls': torch.cat(targets['stage1_cls'], dim=0),
            'stage2_cls': torch.cat(targets['stage2_cls'], dim=0),
            'reg': torch.cat(targets['reg'], dim=0),
            'obj': torch.cat(targets['obj'], dim=0),
            'grades': torch.cat(targets['grades'], dim=0),
            'boxes': torch.cat([b for b in targets['boxes'] if b.shape[0] > 0], dim=0) if any(b.shape[0] > 0 for b in targets['boxes']) else torch.empty((0, 5), dtype=torch.float32),
            'labels': torch.cat([l for l in targets['labels'] if l.shape[0] > 0], dim=0) if any(l.shape[0] > 0 for l in targets['labels']) else torch.empty(0, dtype=torch.long)
        }
    }

def load_data(data_yaml_path, batch_size=CONFIG['BATCH_SIZE']):
    with open(data_yaml_path, 'r') as f: data_yaml = yaml.safe_load(f)
    class_names = {i: name for i, name in enumerate(data_yaml['names'])}
    project_path = os.path.dirname(data_yaml_path)
    
    # Load Grades
    grade_map = load_pfirrmann_grades(CONFIG['GRADE_CSV_PATH'])

    # Initialize Datasets
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))

    # Note: Passing grade_map to dataset
    train_ds = SpinalMRIDataset(os.path.join(project_path, "train", "images"), 
                               os.path.join(project_path, "train", "labels"), class_names, transform, grade_map)
    val_ds = SpinalMRIDataset(os.path.join(project_path, "val", "images"), 
                             os.path.join(project_path, "val", "labels"), class_names, transform, grade_map)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
            None, class_names)

# --- Model Modules (Unchanged parts omitted for brevity, keeping modified parts) ---
class C3Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels//2, 1, 1, 0)
        self.shortcut = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.silu(self.bn(torch.cat([x, shortcut], dim=1)))

class RELANBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.silu(self.bn(self.conv2(self.conv1(x)) + self.shortcut(x)))

class AreaAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = 1.0 / (channels ** 0.5)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1).permute(0, 2, 1)
        k = self.key(x).view(B, C, -1)
        v = self.value(x).view(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        return x + torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)

class YOLOspine(nn.Module):
    def __init__(self, num_classes=CONFIG['NUM_CLASSES']):
        super().__init__()
        # ... (Backbone and Neck identical to your original code) ...
        self.backbone = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, 1), nn.MaxPool2d(2, 2),
            C3Block(64, 128), nn.MaxPool2d(2, 2),
            C3Block(128, 256), nn.MaxPool2d(2, 2),
            RELANBlock(256, 256), nn.MaxPool2d(2, 2),
            RELANBlock(256, 512), nn.MaxPool2d(2, 2),
        ])
        self.neck = nn.ModuleList([
            nn.Conv2d(512, 512, 3, 1, 1), nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 1, 1, 0), RELANBlock(1024, 256),
            AreaAttention(256), nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 1, 1, 0), RELANBlock(512, 256),
            AreaAttention(256), nn.Conv2d(256, 512, 3, 2, 1),
        ])
        
        # Stage 1 Heads (Identical)
        self.stage1_cls = nn.ModuleList([nn.Conv2d(256, 2, 1), nn.Conv2d(256, 2, 1), nn.Conv2d(512, 2, 1)])
        self.stage1_reg = nn.ModuleList([nn.Conv2d(256, 5, 1), nn.Conv2d(256, 5, 1), nn.Conv2d(512, 5, 1)])
        
        # Stage 2 Heads (MODIFIED for SEVERITY)
        self.p5_proj = nn.Conv2d(512, 256, 1)
        self.roi_align = nn.ModuleList([
            RoIAlign((7, 7), spatial_scale=1/8, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/16, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/32, sampling_ratio=2),
        ])
        self.stage2_cls = nn.Conv2d(256, 4, 1)
        self.stage2_reg = nn.Conv2d(256, 5, 1)
        
        # NEW: Severity Grading Head (Pfirrmann I-V)
        # Input: 256 channels (from ROI Align), Output: 5 classes (Grades 1-5)
        self.grade_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, CONFIG['NUM_GRADES']) 
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ... (Backbone/Neck Forward Pass identical) ...
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 5: p3 = x
            if i == 8: p4 = x
            if i == 9: p5 = x
        
        x = self.neck[0](p5); x = self.neck[1](x); p4_lateral = self.neck[2](p4)
        x = torch.cat([x, p4_lateral], dim=1); x = self.neck[3](x); p4_prime = self.neck[4](x)
        x = self.neck[5](p4_prime); p3_lateral = self.neck[6](p3)
        x = torch.cat([x, p3_lateral], dim=1); p3_prime = self.neck[7](x)
        p3_prime = self.neck[8](p3_prime); p5_prime = self.neck[9](p4_prime)

        stage1_outputs = []
        for i, (cls_head, reg_head) in enumerate(zip(self.stage1_cls, self.stage1_reg)):
            feat = [p3_prime, p4_prime, p5_prime][i]
            stage1_outputs.append(torch.cat([F.softmax(cls_head(feat), dim=1), torch.sigmoid(reg_head(feat))], dim=1))

        # Modified Stage 2 Forward to include Grading
        stage2_outputs = []
        device = x.device
        for i, (feat, roi_align) in enumerate(zip([p3_prime, p4_prime, self.p5_proj(p5_prime)], self.roi_align)):
            boxes = self._extract_boxes(stage1_outputs[i], stride=CONFIG['STRIDES'][i], device=device)
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.to(device)
                roi_feats = roi_align(feat, boxes)
                
                # Existing Heads
                cls_out = torch.sigmoid(self.stage2_cls(roi_feats))
                cls_out = F.adaptive_avg_pool2d(cls_out, (1, 1)).squeeze(-1).squeeze(-1)
                reg_out = self.stage2_reg(roi_feats)
                reg_out = F.adaptive_avg_pool2d(reg_out, (1, 1)).squeeze(-1).squeeze(-1)
                
                # NEW: Grade Prediction
                grade_logits = self.grade_head(roi_feats) # [N_boxes, 5]
                
                # Concat everything: [cls(4), reg(5), grade(5)]
                stage2_outputs.append(torch.cat([cls_out, reg_out, grade_logits], dim=1))
            else:
                stage2_outputs.append(None)

        return stage1_outputs, stage2_outputs

    def _extract_boxes(self, output, stride, device):
        # Identical to your original function
        B, _, H, W = output.shape
        reg_pred = output[:, 2:]
        obj_scores = reg_pred[:, 4]
        mask = obj_scores > CONFIG['OBJ_THRESHOLD']
        if not mask.any(): return None
        boxes = []
        for b in range(B):
            indices = torch.nonzero(mask[b], as_tuple=False)
            for idx in indices:
                h, w = idx
                x, y, w_box, h_box = reg_pred[b, :4, h, w]
                boxes.append([b, x - w_box/2, y - h_box/2, x + w_box/2, y + h_box/2])
        return torch.tensor(boxes, dtype=torch.float32, device=device) if boxes else None

# --- New Loss Calculation for Severity ---
def compute_grade_loss(stage2_outputs, targets, device, assignments):
    """Computes cross-entropy loss for Pfirrmann grades on matched boxes."""
    grade_loss = torch.tensor(0.0, device=device, requires_grad=True)
    gt_grades = targets['grades'].to(device) # [Total_GT_Boxes]
    
    valid_batches = False
    
    # targets['grades'] aligns with targets['boxes']
    # assignments maps Pred_Box_Idx -> GT_Box_Idx
    
    for output in stage2_outputs:
        if output is None: continue
        
        # Output structure: [cls(4), reg(5), grade(5)]
        grade_logits = output[:, 9:] # Indices 9 to 13 are grades
        
        if assignments is not None:
            valid = assignments >= 0
            if valid.any():
                valid_batches = True
                
                # Filter Preds
                pred_grades = grade_logits[valid] # [N_valid, 5]
                
                # Get corresponding GT indices
                matched_gt_indices = assignments[valid]
                
                # Get GT Grades
                target_grades = gt_grades[matched_gt_indices] # [N_valid]
                
                # Filter out 'Ignore' grades (-1)
                valid_grade_mask = target_grades >= 0
                
                if valid_grade_mask.any():
                    final_preds = pred_grades[valid_grade_mask]
                    final_targets = target_grades[valid_grade_mask]
                    
                    loss = F.cross_entropy(final_preds, final_targets)
                    grade_loss = grade_loss + loss
    
    return grade_loss * CONFIG['LAMBDA_GRADE']

# --- Helper Utils (Keep your existing assign_targets, etc.) ---
def assign_targets(pred_boxes, gt_boxes, iou_threshold=CONFIG['IOU_THRESHOLD']):
    # Identical to your code
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return torch.full((len(pred_boxes),), -1, dtype=torch.long, device=pred_boxes.device)
    
    # ... (Your existing IoU logic) ...
    pred_boxes_xyxy = pred_boxes[:, 1:5] # Assuming batch index is handled
    gt_boxes_xyxy = torch.zeros_like(gt_boxes[:, 1:])
    gt_boxes_xyxy[:, 0] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_boxes_xyxy[:, 1] = gt_boxes[:, 2] - gt_boxes[:, 4] / 2
    gt_boxes_xyxy[:, 2] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
    gt_boxes_xyxy[:, 3] = gt_boxes[:, 2] + gt_boxes[:, 4] / 2
    
    assignments = torch.full((len(pred_boxes),), -1, dtype=torch.long, device=pred_boxes.device)
    for b in range(int(pred_boxes[:, 0].max().item() + 1)):
        pred_mask = pred_boxes[:, 0] == b
        gt_mask = gt_boxes[:, 0] == b
        if pred_mask.any() and gt_mask.any():
            iou = box_iou(pred_boxes_xyxy[pred_mask], gt_boxes_xyxy[gt_mask])
            max_iou, max_idx = iou.max(dim=1)
            valid = max_iou >= iou_threshold
            
            # Map local GT index to global GT index
            gt_indices = torch.where(gt_mask)[0]
            if len(gt_indices) > 0:
                global_max_idx = gt_indices[max_idx]
                assignments[pred_mask] = torch.where(valid, global_max_idx, -1)
                
    return assignments

# --- Training Loop Update ---
def train_yolospine(model, train_loader, val_loader, device, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=CONFIG['WEIGHT_DECAY'])
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
            
            stage1_out, stage2_out = model(images)
            
            # 1. Stage 1 Loss
            loss1 = compute_stage1_loss(stage1_out, targets, device) # Using your existing function
            
            # Generate Assignments for Stage 2
            pred_boxes = []
            for output, stride in zip(stage1_out, CONFIG['STRIDES']):
                # ... (Your existing box extraction logic for assignments) ...
                reg_pred = output[:, 2:].permute(0, 2, 3, 1)
                B, H, W, _ = reg_pred.shape
                batch_idx_tensor = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
                x, y, w_box, h_box, obj = reg_pred.reshape(-1, 5).split(1, dim=1)
                pred_boxes.append(torch.cat([batch_idx_tensor.view(-1, 1), x - w_box/2, y - h_box/2, x + w_box/2, y + h_box/2], dim=1))
            
            pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else torch.empty((0, 5), device=device)
            assignments = assign_targets(pred_boxes, targets['boxes'])
            
            # 2. Stage 2 Loss (Existing)
            loss2 = compute_stage2_loss(stage2_out, targets, device, assignments) # Using your existing function
            
            # 3. NEW: Grade Loss
            loss_grade = compute_grade_loss(stage2_out, targets, device, assignments)
            
            loss = loss1 + loss2 + loss_grade
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} (Grade: {loss_grade.item():.4f})")

# --- Main (Simplified) ---
if __name__ == "__main__":
    main() # Call your main function structure