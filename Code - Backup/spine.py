# YOLOSPINE - MY PROPOSED


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, box_iou
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import yaml
import logging
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
CONFIG = {
    'IMG_SIZE': (384, 384),
    'NUM_CLASSES': 6,
    'STRIDES': [8, 16, 32],
    'CLASS_WEIGHTS': torch.tensor([1/0.459, 1/0.1, 1/0.214, 1/0.226, 1/0.017, 1/0.014]),
    'SIGMA': 0.5,
    'LAMBDA_REG': 1.0,
    'BATCH_SIZE': 8,
    'ACCUMULATION_STEPS': 2,
    'IOU_THRESHOLD': 0.2,
    'OBJ_THRESHOLD': 0.3,
    'CHECKPOINT_PATH': r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\checkpoints",
    'DATA_YAML_PATH': r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\data.yaml",
    'MIN_BOX_SIZE': 1e-4,
    'ANCHORS': [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]],
    'ANCHOR_LOSS_WEIGHT': 1.0,  # Increased
    'WEIGHT_DECAY': 1e-4  # Added L2 regularization
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dataset Module ---
def preprocess_labels(label_dir):
    """Preprocess label files to clamp coordinates and ensure validity."""
    fixed_files = []
    invalid_files = []
    min_box_size = CONFIG['MIN_BOX_SIZE']
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            new_lines = []
            modified = False
            for line in lines:
                try:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    orig = [x, y, w, h]
                    x, y = np.clip([x, y], 0, 1)
                    w = max(min_box_size, min(w, 1 - x))
                    h = max(min_box_size, min(h, 1 - y))
                    if w <= min_box_size or h <= min_box_size:
                        logger.warning(f"Skipping near-zero box in {label_path}: {line.strip()}")
                        continue
                    if orig != [x, y, w, h]:
                        modified = True
                    new_lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                except ValueError:
                    logger.warning(f"Skipping malformed line in {label_path}: {line.strip()}")
                    continue
            if not new_lines:
                logger.error(f"Label file {label_path} has no valid boxes. Marking as invalid.")
                invalid_files.append(label_file)
                continue
            if modified:
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
                fixed_files.append(label_file)
        except Exception as e:
            logger.error(f"Failed to process {label_path}: {str(e)}")
            invalid_files.append(label_file)
    if fixed_files:
        logger.info(f"Fixed {len(fixed_files)} label files: {fixed_files}")
    if invalid_files:
        logger.warning(f"Found {len(invalid_files)} invalid label files: {invalid_files}")
    return fixed_files, invalid_files

def validate_dataset(img_dir, label_dir):
    """Validate dataset integrity and log statistics."""
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    missing_labels = [f for f in img_files if f.rsplit('.', 1)[0] + '.txt' not in label_files]
    missing_images = [f for f in label_files if f.rsplit('.', 1)[0] + ('.jpg' if os.path.exists(os.path.join(img_dir, f.rsplit('.', 1)[0] + '.jpg')) else '.png') not in img_files]
    
    logger.info(f"Dataset stats: {len(img_files)} images, {len(label_files)} labels")
    if missing_labels:
        logger.warning(f"Missing labels for {len(missing_labels)} images: {missing_labels[:5]}...")
    if missing_images:
        logger.warning(f"Missing images for {len(missing_images)} labels: {missing_images[:5]}...")
    
    invalid_labels = []
    box_stats = {'num_boxes': 0, 'min_w': float('inf'), 'max_w': 0, 'min_h': float('inf'), 'max_h': 0}
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    if not (0 <= x <= 1 and 0 <= y <= 1 and w > CONFIG['MIN_BOX_SIZE'] and h > CONFIG['MIN_BOX_SIZE']):
                        invalid_labels.append((label_file, i+1, line.strip()))
                    else:
                        box_stats['num_boxes'] += 1
                        box_stats['min_w'] = min(box_stats['min_w'], w)
                        box_stats['max_w'] = max(box_stats['max_w'], w)
                        box_stats['min_h'] = min(box_stats['min_h'], h)
                        box_stats['max_h'] = max(box_stats['max_h'], h)
                except ValueError:
                    invalid_labels.append((label_file, i+1, line.strip()))
    if invalid_labels:
        logger.warning(f"Found {len(invalid_labels)} invalid labels: {invalid_labels[:5]}...")
    logger.info(f"Box stats: {box_stats['num_boxes']} boxes, width [{box_stats['min_w']:.6f}, {box_stats['max_w']:.6f}], height [{box_stats['min_h']:.6f}, {box_stats['max_h']:.6f}]")

class SpinalMRIDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        logger.info(f"Loaded {len(self.img_files)} images from {img_dir}")
        fixed_files, invalid_files = preprocess_labels(label_dir)
        if invalid_files:
            self.img_files = [f for f in self.img_files if f.rsplit('.', 1)[0] + '.txt' not in invalid_files]
            logger.info(f"Excluded {len(invalid_files)} images due to invalid labels. Remaining: {len(self.img_files)}")
        validate_dataset(img_dir, label_dir)

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id, x, y, w, h = map(float, line.strip().split())
                        x, y, w, h = np.clip([x, y, w, h], 0, 1)
                        if w <= CONFIG['MIN_BOX_SIZE'] or h <= CONFIG['MIN_BOX_SIZE']:
                            logger.warning(f"Skipping invalid box in {label_path}: {line.strip()}")
                            continue
                        boxes.append([x, y, w, h])
                        labels.append(int(class_id))
                    except ValueError:
                        logger.warning(f"Skipping malformed line in {label_path}: {line.strip()}")
                        continue
        else:
            logger.debug(f"No label file for {img_file}")
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        stage1_cls = np.zeros((max(1, len(boxes)), 2), dtype=np.float32)
        stage2_cls = np.zeros((max(1, len(boxes)), 4), dtype=np.float32)
        if len(boxes) > 0:
            for i, class_id in enumerate(labels):
                if class_id in [0, 1]:
                    stage1_cls[i, class_id] = 1.0
                elif class_id in [2, 3, 4, 5]:
                    stage2_cls[i, class_id - 2] = 1.0
        else:
            logger.debug(f"No valid boxes for {img_file}")
        
        if self.transform:
            try:
                augmented = self.transform(image=img, bboxes=boxes, labels=labels)
                img = augmented['image']
                boxes = np.array(augmented['bboxes'], dtype=np.float32)
                labels = np.array(augmented['labels'], dtype=np.int32)
            except Exception as e:
                logger.warning(f"Transformation failed for {img_path}: {str(e)}. Using dummy data.")
                img = np.zeros((CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1], 3), dtype=np.uint8)
                boxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
                labels = np.array([0], dtype=np.int32)
                stage1_cls = np.array([[1.0, 0.0]], dtype=np.float32)
                stage2_cls = np.zeros((1, 4), dtype=np.float32)
                try:
                    augmented = self.transform(image=img, bboxes=boxes, labels=labels)
                    img = augmented['image']
                    boxes = np.array(augmented['bboxes'], dtype=np.float32)
                    labels = np.array(augmented['labels'], dtype=np.int32)
                except Exception as e:
                    logger.error(f"Dummy data transformation failed for {img_path}: {str(e)}")
                    raise
        
        reg_targets = boxes if len(boxes) > 0 else np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        obj_targets = np.ones(len(boxes), dtype=np.float32) if len(boxes) > 0 else np.array([1.0], dtype=np.float32)
        
        return {
            'image': img,
            'stage1_cls': stage1_cls,
            'stage2_cls': stage2_cls,
            'reg': reg_targets,
            'obj': obj_targets,
            'boxes': boxes,
            'labels': labels
        }

def get_transform():
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))

def collate_fn(batch):
    images = []
    targets = {'stage1_cls': [], 'stage2_cls': [], 'reg': [], 'obj': [], 'boxes': [], 'labels': []}
    for b, item in enumerate(batch):
        images.append(item['image'])
        targets['stage1_cls'].append(torch.tensor(item['stage1_cls'], dtype=torch.float32))
        targets['stage2_cls'].append(torch.tensor(item['stage2_cls'], dtype=torch.float32))
        targets['reg'].append(torch.tensor(item['reg'], dtype=torch.float32))
        targets['obj'].append(torch.tensor(item['obj'], dtype=torch.float32))
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
            'boxes': torch.cat([b for b in targets['boxes'] if b.shape[0] > 0], dim=0) if any(b.shape[0] > 0 for b in targets['boxes']) else torch.empty((0, 5), dtype=torch.float32),
            'labels': torch.cat([l for l in targets['labels'] if l.shape[0] > 0], dim=0) if any(l.shape[0] > 0 for l in targets['labels']) else torch.empty(0, dtype=torch.long)
        }
    }

def load_data(data_yaml_path, batch_size=CONFIG['BATCH_SIZE']):
    try:
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load data.yaml: {str(e)}")
        raise
    
    class_names = {i: name for i, name in enumerate(data_yaml['names'])}
    project_path = os.path.dirname(data_yaml_path)
    
    train_img_dir = os.path.join(project_path, "train", "images")
    val_img_dir = os.path.join(project_path, "val", "images")
    test_img_dir = os.path.join(project_path, "test", "images")
    train_label_dir = os.path.join(project_path, "train", "labels")
    val_label_dir = os.path.join(project_path, "val", "labels")
    test_label_dir = os.path.join(project_path, "test", "labels")
    
    transform = get_transform()
    
    try:
        train_dataset = SpinalMRIDataset(train_img_dir, train_label_dir, class_names, transform)
        val_dataset = SpinalMRIDataset(val_img_dir, val_label_dir, class_names, transform)
        test_dataset = SpinalMRIDataset(test_img_dir, test_label_dir, class_names, transform)
    except Exception as e:
        logger.error(f"Dataset initialization failed: {str(e)}")
        raise
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    logger.info(f"Data loaders created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    return train_loader, val_loader, test_loader, class_names

# --- Model Module ---
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
        x = torch.cat([x, shortcut], dim=1)
        return self.silu(self.bn(x))

class RELANBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        return self.silu(self.bn(x))

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
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + out

class YOLOspine(nn.Module):
    def __init__(self, num_classes=CONFIG['NUM_CLASSES']):
        super().__init__()
        self.backbone = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            C3Block(64, 128),
            nn.MaxPool2d(2, 2),
            C3Block(128, 256),
            nn.MaxPool2d(2, 2),
            RELANBlock(256, 256),
            nn.MaxPool2d(2, 2),
            RELANBlock(256, 512),
            nn.MaxPool2d(2, 2),
        ])

        self.neck = nn.ModuleList([
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 1, 1, 0),
            RELANBlock(1024, 256),
            AreaAttention(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 1, 1, 0),
            RELANBlock(512, 256),
            AreaAttention(256),
            nn.Conv2d(256, 512, 3, 2, 1),
        ])

        self.stage1_cls = nn.ModuleList([
            nn.Conv2d(256, 2, 1),
            nn.Conv2d(256, 2, 1),
            nn.Conv2d(512, 2, 1),
        ])
        self.stage1_reg = nn.ModuleList([
            nn.Conv2d(256, 5, 1),
            nn.Conv2d(256, 5, 1),
            nn.Conv2d(512, 5, 1),
        ])

        self.p5_proj = nn.Conv2d(512, 256, 1)
        self.roi_align = nn.ModuleList([
            RoIAlign((7, 7), spatial_scale=1/8, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/16, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/32, sampling_ratio=2),
        ])
        self.stage2_cls = nn.Conv2d(256, 4, 1)
        self.stage2_reg = nn.Conv2d(256, 5, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 5: p3 = x
            if i == 8: p4 = x
            if i == 9: p5 = x

        x = self.neck[0](p5)
        x = self.neck[1](x)
        p4_lateral = self.neck[2](p4)
        x = torch.cat([x, p4_lateral], dim=1)
        x = self.neck[3](x)
        p4_prime = self.neck[4](x)
        x = self.neck[5](p4_prime)
        p3_lateral = self.neck[6](p3)
        x = torch.cat([x, p3_lateral], dim=1)
        p3_prime = self.neck[7](x)
        p3_prime = self.neck[8](p3_prime)
        p5_prime = self.neck[9](p4_prime)

        stage1_outputs = []
        for i, (cls_head, reg_head) in enumerate(zip(self.stage1_cls, self.stage1_reg)):
            feat = [p3_prime, p4_prime, p5_prime][i]
            cls_out = F.softmax(cls_head(feat), dim=1)
            reg_out = torch.sigmoid(reg_head(feat))
            stage1_outputs.append(torch.cat([cls_out, reg_out], dim=1))

        stage2_outputs = []
        device = x.device
        for i, (feat, roi_align) in enumerate(zip([p3_prime, p4_prime, self.p5_proj(p5_prime)], self.roi_align)):
            boxes = self._extract_boxes(stage1_outputs[i], stride=CONFIG['STRIDES'][i], device=device)
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.to(device)
                roi_feats = roi_align(feat, boxes)
                cls_out = torch.sigmoid(self.stage2_cls(roi_feats))
                cls_out = F.adaptive_avg_pool2d(cls_out, (1, 1)).squeeze(-1).squeeze(-1)
                reg_out = self.stage2_reg(roi_feats)
                reg_out = F.adaptive_avg_pool2d(reg_out, (1, 1)).squeeze(-1).squeeze(-1)
                stage2_outputs.append(torch.cat([cls_out, reg_out], dim=1))
            else:
                stage2_outputs.append(None)

        return stage1_outputs, stage2_outputs

    def _extract_boxes(self, output, stride, device):
        B, _, H, W = output.shape
        cls_pred = output[:, :2]
        reg_pred = output[:, 2:]
        obj_scores = reg_pred[:, 4]
        mask = obj_scores > CONFIG['OBJ_THRESHOLD']
        if not mask.any():
            return None
        boxes = []
        for b in range(B):
            indices = torch.nonzero(mask[b], as_tuple=False)
            for idx in indices:
                h, w = idx
                x, y, w_box, h_box = reg_pred[b, :4, h, w]
                x1 = (x - w_box / 2)
                y1 = (y - h_box / 2)
                x2 = (x + w_box / 2)
                y2 = (y + h_box / 2)
                boxes.append([b, x1, y1, x2, y2])
        return torch.tensor(boxes, dtype=torch.float32, device=device) if boxes else None

# --- Utils Module ---
def assign_targets(pred_boxes, gt_boxes, iou_threshold=CONFIG['IOU_THRESHOLD']):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return torch.zeros(len(pred_boxes), dtype=torch.long, device=pred_boxes.device)
    
    pred_boxes_xyxy = torch.zeros_like(pred_boxes[:, 1:])
    pred_boxes_xyxy[:, 0] = pred_boxes[:, 1]
    pred_boxes_xyxy[:, 1] = pred_boxes[:, 2]
    pred_boxes_xyxy[:, 2] = pred_boxes[:, 3]
    pred_boxes_xyxy[:, 3] = pred_boxes[:, 4]
    
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
            logger.debug(f"IoU stats: min={max_iou.min().item():.4f}, max={max_iou.max().item():.4f}, mean={max_iou.mean().item():.4f}")
            valid = max_iou >= iou_threshold
            assignments[pred_mask] = torch.where(valid, max_idx, -1)
    
    logger.debug(f"Valid assignments: {(assignments >= 0).sum().item()} out of {len(assignments)}")
    return assignments

def compute_stage1_loss(stage1_outputs, targets, device):
    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    anchor_loss = torch.tensor(0.0, device=device, requires_grad=True)
    cls_target = targets['stage1_cls'].to(device)
    reg_target = targets['reg'].to(device)
    obj_target = targets['obj'].to(device)
    gt_boxes = targets['boxes'].to(device)
    
    valid_assignments_found = False
    for output, stride in zip(stage1_outputs, CONFIG['STRIDES']):
        cls_pred = output[:, :2].permute(0, 2, 3, 1)
        reg_pred = output[:, 2:].permute(0, 2, 3, 1)
        B, H, W, _ = cls_pred.shape
        
        logger.debug(f"reg_pred: min={reg_pred.min().item():.4f}, max={reg_pred.max().item():.4f}, nan={torch.isnan(reg_pred).any()}, inf={torch.isinf(reg_pred).any()}")
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()
        batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
        
        x, y, w_box, h_box, obj = reg_pred.permute(0, 2, 3, 1).reshape(-1, 5).split(1, dim=1)
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        w_box = torch.clamp(w_box, CONFIG['MIN_BOX_SIZE'], 1)
        h_box = torch.clamp(h_box, CONFIG['MIN_BOX_SIZE'], 1)
        invalid_preds = (w_box <= CONFIG['MIN_BOX_SIZE']) | (h_box <= CONFIG['MIN_BOX_SIZE']) | torch.isnan(x) | torch.isinf(x)
        if invalid_preds.any():
            logger.warning(f"Invalid predicted boxes detected: {invalid_preds.sum().item()} out of {len(x)}")
        
        x1 = x - w_box / 2
        y1 = y - h_box / 2
        x2 = x + w_box / 2
        y2 = y + h_box / 2
        
        pred_boxes = torch.cat([batch_idx.view(-1, 1), x1, y1, x2, y2], dim=1)
        if pred_boxes.shape[0] == 0:
            pred_boxes = torch.empty((0, 5), device=device)
        
        if pred_boxes.shape[0] > 0:
            logger.debug(f"pred_boxes: min={pred_boxes[:, 1:].min().item():.4f}, max={pred_boxes[:, 1:].max().item():.4f}, shape={pred_boxes.shape}")
            logger.debug(f"gt_boxes: min={gt_boxes[:, 1:].min().item():.4f}, max={gt_boxes[:, 1:].max().item():.4f}, shape={gt_boxes.shape}")
        
        assignments = assign_targets(pred_boxes, gt_boxes)
        valid = assignments >= 0
        
        if valid.any():
            valid_assignments_found = True
            valid_idx = torch.where(valid)[0]
            valid_assignments = assignments[valid]
            cls_loss_batch = F.cross_entropy(cls_pred.reshape(-1, 2)[valid_idx], 
                                           cls_target[valid_assignments].argmax(dim=1), 
                                           weight=CONFIG['CLASS_WEIGHTS'][:2].to(device), reduction='mean')
            reg_loss_batch = F.smooth_l1_loss(reg_pred.reshape(-1, 5)[valid_idx], 
                                            torch.cat([reg_target[valid_assignments], 
                                                      obj_target[valid_assignments].unsqueeze(1)], dim=1), 
                                            reduction='mean')
            
            if torch.isnan(cls_loss_batch) or torch.isinf(cls_loss_batch) or torch.isnan(reg_loss_batch) or torch.isinf(reg_loss_batch):
                logger.warning(f"Invalid loss detected for stride {stride}: cls_loss={cls_loss_batch}, reg_loss={reg_loss_batch}")
                continue
            
            cls_loss = cls_loss + cls_loss_batch
            reg_loss = reg_loss + reg_loss_batch
        else:
            logger.debug(f"No valid assignments for stride {stride}")
        
        anchors = torch.tensor(CONFIG['ANCHORS'], device=device)
        anchor_targets = torch.zeros_like(reg_pred.reshape(-1, 5))
        anchor_targets[:, 2:4] = anchors[0]
        anchor_loss_batch = F.smooth_l1_loss(reg_pred.reshape(-1, 5), anchor_targets, reduction='mean') * CONFIG['ANCHOR_LOSS_WEIGHT']
        anchor_loss = anchor_loss + anchor_loss_batch
    
    total_loss = cls_loss + CONFIG['LAMBDA_REG'] * reg_loss + anchor_loss
    if not valid_assignments_found:
        logger.warning("No valid assignments found for any stride. Using anchor loss and fallback.")
        total_loss = total_loss + torch.tensor(1e-4, device=device, requires_grad=True)
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"Total stage1 loss is invalid: {total_loss}")
        return torch.tensor(1e-4, device=device, requires_grad=True)
    
    return total_loss

def compute_stage2_loss(stage2_outputs, targets, device, assignments=None):
    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    cls_target = targets['stage2_cls'].to(device)
    reg_target = targets['reg'].to(device)
    obj_target = targets['obj'].to(device)
    
    valid_outputs = False
    for output in stage2_outputs:
        if output is None:
            logger.debug("Skipping None output in stage2 loss")
            continue
        valid_outputs = True
        cls_pred = output[:, :4]
        reg_pred = output[:, 4:]
        
        if assignments is not None:
            valid = assignments >= 0
            if valid.any():
                valid_idx = torch.where(valid)[0]
                valid_assignments = assignments[valid]
                cls_pred = cls_pred[valid_idx]
                reg_pred = reg_pred[valid_idx]
                cls_target = cls_target[valid_assignments]
                reg_target = reg_target[valid_assignments]
                obj_target = obj_target[valid_assignments]
            else:
                logger.debug("No valid assignments for stage2 loss")
                continue
        
        cls_loss_batch = F.binary_cross_entropy(cls_pred, cls_target, 
                                              weight=CONFIG['CLASS_WEIGHTS'][2:].to(device), reduction='mean')
        mask = obj_target > 0
        if mask.any():
            reg_loss_batch = F.smooth_l1_loss(reg_pred[mask], 
                                            torch.cat([reg_target[mask], obj_target[mask].unsqueeze(1)], dim=1), 
                                            reduction='mean')
            if torch.isnan(reg_loss_batch) or torch.isinf(reg_loss_batch):
                logger.warning(f"Invalid reg_loss in stage2: {reg_loss_batch}")
                continue
            reg_loss = reg_loss + reg_loss_batch
        else:
            logger.debug("No objects detected in stage2")
        
        if torch.isnan(cls_loss_batch) or torch.isinf(cls_loss_batch):
            logger.warning(f"Invalid cls_loss in stage2: {cls_loss_batch}")
            continue
        cls_loss = cls_loss + cls_loss_batch
    
    total_loss = cls_loss + CONFIG['LAMBDA_REG'] * reg_loss
    if not valid_outputs:
        logger.warning("No valid stage2 outputs. Adding small fallback loss.")
        total_loss = total_loss + torch.tensor(1e-4, device=device, requires_grad=True)
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"Total stage2 loss is invalid: {total_loss}")
        return torch.tensor(1e-4, device=device, requires_grad=True)
    
    return total_loss

def soft_nms(boxes, scores, sigma=CONFIG['SIGMA'], score_threshold=0.3):
    if boxes is None or len(boxes) == 0:
        return boxes, scores
    
    boxes = boxes.clone()
    scores = scores.clone()
    N = boxes.shape[0]
    keep = []
    indices = torch.arange(N)
    
    while len(indices) > 0:
        max_score_idx = scores[indices].argmax()
        max_score_idx = indices[max_score_idx]
        keep.append(max_score_idx)
        
        if len(indices) == 1:
            break
            
        curr_box = boxes[max_score_idx:max_score_idx+1, 1:]
        other_boxes = boxes[indices, 1:]
        iou = box_iou(curr_box, other_boxes)[0]
        
        weights = torch.exp(-(iou ** 2) / sigma)
        scores[indices] *= weights
        
        mask = scores[indices] > score_threshold
        indices = indices[mask]
    
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep], scores[keep]

def save_checkpoint(model, optimizer, epoch, phase, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'phase': phase
    }, path)
    logger.info(f"Saved checkpoint: {path}")

def check_gradients(model):
    """Check for invalid gradients in model parameters."""
    total_norm = 0.0
    has_inf_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm()
            total_norm += norm.item() ** 2
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger.warning(f"Invalid gradient in {name}: {param.grad}")
                has_inf_nan = True
    total_norm = total_norm ** 0.5
    logger.debug(f"Gradient norm: {total_norm:.4f}")
    return not has_inf_nan, total_norm

# --- Training Module ---
def train_yolospine(model, train_loader, val_loader, device, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=CONFIG['WEIGHT_DECAY'])
    model.train()
    
    start_phase = 1
    start_epoch = 0
    checkpoint_path = os.path.join(CONFIG['CHECKPOINT_PATH'], 'yolospine_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_phase = checkpoint['phase']
        logger.info(f"Resumed from checkpoint: phase {start_phase}, epoch {start_epoch}")

    for phase, lr in [(1, 0.001), (2, 0.001)]:
        if phase < start_phase:
            continue
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG['WEIGHT_DECAY'])
        logger.info(f"Starting phase {phase} with learning rate {lr}")
        for epoch in range(start_epoch if phase == start_phase else 0, epochs):
            model.train()
            total_loss = 0
            step_count = 0
            optimizer.zero_grad()
            invalid_loss_count = 0
            
            try:
                for batch_idx, batch in enumerate(train_loader):
                    batch_idx = int(batch_idx)  # Ensure batch_idx is an integer
                    logger.debug(f"Batch index type: {type(batch_idx)}")
                    images = batch['image'].to(device)
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch['targets'].items()}
                    
                    stage1_out, stage2_out = model(images)
                    loss1 = compute_stage1_loss(stage1_out, targets, device)
                    
                    pred_boxes = []
                    for output, stride in zip(stage1_out, CONFIG['STRIDES']):
                        reg_pred = output[:, 2:].permute(0, 2, 3, 1)
                        B, H, W, _ = reg_pred.shape
                        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                        grid_y = grid_y.flatten()
                        grid_x = grid_x.flatten()
                        batch_idx_tensor = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
                        x, y, w_box, h_box, obj = reg_pred.reshape(-1, 5).split(1, dim=1)
                        x1 = x - w_box / 2
                        y1 = y - h_box / 2
                        x2 = x + w_box / 2
                        y2 = y + h_box / 2
                        pred_boxes.append(torch.cat([batch_idx_tensor.view(-1, 1), x1, y1, x2, y2], dim=1))
                    pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else torch.empty((0, 5), device=device)
                    assignments = assign_targets(pred_boxes, targets['boxes'].to(device))
                    
                    if phase == 1:
                        loss = loss1
                    else:
                        loss2 = compute_stage2_loss(stage2_out, targets, device, assignments)
                        loss = loss1 + loss2
                
                    if not torch.is_tensor(loss) or not loss.requires_grad or torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss at batch {batch_idx}, phase {phase}, epoch {epoch+1}: {loss}")
                        invalid_loss_count += 1
                        if invalid_loss_count > 10:
                            logger.error("Too many invalid losses. Stopping training.")
                            raise RuntimeError("Training stopped due to persistent invalid losses")
                        continue
                
                    loss = loss / CONFIG['ACCUMULATION_STEPS']
                    loss.backward()
                    valid_grads, grad_norm = check_gradients(model)
                    if not valid_grads:
                        logger.warning(f"Skipping optimization step due to invalid gradients at batch {batch_idx}")
                        optimizer.zero_grad()
                        continue
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Strengthened
                    step_count += 1
                    
                    if step_count % CONFIG['ACCUMULATION_STEPS'] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += loss.item() * CONFIG['ACCUMULATION_STEPS']
                    if batch_idx % 10 == 0:  # Use integer batch_idx
                        logger.info(f"Phase {phase}, Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * CONFIG['ACCUMULATION_STEPS']:.4f}, Grad Norm: {grad_norm:.4f}")
                
                torch.cuda.empty_cache()
                avg_loss = total_loss / max(1, len(train_loader))
                logger.info(f"Phase {phase}, Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        batch_idx = int(batch_idx)
                        images = batch['image'].to(device)
                        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch['targets'].items()}
                        
                        stage1_out, stage2_out = model(images)
                        loss1 = compute_stage1_loss(stage1_out, targets, device)
                        
                        pred_boxes = []
                        for output, stride in zip(stage1_out, CONFIG['STRIDES']):
                            reg_pred = output[:, 2:].permute(0, 2, 3, 1)
                            B, H, W, _ = reg_pred.shape
                            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                            grid_y = grid_y.flatten()
                            grid_x = grid_x.flatten()
                            batch_idx_tensor = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
                            x, y, w_box, h_box, obj = reg_pred.reshape(-1, 5).split(1, dim=1)
                            x1 = x - w_box / 2
                            y1 = y - h_box / 2
                            x2 = x + w_box / 2
                            y2 = y + h_box / 2
                            pred_boxes.append(torch.cat([batch_idx_tensor.view(-1, 1), x1, y1, x2, y2], dim=1))
                        pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else torch.empty((0, 5), device=device)
                        assignments = assign_targets(pred_boxes, targets['boxes'].to(device))
                        
                        if phase == 1:
                            loss = loss1
                        else:
                            loss2 = compute_stage2_loss(stage2_out, targets, device, assignments)
                            loss = loss1 + loss2
                        
                        val_loss += loss.item()
                
                torch.cuda.empty_cache()
                avg_val_loss = val_loss / max(1, len(val_loader))
                logger.info(f"Phase {phase}, Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
                
                save_checkpoint(model, optimizer, epoch+1, phase, checkpoint_path)
            
            except KeyboardInterrupt:
                logger.info("Training interrupted. Saving checkpoint...")
                save_checkpoint(model, optimizer, epoch+1, phase, checkpoint_path)
                raise
    
    return model

# --- Main Execution ---
def main():
    os.chdir(os.path.dirname(CONFIG['DATA_YAML_PATH']))
    logger.info(f"Working directory set to: {os.getcwd()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        train_loader, val_loader, test_loader, class_names = load_data(CONFIG['DATA_YAML_PATH'])
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise
    
    model = YOLOspine(num_classes=CONFIG['NUM_CLASSES']).to(device)
    logger.info("Model initialized and moved to device")

    try:
        train_yolospine(model, train_loader, val_loader, device, epochs=1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()