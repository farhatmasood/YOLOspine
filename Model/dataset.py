import torch
import cv2
import numpy as np
import pandas as pd
import os
import logging
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def load_pfirrmann_grades(csv_path):
    """
    Loads Pfirrmann grades from CSV file.
    
    Args:
        csv_path: Path to PfirrmannGrade.csv
    
    Returns:
        Dictionary mapping patient_id -> [D3_grade, D4_grade, D5_grade]
        Grades are 0-indexed (0-4) for use with CrossEntropyLoss
    """
    try:
        df = pd.read_csv(csv_path)
        grade_map = {}
        for _, row in df.iterrows():
            pid = str(int(row['Patient_ID'])).zfill(4)
            grades = [
                int(row['D3']) - 1,
                int(row['D4']) - 1,
                int(row['D5']) - 1
            ]
            grade_map[pid] = grades
        logger.info(f"Loaded Pfirrmann grades for {len(grade_map)} patients")
        return grade_map
    except Exception as e:
        logger.warning(f"Could not load grade CSV: {e}. Severity training will be skipped.")
        return {}


class SpinalMRIDataset(Dataset):
    """
    Dataset for spinal MRI with anatomically-sorted Pfirrmann grade mapping.
    
    Anatomical Sorting:
    - Boxes are sorted by Y-coordinate (vertical position)
    - Lower Y = higher in image = superior vertebrae (L1-L3)
    - Higher Y = lower in image = inferior vertebrae (L3-L5)
    - Grades mapped: D3 (superior), D4 (middle), D5 (inferior)
    """
    def __init__(self, img_dir, label_dir, class_names, transform=None, grade_map=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.transform = transform
        self.grade_map = grade_map
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        logger.info(f"Loaded {len(self.img_files)} images from {img_dir}")
        
    def __len__(self):
        return len(self.img_files)
    
    def _extract_patient_id(self, filename):
        """
        Extract patient ID from filename
        
        Handles two formats:
        1. NNNN_T2_TSE_SAG_384_NNNN_PPPP_NNN_R.png -> PPPP (6th component)
        2. T1_PPPP_SN .png -> PPPP (2nd component)
        """
        name = filename.split('.')[0]
        parts = name.split('_')
        
        if parts[0].startswith('T1'):
            patient_id = parts[1].strip()
        else:
            if len(parts) >= 6:
                patient_id = parts[5].strip()
            else:
                patient_id = ''.join(filter(str.isdigit, name))
        
        return patient_id.zfill(4)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        
        patient_id = self._extract_patient_id(img_file)
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c, x, y, w, h = map(float, parts[:5])
                        x = max(1e-6, min(1.0 - 1e-6, x))
                        y = max(1e-6, min(1.0 - 1e-6, y))
                        w = max(1e-6, min(1.0 - 1e-6, w))
                        h = max(1e-6, min(1.0 - 1e-6, h))
                        boxes.append([x, y, w, h])
                        labels.append(int(c))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        box_grades = np.full(len(boxes), -1, dtype=np.int64)
        
        if self.grade_map and patient_id in self.grade_map and len(boxes) > 0:
            patient_grades = self.grade_map[patient_id]
            
            ivd_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
            
            if len(ivd_indices) > 0:
                ivd_boxes = boxes[ivd_indices]
                ivd_y_centers = ivd_boxes[:, 1]
                sorted_ivd_indices = np.argsort(ivd_y_centers)
                
                num_ivd = len(ivd_indices)
                if num_ivd >= 3:
                    box_grades[ivd_indices[sorted_ivd_indices[-3]]] = patient_grades[0]
                    box_grades[ivd_indices[sorted_ivd_indices[-2]]] = patient_grades[1]
                    box_grades[ivd_indices[sorted_ivd_indices[-1]]] = patient_grades[2]
                elif num_ivd == 2:
                    box_grades[ivd_indices[sorted_ivd_indices[-2]]] = patient_grades[1]
                    box_grades[ivd_indices[sorted_ivd_indices[-1]]] = patient_grades[2]
                elif num_ivd == 1:
                    box_grades[ivd_indices[sorted_ivd_indices[-1]]] = patient_grades[2]
        
        stage1_cls = np.zeros((max(1, len(boxes)), 3), dtype=np.float32)
        stage2_cls = np.zeros((max(1, len(boxes)), 3), dtype=np.float32)
        
        if len(boxes) > 0:
            for i, class_id in enumerate(labels):
                if class_id in [0, 1, 2]:
                    stage1_cls[i, class_id] = 1.0
                elif class_id in [3, 4, 5]:
                    stage2_cls[i, class_id - 3] = 1.0

        if self.transform:
            if len(boxes) > 0:
                try:
                    boxes_clipped = np.round(boxes, decimals=6)
                    boxes_clipped = np.clip(boxes_clipped, 0.0, 1.0)
                    boxes_clipped[:, 2:] = np.maximum(boxes_clipped[:, 2:], 1e-6)
                    augmented = self.transform(image=img, bboxes=boxes_clipped.tolist(), labels=labels.tolist())
                    img = augmented['image']
                    boxes = np.array(augmented['bboxes'], dtype=np.float32)
                    labels = np.array(augmented['labels'], dtype=np.int32)
                    box_grades = box_grades[:len(boxes)]
                except ValueError as e:
                    if 'bbox' in str(e):
                        from albumentations.pytorch import ToTensorV2
                        import albumentations as A
                        simple_transform = A.Compose([
                            A.Resize(384, 384),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()
                        ])
                        augmented = simple_transform(image=img)
                        img = augmented['image']
                    else:
                        raise
            else:
                from albumentations.pytorch import ToTensorV2
                import albumentations as A
                simple_transform = A.Compose([
                    A.Resize(384, 384),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
                augmented = simple_transform(image=img)
                img = augmented['image']

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
            'grades': box_grades
        }


def collate_fn(batch):
    """Custom collate function to handle variable number of boxes per image"""
    images = []
    targets = {
        'stage1_cls': [],
        'stage2_cls': [],
        'reg': [],
        'obj': [],
        'boxes': [],
        'labels': [],
        'grades': []
    }
    
    for b, item in enumerate(batch):
        images.append(item['image'])
        targets['stage1_cls'].append(torch.tensor(item['stage1_cls'], dtype=torch.float32))
        targets['stage2_cls'].append(torch.tensor(item['stage2_cls'], dtype=torch.float32))
        targets['reg'].append(torch.tensor(item['reg'], dtype=torch.float32))
        targets['obj'].append(torch.tensor(item['obj'], dtype=torch.float32))
        targets['grades'].append(torch.tensor(item['grades'], dtype=torch.long))

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


def get_transform(train=True):
    """Get data augmentation pipeline"""
    if train:
        return A.Compose([
            A.Resize(384, 384),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))
    else:
        return A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))


def create_dataloaders(data_dir, grade_csv_path, batch_size=8, num_workers=0):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing train/val/test folders
        grade_csv_path: Path to PfirrmannGrade.csv
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # CORRECTED CLASS MAPPING (line ~147 in create_dataloaders)
    class_names = {
    0: 'DDD',                    # Was: Normal_IVD (WRONG)
    1: 'Normal_IVD',             # Was: LDB (WRONG)
    2: 'SS',                     # Correct
    3: 'Spondylolisthesis',      # Was: DDD (WRONG)
    4: 'LDB',                    # Was: TDB (WRONG)
    5: 'TDB'                     # Was: Spondylolisthesis (WRONG)
    }

    
    grade_map = load_pfirrmann_grades(grade_csv_path)
    
    train_dataset = SpinalMRIDataset(
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'train', 'labels'),
        class_names,
        transform=get_transform(train=True),
        grade_map=grade_map
    )
    
    val_dataset = SpinalMRIDataset(
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', 'labels'),
        class_names,
        transform=get_transform(train=False),
        grade_map=grade_map
    )
    
    test_dataset = SpinalMRIDataset(
        os.path.join(data_dir, 'test', 'images'),
        os.path.join(data_dir, 'test', 'labels'),
        class_names,
        transform=get_transform(train=False),
        grade_map=grade_map
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    
    return train_loader, val_loader, test_loader, class_names
