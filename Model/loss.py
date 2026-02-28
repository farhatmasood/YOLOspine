import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou
import logging

logger = logging.getLogger(__name__)


def assign_targets(pred_boxes, gt_boxes, iou_threshold=0.2):
    """
    Assign ground truth boxes to predicted boxes based on IoU matching.
    
    Args:
        pred_boxes: [N, 5] tensor (batch_idx, x1, y1, x2, y2)
        gt_boxes: [M, 5] tensor (batch_idx, cx, cy, w, h)
        iou_threshold: Minimum IoU for positive assignment
    
    Returns:
        assignments: [N] tensor mapping pred_idx -> gt_idx (-1 for unmatched)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return torch.full((len(pred_boxes),), -1, dtype=torch.long, device=pred_boxes.device)
    
    pred_boxes_xyxy = pred_boxes[:, 1:5]
    
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
            
            gt_indices = torch.where(gt_mask)[0]
            if len(gt_indices) > 0:
                global_max_idx = gt_indices[max_idx]
                assignments[pred_mask] = torch.where(valid, global_max_idx, -1)
    
    logger.debug(f"Valid assignments: {(assignments >= 0).sum().item()} out of {len(assignments)}")
    return assignments


def compute_stage1_loss(stage1_outputs, targets, device, strides=[8, 16, 32], lambda_reg=1.0, min_box_size=1e-4):
    """
    Compute Stage 1 loss (coarse detection + 3-class classification).
    
    Args:
        stage1_outputs: List of [B, 8, H, W] tensors (cls:3, reg:5)
        targets: Dictionary with 'stage1_cls', 'reg', 'obj', 'boxes'
        device: Torch device
        strides: Feature map strides
        lambda_reg: Regression loss weight
        min_box_size: Minimum valid box dimension
    
    Returns:
        total_loss: Combined classification and regression loss
    """
    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    cls_target = targets['stage1_cls'].to(device)
    reg_target = targets['reg'].to(device)
    obj_target = targets['obj'].to(device)
    gt_boxes = targets['boxes'].to(device)
    
    valid_assignments_found = False
    
    for output, stride in zip(stage1_outputs, strides):
        cls_pred = output[:, :3].permute(0, 2, 3, 1)
        reg_pred = output[:, 3:].permute(0, 2, 3, 1)
        B, H, W, _ = cls_pred.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()
        batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
        
        x, y, w_box, h_box, obj = reg_pred.reshape(-1, 5).split(1, dim=1)
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        w_box = torch.clamp(w_box, min_box_size, 1)
        h_box = torch.clamp(h_box, min_box_size, 1)
        
        x1 = x - w_box / 2
        y1 = y - h_box / 2
        x2 = x + w_box / 2
        y2 = y + h_box / 2
        
        pred_boxes = torch.cat([batch_idx.view(-1, 1), x1, y1, x2, y2], dim=1)
        
        if pred_boxes.shape[0] > 0:
            assignments = assign_targets(pred_boxes, gt_boxes)
            valid = assignments >= 0
            
            if valid.any():
                valid_assignments_found = True
                
                matched_cls_target = cls_target[assignments[valid]]
                matched_cls_pred = cls_pred.reshape(-1, 3)[valid]
                cls_loss = cls_loss + F.binary_cross_entropy(matched_cls_pred, matched_cls_target)
                
                matched_reg_target = reg_target[assignments[valid]]
                matched_reg_pred = reg_pred.reshape(-1, 5)[valid, :4]
                reg_loss = reg_loss + F.smooth_l1_loss(matched_reg_pred, matched_reg_target)
    
    total_loss = cls_loss + lambda_reg * reg_loss
    
    if not valid_assignments_found:
        logger.warning("No valid assignments found in Stage 1")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss


def compute_stage2_loss(stage2_outputs, targets, device, assignments=None, lambda_reg=1.0):
    """
    Compute Stage 2 loss (fine-grained classification + refined regression).
    
    Args:
        stage2_outputs: List of [N, 13] tensors (cls:3, reg:5, grade:5) or None
        targets: Dictionary with 'stage2_cls', 'reg', 'obj'
        device: Torch device
        assignments: Box assignment indices from Stage 1
        lambda_reg: Regression loss weight
    
    Returns:
        total_loss: Combined classification and regression loss
    """
    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    cls_target = targets['stage2_cls'].to(device)
    reg_target = targets['reg'].to(device)
    obj_target = targets['obj'].to(device)
    
    valid_outputs = False
    total_stage2_preds = 0
    
    for output in stage2_outputs:
        if output is None:
            continue
        
        num_preds = output.shape[0]
        cls_pred = output[:, :3]
        reg_pred = output[:, 3:8]
        
        if assignments is not None:
            valid = assignments >= 0
            
            if valid.any():
                valid_outputs = True
                
                valid_indices = torch.where(valid)[0]
                valid_in_range = valid_indices[valid_indices >= total_stage2_preds]
                valid_in_range = valid_in_range[valid_in_range < total_stage2_preds + num_preds]
                
                if len(valid_in_range) > 0:
                    local_indices = valid_in_range - total_stage2_preds
                    gt_indices = assignments[valid_in_range]
                    
                    matched_cls_target = cls_target[gt_indices]
                    matched_cls_pred = cls_pred[local_indices]
                    cls_loss = cls_loss + F.binary_cross_entropy(matched_cls_pred, matched_cls_target)
                    
                    matched_reg_target = reg_target[gt_indices]
                    matched_reg_pred = reg_pred[local_indices, :4]
                    reg_loss = reg_loss + F.smooth_l1_loss(matched_reg_pred, matched_reg_target)
        
        total_stage2_preds += num_preds
    
    total_loss = cls_loss + lambda_reg * reg_loss
    
    if not valid_outputs:
        logger.warning("No valid outputs found in Stage 2")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss


def compute_grade_loss(stage2_outputs, targets, device, assignments, lambda_grade=0.5):
    """
    Compute Pfirrmann grade classification loss.
    
    Args:
        stage2_outputs: List of [N, 13] tensors where indices 8:13 are grade logits
        targets: Dictionary with 'grades' [M] tensor (-1 for ignore)
        device: Torch device
        assignments: [N] tensor mapping pred_idx -> gt_idx
        lambda_grade: Grade loss weight
    
    Returns:
        grade_loss: Weighted cross-entropy loss for grade classification
    """
    grade_loss = torch.tensor(0.0, device=device, requires_grad=True)
    gt_grades = targets['grades'].to(device)
    
    valid_batches = False
    total_stage2_preds = 0
    
    for output in stage2_outputs:
        if output is None:
            continue
        
        num_preds = output.shape[0]
        grade_logits = output[:, 8:]
        
        if assignments is not None:
            valid = assignments >= 0
            
            if valid.any():
                valid_indices = torch.where(valid)[0]
                valid_in_range = valid_indices[valid_indices >= total_stage2_preds]
                valid_in_range = valid_in_range[valid_in_range < total_stage2_preds + num_preds]
                
                if len(valid_in_range) > 0:
                    local_indices = valid_in_range - total_stage2_preds
                    gt_indices = assignments[valid_in_range]
                    
                    pred_grades = grade_logits[local_indices]
                    target_grades = gt_grades[gt_indices]
                    
                    valid_grade_mask = target_grades >= 0
                    
                    if valid_grade_mask.any():
                        valid_batches = True
                        final_preds = pred_grades[valid_grade_mask]
                        final_targets = target_grades[valid_grade_mask]
                        
                        loss = F.cross_entropy(final_preds, final_targets)
                        grade_loss = grade_loss + loss
        
        total_stage2_preds += num_preds
    
    if not valid_batches:
        logger.debug("No valid grade targets found")
    
    return grade_loss * lambda_grade


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss
