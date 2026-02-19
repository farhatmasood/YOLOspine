"""
Two-stage detection loss for YOLOspine V2 (DDE + MLDR).

Stage 1 – Dense Disorder Estimation (DDE):
    Grid-based assignment of ground truth to feature-map cells.
    Classification (BCE) + CIoU box regression + objectness.

Stage 2 – Multi-Level Disorder Refinement (MLDR):
    Proposal-based refinement with IoU-based positive matching.
    Classification (BCE) + Smooth-L1 box delta regression.

The ``YOLOspineLoss`` class combines both stages with configurable
weights and a stage-2 warm-up schedule.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# IoU computation (supports GIoU / DIoU / CIoU)
# --------------------------------------------------------------------------

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor,
             x1y1x2y2: bool = True,
             GIoU: bool = False,
             DIoU: bool = False,
             CIoU: bool = True,
             eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU between two aligned sets of boxes.

    Args:
        box1, box2: ``[N, 4]`` tensors.
        x1y1x2y2: If True, format is (x1, y1, x2, y2).
        GIoU / DIoU / CIoU: Select IoU variant.

    Returns:
        ``[N]`` tensor of IoU values.
    """
    if not x1y1x2y2:
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1 = box1[..., 0], box1[..., 1]
        b1_x2, b1_y2 = box1[..., 2], box1[..., 3]
        b2_x1, b2_y1 = box2[..., 0], box2[..., 1]
        b2_x2, b2_y2 = box2[..., 2], box2[..., 3]

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter_area = inter_w * inter_h

    b1_w = (b1_x2 - b1_x1).clamp(min=eps)
    b1_h = (b1_y2 - b1_y1).clamp(min=eps)
    b2_w = (b2_x2 - b2_x1).clamp(min=eps)
    b2_h = (b2_y2 - b2_y1).clamp(min=eps)

    union = b1_w * b1_h + b2_w * b2_h - inter_area + eps
    iou = inter_area / union

    if GIoU or DIoU or CIoU:
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=eps)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=eps)

        if CIoU or DIoU:
            c2 = enclose_w ** 2 + enclose_h ** 2 + eps
            b1_cx = (b1_x1 + b1_x2) / 2
            b1_cy = (b1_y1 + b1_y2) / 2
            b2_cx = (b2_x1 + b2_x2) / 2
            b2_cy = (b2_y1 + b2_y2) / 2
            rho2 = (b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2

            if DIoU:
                return iou - rho2 / c2
            # CIoU
            v = (4 / math.pi ** 2) * torch.pow(
                torch.atan(b2_w / (b2_h + eps)) -
                torch.atan(b1_w / (b1_h + eps)), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - (rho2 / c2 + v * alpha)
        else:
            # GIoU
            enclose_area = enclose_w * enclose_h + eps
            return iou - (enclose_area - union) / enclose_area

    return iou


# --------------------------------------------------------------------------
# Auxiliary loss modules
# --------------------------------------------------------------------------

class FocalLoss(nn.Module):
    r"""Focal Loss: :math:`FL(p) = -\alpha (1-p)^\gamma \log(p)`.

    Args:
        alpha: Balancing factor for positives.
        gamma: Focusing parameter.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        loss = alpha_t * focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss using IoU as soft label.

    :math:`QFL(\sigma)=-|y-\sigma|^\beta\bigl((1-y)\log(1-\sigma)+y\log\sigma\bigr)`

    Args:
        beta: Focusing parameter.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, beta: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        scale_factor = (pred_sigmoid - target).abs().pow(self.beta)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        loss = scale_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------------------------------------------
# Target assignment
# --------------------------------------------------------------------------

def assign_targets_to_grid(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    grid_h: int,
    grid_w: int,
    stride: int,
    num_classes: int,
    image_size: int = 384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign ground truth targets to grid cells.

    Each GT box is assigned to the cell containing its center.

    Args:
        gt_boxes: ``[N, 4]`` boxes in (cx, cy, w, h) normalised format.
        gt_labels: ``[N]`` class indices.
        grid_h, grid_w: Feature-map spatial dimensions.
        stride: Down-sampling stride for this scale.
        num_classes: Total number of classes.
        image_size: Input image resolution.

    Returns:
        ``(cls_targets, box_targets, obj_targets, pos_mask)``
    """
    device = gt_boxes.device

    cls_targets = torch.zeros(num_classes, grid_h, grid_w, device=device)
    box_targets = torch.zeros(4, grid_h, grid_w, device=device)
    obj_targets = torch.zeros(1, grid_h, grid_w, device=device)
    pos_mask = torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device)

    if len(gt_boxes) == 0:
        return cls_targets, box_targets, obj_targets, pos_mask

    cx = gt_boxes[:, 0] * image_size
    cy = gt_boxes[:, 1] * image_size
    w = gt_boxes[:, 2] * image_size
    h = gt_boxes[:, 3] * image_size

    grid_x = (cx / stride).long().clamp(0, grid_w - 1)
    grid_y = (cy / stride).long().clamp(0, grid_h - 1)

    for i in range(len(gt_boxes)):
        gx, gy = grid_x[i], grid_y[i]
        label = gt_labels[i].long()

        cls_targets[label, gy, gx] = 1.0

        cell_cx = gx.float() * stride
        cell_cy = gy.float() * stride
        offset_x = (cx[i] - cell_cx) / stride
        offset_y = (cy[i] - cell_cy) / stride

        box_targets[0, gy, gx] = offset_x.clamp(0, 1)
        box_targets[1, gy, gx] = offset_y.clamp(0, 1)
        box_targets[2, gy, gx] = (w[i] / image_size).clamp(1e-4, 1)
        box_targets[3, gy, gx] = (h[i] / image_size).clamp(1e-4, 1)

        obj_targets[0, gy, gx] = 1.0
        pos_mask[gy, gx] = True

    return cls_targets, box_targets, obj_targets, pos_mask


# --------------------------------------------------------------------------
# Stage losses
# --------------------------------------------------------------------------

def compute_stage1_loss(
    predictions: List[torch.Tensor],
    targets: Dict[str, torch.Tensor],
    num_classes: int = 6,
    image_size: int = 384,
    cls_weight: float = 1.0,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    obj_focal_alpha: Optional[float] = None,
    obj_focal_gamma: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Stage-1 (DDE) loss across all FPN scales.

    Args:
        predictions: List of ``[B, 11, H, W]`` tensors (P3, P4, P5).
            Channels: ``num_classes`` classification + 4 box + 1 objectness.
        targets: Dict with ``'boxes'`` ``[N, 5]`` (batch_idx, cx, cy, w, h)
            normalised and ``'labels'`` ``[N]``.
        cls_weight, box_weight, obj_weight: Loss weights.
        obj_focal_alpha: If set, use Focal Loss for objectness.
        obj_focal_gamma: Focal Loss gamma.

    Returns:
        ``(total_loss, loss_dict)``
    """
    strides = [8, 16, 32]
    device = predictions[0].device
    batch_size = predictions[0].shape[0]

    total_cls = torch.tensor(0.0, device=device)
    total_box = torch.tensor(0.0, device=device)
    total_obj = torch.tensor(0.0, device=device)
    num_pos = 0

    gt_boxes = targets["boxes"]
    gt_labels = targets["labels"]

    focal_obj = None
    if obj_focal_alpha is not None:
        focal_obj = FocalLoss(
            alpha=obj_focal_alpha, gamma=obj_focal_gamma, reduction="sum")

    for pred, stride in zip(predictions, strides):
        B, C, H, W = pred.shape

        pred_cls = pred[:, :num_classes]
        pred_box = pred[:, num_classes:num_classes + 4]
        pred_obj = pred[:, num_classes + 4:]

        for b in range(B):
            b_mask = gt_boxes[:, 0] == b
            b_gt_boxes = gt_boxes[b_mask, 1:]
            b_gt_labels = gt_labels[b_mask]

            cls_tgt, box_tgt, obj_tgt, pos_mask = assign_targets_to_grid(
                b_gt_boxes, b_gt_labels, H, W, stride, num_classes,
                image_size)

            total_cls = total_cls + F.binary_cross_entropy_with_logits(
                pred_cls[b], cls_tgt, reduction="sum")

            if focal_obj is not None:
                total_obj = total_obj + focal_obj(pred_obj[b], obj_tgt)
            else:
                total_obj = total_obj + F.binary_cross_entropy_with_logits(
                    pred_obj[b], obj_tgt, reduction="sum")

            n_pos = pos_mask.sum().item()
            num_pos += n_pos

            if n_pos > 0:
                pred_box_pos = pred_box[b, :, pos_mask].permute(1, 0)
                target_box_pos = box_tgt[:, pos_mask].permute(1, 0)
                pred_box_sig = torch.sigmoid(pred_box_pos)

                pos_y, pos_x = torch.where(pos_mask)

                pred_cx = (pred_box_sig[:, 0] + pos_x.float()) * stride
                pred_cy = (pred_box_sig[:, 1] + pos_y.float()) * stride
                pred_w = pred_box_sig[:, 2] * image_size
                pred_h = pred_box_sig[:, 3] * image_size
                pred_xyxy = torch.stack([
                    pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                    pred_cx + pred_w / 2, pred_cy + pred_h / 2], dim=1)

                tgt_cx = (target_box_pos[:, 0] + pos_x.float()) * stride
                tgt_cy = (target_box_pos[:, 1] + pos_y.float()) * stride
                tgt_w = target_box_pos[:, 2] * image_size
                tgt_h = target_box_pos[:, 3] * image_size
                tgt_xyxy = torch.stack([
                    tgt_cx - tgt_w / 2, tgt_cy - tgt_h / 2,
                    tgt_cx + tgt_w / 2, tgt_cy + tgt_h / 2], dim=1)

                iou = bbox_iou(pred_xyxy, tgt_xyxy, CIoU=True)
                total_box = total_box + (1.0 - iou).sum()

    num_pos = max(num_pos, 1)

    total_cls = cls_weight * total_cls / (
        batch_size * len(strides) * num_classes)
    total_box = box_weight * total_box / num_pos
    total_obj = obj_weight * total_obj / (batch_size * len(strides))

    total = total_cls + total_box + total_obj

    loss_dict = {
        "cls_loss": total_cls.item(),
        "box_loss": total_box.item(),
        "obj_loss": total_obj.item(),
        "num_pos": num_pos,
    }
    return total, loss_dict


def compute_stage2_loss(
    stage2_outputs: List[Optional[torch.Tensor]],
    proposals: List[torch.Tensor],
    targets: Dict[str, torch.Tensor],
    num_classes: int = 6,
    image_size: int = 384,
    iou_threshold: float = 0.5,
    cls_weight: float = 1.0,
    box_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Stage-2 (MLDR) refinement loss.

    Args:
        stage2_outputs: Per-scale ``[num_proposals, 10]``
            (6 cls + 4 box deltas).
        proposals: Per-scale ``[num_proposals, 5]``
            (batch_idx, x1, y1, x2, y2).
        targets: Same as ``compute_stage1_loss``.
        iou_threshold: Minimum IoU to count as positive.

    Returns:
        ``(total_loss, loss_dict)``
    """
    from torchvision.ops import box_iou as tv_box_iou

    device = (proposals[0].device
              if proposals and proposals[0] is not None else "cpu")
    total_cls = torch.tensor(0.0, device=device)
    total_box = torch.tensor(0.0, device=device)
    num_pos = 0
    num_total = 0

    gt_boxes = targets["boxes"]
    gt_labels = targets["labels"]

    for s2_out, props in zip(stage2_outputs, proposals):
        if s2_out is None or props is None or len(props) == 0:
            continue

        pred_cls = s2_out[:, :num_classes]
        pred_box = s2_out[:, num_classes:]

        prop_batch = props[:, 0].long()
        prop_boxes = props[:, 1:5]

        cls_tgt = torch.zeros_like(pred_cls)
        box_tgt = torch.zeros_like(pred_box)
        pos_mask = torch.zeros(len(props), dtype=torch.bool, device=device)

        for b in prop_batch.unique().tolist():
            b_gt_mask = gt_boxes[:, 0] == b
            b_gt_boxes = gt_boxes[b_gt_mask, 1:]
            b_gt_labels = gt_labels[b_gt_mask]

            if len(b_gt_boxes) == 0:
                continue

            # Convert normalised cxcywh to pixel xyxy
            b_gt_cx = b_gt_boxes[:, 0] * image_size
            b_gt_cy = b_gt_boxes[:, 1] * image_size
            b_gt_w = b_gt_boxes[:, 2] * image_size
            b_gt_h = b_gt_boxes[:, 3] * image_size
            gt_xyxy = torch.stack([
                b_gt_cx - b_gt_w / 2, b_gt_cy - b_gt_h / 2,
                b_gt_cx + b_gt_w / 2, b_gt_cy + b_gt_h / 2], dim=1)

            b_prop_mask = prop_batch == b
            b_prop_boxes = prop_boxes[b_prop_mask]
            if len(b_prop_boxes) == 0:
                continue

            ious = tv_box_iou(b_prop_boxes, gt_xyxy)
            max_ious, matched_gt = ious.max(dim=1)
            pos_props = max_ious >= iou_threshold

            b_indices = torch.where(b_prop_mask)[0]
            for idx, (is_pos, gt_idx) in enumerate(
                    zip(pos_props, matched_gt)):
                prop_idx = b_indices[idx]
                if is_pos:
                    pos_mask[prop_idx] = True
                    label = b_gt_labels[gt_idx].long()
                    cls_tgt[prop_idx, label] = 1.0

                    # Box delta targets
                    pbox = prop_boxes[prop_idx]
                    gbox = gt_xyxy[gt_idx]
                    p_cx = (pbox[0] + pbox[2]) / 2
                    p_cy = (pbox[1] + pbox[3]) / 2
                    p_w = pbox[2] - pbox[0]
                    p_h = pbox[3] - pbox[1]
                    g_cx = (gbox[0] + gbox[2]) / 2
                    g_cy = (gbox[1] + gbox[3]) / 2
                    g_w = gbox[2] - gbox[0]
                    g_h = gbox[3] - gbox[1]
                    _eps = 1e-7
                    tx = (g_cx - p_cx) / (p_w + _eps)
                    ty = (g_cy - p_cy) / (p_h + _eps)
                    tw = torch.log((g_w + _eps) / (p_w + _eps))
                    th = torch.log((g_h + _eps) / (p_h + _eps))
                    box_tgt[prop_idx] = torch.stack([tx, ty, tw, th])

        total_cls = total_cls + F.binary_cross_entropy_with_logits(
            pred_cls, cls_tgt, reduction="sum")

        n_pos = pos_mask.sum().item()
        num_pos += n_pos
        num_total += len(props)

        if n_pos > 0:
            total_box = total_box + F.smooth_l1_loss(
                pred_box[pos_mask], box_tgt[pos_mask],
                reduction="sum", beta=0.1)

    num_pos = max(num_pos, 1)
    num_total = max(num_total, 1)

    total_cls = cls_weight * total_cls / num_total
    total_box = box_weight * total_box / num_pos
    total = total_cls + total_box

    loss_dict = {
        "s2_cls_loss": total_cls.item(),
        "s2_box_loss": total_box.item(),
        "s2_num_pos": num_pos,
    }
    return total, loss_dict


# --------------------------------------------------------------------------
# Combined loss module
# --------------------------------------------------------------------------

class YOLOspineLoss(nn.Module):
    """
    Combined loss for YOLOspine V2 (DDE + MLDR).

    Stage-2 loss is activated after ``s2_warmup_epochs`` and gradually
    ramped up over 5 subsequent epochs.

    Args:
        num_classes: Number of target classes.
        image_size: Input resolution.
        s1_*: Stage-1 loss hyper-parameters.
        s2_*: Stage-2 loss hyper-parameters.
        s2_warmup_epochs: Epochs before stage-2 loss kicks in.
    """

    def __init__(
        self,
        num_classes: int = 6,
        image_size: int = 384,
        s1_cls_weight: float = 1.0,
        s1_box_weight: float = 5.0,
        s1_obj_weight: float = 1.0,
        s1_obj_focal_alpha: Optional[float] = None,
        s1_obj_focal_gamma: float = 2.0,
        s2_cls_weight: float = 1.0,
        s2_box_weight: float = 2.0,
        s2_weight: float = 0.5,
        s2_warmup_epochs: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.s1_cls_weight = s1_cls_weight
        self.s1_box_weight = s1_box_weight
        self.s1_obj_weight = s1_obj_weight
        self.s1_obj_focal_alpha = s1_obj_focal_alpha
        self.s1_obj_focal_gamma = s1_obj_focal_gamma

        self.s2_cls_weight = s2_cls_weight
        self.s2_box_weight = s2_box_weight
        self.s2_weight = s2_weight
        self.s2_warmup_epochs = s2_warmup_epochs

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for stage-2 warm-up scheduling."""
        self.current_epoch = epoch

    def forward(
        self,
        stage1_outputs: List[torch.Tensor],
        stage2_outputs: List[Optional[torch.Tensor]],
        proposals: List[torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined two-stage loss.

        Args:
            stage1_outputs: Per-scale DDE feature maps.
            stage2_outputs: Per-scale MLDR refinement outputs.
            proposals: Per-scale proposal boxes.
            targets: Ground truth dict.

        Returns:
            ``(total_loss, loss_dict)``
        """
        s1_loss, s1_dict = compute_stage1_loss(
            stage1_outputs, targets, self.num_classes, self.image_size,
            self.s1_cls_weight, self.s1_box_weight, self.s1_obj_weight,
            self.s1_obj_focal_alpha, self.s1_obj_focal_gamma)

        if (self.current_epoch >= self.s2_warmup_epochs and
                any(s is not None for s in stage2_outputs)):
            s2_loss, s2_dict = compute_stage2_loss(
                stage2_outputs, proposals, targets,
                self.num_classes, self.image_size, 0.5,
                self.s2_cls_weight, self.s2_box_weight)

            warmup_factor = min(
                1.0,
                (self.current_epoch - self.s2_warmup_epochs + 1) / 5)
            s2_loss = self.s2_weight * warmup_factor * s2_loss
        else:
            s2_loss = torch.tensor(0.0, device=stage1_outputs[0].device)
            s2_dict = {
                "s2_cls_loss": 0.0, "s2_box_loss": 0.0, "s2_num_pos": 0}

        total_loss = s1_loss + s2_loss

        loss_dict = {
            "total": total_loss.item(),
            "s1_total": s1_loss.item(),
            "s2_total": (s2_loss.item() if isinstance(s2_loss, torch.Tensor)
                         else s2_loss),
            **s1_dict,
            **s2_dict,
        }

        return total_loss, loss_dict
