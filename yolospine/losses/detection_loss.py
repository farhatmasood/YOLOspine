"""
Single-stage detection loss for YOLOspine V1 / V3.3.

Implements task-aligned label assignment and DFL-based bounding box regression
following YOLOv8-style detection heads.

Components:
    - ``TaskAlignedAssigner``: Dynamic positive sample assignment using
      alignment metric = cls_score^alpha * iou^beta.
    - ``BboxLoss``: CIoU regression loss + optional Distribution Focal Loss (DFL).
    - ``v8DetectionLoss``: End-to-end loss that ties anchor generation,
      target assignment, classification BCE, and box losses together.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Box utilities
# --------------------------------------------------------------------------

def bbox_iou(box1, box2, xywh=True, CIoU=True, eps=1e-7):
    """
    Compute IoU (optionally CIoU) between two sets of boxes.

    Args:
        box1, box2: ``[..., 4]`` tensors.
        xywh: If True, boxes are (cx, cy, w, h); else (x1, y1, x2, y2).
        CIoU: If True, return Complete IoU.
    """
    if xywh:
        (x1, y1, w1, h1) = box1.chunk(4, -1)
        (x2, y2, w2, h2) = box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = ((b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) *
             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / math.pi ** 2) * (
            torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
        ) ** 2
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    return iou


def bbox2dist(anchor_points, bbox, reg_max):
    """Convert bounding box to distance from anchor points."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat(
        (anchor_points - x1y1, x2y2 - anchor_points), -1
    ).clamp_(0, reg_max - 0.01)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert distance predictions to bounding boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensors from multi-scale features.

    Args:
        feats: List of feature maps ``[B, C, H_i, W_i]``.
        strides: List of stride values for each scale.
        grid_cell_offset: Offset added to grid coordinates.

    Returns:
        ``(anchor_points, stride_tensor)`` each ``[total_anchors, ...]``.
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


# --------------------------------------------------------------------------
# Task-Aligned Assigner
# --------------------------------------------------------------------------

class TaskAlignedAssigner(nn.Module):
    """
    Dynamic label assignment using alignment metric.

    Alignment metric = ``cls_score^alpha * iou^beta``, selecting top-k
    anchors per ground truth for positive assignment.

    Args:
        topk: Number of top candidates per GT.
        num_classes: Total number of classes.
        alpha: Exponent for classification score.
        beta: Exponent for IoU overlap.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels,
                gt_bboxes, mask_gt):
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        device = pd_scores.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes).long(),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros(pd_scores.shape[:2], dtype=torch.bool,
                            device=device),
                torch.zeros(pd_scores.shape[:2], dtype=torch.long,
                            device=device),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, align_metric,
            overlaps)

        return target_labels, target_bboxes, target_scores, fg_mask, \
            target_gt_idx

    # -- internal helpers --------------------------------------------------

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                      anc_points, mask_gt):
        mask_in_gts = self._select_candidates_in_gts(anc_points, gt_bboxes)
        mask_gt_expanded = mask_gt.squeeze(-1).unsqueeze(1)
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes,
            mask_in_gts * mask_gt_expanded)
        mask_topk = self._select_topk_candidates(align_metric, mask_gt)
        mask_pos = mask_topk * mask_in_gts * mask_gt_expanded
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                         mask_gt):
        na = pd_bboxes.shape[1]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            [self.bs, na, self.n_max_boxes],
            dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros(
            [self.bs, na, self.n_max_boxes],
            dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros(
            [2, self.bs, self.n_max_boxes],
            dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(self.bs, device=pd_scores.device).view(
            -1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()

        bbox_scores[mask_gt] = pd_scores[
            ind[0], :, ind[1]].permute(0, 2, 1)[mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(2).expand(
            -1, -1, self.n_max_boxes, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(1).expand(
            -1, na, -1, -1)[mask_gt]
        overlaps[mask_gt] = self._iou_calculation(pd_boxes, gt_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    @staticmethod
    def _iou_calculation(box1, box2, eps=1e-7):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        inter = ((b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) *
                 (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0))
        union = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) +
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps)
        return (inter / union).squeeze(-1)

    def _select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), dim=2
        ).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps).permute(0, 2, 1)

    def _select_topk_candidates(self, metrics, mask_gt):
        num_anchors = metrics.shape[1]
        topk_metrics, topk_idxs = torch.topk(
            metrics, min(self.topk, num_anchors), dim=1, largest=True)

        is_in_topk = torch.zeros(
            metrics.shape, dtype=torch.bool, device=metrics.device)
        for b in range(self.bs):
            is_in_topk[b].scatter_(0, topk_idxs[b], True)

        mask_gt_bool = mask_gt.squeeze(-1).bool()
        is_in_topk = is_in_topk & mask_gt_bool.unsqueeze(1)
        return is_in_topk.float()

    def _select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-1) > 0
        if fg_mask.sum() > 0:
            mask_multi = (mask_pos.sum(-1).unsqueeze(-1) > 1).expand(
                -1, -1, n_max_boxes)
            max_overlaps_idx = overlaps.argmax(-1)
            is_max = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max.scatter_(-1, max_overlaps_idx.unsqueeze(-1), 1)
            mask_pos = torch.where(mask_multi, is_max, mask_pos).float()
            fg_mask = mask_pos.sum(-1) > 0
        target_gt_idx = mask_pos.argmax(-1)
        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask,
                     align_metric, overlaps):
        batch_ind = torch.arange(
            self.bs, dtype=torch.int64,
            device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1],
             self.num_classes),
            dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        norm_align = (
            (align_metric * fg_mask.unsqueeze(-1).float())
            .amax(-1, keepdim=True).clamp_(min=self.eps))
        norm_overlap = (
            (overlaps * fg_mask.unsqueeze(-1).float())
            .amax(-1, keepdim=True).clamp_(min=self.eps))

        t_scores = ((align_metric / norm_align) *
                     (overlaps / norm_overlap) *
                     fg_mask.unsqueeze(-1).float())
        t_scores = t_scores.amax(-1)
        target_scores = target_scores * t_scores.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores.float()


# --------------------------------------------------------------------------
# Bounding-box loss
# --------------------------------------------------------------------------

class BboxLoss(nn.Module):
    """CIoU regression loss + optional Distribution Focal Loss (DFL).

    Args:
        reg_max: Maximum register for DFL.
        use_dfl: Whether to compute DFL loss.
    """

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        if fg_mask.sum() == 0:
            z = torch.tensor(0.0, device=pred_dist.device)
            return z, z.clone()

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        weight = weight.clamp(min=1e-6)

        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask],
            xywh=False, CIoU=True)
        iou = iou.clamp(min=-1.0, max=1.0)
        loss_iou = ((1.0 - iou) * weight).sum() / max(target_scores_sum, 1)
        loss_iou = loss_iou.clamp(max=100.0)

        if self.use_dfl:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.reg_max)
            target_ltrb = target_ltrb.clamp(0, self.reg_max - 1.01)
            loss_dfl = (
                self._df_loss(
                    pred_dist[fg_mask].view(-1, self.reg_max),
                    target_ltrb[fg_mask])
                * weight)
            loss_dfl = loss_dfl.sum() / max(target_scores_sum, 1)
            loss_dfl = loss_dfl.clamp(max=100.0)
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long().clamp(0, pred_dist.shape[-1] - 2)
        tr = (tl + 1).clamp(max=pred_dist.shape[-1] - 1)
        wl = (tr.float() - target).clamp(0, 1)
        wr = (target - tl.float()).clamp(0, 1)
        left = F.cross_entropy(
            pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        right = F.cross_entropy(
            pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        loss = left * wl + right * wr
        return loss.mean(-1, keepdim=True).clamp(max=100.0)


# --------------------------------------------------------------------------
# End-to-end v8-style detection loss
# --------------------------------------------------------------------------

class v8DetectionLoss(nn.Module):
    """
    YOLOv8-style detection loss for single-stage models (V1, V3.3).

    Combines BCE classification, CIoU regression, and DFL losses with
    task-aligned dynamic label assignment.

    Args:
        model: YOLOspine model instance (needs ``stride`` and ``head``).
        num_classes: Number of object classes.
    """

    def __init__(self, model, num_classes=6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        self.stride = model.stride
        self.nc = num_classes
        self.reg_max = model.head.reg_max
        self.no = num_classes + self.reg_max * 4
        self.device = next(model.parameters()).device

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max, use_dfl=self.reg_max > 1)
        self.proj = torch.arange(
            self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Pad and convert targets to ``[B, max_gt, 5]``."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]

        cxcywh = out[..., 1:5] * scale_tensor
        cx, cy, w, h = cxcywh.unbind(-1)
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        out[..., 1:5] = torch.stack([x1, y1, x2, y2], -1)
        return out

    def forward(self, preds, batch):
        """
        Compute loss.

        Args:
            preds: List of multi-scale feature maps from detection head.
            batch: Dict with ``'targets'`` containing ``'boxes'`` and
                ``'labels'``.

        Returns:
            Scalar loss tensor.
        """
        loss = torch.zeros(3, device=self.device)
        feats = preds

        pred_distri, pred_scores = torch.cat(
            [xi.view(xi.shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (torch.tensor(feats[0].shape[2:], device=self.device,
                               dtype=dtype) * self.stride[0])

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets_raw = batch["targets"]
        gt_boxes_all = targets_raw["boxes"].to(self.device)
        cls_all = targets_raw["labels"].to(self.device)

        if cls_all.dim() > 1:
            cls_all = cls_all.squeeze(-1)

        targets = torch.cat(
            [gt_boxes_all[:, 0:1], cls_all.unsqueeze(-1).float(),
             gt_boxes_all[:, 1:5]], dim=-1)

        gt = self.preprocess(
            targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_distri_sm = pred_distri.view(
            batch_size, -1, 4, self.reg_max).softmax(3)
        pred_distri_dec = (
            pred_distri_sm @ self.proj.type(pred_distri_sm.dtype)
        ).contiguous()

        pred_bboxes = dist2bbox(
            pred_distri_dec, anchor_points.unsqueeze(0), xywh=False)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)).sum()
                   / target_scores_sum)

        # Box regression loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["cls"]
        loss[2] *= self.hyp["dfl"]

        return loss.sum() * batch_size
