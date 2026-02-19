"""
Decode raw model outputs to bounding-box predictions.

Supports both:
- Single-stage (V1 / V3.3): multi-scale feature maps with
  ``[B, num_classes + 4 + 1, H, W]`` format.
- DFL-based heads: decoding distribution-to-box via softmax projection.
"""

import torch
from torchvision.ops import nms


def decode_predictions(
    stage1_outputs,
    num_classes=6,
    image_size=384,
    conf_thresh=0.001,
    nms_thresh=0.65,
    max_det=300,
):
    """
    Decode Stage-1 feature maps to per-image detections.

    Args:
        stage1_outputs: List of ``[B, C, H, W]`` tensors for P3, P4, P5.
            Channel layout: ``num_classes`` cls + 4 box + 1 obj.
        num_classes: Number of object classes.
        image_size: Input image resolution.
        conf_thresh: Minimum confidence to retain.
        nms_thresh: NMS IoU threshold.
        max_det: Maximum detections per image.

    Returns:
        List of ``(boxes, scores, labels)`` tuples per batch item.
        ``boxes``: ``[N, 4]`` in (x1, y1, x2, y2) pixel coordinates.
        ``scores``: ``[N]`` confidence scores.
        ``labels``: ``[N]`` class indices (long).
    """
    strides = [8, 16, 32]
    batch_size = stage1_outputs[0].shape[0]
    device = stage1_outputs[0].device

    all_detections = []

    for b in range(batch_size):
        batch_boxes, batch_scores, batch_labels = [], [], []

        for output, stride in zip(stage1_outputs, strides):
            _, C, H, W = output.shape

            pred_cls = output[b, :num_classes]       # [nc, H, W]
            pred_box = output[b, num_classes:num_classes + 4]  # [4, H, W]
            pred_obj = output[b, num_classes + 4:]   # [1, H, W]

            cls_scores = torch.sigmoid(pred_cls)
            obj_scores = torch.sigmoid(pred_obj)
            combined = cls_scores * obj_scores       # [nc, H, W]

            max_scores, max_labels = combined.max(dim=0)  # [H, W]
            mask = max_scores > conf_thresh

            if not mask.any():
                continue

            y_idx, x_idx = torch.where(mask)
            scores_sel = max_scores[mask]
            labels_sel = max_labels[mask]
            boxes_sel = pred_box[:, mask].permute(1, 0)  # [n, 4]
            boxes_sig = torch.sigmoid(boxes_sel)

            cx = (boxes_sig[:, 0] + x_idx.float()) * stride
            cy = (boxes_sig[:, 1] + y_idx.float()) * stride
            w = boxes_sig[:, 2] * image_size
            h = boxes_sig[:, 3] * image_size

            x1 = (cx - w / 2).clamp(0, image_size)
            y1 = (cy - h / 2).clamp(0, image_size)
            x2 = (cx + w / 2).clamp(0, image_size)
            y2 = (cy + h / 2).clamp(0, image_size)

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            valid = (x2 > x1) & (y2 > y1)
            batch_boxes.append(boxes_xyxy[valid])
            batch_scores.append(scores_sel[valid])
            batch_labels.append(labels_sel[valid])

        if batch_boxes:
            boxes = torch.cat(batch_boxes)
            scores = torch.cat(batch_scores)
            labels = torch.cat(batch_labels)

            keep = nms(boxes, scores, nms_thresh)
            if len(keep) > max_det:
                keep = keep[:max_det]

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            boxes = torch.empty((0, 4), device=device)
            scores = torch.empty(0, device=device)
            labels = torch.empty(0, dtype=torch.long, device=device)

        all_detections.append((boxes, scores, labels))

    return all_detections
