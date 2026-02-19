"""
Detection metrics: Average Precision (AP), mAP, Precision, Recall.

All functions operate on per-image ``(boxes, scores, labels)`` tuples
produced by :func:`yolospine.utils.decode.decode_predictions`.
"""

from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_ap(recall, precision):
    """
    Compute Average Precision using the all-point interpolation method.

    Args:
        recall: 1-D array of recall values (ascending).
        precision: 1-D array of precision values.

    Returns:
        Scalar AP value.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Monotonically decreasing envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integration at recall-change points
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_metrics(predictions, targets, iou_threshold=0.5,
                    num_classes=6, image_size=384):
    """
    Compute mAP, Precision, Recall for a set of images.

    Args:
        predictions: List of ``(boxes, scores, labels)`` tuples
            (one per image, pixel-coordinate boxes).
        targets: Dict with ``'boxes'`` ``[N, 5]``
            ``(batch_idx, cx, cy, w, h)`` normalised and
            ``'labels'`` ``[N]``.
        iou_threshold: IoU threshold for true-positive matching.
        num_classes: Number of object classes.
        image_size: Image resolution (for gt box conversion).

    Returns:
        Dict with ``'mAP'``, ``'precision'``, ``'recall'``,
        ``'aps'`` (per-class), ``'num_gt'``.
    """
    all_tp = defaultdict(list)
    all_fp = defaultdict(list)
    all_scores = defaultdict(list)
    num_gt_per_class = defaultdict(int)

    gt_boxes = targets["boxes"]
    gt_labels = targets["labels"]
    batch_size = len(predictions)

    for b in range(batch_size):
        pred_boxes, pred_scores, pred_labels = predictions[b]

        b_mask = gt_boxes[:, 0] == b
        b_gt_cxcywh = gt_boxes[b_mask, 1:]
        b_gt_labels = gt_labels[b_mask]

        if len(b_gt_cxcywh) > 0:
            cx = b_gt_cxcywh[:, 0] * image_size
            cy = b_gt_cxcywh[:, 1] * image_size
            w = b_gt_cxcywh[:, 2] * image_size
            h = b_gt_cxcywh[:, 3] * image_size
            gt_xyxy = torch.stack([
                cx - w / 2, cy - h / 2,
                cx + w / 2, cy + h / 2], dim=1)
            for c in range(num_classes):
                num_gt_per_class[c] += (b_gt_labels == c).sum().item()
        else:
            gt_xyxy = torch.empty((0, 4), device=pred_boxes.device)

        if len(pred_boxes) == 0:
            continue

        if len(gt_xyxy) == 0:
            for score, label in zip(pred_scores, pred_labels):
                c = label.item()
                all_fp[c].append(1)
                all_tp[c].append(0)
                all_scores[c].append(score.item())
            continue

        matched_gt = set()
        sorted_idx = torch.argsort(pred_scores, descending=True)

        for idx in sorted_idx:
            p_box = pred_boxes[idx:idx + 1]
            p_score = pred_scores[idx].item()
            p_label = pred_labels[idx].item()

            c_mask = b_gt_labels == p_label
            if not c_mask.any():
                all_fp[p_label].append(1)
                all_tp[p_label].append(0)
                all_scores[p_label].append(p_score)
                continue

            c_gt_boxes = gt_xyxy[c_mask]
            c_gt_indices = torch.where(c_mask)[0]
            ious = box_iou(p_box, c_gt_boxes)[0]

            if len(ious) == 0:
                all_fp[p_label].append(1)
                all_tp[p_label].append(0)
                all_scores[p_label].append(p_score)
                continue

            best_iou, best_idx = ious.max(dim=0)
            gt_idx = c_gt_indices[best_idx].item()

            if best_iou >= iou_threshold and gt_idx not in matched_gt:
                matched_gt.add(gt_idx)
                all_tp[p_label].append(1)
                all_fp[p_label].append(0)
            else:
                all_tp[p_label].append(0)
                all_fp[p_label].append(1)

            all_scores[p_label].append(p_score)

    aps, prec_list, rec_list = [], [], []

    for c in range(num_classes):
        if num_gt_per_class[c] == 0:
            continue
        if not all_tp[c]:
            aps.append(0.0)
            prec_list.append(0.0)
            rec_list.append(0.0)
            continue

        scores = np.array(all_scores[c])
        tp = np.array(all_tp[c])
        fp = np.array(all_fp[c])
        order = np.argsort(-scores)
        tp, fp = tp[order], fp[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_gt_per_class[c]
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)

        aps.append(compute_ap(recall, precision))
        prec_list.append(float(precision[-1]) if len(precision) else 0.0)
        rec_list.append(float(recall[-1]) if len(recall) else 0.0)

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "precision": float(np.mean(prec_list)) if prec_list else 0.0,
        "recall": float(np.mean(rec_list)) if rec_list else 0.0,
        "aps": aps,
        "num_gt": sum(num_gt_per_class.values()),
    }
