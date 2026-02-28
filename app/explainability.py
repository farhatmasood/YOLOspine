"""
Explainability Module - Focus Map Visualization
================================================
Attention-like heatmaps from detection bounding boxes and scores.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


def generate_focus_map(detections: List[Dict], image_shape: Tuple[int, ...],
                       blur_kernel: int = 51, weight_by_confidence: bool = True) -> np.ndarray:
    """
    Gaussian-weighted focus map from detection boxes.
    Returns (H, W) normalized heatmap in [0, 1].
    """
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    if not detections:
        return heatmap

    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(w, int(bbox[2])), min(h, int(bbox[3]))
        score = det.get('score', 1.0) if weight_by_confidence else 1.0

        if x2 > x1 and y2 > y1:
            bh, bw = y2 - y1, x2 - x1
            if bh > 4 and bw > 4:
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                yc, xc = np.ogrid[y1:y2, x1:x2]
                dist = np.sqrt(((yc - cy) / (bh / 2)) ** 2 + ((xc - cx) / (bw / 2)) ** 2)
                wmap = np.exp(-dist ** 2 / 2) * score
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], wmap)
            else:
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], score)

    if heatmap.max() > 0:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def apply_heatmap(image: np.ndarray, heatmap: np.ndarray,
                  alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay heatmap on BGR image. Returns BGR."""
    h_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * h_resized), colormap)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def generate_class_specific_map(detections: List[Dict], image_shape: Tuple[int, ...],
                                target_class: Optional[str] = None) -> np.ndarray:
    """Focus map for a specific class only."""
    filtered = detections if target_class is None else [d for d in detections if d.get('class_name') == target_class]
    return generate_focus_map(filtered, image_shape)
