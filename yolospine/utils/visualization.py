"""
Visualization utilities for YOLOspine predictions.

Provides functions to draw bounding boxes with class labels and
confidence scores on spinal MRI images.
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = {
    0: "DDD",
    1: "LDB",
    2: "Normal_IVD",
    3: "SS",
    4: "TDB",
    5: "Spondylolisthesis",
}

COLORS = {
    0: (255, 0, 0),       # red
    1: (0, 255, 0),       # green
    2: (0, 0, 255),       # blue
    3: (255, 255, 0),     # yellow
    4: (255, 0, 255),     # magenta
    5: (0, 255, 255),     # cyan
}


def draw_predictions(image, boxes, scores, labels, class_names=None,
                     score_thresh=0.25):
    """
    Draw bounding boxes on an image (numpy RGB array).

    Args:
        image: ``[H, W, 3]`` uint8 RGB array.
        boxes: ``[N, 4]`` array/tensor in (x1, y1, x2, y2) pixels.
        scores: ``[N]`` confidence scores.
        labels: ``[N]`` class indices.
        class_names: Optional dict ``{idx: name}``.
        score_thresh: Minimum score to draw.

    Returns:
        Annotated image (RGB numpy array).
    """
    class_names = class_names or CLASS_NAMES
    img = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        if hasattr(score, "item"):
            score = score.item()
        if hasattr(label, "item"):
            label = label.item()
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = class_names.get(label, str(label))
        text = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def plot_predictions(image, boxes, scores, labels, title="",
                     class_names=None, save_path=None):
    """
    Plot predictions using matplotlib.

    Args:
        image: ``[H, W, 3]`` RGB array or ``[3, H, W]`` tensor.
        boxes: ``[N, 4]`` (x1, y1, x2, y2) pixels.
        scores: ``[N]`` scores.
        labels: ``[N]`` class indices.
        title: Plot title.
        class_names: Optional ``{idx: name}`` dict.
        save_path: If set, save figure to this path.
    """
    class_names = class_names or CLASS_NAMES

    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box, score, cls in zip(boxes, scores, labels):
        if hasattr(box, "cpu"):
            box = box.cpu().numpy()
        if hasattr(score, "item"):
            score = score.item()
        if hasattr(cls, "item"):
            cls = cls.item()

        x1, y1, x2, y2 = box
        color = np.array(COLORS.get(cls, (255, 255, 255))) / 255.0
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5,
                f"{class_names.get(cls, str(cls))}: {score:.2f}",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8))

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
