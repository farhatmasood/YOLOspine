"""
Segmentation-based baseline models via ``segmentation-models-pytorch``.

Provides U-Net++, Swin-UNet, and TransUNet proxies for comparison
against YOLOspine. Segmentation masks are converted to bounding boxes
for detection-metric evaluation.

Requirements::

    pip install segmentation-models-pytorch timm
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logger.warning(
        "segmentation_models_pytorch not installed. "
        "U-Net++ / Swin-UNet / TransUNet baselines unavailable.")

try:
    from transformers import DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def get_unet_plusplus(num_classes=7, encoder_name="resnet50",
                      encoder_weights="imagenet"):
    """U-Net++ with the specified encoder backbone."""
    if not SMP_AVAILABLE:
        raise ImportError("Install segmentation_models_pytorch")
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def get_swin_unet(num_classes=7):
    """Swin-UNet equivalent (Swin-Tiny encoder via timm)."""
    if not SMP_AVAILABLE:
        raise ImportError("Install segmentation_models_pytorch and timm")
    return smp.Unet(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def get_transunet(num_classes=7):
    """TransUNet proxy (PVTv2 encoder + U-Net decoder)."""
    if not SMP_AVAILABLE:
        raise ImportError("Install segmentation_models_pytorch and timm")
    return smp.Unet(
        encoder_name="tu-pvt_v2_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


class DetrWrapper(nn.Module):
    """Hugging Face DETR wrapper (ResNet-50 backbone)."""

    def __init__(self, num_classes=6):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install transformers")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )


def get_baseline_model(model_name, num_classes=7):
    """
    Factory for baseline models.

    Args:
        model_name: One of ``'unetplusplus'``, ``'swinunet'``,
            ``'transunet'``, ``'detr'``.
        num_classes: Number of output classes (segmentation models
            use ``disorder_classes + 1`` for background).
    """
    builders = {
        "unetplusplus": get_unet_plusplus,
        "swinunet": get_swin_unet,
        "transunet": get_transunet,
        "detr": DetrWrapper,
    }
    if model_name not in builders:
        raise ValueError(
            f"Unknown baseline: {model_name}. "
            f"Choose from {list(builders)}")
    print(f"Initialising baseline: {model_name}")
    return builders[model_name](num_classes)
