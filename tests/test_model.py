"""Smoke tests for model instantiation and forward pass."""

import torch
import pytest


def test_yolospine_v1_forward():
    from yolospine.models import YOLOspine
    model = YOLOspine(num_classes=6)
    x = torch.randn(1, 3, 384, 384)
    out = model(x)
    assert isinstance(out, list)
    assert len(out) == 3  # multi-scale


def test_yolospine_v2_forward():
    from yolospine.models import YOLOspineV2
    model = YOLOspineV2(num_classes=6)
    x = torch.randn(1, 3, 384, 384)
    s1, s2, proposals = model(x)
    assert s1.dim() == 4
    assert s2.dim() == 3


def test_yolospine_v33_forward():
    from yolospine.models import YOLOspineV33
    model = YOLOspineV33(num_classes=6)
    x = torch.randn(1, 3, 384, 384)
    out = model(x)
    assert isinstance(out, list)
    assert len(out) == 3


def test_dataset_collate():
    from yolospine.data.dataset import collate_fn
    # Minimal test — collate_fn should handle empty batch
    batch = []
    imgs, targets = collate_fn(batch)
    assert imgs.numel() == 0


def test_decode_smoke():
    from yolospine.utils.decode import decode_predictions
    feats = [torch.randn(1, 70 + 6, 48, 48)]  # 64 dfl + 6 cls
    dets = decode_predictions(feats, conf_thresh=0.5, num_classes=6)
    assert isinstance(dets, list)


def test_metrics_ap():
    from yolospine.utils.metrics import compute_ap
    import numpy as np
    recalls = np.array([0.1, 0.2, 0.3])
    precisions = np.array([1.0, 1.0, 0.5])
    ap = compute_ap(recalls, precisions)
    assert 0 <= ap <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
