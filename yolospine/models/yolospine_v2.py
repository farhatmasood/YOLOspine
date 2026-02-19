"""
YOLOspine V2: Two-Stage Spinal Disorder Detection Model
=========================================================

Two-stage architecture combining a Disjoint Disorder Extractor (DDE)
with a Multi-Label Disorder Refiner (MLDR).

Architecture:
    Stage 1 (DDE):
        - C2f backbone with SPPF
        - FPN + PAN neck for multi-scale features
        - Decoupled detection heads at P3, P4, P5
        - Outputs: classification + bbox regression + objectness

    Stage 2 (MLDR):
        - RoI-Align pooling from top-k proposals
        - MLP-based refinement heads per scale
        - Outputs: refined class scores + box deltas

Input:  [B, 3, 384, 384]
Output:
    stage1_outputs: List of 3 tensors [B, 11, H, W]
                    (6 classes + 4 bbox + 1 objectness)
    stage2_outputs: List of 3 tensors [num_proposals, 10] or None
    proposals:      List of 3 tensors [num_proposals, 5]
                    (batch_idx, x1, y1, x2, y2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms


def autopad(k, p=None):
    """Compute padding for 'same' shape."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU activation."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=None, groups=1):
        super().__init__()
        padding = autopad(kernel, padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Bottleneck with optional residual connection."""

    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = ConvBNSiLU(in_ch, hidden, kernel=1)
        self.cv2 = ConvBNSiLU(hidden, out_ch, kernel=3)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, in_ch, out_ch, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.c = int(out_ch * expansion)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c, kernel=1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch, kernel=1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        c_ = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, c_, kernel=1)
        self.cv2 = ConvBNSiLU(c_ * 4, out_ch, kernel=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DetectionHead(nn.Module):
    """
    Stage 1 detection head for a single scale.

    Outputs [B, num_classes + 4 + 1, H, W] with decoupled
    classification, regression, and objectness branches.
    """

    def __init__(self, in_ch, num_classes=6, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_anchors * (num_classes + 5)

        self.stem = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch, kernel=3),
            ConvBNSiLU(in_ch, in_ch, kernel=3),
        )
        self.cls_conv = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch, kernel=3),
            nn.Conv2d(in_ch, num_anchors * num_classes, kernel_size=1),
        )
        self.reg_conv = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch, kernel=3),
            nn.Conv2d(in_ch, num_anchors * 4, kernel_size=1),
        )
        self.obj_conv = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch // 2, kernel=3),
            nn.Conv2d(in_ch // 2, num_anchors, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        bias_init = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_conv[-1].bias, bias_init)
        nn.init.constant_(self.obj_conv[-1].bias, bias_init)

    def forward(self, x):
        x = self.stem(x)
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        obj_out = self.obj_conv(x)
        return torch.cat([cls_out, reg_out, obj_out], dim=1)


class RefinementHead(nn.Module):
    """
    Stage 2 refinement head.

    Takes RoI-pooled features and outputs refined class scores
    and box regression deltas.
    """

    def __init__(self, in_features, num_classes=6, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, roi_features):
        x = roi_features.flatten(1)
        x = self.fc(x)
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        return torch.cat([cls_out, reg_out], dim=1)


class YOLOspineV2(nn.Module):
    """
    Two-stage YOLOspine detector (DDE + MLDR).

    Stage 1 (DDE): Multi-scale feature pyramid detection with
    decoupled classification, regression, objectness heads.

    Stage 2 (MLDR): RoI-based proposal refinement using top-k
    detections from Stage 1.

    Args:
        num_classes: Number of disorder classes (default: 6).
        image_size:  Input resolution (default: 384).
    """

    def __init__(self, num_classes=6, image_size=384):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = [8, 16, 32]
        self.roi_size = 7
        self.top_k = 100

        # Backbone
        self.stem = ConvBNSiLU(3, 64, kernel=3, stride=2)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(64, 128, kernel=3, stride=2), C2f(128, 128, n=3))
        self.stage2 = nn.Sequential(
            ConvBNSiLU(128, 256, kernel=3, stride=2), C2f(256, 256, n=6))
        self.stage3 = nn.Sequential(
            ConvBNSiLU(256, 512, kernel=3, stride=2), C2f(512, 512, n=6))
        self.stage4 = nn.Sequential(
            ConvBNSiLU(512, 512, kernel=3, stride=2),
            C2f(512, 512, n=3), SPPF(512, 512))

        # FPN (top-down)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral1 = ConvBNSiLU(512, 512, kernel=1)
        self.fpn1 = C2f(1024, 512, n=3, shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral2 = ConvBNSiLU(256, 256, kernel=1)
        self.fpn2 = C2f(768, 256, n=3, shortcut=False)

        # PAN (bottom-up)
        self.down1 = ConvBNSiLU(256, 256, kernel=3, stride=2)
        self.pan1 = C2f(768, 512, n=3, shortcut=False)
        self.down2 = ConvBNSiLU(512, 512, kernel=3, stride=2)
        self.pan2 = C2f(1024, 512, n=3, shortcut=False)

        # Detection heads (Stage 1)
        self.detect_p3 = DetectionHead(256, num_classes)
        self.detect_p4 = DetectionHead(512, num_classes)
        self.detect_p5 = DetectionHead(512, num_classes)

        # Refinement heads (Stage 2)
        roi_dim = 512 * self.roi_size * self.roi_size
        self.refine_p3 = RefinementHead(
            256 * self.roi_size * self.roi_size, num_classes)
        self.refine_p4 = RefinementHead(roi_dim, num_classes)
        self.refine_p5 = RefinementHead(roi_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Backbone
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # FPN
        p5 = c5
        p4 = self.fpn1(torch.cat([self.up1(p5), self.lateral1(c4)], dim=1))
        p3 = self.fpn2(torch.cat([self.up2(p4), self.lateral2(c3)], dim=1))

        # PAN
        n3 = p3
        n4 = self.pan1(torch.cat([self.down1(n3), p4], dim=1))
        n5 = self.pan2(torch.cat([self.down2(n4), p5], dim=1))

        # Stage 1 detection
        det_p3 = self.detect_p3(n3)
        det_p4 = self.detect_p4(n4)
        det_p5 = self.detect_p5(n5)
        stage1_outputs = [det_p3, det_p4, det_p5]

        if self.training:
            stage2_outputs = []
            proposals_list = []
            feature_maps = [n3, n4, n5]
            refine_heads = [self.refine_p3, self.refine_p4, self.refine_p5]

            for scale_idx, (det, feat, stride, refine) in enumerate(
                zip(stage1_outputs, feature_maps, self.strides, refine_heads)
            ):
                proposals = self._extract_proposals(det, stride, B)
                if proposals.shape[0] > 0:
                    roi_feats = roi_align(
                        feat, proposals,
                        output_size=(self.roi_size, self.roi_size),
                        spatial_scale=1.0 / stride,
                        aligned=True,
                    )
                    refined = refine(roi_feats)
                    stage2_outputs.append(refined)
                else:
                    stage2_outputs.append(None)
                proposals_list.append(proposals)

            return stage1_outputs, stage2_outputs, proposals_list
        else:
            return stage1_outputs, [None, None, None], [None, None, None]

    def _extract_proposals(self, detection, stride, batch_size):
        """Extract top-k proposals from detection head output."""
        B, C, H, W = detection.shape
        device = detection.device

        cls_logits = detection[:, :self.num_classes]
        box_pred = detection[:, self.num_classes:self.num_classes + 4]
        obj_logits = detection[:, self.num_classes + 4:]

        cls_scores = torch.sigmoid(cls_logits)
        obj_scores = torch.sigmoid(obj_logits)
        scores = cls_scores.max(dim=1)[0] * obj_scores.squeeze(1)

        scores_flat = scores.view(B, -1)
        box_flat = box_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

        k = min(self.top_k, scores_flat.shape[1])
        _, topk_indices = scores_flat.topk(k, dim=1)

        proposals_list = []
        for b in range(B):
            indices = topk_indices[b]
            y_idx = indices // W
            x_idx = indices % W
            boxes = box_flat[b, indices]

            cx = (torch.sigmoid(boxes[:, 0]) + x_idx.float()) * stride
            cy = (torch.sigmoid(boxes[:, 1]) + y_idx.float()) * stride
            w = torch.sigmoid(boxes[:, 2]) * self.image_size * 0.5
            h = torch.sigmoid(boxes[:, 3]) * self.image_size * 0.5

            x1 = (cx - w / 2).clamp(0, self.image_size)
            y1 = (cy - h / 2).clamp(0, self.image_size)
            x2 = (cx + w / 2).clamp(0, self.image_size)
            y2 = (cy + h / 2).clamp(0, self.image_size)

            batch_idx = torch.full((k,), b, dtype=torch.float32, device=device)
            proposals = torch.stack([batch_idx, x1, y1, x2, y2], dim=1)
            proposals_list.append(proposals)

        return torch.cat(proposals_list, dim=0)


def build_model(num_classes=6, image_size=384):
    """Construct a YOLOspineV2 model instance."""
    return YOLOspineV2(num_classes=num_classes, image_size=image_size)


if __name__ == "__main__":
    model = build_model(num_classes=6)

    x = torch.randn(2, 3, 384, 384)

    model.train()
    s1, s2, props = model(x)
    print("Stage 1 outputs:")
    for i, s in enumerate(s1):
        print(f"  P{i+3}: {s.shape}  (stride={model.strides[i]})")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
