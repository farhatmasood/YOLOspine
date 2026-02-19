"""
YOLOspine V3.3: DenseC2f Architecture with Improved Normalization
==================================================================

Variant using DenseNet-style bottleneck blocks (DenseC2f) instead of
standard C2f, with AreaAttention at FPN boundaries and proper
normalization (LayerNorm after output projections, BatchNorm on SPPF).

Key improvements over V2:
    1. DenseBottleneck with growth-rate based feature concatenation
    2. LayerNorm after DenseC2f output projections
    3. Feature normalization at FPN boundaries (adaptive)
    4. Proper Kaiming/Xavier initialization for all layers

Input:  [B, 3, 384, 384]
Output:
    stage1_outputs: List of 3 tensors [B, 11, H, W]
    stage2_outputs: List of 3 tensors or None
    proposals:      List of 3 tensors or None
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign


class DenseBottleneck(nn.Module):
    """DenseNet-style bottleneck with feature concatenation."""

    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1,
                               bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        return torch.cat([x, out], 1)


class DenseC2f(nn.Module):
    """
    CSP module with DenseNet-style bottlenecks.

    Splits input, passes one branch through dense layers,
    concatenates with the other branch, and projects.
    """

    def __init__(self, in_channels, out_channels, n=3, growth_rate=32):
        super().__init__()
        self.c = in_channels // 2
        self.cv1 = nn.Conv2d(in_channels, self.c * 2, 1)

        self.dense_layers = nn.ModuleList()
        current_channels = self.c
        for _ in range(n):
            self.dense_layers.append(
                DenseBottleneck(current_channels, growth_rate))
            current_channels += growth_rate

        self.cv2 = nn.Conv2d(self.c + current_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        dense_out = y2
        for layer in self.dense_layers:
            dense_out = layer(dense_out)
        out = self.cv2(torch.cat([y1, dense_out], 1))
        return self.norm(out)


class AreaAttention(nn.Module):
    """
    Area Attention for long-range spatial context.

    Uses query-key-value attention with a learnable residual scaling
    factor (gamma) and post-attention LayerNorm.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1.0 / ((channels // reduction) ** 0.5)
        self.norm = nn.LayerNorm([channels])

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.float()

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, C, H * W)

        attn_scores = torch.bmm(q, k) * self.scale
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
        attn = torch.softmax(attn_scores, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        result = self.gamma * out + x
        result = result.permute(0, 2, 3, 1)
        result = self.norm(result)
        result = result.permute(0, 3, 1, 2)
        return result


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast with output normalization."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.norm = nn.BatchNorm2d(c2)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        return self.norm(out)


class NormConv(nn.Module):
    """Conv2d + BatchNorm (no activation)."""

    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class YOLOspineV33(nn.Module):
    """
    YOLOspine V3.3 with DenseC2f blocks and improved normalization.

    Architecture:
        Backbone: Stem → 4 DenseC2f stages → SPPF
        Neck:     Top-down FPN with AreaAttention at P5, P4, P3
        Head:     Decoupled Stage 1 detection + RoI-based Stage 2

    Args:
        num_classes: Number of disorder classes (default: 6).
    """

    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
        )

        # Backbone
        self.stage1 = DenseC2f(128, 128, n=2, growth_rate=32)
        self.down1 = NormConv(128, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = DenseC2f(256, 256, n=4, growth_rate=48)
        self.down2 = NormConv(256, 512, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = DenseC2f(512, 512, n=4, growth_rate=64)
        self.down3 = NormConv(512, 1024, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = DenseC2f(1024, 1024, n=2, growth_rate=96)
        self.sppf = SPPF(1024, 1024)

        # Neck (FPN with attention)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce_c5 = NormConv(1024, 512)
        self.attn_p5 = AreaAttention(512)

        self.fpn_c4 = DenseC2f(1024, 512, n=2, growth_rate=48)
        self.attn_p4 = AreaAttention(512)

        self.reduce_c4 = NormConv(512, 256)
        self.fpn_c3 = DenseC2f(512, 256, n=2, growth_rate=32)
        self.attn_p3 = AreaAttention(256)

        # Stage 1 heads
        self.stage1_cls = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(512, num_classes, 1),
            nn.Conv2d(1024, num_classes, 1),
        ])
        self.stage1_reg = nn.ModuleList([
            nn.Conv2d(256, 4, 1),
            nn.Conv2d(512, 4, 1),
            nn.Conv2d(1024, 4, 1),
        ])
        self.stage1_obj = nn.ModuleList([
            nn.Conv2d(256, 1, 1),
            nn.Conv2d(512, 1, 1),
            nn.Conv2d(1024, 1, 1),
        ])

        # Stage 2 RoI components
        self.roi_align = nn.ModuleList([
            RoIAlign((7, 7), spatial_scale=1 / 8, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1 / 16, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1 / 32, sampling_ratio=2),
        ])
        self.proj_p4 = nn.Conv2d(512, 256, 1)
        self.proj_p5 = nn.Conv2d(1024, 256, 1)

        self.stage2_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(),
        )
        self.stage2_cls = nn.Linear(256, num_classes)
        self.stage2_reg = nn.Linear(256, 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for conv in self.stage1_cls:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, -2.0)
        for conv in self.stage1_reg:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.zeros_(conv.bias)
        for conv in self.stage1_obj:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, -4.0)

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        c2 = self.stage1(x)
        x = self.pool1(self.down1(c2))
        c3 = self.stage2(x)
        x = self.pool2(self.down2(c3))
        c4 = self.stage3(x)
        x = self.pool3(self.down3(c4))
        c5 = self.stage4(x)
        c5 = self.sppf(c5)

        # FPN with attention
        p5_reduced = self.reduce_c5(c5)
        p5 = self.attn_p5(p5_reduced)

        p4_up = self.up(p5)
        p4 = self.fpn_c4(torch.cat([p4_up, c4], 1))
        p4 = self.attn_p4(p4)

        p4_reduced = self.reduce_c4(p4)
        p3_up = self.up(p4_reduced)
        p3 = self.fpn_c3(torch.cat([p3_up, c3], 1))
        p3 = self.attn_p3(p3)

        features = [p3, p4, c5]
        feature_channels = [256, 512, 1024]

        # Adaptive feature normalization
        normalized_features = []
        for feat, ch in zip(features, feature_channels):
            if feat.std() > 5.0:
                feat = F.layer_norm(
                    feat.permute(0, 2, 3, 1), [ch]
                ).permute(0, 3, 1, 2)
            normalized_features.append(feat)
        features = normalized_features

        # Stage 1 outputs
        stage1_outputs = []
        for i, (feat, cls_h, reg_h, obj_h) in enumerate(zip(
            features, self.stage1_cls, self.stage1_reg, self.stage1_obj
        )):
            stage1_outputs.append(torch.cat([
                cls_h(feat), reg_h(feat), obj_h(feat)
            ], dim=1))

        # Stage 2 RoI refinement
        stage2_outputs = []
        stage2_proposals = []
        device = x.device

        for i, (feat, roi_op) in enumerate(zip(features, self.roi_align)):
            if i == 0:
                proj_feat = feat
            elif i == 1:
                proj_feat = self.proj_p4(feat)
            else:
                proj_feat = self.proj_p5(feat)

            boxes = self._extract_boxes(
                stage1_outputs[i], [8, 16, 32][i], device)

            if boxes is not None and len(boxes) > 0:
                roi_feats = roi_op(proj_feat, boxes)
                shared_feats = self.stage2_head(roi_feats)
                cls_logits = self.stage2_cls(shared_feats)
                reg_deltas = self.stage2_reg(shared_feats)
                stage2_outputs.append(
                    torch.cat([cls_logits, reg_deltas], dim=1))
                stage2_proposals.append(boxes)
            else:
                stage2_outputs.append(None)
                stage2_proposals.append(None)

        return stage1_outputs, stage2_outputs, stage2_proposals

    def _extract_boxes(self, output, stride, device, obj_threshold=0.05):
        """Extract proposal boxes from Stage 1 output."""
        B, _, H, W = output.shape
        nc = self.num_classes

        bbox_pred = output[:, nc:nc + 4]
        obj_logits = output[:, nc + 4:nc + 5]
        obj_scores = torch.sigmoid(obj_logits).squeeze(1)
        mask = obj_scores > obj_threshold

        if not mask.any():
            return None

        box_pred = torch.sigmoid(bbox_pred)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()

        all_boxes = []
        all_batch_idx = []

        for b in range(B):
            b_mask = mask[b]
            if not b_mask.any():
                continue

            b_boxes = box_pred[b].permute(1, 2, 0)[b_mask]
            b_grid = grid[b_mask]

            xy = (b_boxes[:, :2] + b_grid) * stride
            wh = b_boxes[:, 2:4] * stride * 4

            x1 = xy[:, 0] - wh[:, 0] / 2
            y1 = xy[:, 1] - wh[:, 1] / 2
            x2 = xy[:, 0] + wh[:, 0] / 2
            y2 = xy[:, 1] + wh[:, 1] / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            batch_idx = torch.full(
                (len(boxes), 1), b, device=device, dtype=boxes.dtype)

            all_boxes.append(boxes)
            all_batch_idx.append(batch_idx)

        if not all_boxes:
            return None

        all_boxes = torch.cat(all_boxes, dim=0)
        all_batch_idx = torch.cat(all_batch_idx, dim=0)
        return torch.cat([all_batch_idx, all_boxes], dim=1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOspineV33(num_classes=6).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")

    x = torch.randn(2, 3, 384, 384, device=device)
    model.eval()
    with torch.no_grad():
        s1, s2, props = model(x)
        print("\nStage 1 outputs:")
        for i, out in enumerate(s1):
            print(f"  Scale {i}: {out.shape}")
