"""
YOLOspine V1: Single-Stage Spinal Disorder Detector
=====================================================

YOLOv8m-style architecture with AreaAttention for spinal context.
Uses C2f backbone, PANet neck, and Distribution Focal Loss (DFL) head.

Architecture:
    Backbone: Conv → C2f × 4 stages → SPPF
    Neck:     PANet (top-down + bottom-up fusion)
    Head:     Decoupled detection (DFL regression + classification)
    Attention: AreaAttention on P5 and P4 output features

Input:  [B, 3, 384, 384]
Output:
    Training  - List of 3 feature maps [B, nc+reg_max*4, H, W]
    Inference - [B, N, nc+4] decoded detections

Classes (6):
    0: DDD (Degenerative Disc Disease)
    1: LDB (Lateral Disc Bulge)
    2: Normal_IVD (Normal Intervertebral Disc)
    3: SS  (Spinal Stenosis)
    4: TDB (Transverse Disc Bulge)
    5: Spondylolisthesis
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Compute padding for 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU block."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True
                    else act if isinstance(act, nn.Module)
                    else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck with optional residual connection."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class AreaAttention(nn.Module):
    """
    Area Attention for capturing long-range spinal context.

    Computes global attention over spatial positions using
    query-key-value projections with a learnable residual gate.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(channels)
        self.scale = (channels // reduction) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        resid = x

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))

        out = out.view(B, C, H, W)
        out = self.gamma * out + resid

        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)

        return out


class DFL(nn.Module):
    """Distribution Focal Loss integral module."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


def _make_anchors(feats, stride, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors for a single feature map."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats.dtype, feats.device
    _, _, h, w = feats.shape
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
    sy, sx = torch.meshgrid(sy, sx, indexing="ij")
    anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
    stride_tensor.append(
        torch.full((h * w, 1), stride, dtype=dtype, device=device)
    )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect(nn.Module):
    """YOLOv8-style decoupled detection head with DFL regression."""

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        y = []
        for i, xi in enumerate(x):
            b, c, h, w = xi.shape
            stride = self.stride[i]
            box_dist, cls = xi.split((self.reg_max * 4, self.nc), 1)
            box_dist = box_dist.view(b, 4 * self.reg_max, -1)
            box = self.dfl(box_dist)

            anchors, strides = _make_anchors(xi, stride, 0.5)

            box = box.view(b, 4, -1)
            lt, rb = box.chunk(2, 1)
            x1y1 = anchors.unsqueeze(0).transpose(1, 2) - lt
            x2y2 = anchors.unsqueeze(0).transpose(1, 2) + rb

            box_decoded = torch.cat((x1y1, x2y2), 1) * stride

            cls = cls.view(b, self.nc, -1)
            cls_scores = cls.sigmoid()

            y.append(torch.cat((box_decoded, cls_scores), 1))

        return torch.cat(y, 2).transpose(1, 2)


class YOLOspine(nn.Module):
    """
    YOLOv8m-Attention spinal detection model.

    Single-stage detector with C2f backbone, PANet neck, AreaAttention
    on P5/P4 features, and a decoupled DFL detection head.

    Args:
        num_classes: Number of disorder classes (default: 6).
        image_size:  Input image resolution (default: 384).
        width:       Channel width multiplier (default: 0.75).
        depth:       Depth multiplier for C2f blocks (default: 0.67).
    """

    def __init__(self, num_classes=6, image_size=384, width=0.75, depth=0.67):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        c1 = 3
        c2 = int(64 * width)
        c3 = int(128 * width)
        c4 = int(256 * width)
        c5 = int(512 * width)
        c6 = int(576 * width)

        n_base = [3, 6, 6, 3]
        n = [max(round(x * depth), 1) for x in n_base]

        # Backbone
        self.conv1 = Conv(c1, c2, 3, 2)
        self.conv2 = Conv(c2, c3, 3, 2)
        self.c2f2 = C2f(c3, c3, n[0], shortcut=True)
        self.conv3 = Conv(c3, c4, 3, 2)
        self.c2f3 = C2f(c4, c4, n[1], shortcut=True)
        self.conv4 = Conv(c4, c5, 3, 2)
        self.c2f4 = C2f(c5, c5, n[2], shortcut=True)
        self.conv5 = Conv(c5, c6, 3, 2)
        self.c2f5 = C2f(c6, c6, n[3], shortcut=True)
        self.sppf = SPPF(c6, c6, 5)

        self.attn5 = AreaAttention(c6)

        # Neck (PANet)
        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p4_up = C2f(c6 + c5, c5, n[0], shortcut=False)
        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p3_up = C2f(c5 + c4, c4, n[0], shortcut=False)
        self.conv_down_p3 = Conv(c4, c4, 3, 2)
        self.c2f_p4_down = C2f(c4 + c5, c5, n[0], shortcut=False)
        self.conv_down_p4 = Conv(c5, c5, 3, 2)
        self.c2f_p5_down = C2f(c5 + c6, c6, n[0], shortcut=False)

        self.attn4_out = AreaAttention(c5)

        # Head
        self.head = Detect(num_classes, ch=(c4, c5, c6))
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        self.stride = self.head.stride

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x):
        # Backbone
        p1 = self.conv1(x)
        p2 = self.c2f2(self.conv2(p1))
        p3 = self.c2f3(self.conv3(p2))
        p4 = self.c2f4(self.conv4(p3))
        p5 = self.sppf(self.c2f5(self.conv5(p4)))
        p5 = self.attn5(p5)

        # Neck - top-down
        p5_up = self.up_p5(p5)
        p4_cat = torch.cat([p5_up, p4], 1)
        p4_head = self.c2f_p4_up(p4_cat)
        p4_up = self.up_p4(p4_head)
        p3_cat = torch.cat([p4_up, p3], 1)
        p3_out = self.c2f_p3_up(p3_cat)

        # Neck - bottom-up
        p3_down = self.conv_down_p3(p3_out)
        p4_cat_down = torch.cat([p3_down, p4_head], 1)
        p4_out = self.c2f_p4_down(p4_cat_down)
        p4_out = self.attn4_out(p4_out)

        p4_down = self.conv_down_p4(p4_out)
        p5_cat_down = torch.cat([p4_down, p5], 1)
        p5_out = self.c2f_p5_down(p5_cat_down)

        return self.head([p3_out, p4_out, p5_out])


if __name__ == "__main__":
    model = YOLOspine(num_classes=6)
    x = torch.randn(2, 3, 384, 384)

    model.train()
    y = model(x)
    print("Training output shapes:")
    for i, t in enumerate(y):
        print(f"  Scale {i}: {t.shape}")

    model.eval()
    with torch.no_grad():
        y_inf = model(x)
    print(f"Inference output: {y_inf.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
