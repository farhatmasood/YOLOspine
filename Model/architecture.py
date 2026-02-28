import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign


class C3Block(nn.Module):
    """C3 Block from YOLOv5 architecture for efficient feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels//2, 1, 1, 0)
        self.shortcut = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.silu(self.bn(torch.cat([x, shortcut], dim=1)))


class RELANBlock(nn.Module):
    """Residual ELAN Block for enhanced feature representation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv2(self.conv1(x)) + self.shortcut(x)))


class AreaAttention(nn.Module):
    """Spatial attention mechanism for region-specific feature enhancement"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = 1.0 / (channels ** 0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1).permute(0, 2, 1)
        k = self.key(x).view(B, C, -1)
        v = self.value(x).view(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        return x + torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)


class YOLOspine(nn.Module):
    """
    Two-stage YOLO-based spine segmentation with Pfirrmann grade prediction
    
    Stage 1: Coarse detection and binary classification (IVD vs PE)
    Stage 2: Fine-grained classification (VB types) and Pfirrmann grading
    """
    def __init__(self, num_classes=6, num_grades=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_grades = num_grades
        
        self.backbone = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            C3Block(64, 128),
            nn.MaxPool2d(2, 2),
            C3Block(128, 256),
            nn.MaxPool2d(2, 2),
            RELANBlock(256, 256),
            nn.MaxPool2d(2, 2),
            RELANBlock(256, 512),
            nn.MaxPool2d(2, 2),
        ])

        self.neck = nn.ModuleList([
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 1, 1, 0),
            RELANBlock(1024, 256),
            AreaAttention(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 1, 1, 0),
            RELANBlock(512, 256),
            AreaAttention(256),
            nn.Conv2d(256, 512, 3, 2, 1),
        ])
        
        self.stage1_cls = nn.ModuleList([
            nn.Conv2d(256, 3, 1),
            nn.Conv2d(256, 3, 1),
            nn.Conv2d(512, 3, 1)
        ])
        self.stage1_reg = nn.ModuleList([
            nn.Conv2d(256, 5, 1),
            nn.Conv2d(256, 5, 1),
            nn.Conv2d(512, 5, 1)
        ])

        self.p5_proj = nn.Conv2d(512, 256, 1)
        self.roi_align = nn.ModuleList([
            RoIAlign((7, 7), spatial_scale=1/8, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/16, sampling_ratio=2),
            RoIAlign((7, 7), spatial_scale=1/32, sampling_ratio=2),
        ])
        
        self.stage2_cls = nn.Conv2d(256, 3, 1)
        self.stage2_reg = nn.Conv2d(256, 5, 1)
        
        self.grade_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_grades)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 5:
                p3 = x
            if i == 8:
                p4 = x
            if i == 9:
                p5 = x

        x = self.neck[0](p5)
        x = self.neck[1](x)
        p4_lateral = self.neck[2](p4)
        x = torch.cat([x, p4_lateral], dim=1)
        x = self.neck[3](x)
        p4_prime = self.neck[4](x)
        x = self.neck[5](p4_prime)
        p3_lateral = self.neck[6](p3)
        x = torch.cat([x, p3_lateral], dim=1)
        p3_prime = self.neck[7](x)
        p3_prime = self.neck[8](p3_prime)
        p5_prime = self.neck[9](p4_prime)

        stage1_outputs = []
        for i, (cls_head, reg_head) in enumerate(zip(self.stage1_cls, self.stage1_reg)):
            feat = [p3_prime, p4_prime, p5_prime][i]
            stage1_outputs.append(torch.cat([
                F.softmax(cls_head(feat), dim=1),
                torch.sigmoid(reg_head(feat))
            ], dim=1))

        stage2_outputs = []
        device = x.device
        for i, (feat, roi_align) in enumerate(zip(
            [p3_prime, p4_prime, self.p5_proj(p5_prime)],
            self.roi_align
        )):
            boxes = self._extract_boxes(
                stage1_outputs[i],
                stride=[8, 16, 32][i],
                device=device
            )
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.to(device)
                roi_feats = roi_align(feat, boxes)
                
                cls_out = torch.sigmoid(self.stage2_cls(roi_feats))
                cls_out = F.adaptive_avg_pool2d(cls_out, (1, 1)).squeeze(-1).squeeze(-1)
                
                reg_out = self.stage2_reg(roi_feats)
                reg_out = F.adaptive_avg_pool2d(reg_out, (1, 1)).squeeze(-1).squeeze(-1)
                
                grade_logits = self.grade_head(roi_feats)
                
                stage2_outputs.append(torch.cat([cls_out, reg_out, grade_logits], dim=1))
            else:
                stage2_outputs.append(None)

        return stage1_outputs, stage2_outputs

    def _extract_boxes(self, output, stride, device, obj_threshold=0.3):
        B, _, H, W = output.shape
        reg_pred = output[:, 3:]
        obj_scores = reg_pred[:, 4]
        mask = obj_scores > obj_threshold
        
        if not mask.any():
            return None
        
        boxes = []
        for b in range(B):
            indices = torch.nonzero(mask[b], as_tuple=False)
            for idx in indices:
                h, w = idx
                x, y, w_box, h_box = reg_pred[b, :4, h, w]
                boxes.append([
                    b,
                    x - w_box/2,
                    y - h_box/2,
                    x + w_box/2,
                    y + h_box/2
                ])
        
        return torch.tensor(boxes, dtype=torch.float32, device=device) if boxes else None
