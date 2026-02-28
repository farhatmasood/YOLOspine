"""
GradCAM Interpretability Module
================================
Gradient-weighted Class Activation Mapping for model explanation.
Supports YOLO, RT-DETR, RF-DETR, segmentation architectures.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image


class GradCAM:
    """
    GradCAM for object detection and segmentation models.
    Generates saliency maps showing which regions drove predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layers: List[str] = None):
        self.model = model
        self.target_layers = target_layers or []
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, inp, out):
                self.activations[name] = out.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0].detach()
            return hook

        if not self.target_layers:
            self._auto_detect_layers()

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(get_activation(name)))
                self.hooks.append(module.register_full_backward_hook(get_gradient(name)))

    def _auto_detect_layers(self):
        candidates = ['AreaAttention', 'c2f_attn', 'attn5', 'attn4_out', 'sppf', 'c2f5', 'c2f4']
        for name, _ in self.model.named_modules():
            if any(c in name for c in candidates):
                self.target_layers.append(name)
                break

        if not self.target_layers:
            last_conv = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = name
            if last_conv:
                self.target_layers.append(last_conv)

    def _extract_target_score(self, output, input_tensor, target_class=None, target_box_idx=None):
        """Extract a differentiable target score from model output."""
        device = input_tensor.device

        if isinstance(output, (list, tuple)):
            if all(isinstance(o, torch.Tensor) for o in output):
                scores = []
                for o in output:
                    if o.dim() == 4:
                        nc = o.shape[1] - 64
                        if nc > 0:
                            scores.append(o[:, 64:, :, :].sigmoid().max())
                if scores:
                    return sum(scores) / len(scores)
                return output[0].mean()

            for o in output:
                if isinstance(o, dict) and 'one2many' in o:
                    aux = o['one2many']
                    if isinstance(aux, list):
                        s = [a.max() for a in aux if isinstance(a, torch.Tensor) and a.requires_grad]
                        return sum(s) / (len(s) + 1e-8) if s else torch.tensor(0.0, device=device, requires_grad=True)
                    elif isinstance(aux, torch.Tensor):
                        return aux.max()

            if isinstance(output[0], torch.Tensor):
                return output[0].mean()
            return torch.tensor(0.0, device=device, requires_grad=True)

        if isinstance(output, dict):
            if 'one2many' in output:
                aux = output['one2many']
                if isinstance(aux, list):
                    s = [a.max() for a in aux if isinstance(a, torch.Tensor) and a.requires_grad]
                    return sum(s) / (len(s) + 1e-8) if s else torch.tensor(0.0, device=device, requires_grad=True)
                elif isinstance(aux, torch.Tensor):
                    return aux.max()
            for key in ('pred_logits', 'logits'):
                if key in output:
                    return output[key].max()
            vals = [v for v in output.values() if isinstance(v, torch.Tensor) and v.requires_grad]
            return vals[0].mean() if vals else torch.tensor(0.0, device=device, requires_grad=True)

        if output.dim() == 4:
            if target_class is not None:
                c = min(target_class, output.shape[1] - 1)
                return output[:, c, :, :].mean()
            return output[:, 1:, :, :].max() if output.shape[1] > 1 else output.max()

        if output.dim() == 3:
            cls_scores = output[0, :, 4:]
            if target_box_idx is not None:
                box_scores = cls_scores[target_box_idx]
            else:
                box_scores = cls_scores.max(dim=0)[0]
            return box_scores[target_class] if target_class is not None else box_scores.max()

        return output.mean() if hasattr(output, 'mean') else torch.tensor(1.0, device=device, requires_grad=True)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 target_box_idx: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Generate GradCAM heatmap. Returns (heatmap_array, metadata_dict)."""
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        target_score = self._extract_target_score(output, input_tensor, target_class, target_box_idx)

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        cam_maps = []
        for layer_name in self.target_layers:
            if layer_name not in self.activations or layer_name not in self.gradients:
                continue
            activation = self.activations[layer_name]
            gradient = self.gradients[layer_name]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam).squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = np.nan_to_num(cam, nan=0.0)
            cam_maps.append(cam)

        if cam_maps:
            final_cam = np.mean(cam_maps, axis=0)
        else:
            h, w = input_tensor.shape[2:]
            final_cam = np.ones((h // 32, w // 32)) * 0.5

        metadata = {
            'target_class': target_class,
            'target_score': target_score.item() if isinstance(target_score, torch.Tensor) else target_score,
            'layers_used': self.target_layers,
        }
        return final_cam, metadata

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Visualization
# ============================================================================
def visualize_gradcam(image: np.ndarray, heatmap: np.ndarray,
                      alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay GradCAM heatmap on image. Returns BGR."""
    h_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    h_resized = np.nan_to_num(np.clip(h_resized, 0, 1), nan=0.0)
    colored = cv2.applyColorMap(np.uint8(255 * h_resized), colormap)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


# ============================================================================
# Attention Report  (keys: class_name, score, alignment_score)
# ============================================================================
def generate_attention_report(detections: List[Dict], heatmap: np.ndarray,
                              image_shape: Tuple[int, int]) -> Dict:
    """
    Quantitative attention analysis.
    Returns dict with:
      - total_detections
      - attention_alignment_scores: list of {class_name, score, alignment_score}
      - pathology_focus_ratio
      - background_attention
      - clinical_validity_score
    """
    h, w = image_shape[:2]
    hmap = cv2.resize(heatmap, (w, h))
    hmap = np.nan_to_num(np.clip(hmap, 0, 1), nan=0.0)

    report = {
        'total_detections': len(detections),
        'attention_alignment_scores': [],
        'pathology_focus_ratio': 0.0,
        'background_attention': 0.0,
        'clinical_validity_score': 0.0,
    }

    if not detections:
        report['background_attention'] = float(hmap.mean())
        return report

    pathology_mask = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        x1, y1, x2, y2 = (max(0, int(v)) for v in det['bbox'])
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            pathology_mask[y1:y2, x1:x2] = 1.0
            box_attn = float(hmap[y1:y2, x1:x2].mean())
            report['attention_alignment_scores'].append({
                'class_name': det['class_name'],
                'score': det['score'],
                'alignment_score': box_attn,
            })

    pa = pathology_mask.sum()
    if pa > 0:
        report['pathology_focus_ratio'] = float((hmap * pathology_mask).sum() / pa)

    bg = 1 - pathology_mask
    ba = bg.sum()
    if ba > 0:
        report['background_attention'] = float((hmap * bg).sum() / ba)

    if report['background_attention'] > 0:
        report['clinical_validity_score'] = min(1.0,
            report['pathology_focus_ratio'] / (report['background_attention'] + 0.1))
    else:
        report['clinical_validity_score'] = report['pathology_focus_ratio']

    return report
