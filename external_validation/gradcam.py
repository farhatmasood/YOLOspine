"""
GradCAM Interpretability Module
================================
Gradient-weighted Class Activation Mapping for YOLOspine models.
Provides visual explanations of model predictions for clinical validation.

References:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017).

    Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual
    Explanations for Deep Convolutional Networks" (WACV 2018).

Usage::

    from external_validation.gradcam import GradCAM, visualize_gradcam

    gradcam = GradCAM(model)
    heatmap, meta = gradcam.generate(input_tensor)
    vis = visualize_gradcam(image_bgr, heatmap)
    gradcam.remove_hooks()
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for object detection models.

    Generates saliency maps showing which regions of the input image
    contributed most to the model's predictions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List[str]] = None,
    ):
        self.model = model
        self.target_layers = target_layers or []
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        def _fwd(name):
            def hook(module, inp, out):
                self.activations[name] = out.detach()
            return hook

        def _bwd(name):
            def hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0].detach()
            return hook

        if not self.target_layers:
            self._auto_detect_layers()

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(_fwd(name)))
                self.hooks.append(module.register_full_backward_hook(_bwd(name)))

    def _auto_detect_layers(self):
        """Best-effort auto-detection of a suitable target layer."""
        candidates = [
            "AreaAttention", "c2f_attn", "attn5", "attn4_out",
            "sppf", "c2f5", "c2f4",
        ]
        for name, _ in self.model.named_modules():
            if any(c in name for c in candidates):
                self.target_layers.append(name)
                return
        # Fallback: last Conv2d
        last_conv = None
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = name
        if last_conv:
            self.target_layers.append(last_conv)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _extract_target_score(self, output, target_class, target_box_idx, device):
        """Extract a differentiable scalar from heterogeneous model outputs."""
        if isinstance(output, (list, tuple)):
            scores = []
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    nc = o.shape[1] - 64
                    if nc > 0:
                        scores.append(o[:, 64:, :, :].sigmoid().max())
            if scores:
                return sum(scores) / len(scores)
            if isinstance(output[0], torch.Tensor):
                return output[0].mean()
            return torch.tensor(0.0, device=device, requires_grad=True)

        if isinstance(output, dict):
            for key in ("pred_logits", "logits"):
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key].max()
            vals = [v for v in output.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad]
            if vals:
                return vals[0].mean()
            return torch.tensor(0.0, device=device, requires_grad=True)

        if output.dim() == 4:
            if target_class is not None:
                c = min(target_class, output.shape[1] - 1)
                return output[:, c, :, :].mean()
            return output[:, 1:, :, :].max() if output.shape[1] > 1 else output.max()

        if output.dim() == 3:
            cls_scores = output[0, :, 4:]
            box_scores = (cls_scores[target_box_idx]
                          if target_box_idx is not None
                          else cls_scores.max(dim=0)[0])
            return (box_scores[target_class]
                    if target_class is not None
                    else box_scores.max())

        return output.mean() if hasattr(output, "mean") else torch.tensor(
            1.0, device=device, requires_grad=True
        )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_box_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a GradCAM heatmap.

        Args:
            input_tensor: ``[1, C, H, W]`` batch.
            target_class: Class index to explain (None = max).
            target_box_idx: Specific box index (None = max).

        Returns:
            ``(heatmap, metadata)`` where *heatmap* is a normalised
            ``[h, w]`` numpy array.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        target_score = self._extract_target_score(
            output, target_class, target_box_idx, input_tensor.device,
        )
        if target_score is None:
            target_score = torch.tensor(
                1.0, device=input_tensor.device, requires_grad=True,
            )

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        cam_maps = []
        for layer in self.target_layers:
            act = self.activations.get(layer)
            grad = self.gradients.get(layer)
            if act is None or grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_maps.append(np.nan_to_num(cam, nan=0.0))

        if cam_maps:
            final = np.mean(cam_maps, axis=0)
        else:
            h, w = input_tensor.shape[2:]
            final = np.ones((h // 32, w // 32)) * 0.5

        meta = {
            "target_class": target_class,
            "target_score": (target_score.item()
                             if isinstance(target_score, torch.Tensor)
                             else target_score),
            "layers_used": list(self.target_layers),
        }
        return final, meta


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ with improved localisation for multiple instances."""

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_box_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        target_score = self._extract_target_score(
            output, target_class, target_box_idx, input_tensor.device,
        )
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        cam_maps = []
        for layer in self.target_layers:
            act = self.activations.get(layer)
            grad = self.gradients.get(layer)
            if act is None or grad is None:
                continue
            g2 = grad ** 2
            g3 = grad ** 3
            alpha = g2 / (2 * g2 + act.sum(dim=(2, 3), keepdim=True) * g3 + 1e-8)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_maps.append(np.nan_to_num(cam, nan=0.0))

        final = np.mean(cam_maps, axis=0) if cam_maps else np.ones(
            (input_tensor.shape[2] // 32, input_tensor.shape[3] // 32)
        ) * 0.5

        return final, {
            "target_class": target_class,
            "target_score": (target_score.item()
                             if isinstance(target_score, torch.Tensor)
                             else target_score),
            "layers_used": list(self.target_layers),
            "method": "GradCAM++",
        }


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a GradCAM heatmap on a BGR image."""
    h_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    h_resized = np.clip(np.nan_to_num(h_resized), 0, 1)
    coloured = cv2.applyColorMap(np.uint8(255 * h_resized), colormap)
    return cv2.addWeighted(image, 1 - alpha, coloured, alpha, 0)


def generate_attention_report(
    detections: List[Dict],
    heatmap: np.ndarray,
    image_shape: Tuple[int, int],
) -> Dict:
    """
    Compute attention alignment metrics between detections and a heatmap.

    Returns a dict with ``pathology_focus_ratio``,
    ``background_attention``, and ``clinical_validity_score``.
    """
    h, w = image_shape[:2]
    hm = np.clip(np.nan_to_num(cv2.resize(heatmap, (w, h))), 0, 1)

    report: Dict = {
        "total_detections": len(detections),
        "attention_alignment_scores": [],
        "pathology_focus_ratio": 0.0,
        "background_attention": 0.0,
        "clinical_validity_score": 0.0,
    }
    if not detections:
        report["background_attention"] = float(hm.mean())
        return report

    mask = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = (max(0, int(v)) for v in det["bbox"])
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
            report["attention_alignment_scores"].append({
                "class": det["class_name"],
                "confidence": det["score"],
                "attention_score": float(hm[y1:y2, x1:x2].mean()),
            })

    pa = mask.sum()
    if pa > 0:
        report["pathology_focus_ratio"] = float((hm * mask).sum() / pa)
    ba = (1 - mask).sum()
    if ba > 0:
        report["background_attention"] = float((hm * (1 - mask)).sum() / ba)
    if report["background_attention"] > 0:
        report["clinical_validity_score"] = min(
            1.0, report["pathology_focus_ratio"] / (report["background_attention"] + 0.1)
        )
    else:
        report["clinical_validity_score"] = report["pathology_focus_ratio"]
    return report
