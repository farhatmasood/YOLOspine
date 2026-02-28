"""
Model Factory — SpineScan AI Platform
=======================================
Unified model loading with auto-discovery.
All models discovered from the ``weights/`` directory.

Supported architectures:
    YOLO (v8–v26), RT-DETR, RF-DETR, Detectron2,
    UNet++, SwinUNet, TransUNet, YOLO-Seg, YOLOspine
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod

import config
from config import (
    BASE_DIR, MODELS_DIR, DEVICE,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, NUM_SEG_CLASSES,
    discover_models,
)

# ---------------------------------------------------------------------------
# Framework Imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base Model Wrapper
# ---------------------------------------------------------------------------
class ModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    model_type: str = "detection"
    supports_gradcam: bool = False

    @abstractmethod
    def predict(
        self, image: Image.Image, threshold: float
    ) -> Tuple[List[Dict], np.ndarray, Any]:
        """Run inference on a PIL image.

        Returns
        -------
        detections : list[dict]
            Each dict: ``{bbox, score, class_id, class_name}``.
        vis_image : np.ndarray
            Visualisation in BGR format.
        raw_result : Any
            Framework-specific raw output.
        """
        ...

    def get_model_for_gradcam(self) -> Optional[nn.Module]:
        return None

    def preprocess_for_gradcam(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError


def _get_font(size: int = 12):
    """Get a TrueType font, falling back to the default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_detections(image: Image.Image, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on a PIL image; return BGR numpy array."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = _get_font(12)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det['class_name']}: {det['score']:.2f}"
        draw.text((x1, max(0, y1 - 14)), label, fill=color, font=font)
    return np.array(vis)[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Ultralytics Wrapper (YOLO v8–v26, RT-DETR)
# ---------------------------------------------------------------------------
class UltralyticsWrapper(ModelWrapper):
    model_type = "detection"
    supports_gradcam = True

    def __init__(self, model_path: str):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is not installed.")
        self.model = YOLO(str(model_path))
        self.class_names = CLASS_NAMES
        self.device = torch.device(DEVICE)
        if hasattr(self.model, "task") and self.model.task == "segment":
            self.model_type = "segmentation"

    def predict(self, image, threshold):
        results = self.model.predict(image, conf=threshold, verbose=False)
        result = results[0]
        vis_image = result.plot()
        detections = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": names.get(int(box.cls[0]), f"class_{int(box.cls[0])}"),
                })
        return detections, vis_image, result

    def get_model_for_gradcam(self):
        return self.model.model

    def preprocess_for_gradcam(self, image):
        img = image.resize((384, 384), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(self.device)


# ---------------------------------------------------------------------------
# RF-DETR Wrapper
# ---------------------------------------------------------------------------
class RFDETRWrapper(ModelWrapper):
    model_type = "detection"
    supports_gradcam = True

    def __init__(self, model_path: str):
        if not RFDETR_AVAILABLE:
            raise ImportError("rfdetr is not installed.")
        self.model = RFDETRBase(checkpoint=str(model_path))
        self.class_names = CLASS_NAMES
        self.device = torch.device(DEVICE)

    def predict(self, image, threshold):
        raw = self.model.predict(image, threshold=threshold)
        detections = []
        if hasattr(raw, "xyxy"):
            for i in range(len(raw)):
                bbox = raw.xyxy[i].tolist()
                score = float(raw.confidence[i])
                cls_id = int(raw.class_id[i])
                detections.append({
                    "bbox": bbox, "score": score, "class_id": cls_id,
                    "class_name": (self.class_names[cls_id]
                                   if cls_id < len(self.class_names)
                                   else f"class_{cls_id}"),
                })
        vis = _draw_detections(image, detections)
        return detections, vis, raw

    def get_model_for_gradcam(self):
        return self.model.model.model

    def preprocess_for_gradcam(self, image):
        img = image.resize((384, 384), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(self.device)


# ---------------------------------------------------------------------------
# ModelV1 Wrapper (RF-DETR with custom checkpoint)
# ---------------------------------------------------------------------------
class ModelV1Wrapper(ModelWrapper):
    model_type = "detection"
    supports_gradcam = True

    def __init__(self, model_path: str):
        if not RFDETR_AVAILABLE:
            raise ImportError("rfdetr is not installed.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        num_classes = checkpoint.get("num_classes", 6)
        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            if "class_embed.weight" in sd:
                num_classes = sd["class_embed.weight"].shape[0]
        self.model = RFDETRBase(
            checkpoint=str(model_path), pretrain=False, num_classes=num_classes
        )
        self.class_names = CLASS_NAMES
        self.device = torch.device(DEVICE)

    def predict(self, image, threshold):
        raw = self.model.predict(image, threshold=threshold)
        detections = []
        if hasattr(raw, "xyxy"):
            for i in range(len(raw)):
                bbox = raw.xyxy[i].tolist()
                score = float(raw.confidence[i])
                cls_id = int(raw.class_id[i])
                detections.append({
                    "bbox": bbox, "score": score, "class_id": cls_id,
                    "class_name": (self.class_names[cls_id]
                                   if cls_id < len(self.class_names)
                                   else f"class_{cls_id}"),
                })
        vis = _draw_detections(image, detections)
        return detections, vis, raw

    def get_model_for_gradcam(self):
        return self.model.model.model

    def preprocess_for_gradcam(self, image):
        img = image.resize((384, 384), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(self.device)


# ---------------------------------------------------------------------------
# YOLOspine Wrapper (custom architecture or Ultralytics fallback)
# ---------------------------------------------------------------------------
try:
    from yolospine.models import YOLOspineV33 as YOLOspineModel
    YOLOSPINE_MODEL_AVAILABLE = True
except ImportError:
    YOLOSPINE_MODEL_AVAILABLE = False


class YOLOspineWrapper(ModelWrapper):
    model_type = "detection"
    supports_gradcam = True

    def __init__(self, model_path: str):
        self.class_names = CLASS_NAMES
        self.device = torch.device(DEVICE)
        self.use_ultralytics = False

        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(str(model_path))
                self.use_ultralytics = True
                return
            except Exception:
                pass

        if not YOLOSPINE_MODEL_AVAILABLE:
            raise ImportError(
                "YOLOspine model architecture not available. "
                "Install the package: pip install -e ."
            )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model" not in checkpoint:
            raise ValueError("YOLOspine checkpoint must contain a 'model' key.")
        self.model = YOLOspineModel(num_classes=NUM_CLASSES)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval().to(self.device)

    def predict(self, image, threshold):
        if self.use_ultralytics:
            results = self.model.predict(image, conf=threshold, verbose=False)
            result = results[0]
            vis_image = result.plot()
            detections = []
            names = result.names
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "class_name": names.get(int(box.cls[0]),
                                                 f"class_{int(box.cls[0])}"),
                    })
            return detections, vis_image, result

        from torchvision import transforms
        orig_size = image.size
        img = image.resize((384, 384), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
        detections = []
        if outputs is not None and len(outputs.shape) == 3:
            preds = outputs[0].cpu().numpy()
            for pred in preds:
                if len(pred) >= 5 + NUM_CLASSES:
                    cx, cy, w, h = pred[:4]
                    obj_conf = pred[4] if len(pred) > 4 else 1.0
                    cls_scores = (pred[5:5 + NUM_CLASSES]
                                  if len(pred) > 5 else pred[4:4 + NUM_CLASSES])
                    cls_id = int(np.argmax(cls_scores))
                    cls_conf = float(cls_scores[cls_id])
                    score = obj_conf * cls_conf if len(pred) > 5 else cls_conf
                    if score >= threshold:
                        sx = orig_size[0] / 384
                        sy = orig_size[1] / 384
                        detections.append({
                            "bbox": [(cx - w / 2) * sx, (cy - h / 2) * sy,
                                     (cx + w / 2) * sx, (cy + h / 2) * sy],
                            "score": score, "class_id": cls_id,
                            "class_name": (self.class_names[cls_id]
                                           if cls_id < len(self.class_names)
                                           else f"class_{cls_id}"),
                        })
        vis = _draw_detections(image, detections)
        return detections, vis, outputs

    def get_model_for_gradcam(self):
        return self.model.model if self.use_ultralytics else self.model

    def preprocess_for_gradcam(self, image):
        img = image.resize((384, 384), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(self.device)


# ---------------------------------------------------------------------------
# Detectron2 Wrapper
# ---------------------------------------------------------------------------
try:
    import detectron2  # noqa: F401
    from detectron2.config import get_cfg, CfgNode
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.modeling import build_model
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False


class Detectron2Wrapper(ModelWrapper):
    model_type = "detection"
    supports_gradcam = False

    def __init__(self, model_path: str):
        if not DETECTRON2_AVAILABLE:
            raise ImportError("detectron2 is not installed.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        num_classes = checkpoint.get("num_classes", 6)
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        saved_config = checkpoint.get("config", None)

        self.class_names = class_names or CLASS_NAMES
        self.device = torch.device(DEVICE)

        cfg = get_cfg()
        if isinstance(saved_config, CfgNode):
            cfg = saved_config.clone()
        elif isinstance(saved_config, str):
            cfg.merge_from_other_cfg(CfgNode.load_cfg(saved_config))
        else:
            cfg.merge_from_file(
                model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        cfg.MODEL.DEVICE = DEVICE
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

        self.model = build_model(cfg)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("Detectron2 checkpoint requires 'model_state_dict'.")
        self.model.eval().to(torch.device(DEVICE))
        self.cfg = cfg

        meta_name = f"detectron2_custom_{id(self)}"
        if meta_name not in MetadataCatalog.list():
            MetadataCatalog.get(meta_name).set(
                thing_classes=list(self.class_names)
            )
        self.metadata = MetadataCatalog.get(meta_name)

    def predict(self, image, threshold):
        img_np = np.array(image)[:, :, ::-1].copy()
        with torch.no_grad():
            inputs = [{"image": torch.from_numpy(img_np.transpose(2, 0, 1)).float()}]
            outputs = self.model(inputs)[0]
        instances = outputs["instances"]
        instances = instances[instances.scores >= threshold]
        detections = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor[0].cpu().numpy().tolist()
            score = float(instances.scores[i].cpu())
            cls_id = int(instances.pred_classes[i].cpu())
            detections.append({
                "bbox": bbox, "score": score, "class_id": cls_id,
                "class_name": (self.class_names[cls_id]
                               if cls_id < len(self.class_names)
                               else f"class_{cls_id}"),
            })
        v = Visualizer(img_np[:, :, ::-1], self.metadata, scale=1.0)
        vis_out = v.draw_instance_predictions(instances.to("cpu"))
        vis_image = vis_out.get_image()[:, :, ::-1]
        return detections, vis_image, outputs


# ---------------------------------------------------------------------------
# Segmentation Wrapper (UNet++, SwinUNet, TransUNet)
# ---------------------------------------------------------------------------
class SegmentationWrapper(ModelWrapper):
    model_type = "segmentation"
    supports_gradcam = True

    def __init__(self, model_path: str, model_name: str):
        if not SMP_AVAILABLE:
            raise ImportError("segmentation-models-pytorch is not installed.")
        self.model_name = model_name
        self.class_names = CLASS_NAMES
        self.device = torch.device(DEVICE)

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        self.num_classes = (
            state_dict["segmentation_head.0.weight"].shape[0]
            if "segmentation_head.0.weight" in state_dict
            else NUM_SEG_CLASSES
        )

        name_lower = model_name.lower()
        if "unetplusplus" in name_lower:
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=3, classes=self.num_classes, activation=None,
            )
            self.img_size = 384
        elif "swin" in name_lower:
            self.model = smp.Unet(
                encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None,
                in_channels=3, classes=self.num_classes, activation=None,
            )
            self.img_size = 224
        elif "trans" in name_lower:
            self.model = smp.Unet(
                encoder_name="tu-pvt_v2_b2", encoder_weights=None,
                in_channels=3, classes=self.num_classes, activation=None,
            )
            self.img_size = 224
        else:
            self.model = smp.Unet(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=3, classes=self.num_classes, activation=None,
            )
            self.img_size = 384

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def predict(self, image, threshold):
        img_np = np.array(image.convert("RGB"))
        orig_h, orig_w = img_np.shape[:2]

        img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
        img_norm = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        img_tensor = (torch.from_numpy(img_norm.transpose(2, 0, 1))
                      .unsqueeze(0).float().to(self.device))

        with torch.no_grad():
            output = self.model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        pred_mask_full = cv2.resize(
            pred_mask.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        detections = []
        for cls_id in range(1, self.num_classes):
            binary = (pred_mask_full == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    area = cv2.contourArea(cnt)
                    conf = min(1.0, area / (w * h * 0.8))
                    mid = cls_id - 1
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "score": conf,
                        "class_id": mid,
                        "class_name": (self.class_names[mid]
                                       if mid < len(self.class_names)
                                       else f"class_{mid}"),
                    })

        vis = img_np.copy()
        overlay = np.zeros_like(vis)
        seg_colors = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
        ]
        for c in range(1, min(self.num_classes, len(seg_colors))):
            if np.any(pred_mask_full == c):
                overlay[pred_mask_full == c] = seg_colors[c]
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        return detections, vis_bgr, pred_mask_full

    def get_model_for_gradcam(self):
        return self.model

    def preprocess_for_gradcam(self, image):
        img_np = np.array(image.convert("RGB"))
        img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
        img_norm = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return t.to(self.device)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
class ModelFactory:
    """Factory for model loading with auto-discovery from ``weights/``."""

    MODEL_PATTERNS = [
        ("detectron2", "Detectron2Wrapper"),
        ("rf-detr", "RFDETRWrapper"),
        ("rfdetr", "RFDETRWrapper"),
        ("yolospine", "YOLOspineWrapper"),
        ("ys", "YOLOspineWrapper"),
        ("modelv1", "ModelV1Wrapper"),
        ("unetplusplus", "SegmentationWrapper"),
        ("swinunet", "SegmentationWrapper"),
        ("transunet", "SegmentationWrapper"),
    ]

    DISPLAY_NAMES = {
        "yolo8": "YOLOv8m", "yolo8-n": "YOLOv8n",
        "yolo8-seg": "YOLOv8m-Seg", "yolo9": "YOLOv9m",
        "yolo10": "YOLOv10m", "yolo11": "YOLO11m",
        "yolo11-seg": "YOLO11m-Seg", "yolo12": "YOLO12m",
        "yolo12-x": "YOLO12x", "yolo26": "YOLO26m",
        "rtdetr-l": "RT-DETR-L", "rtdetr-l_50": "RT-DETR-L-50",
        "rtdetr-x": "RT-DETR-X", "rf-detr-base": "RF-DETR-Base",
        "unetplusplus": "UNet++", "swinunet": "SwinUNet",
        "transunet": "TransUNet", "yolospine": "YOLOspine",
        "ys": "YOLOspine-v2", "modelv1": "ModelV1",
        "detectron2_modelv1": "Detectron2-CascadeRCNN",
    }

    @staticmethod
    def _get_wrapper_type(model_name: str) -> str:
        name_lower = model_name.lower()
        for pattern, wrapper in ModelFactory.MODEL_PATTERNS:
            if pattern in name_lower:
                return wrapper
        return "UltralyticsWrapper"

    @staticmethod
    def _format_name(name: str) -> str:
        return ModelFactory.DISPLAY_NAMES.get(name.lower(), name.replace("_", "-"))

    @staticmethod
    def get_valid_models() -> Dict[str, Dict]:
        discovered = discover_models()
        valid: Dict[str, Dict] = {}
        for name, path in discovered.items():
            display = ModelFactory._format_name(name)
            valid[display] = {
                "path": str(path),
                "wrapper": ModelFactory._get_wrapper_type(name),
                "model_name": name,
            }
        return valid

    @staticmethod
    def load_model(model_name: str, model_config: Dict) -> ModelWrapper:
        wrapper_name = model_config["wrapper"]
        path = model_config["path"]
        original = model_config.get("model_name", "")
        wrappers = {
            "UltralyticsWrapper": lambda: UltralyticsWrapper(path),
            "RFDETRWrapper": lambda: RFDETRWrapper(path),
            "ModelV1Wrapper": lambda: ModelV1Wrapper(path),
            "YOLOspineWrapper": lambda: YOLOspineWrapper(path),
            "Detectron2Wrapper": lambda: Detectron2Wrapper(path),
            "SegmentationWrapper": lambda: SegmentationWrapper(path, original),
        }
        factory_fn = wrappers.get(wrapper_name)
        if not factory_fn:
            raise ValueError(f"Unknown wrapper: {wrapper_name}")
        return factory_fn()

    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        models = ModelFactory.get_valid_models()
        if model_name not in models:
            return {}
        cfg = models[model_name]
        is_seg = any(
            x in model_name.lower() for x in ["seg", "unet", "swin", "trans"]
        )
        return {
            "name": model_name,
            "path": cfg["path"],
            "exists": Path(cfg["path"]).exists(),
            "wrapper": cfg["wrapper"],
            "supports_gradcam": cfg["wrapper"] != "Detectron2Wrapper",
            "model_type": "segmentation" if is_seg else "detection",
        }


if __name__ == "__main__":
    models = ModelFactory.get_valid_models()
    print(f"Found {len(models)} models:")
    for name in sorted(models):
        info = ModelFactory.get_model_info(name)
        print(f"  {name:25s}  type={info['model_type']:15s}  "
              f"gradcam={info['supports_gradcam']}")
