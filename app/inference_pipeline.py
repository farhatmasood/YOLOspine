"""
Inference Pipeline - SpineScan AI Platform
============================================
Manages model inference for single/batch with GradCAM integration.
"""

import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Callable, Optional
import cv2

from model_factory import ModelWrapper
from gradcam import GradCAM, visualize_gradcam
from explainability import generate_focus_map, apply_heatmap


class InferenceManager:
    """Decouples inference logic from UI."""

    def __init__(self, model: ModelWrapper):
        self.model = model

    def predict_single(self, image: Image.Image, conf_threshold: float = 0.25,
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run inference with optional GradCAM.
        Returns dict with: detections, vis_image, raw_result, inference_time,
        and optionally gradcam_vis, gradcam_heatmap, focus_map.
        """
        options = options or {}
        t0 = time.time()
        detections, vis_image, raw_result = self.model.predict(image, conf_threshold)
        inference_time = time.time() - t0

        result = {
            "detections": detections,
            "vis_image": vis_image,
            "raw_result": raw_result,
            "inference_time": inference_time,
        }

        # Focus map (lightweight, always available)
        img_bgr = np.array(image)[:, :, ::-1].copy()
        heatmap = generate_focus_map(detections, img_bgr.shape)
        alpha = options.get("gradcam_alpha", 0.5)
        result["focus_map"] = apply_heatmap(img_bgr, heatmap, alpha=alpha)

        # GradCAM (optional)
        if options.get("enable_gradcam") and self.model.supports_gradcam:
            try:
                pytorch_model = self.model.get_model_for_gradcam()
                if pytorch_model is not None:
                    gradcam = GradCAM(pytorch_model)
                    img_tensor = self.model.preprocess_for_gradcam(image)
                    gc_heatmap, gc_meta = gradcam.generate(img_tensor)
                    gc_vis = visualize_gradcam(img_bgr, gc_heatmap, alpha=alpha)
                    result["gradcam_vis"] = gc_vis
                    result["gradcam_heatmap"] = gc_heatmap
                    result["gradcam_metadata"] = gc_meta
                    gradcam.remove_hooks()
            except Exception as e:
                result["gradcam_error"] = str(e)

        return result

    def predict_batch(self, image_paths: List[Path], conf_threshold: float = 0.25,
                      output_dir: Optional[Path] = None,
                      callback: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        """Batch inference. Returns DataFrame of all detections."""
        all_dets = []
        total = len(image_paths)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for i, img_path in enumerate(image_paths):
            try:
                if callback:
                    callback(i, total, f"Processing {img_path.name}...")
                image = Image.open(img_path).convert("RGB")
                t0 = time.time()
                dets, vis, _ = self.model.predict(image, conf_threshold)
                dt = time.time() - t0

                for d in dets:
                    rec = d.copy()
                    rec.update({'filename': img_path.name, 'inference_time': dt,
                                'image_width': image.width, 'image_height': image.height})
                    all_dets.append(rec)

                if output_dir and vis is not None:
                    cv2.imwrite(str(output_dir / f"pred_{img_path.name}"), vis)
            except Exception as e:
                if callback:
                    callback(i, total, f"Error: {img_path.name}: {e}")

        if callback:
            callback(total, total, "Complete")
        return pd.DataFrame(all_dets) if all_dets else pd.DataFrame()
