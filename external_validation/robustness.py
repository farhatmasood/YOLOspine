"""
Robustness Testing Module
==========================
Evaluates model performance under realistic MRI degradation conditions.

Simulates:
    1. Gaussian Noise — thermal / electronic noise in MRI acquisition
    2. Rician Noise — magnitude-based MRI noise model
    3. Motion Artifacts — patient movement during scan
    4. Intensity Bias Field — RF coil inhomogeneity (B1 field)
    5. Gibbs Ringing — truncation artifacts from k-space
    6. Contrast Variation — different scan protocols

References:
    Gudbjartsson & Patz, "The Rician Distribution of Noisy MRI Data",
    MRM (1995).

    Sled et al., "A Nonparametric Method for Automatic Correction of
    Intensity Nonuniformity in MRI Data", TMI (1998).

Usage::

    from external_validation.robustness import (
        MRIDegradationSimulator, RobustnessEvaluator,
    )

    evaluator = RobustnessEvaluator(model.predict)
    results = evaluator.evaluate_robustness_suite(image_bgr)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------

@dataclass
class DegradationResult:
    """Container for a single degradation experiment."""

    original_image: np.ndarray
    degraded_image: np.ndarray
    degradation_type: str
    severity: float
    original_detections: List[Dict]
    degraded_detections: List[Dict]
    metrics: Dict


# ------------------------------------------------------------------
# MRI degradation simulator
# ------------------------------------------------------------------

class MRIDegradationSimulator:
    """
    Simulates realistic MRI degradation artifacts.

    Each method models a specific physical phenomenon:
    - Noise: thermal noise from receiver coils
    - Motion: patient / physiological movement
    - Bias field: B1 RF inhomogeneity
    - Gibbs: k-space truncation
    """

    # -- noise --------------------------------------------------------

    @staticmethod
    def add_gaussian_noise(
        image: np.ndarray, sigma: float = 25.0, seed: Optional[int] = None,
    ) -> np.ndarray:
        """Gaussian thermal / electronic noise (sigma in 0-255 scale)."""
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def add_rician_noise(
        image: np.ndarray, sigma: float = 20.0, seed: Optional[int] = None,
    ) -> np.ndarray:
        """Rician noise — the true magnitude-MRI noise model."""
        if seed is not None:
            np.random.seed(seed)
        f = image.astype(np.float32)
        nr = np.random.normal(0, sigma, image.shape)
        ni = np.random.normal(0, sigma, image.shape)
        return np.clip(np.sqrt((f + nr) ** 2 + ni ** 2), 0, 255).astype(np.uint8)

    # -- motion -------------------------------------------------------

    @staticmethod
    def add_motion_artifact(
        image: np.ndarray,
        severity: float = 0.3,
        direction: str = "horizontal",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate patient-movement–induced k-space phase errors."""
        if seed is not None:
            np.random.seed(seed)
        if image.ndim == 3:
            chs = cv2.split(image)
            return cv2.merge([
                MRIDegradationSimulator._motion_kspace(c, severity, direction)
                for c in chs
            ])
        return MRIDegradationSimulator._motion_kspace(image, severity, direction)

    @staticmethod
    def _motion_kspace(img: np.ndarray, severity: float, direction: str) -> np.ndarray:
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        h, w = img.shape
        n_lines = int(severity * (h if direction == "horizontal" else w) * 0.5)
        for _ in range(n_lines):
            if direction in ("horizontal", "random"):
                idx = np.random.randint(0, h)
                f[idx, :] *= np.exp(1j * np.random.uniform(-np.pi, np.pi))
            if direction in ("vertical", "random"):
                idx = np.random.randint(0, w)
                f[:, idx] *= np.exp(1j * np.random.uniform(-np.pi, np.pi))
        out = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
        out = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
        return out.astype(np.uint8)

    # -- bias field ---------------------------------------------------

    @staticmethod
    def add_intensity_bias_field(
        image: np.ndarray, severity: float = 0.4, seed: Optional[int] = None,
    ) -> np.ndarray:
        """B1 field inhomogeneity (second-order polynomial bias)."""
        if seed is not None:
            np.random.seed(seed)
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        y, x = (y - h / 2) / h, (x - w / 2) / w
        c = np.random.uniform(-1, 1, 6) * severity
        bias = np.clip(
            1 + c[0] * x + c[1] * y + c[2] * x * y
            + c[3] * x ** 2 + c[4] * y ** 2 + c[5] * (x ** 2 + y ** 2),
            0.5, 1.5,
        )
        if image.ndim == 3:
            bias = bias[:, :, np.newaxis]
        return np.clip(image.astype(np.float32) * bias, 0, 255).astype(np.uint8)

    # -- Gibbs ringing ------------------------------------------------

    @staticmethod
    def add_gibbs_ringing(image: np.ndarray, severity: float = 0.3) -> np.ndarray:
        """K-space truncation artifact."""
        if image.ndim == 3:
            return cv2.merge([
                MRIDegradationSimulator._gibbs_ch(c, severity)
                for c in cv2.split(image)
            ])
        return MRIDegradationSimulator._gibbs_ch(image, severity)

    @staticmethod
    def _gibbs_ch(img: np.ndarray, severity: float) -> np.ndarray:
        h, w = img.shape
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        cy, cx = h // 2, w // 2
        t = 1 - severity
        ry, rx = int(cy * t), int(cx * t)
        mask = np.zeros((h, w), np.float32)
        mask[cy - ry:cy + ry, cx - rx:cx + rx] = 1
        out = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
        return ((out / (out.max() + 1e-8)) * 255).astype(np.uint8)

    # -- contrast variation -------------------------------------------

    @staticmethod
    def add_contrast_variation(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
        """Simulate scan-to-scan contrast differences."""
        mean = image.mean()
        return np.clip(mean + (image.astype(np.float32) - mean) * factor, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Robustness evaluator
# ------------------------------------------------------------------

class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation framework.

    Tests model performance across multiple degradation types and severity
    levels to assess clinical deployability.
    """

    DEGRADATION_TYPES = {
        "gaussian_noise": {
            "function": MRIDegradationSimulator.add_gaussian_noise,
            "param_name": "sigma",
            "severities": [10, 25, 50, 75],
            "description": "Gaussian thermal noise (sigma)",
        },
        "rician_noise": {
            "function": MRIDegradationSimulator.add_rician_noise,
            "param_name": "sigma",
            "severities": [10, 20, 35, 50],
            "description": "Rician MRI noise (sigma)",
        },
        "motion_artifact": {
            "function": MRIDegradationSimulator.add_motion_artifact,
            "param_name": "severity",
            "severities": [0.1, 0.2, 0.3, 0.5],
            "description": "Motion artifact severity",
        },
        "intensity_bias": {
            "function": MRIDegradationSimulator.add_intensity_bias_field,
            "param_name": "severity",
            "severities": [0.2, 0.4, 0.6, 0.8],
            "description": "B1 bias field strength",
        },
        "gibbs_ringing": {
            "function": MRIDegradationSimulator.add_gibbs_ringing,
            "param_name": "severity",
            "severities": [0.1, 0.2, 0.3, 0.4],
            "description": "K-space truncation ratio",
        },
        "low_contrast": {
            "function": MRIDegradationSimulator.add_contrast_variation,
            "param_name": "factor",
            "severities": [0.5, 0.6, 0.7, 0.8],
            "description": "Contrast reduction factor",
        },
    }

    def __init__(self, model_predict_fn: Callable):
        """
        Args:
            model_predict_fn: ``(PIL.Image, threshold) -> (detections, vis, raw)``
        """
        self.predict_fn = model_predict_fn

    # ------------------------------------------------------------------

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / (union + 1e-8)

    def _compare(
        self, orig: List[Dict], deg: List[Dict], iou_thresh: float = 0.5,
    ) -> Dict:
        m: Dict = {
            "original_count": len(orig),
            "degraded_count": len(deg),
            "detection_retention": 0.0,
            "false_positives": 0,
            "false_negatives": 0,
            "avg_iou_matched": 0.0,
            "avg_confidence_drop": 0.0,
        }
        if not orig:
            m["false_positives"] = len(deg)
            return m
        if not deg:
            m["false_negatives"] = len(orig)
            return m

        used: set = set()
        matches = []
        for o in orig:
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(deg):
                if j in used or o["class_name"] != d["class_name"]:
                    continue
                v = self._iou(o["bbox"], d["bbox"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh and best_j >= 0:
                used.add(best_j)
                matches.append((o, deg[best_j], best_iou))

        m["detection_retention"] = len(matches) / len(orig)
        m["false_positives"] = len(deg) - len(used)
        m["false_negatives"] = len(orig) - len(matches)
        if matches:
            m["avg_iou_matched"] = float(np.mean([x[2] for x in matches]))
            m["avg_confidence_drop"] = float(
                np.mean([x[0]["score"] - x[1]["score"] for x in matches])
            )
        return m

    @staticmethod
    def _quality(original: np.ndarray, degraded: np.ndarray) -> Dict:
        out: Dict = {"psnr": 0.0, "ssim": 0.0}
        mse = np.mean((original.astype(np.float64) - degraded.astype(np.float64)) ** 2)
        out["psnr"] = 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
        if _SKIMAGE:
            try:
                channel_axis = -1 if original.ndim == 3 else None
                out["ssim"] = float(ssim(
                    original, degraded,
                    data_range=float(degraded.max() - degraded.min()),
                    channel_axis=channel_axis,
                ))
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        image: np.ndarray,
        degradation_type: str,
        severity: float,
        threshold: float = 0.3,
        seed: int = 42,
    ) -> DegradationResult:
        """Evaluate on one degraded image."""
        cfg = self.DEGRADATION_TYPES[degradation_type]
        fn = cfg["function"]
        pname = cfg["param_name"]

        kwargs = {pname: severity}
        if "seed" in fn.__code__.co_varnames:
            kwargs["seed"] = seed
        degraded = fn(image, **kwargs)

        pil_o = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_d = Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))

        od, _, _ = self.predict_fn(pil_o, threshold)
        dd, _, _ = self.predict_fn(pil_d, threshold)

        metrics = self._compare(od, dd)
        metrics.update(self._quality(image, degraded))

        return DegradationResult(
            original_image=image,
            degraded_image=degraded,
            degradation_type=degradation_type,
            severity=severity,
            original_detections=od,
            degraded_detections=dd,
            metrics=metrics,
        )

    def evaluate_robustness_suite(
        self,
        image: np.ndarray,
        threshold: float = 0.3,
        seed: int = 42,
    ) -> Dict:
        """Run all degradation types at all severity levels."""
        results: Dict = {"summary": {}, "detailed": {}}
        for dt, cfg in self.DEGRADATION_TYPES.items():
            detailed = []
            for sev in cfg["severities"]:
                try:
                    r = self.evaluate_single(image, dt, sev, threshold, seed)
                    detailed.append({
                        "severity": sev,
                        "metrics": r.metrics,
                        "original_count": len(r.original_detections),
                        "degraded_count": len(r.degraded_detections),
                    })
                except Exception as exc:
                    detailed.append({"severity": sev, "error": str(exc)})
            results["detailed"][dt] = detailed

            valid = [d for d in detailed if "metrics" in d]
            if valid:
                avg_ret = float(np.mean([d["metrics"]["detection_retention"] for d in valid]))
                avg_iou_vals = [d["metrics"]["avg_iou_matched"] for d in valid
                                if d["metrics"]["avg_iou_matched"] > 0]
                avg_iou = float(np.mean(avg_iou_vals)) if avg_iou_vals else 0.0
                results["summary"][dt] = {
                    "avg_detection_retention": avg_ret,
                    "avg_iou": avg_iou,
                    "description": cfg["description"],
                }
        return results
