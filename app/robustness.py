"""
Robustness Testing Module
==========================
MRI degradation simulation and model robustness evaluation.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from PIL import Image
import torch

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


@dataclass
class DegradationResult:
    original_image: np.ndarray
    degraded_image: np.ndarray
    degradation_type: str
    severity: float
    original_detections: List[Dict]
    degraded_detections: List[Dict]
    metrics: Dict


class MRIDegradationSimulator:
    """Simulates realistic MRI degradation artifacts."""

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def add_rician_noise(image: np.ndarray, sigma: float = 20.0, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        img_f = image.astype(np.float32)
        nr = np.random.normal(0, sigma, image.shape)
        ni = np.random.normal(0, sigma, image.shape)
        return np.clip(np.sqrt((img_f + nr) ** 2 + ni ** 2), 0, 255).astype(np.uint8)

    @staticmethod
    def add_motion_artifact(image: np.ndarray, severity: float = 0.3,
                            direction: str = 'horizontal', seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        if len(image.shape) == 3:
            chs = cv2.split(image)
            return cv2.merge([MRIDegradationSimulator._motion_kspace(c, severity, direction, seed) for c in chs])
        return MRIDegradationSimulator._motion_kspace(image, severity, direction, seed)

    @staticmethod
    def _motion_kspace(image: np.ndarray, severity: float, direction: str, seed) -> np.ndarray:
        f = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
        h, w = image.shape
        n_lines = int(severity * (h if direction == 'horizontal' else w) * 0.5)
        if direction in ('horizontal', 'random'):
            for _ in range(n_lines):
                idx = np.random.randint(0, h)
                f[idx, :] *= np.exp(1j * np.random.uniform(-np.pi, np.pi))
        if direction in ('vertical', 'random'):
            for _ in range(n_lines):
                idx = np.random.randint(0, w)
                f[:, idx] *= np.exp(1j * np.random.uniform(-np.pi, np.pi))
        result = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        return (result * 255).astype(np.uint8)

    @staticmethod
    def add_intensity_bias_field(image: np.ndarray, severity: float = 0.4,
                                 seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        y, x = (y - h / 2) / h, (x - w / 2) / w
        c = np.random.uniform(-1, 1, 6) * severity
        bias = np.clip(1 + c[0]*x + c[1]*y + c[2]*x*y + c[3]*x**2 + c[4]*y**2 + c[5]*(x**2+y**2), 0.5, 1.5)
        if len(image.shape) == 3:
            bias = bias[:, :, np.newaxis]
        return np.clip(image.astype(np.float32) * bias, 0, 255).astype(np.uint8)

    @staticmethod
    def add_gibbs_ringing(image: np.ndarray, severity: float = 0.3) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.merge([MRIDegradationSimulator._gibbs_ch(c, severity) for c in cv2.split(image)])
        return MRIDegradationSimulator._gibbs_ch(image, severity)

    @staticmethod
    def _gibbs_ch(image: np.ndarray, severity: float) -> np.ndarray:
        h, w = image.shape
        f = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
        cy, cx = h // 2, w // 2
        t = 1 - severity
        ry, rx = int(cy * t), int(cx * t)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[cy-ry:cy+ry, cx-rx:cx+rx] = 1
        result = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
        return (result / (result.max() + 1e-8) * 255).astype(np.uint8) if result.max() > 0 else np.zeros_like(image)

    @staticmethod
    def add_contrast_variation(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
        mean = image.mean()
        return np.clip(mean + (image.astype(np.float32) - mean) * factor, 0, 255).astype(np.uint8)


class RobustnessEvaluator:
    """Robustness evaluation across degradation types and severities."""

    DEGRADATION_TYPES = {
        'gaussian_noise': {
            'function': MRIDegradationSimulator.add_gaussian_noise,
            'param_name': 'sigma',
            'severities': [10, 25, 50, 75],
            'description': 'Gaussian thermal noise',
            'label': 'Gaussian Noise',
        },
        'rician_noise': {
            'function': MRIDegradationSimulator.add_rician_noise,
            'param_name': 'sigma',
            'severities': [10, 20, 35, 50],
            'description': 'Rician MRI noise',
            'label': 'Rician Noise (MRI)',
        },
        'motion_artifact': {
            'function': MRIDegradationSimulator.add_motion_artifact,
            'param_name': 'severity',
            'severities': [0.1, 0.2, 0.3, 0.5],
            'description': 'Motion artifact severity',
            'label': 'Patient Motion',
        },
        'intensity_bias': {
            'function': MRIDegradationSimulator.add_intensity_bias_field,
            'param_name': 'severity',
            'severities': [0.2, 0.4, 0.6, 0.8],
            'description': 'B1 bias field strength',
            'label': 'Bias Field',
        },
        'gibbs_ringing': {
            'function': MRIDegradationSimulator.add_gibbs_ringing,
            'param_name': 'severity',
            'severities': [0.1, 0.2, 0.3, 0.4],
            'description': 'K-space truncation ratio',
            'label': 'Gibbs Ringing',
        },
        'low_contrast': {
            'function': MRIDegradationSimulator.add_contrast_variation,
            'param_name': 'factor',
            'severities': [0.5, 0.6, 0.7, 0.8],
            'description': 'Contrast reduction factor',
            'label': 'Low Contrast',
        },
    }

    def __init__(self, model_predict_fn: Callable):
        self.predict_fn = model_predict_fn

    def evaluate_single(self, image: np.ndarray, degradation_type: str,
                        severity: float, threshold: float = 0.3, seed: int = 42) -> DegradationResult:
        if degradation_type not in self.DEGRADATION_TYPES:
            raise ValueError(f"Unknown degradation: {degradation_type}")

        cfg = self.DEGRADATION_TYPES[degradation_type]
        fn = cfg['function']
        pn = cfg['param_name']

        kwargs = {pn: severity}
        if 'seed' in fn.__code__.co_varnames:
            kwargs['seed'] = seed
        degraded = fn(image, **kwargs)

        pil_orig = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_deg = Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))

        orig_dets, _, _ = self.predict_fn(pil_orig, threshold)
        deg_dets, _, _ = self.predict_fn(pil_deg, threshold)

        metrics = self._compare(orig_dets, deg_dets)
        metrics.update(self._quality(image, degraded))

        return DegradationResult(
            original_image=image, degraded_image=degraded,
            degradation_type=degradation_type, severity=severity,
            original_detections=orig_dets, degraded_detections=deg_dets,
            metrics=metrics
        )

    def _quality(self, original: np.ndarray, degraded: np.ndarray) -> Dict:
        m = {'psnr': 0.0, 'ssim': 0.0}
        try:
            mse = np.mean((original.astype(np.float64) - degraded.astype(np.float64)) ** 2)
            m['psnr'] = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        except Exception:
            pass
        if SKIMAGE_AVAILABLE:
            try:
                axis = -1 if original.ndim == 3 else None
                m['ssim'] = ssim(original, degraded, data_range=degraded.max() - degraded.min(), channel_axis=axis)
            except Exception:
                pass
        return m

    def evaluate_robustness_suite(self, image: np.ndarray, threshold: float = 0.3, seed: int = 42) -> Dict:
        results = {'summary': {}, 'detailed': {}}
        for deg_type, cfg in self.DEGRADATION_TYPES.items():
            results['detailed'][deg_type] = []
            for sev in cfg['severities']:
                try:
                    r = self.evaluate_single(image, deg_type, sev, threshold, seed)
                    results['detailed'][deg_type].append({
                        'severity': sev, 'metrics': r.metrics,
                        'original_count': len(r.original_detections),
                        'degraded_count': len(r.degraded_detections),
                    })
                except Exception as e:
                    results['detailed'][deg_type].append({'severity': sev, 'error': str(e)})

            valid = [r for r in results['detailed'][deg_type] if 'metrics' in r]
            if valid:
                avg_ret = np.mean([r['metrics']['detection_retention'] for r in valid])
                ious = [r['metrics']['avg_iou_matched'] for r in valid if r['metrics']['avg_iou_matched'] > 0]
                avg_iou = np.mean(ious) if ious else 0.0
                results['summary'][deg_type] = {
                    'avg_detection_retention': float(avg_ret),
                    'avg_iou': float(avg_iou) if not np.isnan(avg_iou) else 0.0,
                    'description': cfg['description'],
                }
        return results

    def _compare(self, orig: List[Dict], deg: List[Dict], iou_thresh: float = 0.5) -> Dict:
        metrics = {
            'original_count': len(orig), 'degraded_count': len(deg),
            'detection_retention': 0.0, 'false_positives': 0,
            'false_negatives': 0, 'avg_iou_matched': 0.0, 'avg_confidence_drop': 0.0,
        }
        if not orig:
            metrics['false_positives'] = len(deg)
            return metrics
        if not deg:
            metrics['false_negatives'] = len(orig)
            return metrics

        matched_d = set()
        matches = []
        for i, o in enumerate(orig):
            best_iou, best_j = 0, -1
            for j, d in enumerate(deg):
                if j in matched_d or o['class_name'] != d['class_name']:
                    continue
                iou = self._iou(o['bbox'], d['bbox'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched_d.add(best_j)
                matches.append({'iou': best_iou, 'conf_drop': o['score'] - deg[best_j]['score']})

        metrics['detection_retention'] = len(matches) / len(orig)
        metrics['false_positives'] = len(deg) - len(matched_d)
        metrics['false_negatives'] = len(orig) - len(matches)
        if matches:
            metrics['avg_iou_matched'] = np.mean([m['iou'] for m in matches])
            metrics['avg_confidence_drop'] = np.mean([m['conf_drop'] for m in matches])
        return metrics

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-8)


def generate_robustness_visualization(original: np.ndarray, results: Dict,
                                      evaluator: RobustnessEvaluator) -> np.ndarray:
    """Grid visualization of degradation effects."""
    h, w = original.shape[:2]
    samples, labels = [], []
    for deg_type, cfg in evaluator.DEGRADATION_TYPES.items():
        sev = cfg['severities'][1]
        fn, pn = cfg['function'], cfg['param_name']
        try:
            kwargs = {pn: sev}
            if 'seed' in fn.__code__.co_varnames:
                kwargs['seed'] = 42
            samples.append(fn(original, **kwargs))
            labels.append(f"{deg_type} ({pn}={sev})")
        except Exception:
            continue

    n = len(samples) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    ch, cw = h // 2, w // 2
    grid = np.ones((rows * (ch + 40), cols * (cw + 10), 3), dtype=np.uint8) * 30

    thumb = cv2.resize(original, (cw, ch))
    grid[20:20+ch, 5:5+cw] = thumb
    cv2.putText(grid, "Original", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for i, (s, l) in enumerate(zip(samples, labels)):
        r, c = (i + 1) // cols, (i + 1) % cols
        y, x = r * (ch + 40) + 20, c * (cw + 10) + 5
        t = cv2.resize(s, (cw, ch))
        if y + ch <= grid.shape[0] and x + cw <= grid.shape[1]:
            grid[y:y+ch, x:x+cw] = t
            cv2.putText(grid, l.split('(')[0].strip(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return grid
