"""
Configuration Module — SpineScan AI Platform
==============================================
Centralized configuration for the Streamlit demo application.
All paths are relative to the ``app/`` directory.
"""

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "weights"
RFDETR_WEIGHTS = BASE_DIR.parent / "rf-detr-base.pth"   # repo root fallback

RESULTS_DIR = BASE_DIR / "runs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_ROOT_ENV = os.getenv("METRICS_ROOT")
METRICS_ROOT = Path(METRICS_ROOT_ENV) if METRICS_ROOT_ENV else METRICS_DIR

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> str:
    import torch
    if os.getenv("FORCE_CPU", "").lower() == "true":
        return "cpu"
    if os.getenv("CUDA_VISIBLE_DEVICES") == "-1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Model Discovery
# ---------------------------------------------------------------------------
def discover_models() -> Dict[str, Path]:
    """Scan ``weights/`` for checkpoint files (.pt / .pth)."""
    models: Dict[str, Path] = {}
    if MODELS_DIR.exists():
        for ext in ("*.pt", "*.pth"):
            for f in MODELS_DIR.glob(ext):
                models[f.stem] = f
    if RFDETR_WEIGHTS.exists():
        models["rf-detr-base"] = RFDETR_WEIGHTS
    return models

# ---------------------------------------------------------------------------
# Class Configuration
# ---------------------------------------------------------------------------
CLASS_NAMES = ["DDD", "LDB", "Normal_IVD", "SS", "TDB", "SP"]

CLASS_FULL_NAMES = {
    "DDD": "Degenerative Disc Disease",
    "LDB": "Lumbar Disc Bulge",
    "Normal_IVD": "Normal Intervertebral Disc",
    "SS": "Spinal Stenosis",
    "TDB": "Thoracic Disc Bulge",
    "SP": "Spondylolisthesis",
}

CLASS_DESCRIPTIONS = {
    "DDD": "Loss of disc height and signal intensity indicating degeneration",
    "LDB": "Disc protrusion extending beyond vertebral body margin in lumbar region",
    "Normal_IVD": "Healthy intervertebral disc with preserved height and signal",
    "SS": "Narrowing of spinal canal compromising neural elements",
    "TDB": "Disc protrusion extending beyond vertebral body margin in thoracic region",
    "SP": "Forward displacement of a vertebral body relative to the one below",
}

CLASS_COLORS = {
    0: (255, 107, 107),   # DDD  — Red
    1: (78, 205, 196),    # LDB  — Teal
    2: (69, 183, 209),    # Normal_IVD — Blue
    3: (150, 206, 180),   # SS   — Green
    4: (255, 234, 167),   # TDB  — Yellow
    5: (221, 160, 221),   # SP   — Purple
}

CLASS_HEX_COLORS = {
    "DDD": "#ff6b6b",
    "LDB": "#4ecdc4",
    "Normal_IVD": "#45b7d1",
    "SS": "#96cea0",
    "TDB": "#ffea9f",
    "SP": "#dda0dd",
}

NUM_CLASSES = 6
NUM_SEG_CLASSES = 7  # 6 disorders + background

# ---------------------------------------------------------------------------
# Inference Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 384

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_setup() -> Dict[str, bool]:
    return {
        "base_dir": BASE_DIR.exists(),
        "models_dir": MODELS_DIR.exists(),
        "results_dir": RESULTS_DIR.exists(),
        "models_found": len(discover_models()) > 0,
    }


if __name__ == "__main__":
    print(f"Base:   {BASE_DIR}")
    print(f"Models: {MODELS_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Found:  {len(discover_models())} models")
    for n, p in discover_models().items():
        print(f"  {n}: {p}")
