# SpineScan AI — Interactive Clinical Research Platform

<div align="center">

**A Streamlit-based demonstration application for the [YOLOspine](https://github.com/farhatmasood/YOLOspine) framework.**

Multi-model spinal disorder detection with GradCAM interpretability, MRI robustness testing, batch processing, and a research performance dashboard.

</div>

---

## Features

| Tab | Capability |
|:----|:-----------|
| **Clinical Inference** | Single-image diagnosis with real-time detection, automated radiological reports, and confidence-aware severity grading |
| **Interpretability** | GradCAM / LayerCAM attention visualisation with per-pathology alignment metrics and clinical validity scoring |
| **Robustness** | MRI degradation simulation (6 corruption types) with detection-retention analysis and radar profiling |
| **Batch Processing** | Directory-based inference with progress tracking, class-distribution charts, and CSV export |
| **Research Dashboard** | Model registry, per-model training curves, mAP comparison charts, and class reference |

---

## Quick Start

### 1. Install Dependencies

```bash
# From the repository root
conda activate YS                       # or your environment
pip install -r requirements.txt         # base dependencies
pip install -r app/requirements.txt     # streamlit + plotly
```

### 2. Add Model Weights

Place `.pt` / `.pth` checkpoint files into `app/weights/`:

```
app/
└── weights/
    ├── yolospine.pt          # YOLOspine (recommended)
    ├── yolo12.pt             # YOLOv12m baseline
    ├── rtdetr-l.pt           # RT-DETR-L
    └── ...                   # any supported architecture
```

> **Note:** Weight files are excluded from version control via `.gitignore`. Download trained checkpoints from the links provided in the main [README](../README.md) or train your own using `tools/train.py`.

### 3. (Optional) Add Training Metrics

Place per-model CSV files into `app/metrics/` for the Research Dashboard:

```
app/
└── metrics/
    ├── yolo8.csv
    ├── yolo12.csv
    └── ...
```

### 4. Launch

```bash
cd app/
streamlit run app.py
```

The application opens at **http://localhost:8501**.

---

## Supported Architectures

The model factory auto-discovers weights by filename pattern:

| Pattern | Architecture | GradCAM |
|:--------|:-------------|:-------:|
| `yolo8*.pt` — `yolo26*.pt` | YOLO v8–v26 (Ultralytics) | ✅ |
| `yolo*-seg.pt` | YOLO Segmentation | ✅ |
| `rtdetr-*.pt` | RT-DETR (Ultralytics) | ✅ |
| `rf-detr*.pth` | RF-DETR | ✅ |
| `yolospine.pt`, `ys.pt` | YOLOspine (custom / Ultralytics) | ✅ |
| `unetplusplus.pt` | UNet++ (SMP) | ✅ |
| `swinunet.pt` | SwinUNet (SMP) | ✅ |
| `transunet.pt` | TransUNet (SMP) | ✅ |
| `detectron2*.pt` | Detectron2 Cascade R-CNN | ❌ |

---

## Directory Structure

```
app/
├── app.py                  # Streamlit entry point (5-tab UI)
├── config.py               # Centralized paths, classes, device
├── model_factory.py        # Auto-discovery model loading factory
├── inference_pipeline.py   # Single / batch inference manager
├── gradcam.py              # GradCAM interpretability engine
├── explainability.py       # Gaussian focus-map visualisation
├── robustness.py           # MRI degradation simulator (6 types)
├── style.css               # Glassmorphism dark academic theme
├── requirements.txt        # App-specific dependencies
├── README.md               # This file
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Plotly chart generators
├── weights/                # Model checkpoints (git-ignored)
│   └── .gitkeep
├── metrics/                # Training CSV files (optional)
│   └── .gitkeep
└── runs/                   # Inference output (git-ignored)
```

---

## Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `FORCE_CPU` | `false` | Force CPU even when CUDA is available |
| `CUDA_VISIBLE_DEVICES` | — | Standard CUDA device selector |
| `METRICS_ROOT` | `app/metrics/` | Override metrics CSV directory |

---

## Citation

If you use this application in your research, please cite the YOLOspine paper:

```bibtex
@article{masood2026yolospine,
  title   = {A Clinically Validated Hierarchical Two-Stage Attention-Enhanced
             Architecture for Multi-Label Overlapping Spinal Disorder Detection},
  author  = {Masood, Rao Farhat and Taj, Imtiaz Ahmad},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026}
}
```

---

## License

Released under the [MIT License](../LICENSE).
