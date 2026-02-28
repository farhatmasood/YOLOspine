"""
SpineScan AI — Clinical Research Platform
==========================================
Interactive Streamlit application for multi-model spinal disorder
detection with GradCAM interpretability, MRI robustness testing,
batch processing, and a research performance dashboard.

Part of the YOLOspine repository:
    https://github.com/farhatmasood/YOLOspine

Usage:
    cd app/
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import os
import time
import cv2
import torch
from datetime import datetime
from typing import Dict, Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import config
from config import CLASS_NAMES, CLASS_FULL_NAMES, CLASS_DESCRIPTIONS, CLASS_HEX_COLORS, METRICS_ROOT
from model_factory import ModelFactory, ModelWrapper
from inference_pipeline import InferenceManager
from gradcam import GradCAM, generate_attention_report, visualize_gradcam
from explainability import generate_focus_map, apply_heatmap
from robustness import RobustnessEvaluator, MRIDegradationSimulator
from utils.visualization import (
    plot_map_comparison, plot_class_distribution,
    plot_robustness_radar, plot_metrics_csv,
)

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="SpineScan AI",
    page_icon="🦴",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================
_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ============================================================================
# Session State Defaults
# ============================================================================
_defaults = {
    "model_loaded": False, "last_model": None, "show_browse_help": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# Header
# ============================================================================
st.markdown("""
<div class="main-header animate-in">
    <h1 class="main-title">SpineScan AI</h1>
    <p class="subtitle">Clinical Research Platform for Spinal Disorder Detection & Analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="icon">🦴</div>
        <h3>SpineScan AI</h3>
        <div class="version">Academic Research v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Model Selection ----
    with st.expander("🤖 Model Configuration", expanded=True):
        available_models = ModelFactory.get_valid_models()
        if not available_models:
            st.error("No models found in weights/ — see app/README.md")
            st.stop()

        det_models = [m for m in available_models if not any(
            x in m.lower() for x in ["seg", "unet", "swin", "trans"])]
        seg_models = [m for m in available_models if m not in det_models]

        task_type = st.radio("Task Type", ["Detection", "Segmentation"], horizontal=True)
        model_options = det_models if task_type == "Detection" else seg_models
        if not model_options:
            model_options = list(available_models.keys())

        selected_model_name = st.selectbox("Architecture", sorted(model_options), index=0)
        selected_model_cfg = available_models[selected_model_name]

        model_info = ModelFactory.get_model_info(selected_model_name)
        gc_badge = "GradCAM" if model_info.get("supports_gradcam") else "Standard"
        gc_color = "#10b981" if model_info.get("supports_gradcam") else "#64748b"
        st.markdown(f"""
        <div style="margin-top:0.5rem;padding:0.6rem;background:rgba(0,0,0,0.2);border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;">Checkpoint</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#e2e8f0;word-break:break-all;margin-top:2px;">{Path(model_info.get('path','')).name}</div>
            <div style="margin-top:0.4rem;"><span style="padding:2px 8px;border-radius:99px;font-size:0.7rem;font-weight:600;background:rgba(16,185,129,0.12);color:{gc_color};">{gc_badge}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Inference Settings ----
    with st.expander("⚙️ Inference Settings", expanded=True):
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
        st.markdown("---")
        st.markdown("<p style='font-size:0.85rem;font-weight:600;color:#e2e8f0;'>Interpretability</p>", unsafe_allow_html=True)
        enable_gradcam = st.checkbox("Enable GradCAM", value=True,
                                     disabled=not model_info.get("supports_gradcam"))
        gradcam_alpha = 0.5
        if enable_gradcam and model_info.get("supports_gradcam"):
            gradcam_alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.5, 0.1)

    # ---- Device Badge ----
    is_gpu = torch.cuda.is_available()
    dev_name = f"GPU ({torch.cuda.get_device_name(0)})" if is_gpu else "CPU"
    dev_cls = "gpu" if is_gpu else "cpu"
    st.markdown(f"""
    <div class="device-badge">
        <div class="indicator {dev_cls}"></div>
        <div>
            <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;">Device</div>
            <div style="font-size:0.82rem;font-weight:600;color:{'#10b981' if is_gpu else '#f59e0b'};">{dev_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Model Loading (cached)
# ============================================================================
@st.cache_resource
def _load_model(name: str, cfg: Dict):
    try:
        return ModelFactory.load_model(name, cfg)
    except Exception as e:
        return str(e)

model: Optional[ModelWrapper] = None
manager: Optional[InferenceManager] = None

with st.spinner(f"Loading {selected_model_name}..."):
    result = _load_model(selected_model_name, selected_model_cfg)
    if isinstance(result, str):
        st.sidebar.error(f"Load Error: {result[:80]}")
    else:
        model = result
        manager = InferenceManager(model)
        st.sidebar.success("Model Ready")

if model is None:
    st.warning("Please select a model to continue.")
    st.stop()

# ============================================================================
# Tabs
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Clinical Inference",
    "🧠 Interpretability",
    "🛡️ Robustness",
    "📁 Batch Processing",
    "📊 Research Dashboard",
])

# ============================================================================
# Tab 1: Clinical Inference
# ============================================================================
with tab1:
    st.markdown("### 🔬 Clinical Single-Image Diagnosis")

    col_up, col_tip = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload MRI (Sagittal)", type=["png", "jpg", "jpeg", "dcm"],
                                    key="tab1_upload")
    with col_tip:
        st.markdown("""
        <div class="info-banner">
            <strong>Tip:</strong> Use mid-sagittal T2-weighted lumbar spine MRI for optimal detection
            of disc disease, stenosis, and bulges.
        </div>
        """, unsafe_allow_html=True)

    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        img_bgr = np.array(image)[:, :, ::-1].copy()

        with st.spinner("Analyzing spinal structures..."):
            res = manager.predict_single(image, conf_threshold, {
                "enable_gradcam": enable_gradcam,
                "gradcam_alpha": gradcam_alpha,
            })
            detections = res["detections"]
            vis_image = res["vis_image"]
            inf_time = res["inference_time"]
            focus_vis = res.get("focus_map", img_bgr)

        # Metrics row
        n_path = len([d for d in detections if d['class_name'] != 'Normal_IVD'])
        avg_conf = np.mean([d['score'] for d in detections]) if detections else 0
        mc = st.columns(4)
        mc[0].metric("Inference Time", f"{inf_time*1000:.0f} ms")
        mc[1].metric("Total Detections", len(detections))
        mc[2].metric("Pathologies", n_path)
        mc[3].metric("Avg Confidence", f"{avg_conf:.1%}")

        # Visual row
        st.markdown("#### Visual Analysis")
        vc = st.columns(3)
        with vc[0]:
            st.markdown("**Original Scan**")
            st.image(image, width='stretch')
        with vc[1]:
            st.markdown(f"**AI Predictions** ({len(detections)})")
            # vis_image could be BGR or RGB depending on wrapper
            if vis_image is not None:
                if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
                    st.image(vis_image[:, :, ::-1], width='stretch')
                else:
                    st.image(vis_image, width='stretch')
        with vc[2]:
            if "gradcam_vis" in res:
                st.markdown("**GradCAM Attention**")
                st.image(res["gradcam_vis"][:, :, ::-1], width='stretch')
            else:
                st.markdown("**Pathology Focus**")
                st.image(focus_vis[:, :, ::-1], width='stretch')

        # Findings
        st.markdown("---")
        dc = st.columns([1, 1])

        with dc[0]:
            st.markdown("#### Detected Pathologies")
            if detections:
                for det in detections:
                    conf = det['score']
                    sev = "high" if conf > 0.7 else "med" if conf > 0.4 else "low"
                    icon = {"high": "🔴", "med": "🟡", "low": "🟢"}[sev]
                    full = CLASS_FULL_NAMES.get(det['class_name'], det['class_name'])
                    st.markdown(f"""
                    <div class="det-card severity-{sev}">
                        <div>
                            <div class="name">{icon} {full}</div>
                            <div class="meta">Conf: {conf:.1%} &bull; Y: {int(det['bbox'][1])}-{int(det['bbox'][3])}</div>
                        </div>
                        <div class="det-badge {sev}">{sev.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-panel">
                    <h4>No Pathology Detected</h4>
                    <p style="color:#94a3b8;">No findings above the current confidence threshold.</p>
                </div>
                """, unsafe_allow_html=True)

        with dc[1]:
            st.markdown("#### 📄 Radiological Report")
            now = datetime.now().strftime("%B %d, %Y")
            st.markdown(f"""
            <div class="clinical-report">
                <div class="report-header">
                    <div>
                        <h3 style="margin:0;font-size:1.1rem;">SpineScan AI Analysis</h3>
                        <p style="margin:2px 0 0;font-size:0.8rem;color:#94a3b8;">Automated Pathological Detection</p>
                    </div>
                    <div style="text-align:right;font-size:0.82rem;color:#cbd5e1;">
                        <div><strong>Date:</strong> {now}</div>
                        <div><strong>Model:</strong> {selected_model_name}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if detections:
                for i, det in enumerate(detections, 1):
                    conf = det['score']
                    sev = "high" if conf > 0.7 else "med" if conf > 0.4 else "low"
                    full = CLASS_FULL_NAMES.get(det['class_name'], det['class_name'])
                    desc = CLASS_DESCRIPTIONS.get(det['class_name'], "")
                    st.markdown(f"""
                    <div class="report-finding {sev}">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-weight:500;color:#f1f5f9;">Finding {i}: {full}</span>
                            <span style="font-size:0.82rem;color:#94a3b8;">{conf:.1%}</span>
                        </div>
                        <div style="font-size:0.78rem;color:#64748b;margin-top:3px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                pathologies = sorted(set(d['class_name'] for d in detections if d['class_name'] != 'Normal_IVD'))
                if pathologies:
                    full_names = [CLASS_FULL_NAMES.get(p, p) for p in pathologies]
                    st.markdown(f"""
                    <div style="margin-top:1rem;">
                        <h4 style="color:#a78bfa;margin-bottom:0.5rem;font-size:0.95rem;">Impression</h4>
                        <p style="color:#e2e8f0;font-size:0.88rem;">Imaging features suggestive of {', '.join(full_names)}.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p style="color:#94a3b8;text-align:center;padding:1rem;">
                    Normal appearance of examined spinal segments within analysis limitations.
                </p>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:1.5rem;padding-top:0.75rem;border-top:1px solid #334155;font-size:0.75rem;color:#64748b;">
                <strong>Disclaimer:</strong> AI research tool — does NOT constitute medical diagnosis. All findings must be verified by a qualified radiologist.
            </div></div>
            """, unsafe_allow_html=True)

# ============================================================================
# Tab 2: Interpretability
# ============================================================================
with tab2:
    st.markdown("### 🧠 Interpretability Analysis")

    with st.expander("ℹ️ About GradCAM & Explainable AI", expanded=False):
        st.markdown("""
        **Gradient-weighted Class Activation Mapping (GradCAM)** visualizes which image regions
        drive model predictions. **Green** = low influence, **Red** = high influence.

        Clinical Relevance:
        - Verifies model focuses on spine/disc anatomy, not artifacts
        - Highlights regions of interest even at low confidence
        - Supports regulatory compliance for explainable AI
        """)

    if not model_info.get("supports_gradcam"):
        st.markdown("""
        <div class="warning-panel">
            <strong>⚠️ Feature Unavailable</strong><br>
            Selected model does not support GradCAM. Choose a compatible architecture (YOLO, RT-DETR, RF-DETR, or segmentation models).
        </div>
        """, unsafe_allow_html=True)
    else:
        gc_upload = st.file_uploader("Upload MRI for Analysis", type=["png", "jpg", "jpeg"],
                                     key="gc_upload")
        if gc_upload and model:
            image = Image.open(gc_upload).convert("RGB")
            img_bgr = np.array(image)[:, :, ::-1].copy()

            with st.spinner("Generating attention maps..."):
                detections, vis_image, raw = model.predict(image, conf_threshold)
                gradcam_vis = img_bgr.copy()
                report = None
                gc_error = None

                try:
                    pm = model.get_model_for_gradcam()
                    if pm:
                        gc = GradCAM(pm)
                        tensor = model.preprocess_for_gradcam(image)
                        heatmap, meta = gc.generate(tensor)
                        gradcam_vis = visualize_gradcam(img_bgr, heatmap, alpha=gradcam_alpha)
                        report = generate_attention_report(detections, heatmap, img_bgr.shape[:2])
                        gc.remove_hooks()
                except Exception as e:
                    gc_error = str(e)

            # Visualization
            st.markdown("#### Visual Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Input**")
                st.image(image, width='stretch')
            with c2:
                st.markdown("**Model Attention (GradCAM)**")
                st.image(gradcam_vis[:, :, ::-1], width='stretch')

            if gc_error:
                st.warning(f"GradCAM encountered an issue: {gc_error}")

            if report:
                st.markdown("---")
                st.markdown("#### 📊 Attention Metrics")
                mc = st.columns(4)
                mc[0].metric("Detections", report['total_detections'])
                mc[1].metric("Focus Ratio", f"{report['pathology_focus_ratio']:.1%}",
                             help="Attention within bounding boxes")
                mc[2].metric("Background Noise", f"{report['background_attention']:.1%}",
                             help="Attention outside relevant regions")
                vs = report['clinical_validity_score']
                mc[3].metric("Validity Score", f"{vs:.2f}",
                             delta="High" if vs > 0.6 else "Low")

                if report['attention_alignment_scores']:
                    st.markdown("**Per-Pathology Alignment**")
                    rows = []
                    for item in report['attention_alignment_scores']:
                        rows.append({
                            "Pathology": CLASS_FULL_NAMES.get(item['class_name'], item['class_name']),
                            "Confidence": f"{item['score']:.1%}",
                            "Attention Alignment": f"{item['alignment_score']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ============================================================================
# Tab 3: Robustness Testing
# ============================================================================
with tab3:
    st.markdown("### 🛡️ Clinical Robustness Testing")

    st.markdown("""
    <div class="info-banner">
        <strong>Clinical Significance:</strong> MRI scans often suffer from acquisition artifacts.
        This module simulates realistic degradations to evaluate diagnostic stability.
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 2])

    with rc1:
        st.markdown("**Degradation Parameters**")
        deg_options = list(RobustnessEvaluator.DEGRADATION_TYPES.keys())
        deg_labels = {k: v.get('label', k.replace('_', ' ').title())
                      for k, v in RobustnessEvaluator.DEGRADATION_TYPES.items()}
        deg_type = st.selectbox("Artifact Type", deg_options,
                                format_func=lambda x: deg_labels[x])
        deg_cfg = RobustnessEvaluator.DEGRADATION_TYPES[deg_type]
        severity = st.select_slider(f"Severity ({deg_cfg['param_name']})",
                                    options=deg_cfg['severities'],
                                    value=deg_cfg['severities'][1])

    with rc2:
        st.markdown("**Input Source**")
        rob_upload = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"],
                                      key="rob_upload")

    if rob_upload and model:
        image = Image.open(rob_upload).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Apply degradation
        fn = deg_cfg['function']
        pn = deg_cfg['param_name']
        kwargs = {pn: severity}
        if 'seed' in fn.__code__.co_varnames:
            kwargs['seed'] = 42
        degraded = fn(img_bgr, **kwargs)

        st.markdown("---")
        st.markdown("#### Stability Analysis")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Original Scan**")
            st.image(image, width='stretch')
            with st.spinner("Analyzing original..."):
                orig_dets, orig_vis, _ = model.predict(image, conf_threshold)
            if orig_vis is not None:
                st.image(orig_vis[:, :, ::-1], width='stretch',
                         caption=f"Baseline ({len(orig_dets)} detections)")

        with c2:
            st.markdown(f"**Degraded** ({deg_labels[deg_type]})")
            deg_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
            st.image(deg_rgb, width='stretch')
            deg_pil = Image.fromarray(deg_rgb)
            with st.spinner("Analyzing degraded..."):
                deg_dets, deg_vis, _ = model.predict(deg_pil, conf_threshold)
            if deg_vis is not None:
                st.image(deg_vis[:, :, ::-1], width='stretch',
                         caption=f"Degraded ({len(deg_dets)} detections)")

        # Metrics
        st.markdown("#### Performance Impact")
        retention = len(deg_dets) / len(orig_dets) if orig_dets else (1.0 if not deg_dets else 0.0)
        conf_drop = 0.0
        if orig_dets and deg_dets:
            conf_drop = np.mean([d['score'] for d in orig_dets]) - np.mean([d['score'] for d in deg_dets])

        # Quality metrics
        qual = {'psnr': 0.0, 'ssim': 0.0}
        try:
            evaluator_tmp = RobustnessEvaluator(model.predict)
            qual = evaluator_tmp._quality(img_bgr, degraded)
        except Exception:
            pass

        qc = st.columns(5)
        qc[0].metric("Detections", f"{len(orig_dets)} → {len(deg_dets)}",
                      delta=f"{len(deg_dets) - len(orig_dets)}")
        qc[1].metric("Retention", f"{retention:.1%}",
                      delta="Stable" if retention >= 0.8 else "Degraded")
        qc[2].metric("Conf Drop", f"{conf_drop:.1%}",
                      delta=f"{-conf_drop:.1%}", delta_color="inverse")
        qc[3].metric("PSNR", f"{qual['psnr']:.1f} dB")
        qc[4].metric("SSIM", f"{qual['ssim']:.3f}")

        # Full suite button
        st.markdown("---")
        if st.button("🔬 Run Full Robustness Suite", type="primary"):
            with st.spinner("Running comprehensive stress test..."):
                evaluator = RobustnessEvaluator(model.predict)
                suite = evaluator.evaluate_robustness_suite(img_bgr, conf_threshold)

            st.success("Robustness profiling complete")

            # Summary table
            rows = []
            for dt, stats in suite['summary'].items():
                rows.append({
                    'Degradation': dt.replace('_', ' ').title(),
                    'Retention': f"{stats['avg_detection_retention']:.1%}",
                    'IoU Stability': f"{stats['avg_iou']:.3f}",
                    'Description': stats['description'],
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            # Radar chart
            if suite['summary']:
                st.plotly_chart(plot_robustness_radar(suite['summary']), width='stretch')

# ============================================================================
# Tab 4: Batch Processing
# ============================================================================
with tab4:
    st.markdown("### 📁 Batch Inference Engine")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("**📂 Input Directory**")
    pc = st.columns([4, 1])
    with pc[0]:
        folder_path = st.text_input("Path", value="", placeholder="Paste folder path containing images...",
                                    label_visibility="collapsed")
    with pc[1]:
        if st.button("📁 Suggestions"):
            st.session_state['show_browse_help'] = not st.session_state.get('show_browse_help', False)

    if st.session_state.get('show_browse_help'):
        parent = Path(config.BASE_DIR).parent
        suggestions = [
            parent / "train_images", parent / "test_images",
            Path(config.BASE_DIR) / "Data_test",
        ]
        existing = [str(s) for s in suggestions if s.exists()]
        if existing:
            st.markdown("**Quick picks:**")
            for sp in existing:
                if st.button(f"📁 {sp}", key=f"qp_{sp}"):
                    folder_path = sp

    if folder_path and os.path.isdir(folder_path):
        imgs = list(Path(folder_path).glob("*.png")) + list(Path(folder_path).glob("*.jpg"))
        st.info(f"Found **{len(imgs)}** images")
    elif folder_path:
        st.error("Invalid directory path")
        imgs = []
    else:
        imgs = []

    bc = st.columns([2, 1, 1])
    with bc[1]:
        max_imgs = st.number_input("Limit", min_value=1, max_value=5000, value=100)
    with bc[2]:
        save_out = st.checkbox("Save Outputs", value=True)

    run_batch = st.button("🚀 Start Batch Processing", type="primary", disabled=model is None)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_batch and imgs and manager:
        images = imgs[:max_imgs]
        st.info(f"Processing **{len(images)}** images...")

        pbar = st.progress(0)
        status = st.empty()

        def _cb(idx, total, msg):
            if total > 0:
                pbar.progress(idx / total)
            status.text(f"[{idx}/{total}] {msg}")

        out_dir = (Path(folder_path).parent / f"batch_{datetime.now():%Y%m%d_%H%M%S}") if save_out else None

        res_df = manager.predict_batch(images, conf_threshold, out_dir, _cb)

        pbar.progress(1.0)
        status.empty()
        st.success(f"Processed {len(images)} images")
        if out_dir:
            st.info(f"Saved to: {out_dir}")

        if not res_df.empty:
            st.markdown("#### 📊 Batch Statistics")
            total_d = len(res_df)
            avg_t = res_df['inference_time'].mean() if 'inference_time' in res_df.columns else 0
            sc = st.columns(4)
            sc[0].metric("Total Detections", total_d)
            sc[1].metric("Avg/Image", f"{total_d/len(images):.1f}")
            sc[2].metric("Avg Time", f"{avg_t*1000:.0f} ms")
            sc[3].metric("Throughput", f"{1/avg_t:.1f} img/s" if avg_t > 0 else "N/A")

            st.markdown("**By Class**")
            summary = res_df.groupby("class_name").agg(
                Count=('score', 'count'),
                Mean_Conf=('score', 'mean'),
                Std_Conf=('score', 'std')
            ).round(3)
            st.dataframe(summary, width='stretch')

            st.plotly_chart(plot_class_distribution(res_df), width='stretch')

            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv,
                               f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv",
                               type="primary")
        else:
            st.warning("No detections generated.")

# ============================================================================
# Tab 5: Research Dashboard
# ============================================================================
with tab5:
    st.markdown("### 📊 Research Performance Dashboard")

    # Model Registry
    st.markdown("#### 🏆 Model Registry")
    reg_rows = []
    for name, mcfg in sorted(ModelFactory.get_valid_models().items()):
        info = ModelFactory.get_model_info(name)
        reg_rows.append({
            'Model': name,
            'Type': info.get('model_type', '?').title(),
            'Available': '✅' if info.get('exists') else '❌',
            'GradCAM': '✅' if info.get('supports_gradcam') else '❌',
            'Checkpoint': Path(info.get('path', '')).name,
        })
    st.dataframe(pd.DataFrame(reg_rows), width='stretch', hide_index=True)

    # Per-model metrics CSVs
    st.markdown("---")
    st.markdown("#### 📈 Training Metrics")

    metrics_dir = METRICS_ROOT
    if metrics_dir.exists():
        csv_files = sorted(metrics_dir.glob("*.csv"))
        if csv_files:
            sel_csv = st.selectbox("Select Model Metrics",
                                   [f.stem for f in csv_files],
                                   index=0, key="metrics_csv_select")
            csv_path = metrics_dir / f"{sel_csv}.csv"
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, width='stretch', hide_index=True)
                fig = plot_metrics_csv(csv_path, sel_csv)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Error reading {sel_csv}: {e}")
        else:
            st.info("No metrics CSV files found.")
    else:
        st.info("Metrics directory not found.")

    # Pre-computed comparison
    det_csv = metrics_dir / "metrics_detection.csv"
    if det_csv.exists():
        st.markdown("---")
        st.markdown("#### 🏅 Model Comparison (mAP)")
        df_det = pd.read_csv(det_csv)
        if 'Model' in df_det.columns and 'mAP50' in df_det.columns:
            st.plotly_chart(plot_map_comparison(df_det), width='stretch')
        else:
            st.dataframe(df_det, width='stretch', hide_index=True)

    # Class Reference
    st.markdown("---")
    st.markdown("#### 📋 Class Reference")
    class_ref = pd.DataFrame({
        'ID': range(6),
        'Abbreviation': CLASS_NAMES,
        'Full Name': [CLASS_FULL_NAMES[c] for c in CLASS_NAMES],
        'Description': [CLASS_DESCRIPTIONS[c] for c in CLASS_NAMES],
    })
    st.dataframe(class_ref, width='stretch', hide_index=True)

# ============================================================================
# Footer
# ============================================================================
st.markdown("""
<div class="app-footer">
    <strong>SpineScan AI</strong> — Clinical Research Platform v3.0<br>
    Spinal Disorder Detection with Interpretability & Robustness Analysis<br>
    © 2026 YOLOspine Research Team &bull; For Research Purposes Only
</div>
""", unsafe_allow_html=True)

