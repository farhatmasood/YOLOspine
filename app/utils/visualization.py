"""Visualization Utilities for SpineScan AI Platform."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional


def plot_map_comparison(model_metrics: pd.DataFrame) -> go.Figure:
    """Bar chart comparing mAP across models."""
    fig = go.Figure()
    if 'mAP50' in model_metrics.columns:
        fig.add_trace(go.Bar(x=model_metrics['Model'], y=model_metrics['mAP50'],
                             name='mAP@0.5', marker_color='#60a5fa'))
    if 'mAP50-95' in model_metrics.columns:
        fig.add_trace(go.Bar(x=model_metrics['Model'], y=model_metrics['mAP50-95'],
                             name='mAP@0.5-0.95', marker_color='#a78bfa'))
    fig.update_layout(
        title='Model Performance Comparison (mAP)', barmode='group',
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Model", yaxis_title="mAP", font=dict(family="Inter, sans-serif"))
    return fig


def plot_confusion_matrix(matrix: np.ndarray, labels: List[str]) -> go.Figure:
    """Interactive confusion matrix heatmap."""
    fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale='Viridis',
                    aspect="auto", labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title='Confusion Matrix', template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(text=matrix, texttemplate="%{text}")
    return fig


def plot_robustness_curve(results: Dict[str, pd.DataFrame]) -> go.Figure:
    """Detection retention vs severity curves."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (deg_type, df) in enumerate(results.items()):
        if 'Severity' in df.columns and 'Retention' in df.columns:
            y = df['Retention']
            if isinstance(y.iloc[0], str):
                y = y.str.rstrip('%').astype('float') / 100.0
            fig.add_trace(go.Scatter(x=df['Severity'], y=y, mode='lines+markers',
                                     name=deg_type.replace('_', ' ').title(),
                                     line=dict(color=colors[i % len(colors)], width=3)))
    fig.update_layout(
        title='Robustness: Performance under Degradation',
        xaxis_title="Severity", yaxis_title="Detection Retention", yaxis_tickformat='.0%',
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"))
    return fig


def plot_class_distribution(detections_df: pd.DataFrame) -> go.Figure:
    """Donut chart of detected pathologies."""
    counts = detections_df['class_name'].value_counts().reset_index()
    counts.columns = ['Pathology', 'Count']
    fig = px.pie(counts, values='Count', names='Pathology',
                 title='Distribution of Detected Pathologies', hole=0.45,
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Inter, sans-serif"))
    return fig


def plot_robustness_radar(summary: Dict) -> go.Figure:
    """Radar chart of robustness across degradation types."""
    categories = [k.replace('_', ' ').title() for k in summary.keys()]
    values = [v['avg_detection_retention'] for v in summary.values()]
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)', line=dict(color='#6366f1', width=2),
        marker=dict(size=8, color='#a78bfa')))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%',
                            gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)'),
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        title='Robustness Profile', font=dict(family="Inter, sans-serif"),
        showlegend=False)
    return fig


def plot_metrics_csv(csv_path, model_name: str = "") -> Optional[go.Figure]:
    """Load a model metrics CSV and plot key columns."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Try to find epoch/step column
    x_col = None
    for candidate in ['epoch', 'step', 'iteration', 'Epoch']:
        if candidate in df.columns:
            x_col = candidate
            break
    if x_col is None and len(df) > 0:
        df['step'] = range(len(df))
        x_col = 'step'

    # Find metric columns
    metric_cols = [c for c in df.columns if c != x_col and df[c].dtype in ('float64', 'float32', 'int64')]
    if not metric_cols:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(metric_cols[:6]):
        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode='lines',
                                 name=col, line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(
        title=f'Training Metrics: {model_name}' if model_name else 'Training Metrics',
        xaxis_title=x_col.title(), yaxis_title="Value",
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"), legend=dict(orientation='h', y=-0.2))
    return fig
