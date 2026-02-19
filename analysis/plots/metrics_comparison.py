"""
Metrics comparison bar charts across all benchmarked models.

Reads a CSV with columns ``Model, mAP50, mAP50_95, Precision, Recall``
and produces a grouped bar chart.

Usage::

    python analysis/plots/metrics_comparison.py \\
        --csv results/benchmark_summary.csv \\
        --output figures/metrics_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(csv_path: str, output: str, figsize=(14, 6)):
    df = pd.read_csv(csv_path)
    metrics = [c for c in df.columns if c != "model" and c != "Model"]
    if not metrics:
        print("No metric columns found.")
        return

    models = df["model"].tolist() if "model" in df.columns else df["Model"].tolist()
    n = len(models)
    x = np.arange(n)
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(metrics):
        vals = df[m].astype(float).tolist()
        ax.bar(x + i * width, vals, width, label=m)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


def main():
    p = argparse.ArgumentParser(description="Metrics comparison plot")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="figures/metrics_comparison.png")
    args = p.parse_args()
    plot(args.csv, args.output)


if __name__ == "__main__":
    main()
