"""
Extract best metrics from YOLO benchmark ``results.csv`` files.

Scans a directory of Ultralytics training runs and extracts the best
mAP@50, mAP@50-95, Precision, and Recall for each model.

Usage::

    python analysis/extract_metrics.py \\
        --runs-dir runs/YOLOspine-Benchmark \\
        --output results/benchmark_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def extract_from_run(run_dir: Path) -> dict | None:
    """Parse a single Ultralytics run's results.csv."""
    csv = run_dir / "results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    # Ultralytics column naming varies across versions
    map50_col = next((c for c in df.columns if "mAP50" in c and "95" not in c), None)
    map5095_col = next((c for c in df.columns if "mAP50-95" in c), None)
    prec_col = next((c for c in df.columns if "precision" in c.lower()), None)
    rec_col = next((c for c in df.columns if "recall" in c.lower()), None)

    out = {"model": run_dir.name}
    if map50_col:
        out["best_mAP50"] = float(df[map50_col].max())
        out["best_epoch_mAP50"] = int(df[map50_col].idxmax()) + 1
    if map5095_col:
        out["best_mAP50_95"] = float(df[map5095_col].max())
    if prec_col:
        out["best_precision"] = float(df[prec_col].max())
    if rec_col:
        out["best_recall"] = float(df[rec_col].max())
    return out


def main():
    p = argparse.ArgumentParser(description="Extract benchmark metrics")
    p.add_argument("--runs-dir", required=True, help="Ultralytics runs root")
    p.add_argument("--output", default=None, help="CSV output path")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    rows = []
    for d in sorted(runs.iterdir()):
        if d.is_dir():
            row = extract_from_run(d)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results.csv files found.")
        return

    print(df.to_string(index=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
