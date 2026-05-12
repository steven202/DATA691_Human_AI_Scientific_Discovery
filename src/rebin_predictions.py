"""
Rebin per-sample predictions into adaptive token-count bins.

Reads per-sample prediction CSVs (produced by evaluate.py) and computes
bin-level BAcc using dataset-adaptive bin widths. Outputs a new
summary_bins.csv that the plot scripts consume.

Strategy:
  - Datasets with max_tokens >= 4000: keep original absolute bins
    (0-500, 500-1000, 1000-2000, 2000-4000, 4000+)
  - Datasets with max_tokens < 4000: use finer bins so we can observe
    the U-shape within the available token range. Bin width is chosen
    to yield 4-6 bins with >= 30 samples each.

Usage:
  python long_context_eval/rebin_predictions.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "final_results"

# Original absolute bins for datasets with wide token range
ABS_BINS = [
    (0, 500, "0-500"),
    (500, 1000, "500-1000"),
    (1000, 2000, "1000-2000"),
    (2000, 4000, "2000-4000"),
    (4000, int(1e9), "4000+"),
]

# Per-dataset fine bin widths (chosen to get 4-6 bins with >= 30 samples)
FINE_BIN_CONFIG = {
    "RAGTruth": 250,
    "ExpertQA": None,       # use absolute bins (wide range)
    "SummHay": None,        # use absolute bins
    "SciFact": None,        # too short, single bin
}

# Default fine bin width for other datasets (TofuEval, Lfqa, etc.)
DEFAULT_FINE_WIDTH = 250
MIN_SAMPLES_PER_BIN = 30


def compute_bacc(labels: np.ndarray, preds: np.ndarray):
    """Balanced accuracy; None if < 2 classes."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    return balanced_accuracy_score(labels, preds)


def build_abs_bins(df: pd.DataFrame, model: str) -> list[dict]:
    """Use absolute bins for wide-range datasets."""
    rows = []
    dataset = df["dataset"].iloc[0]
    for lo, hi, label in ABS_BINS:
        mask = (df["doc_tokens"] >= lo) & (df["doc_tokens"] < hi)
        sub = df[mask]
        n = len(sub)
        if n == 0:
            continue
        bacc = compute_bacc(sub["label"].values, sub["pred_label"].values)
        rows.append({
            "model": model,
            "dataset": dataset,
            "bin_label": label,
            "bin_min": lo,
            "bin_max": hi if hi < 1e8 else -1,
            "n": n,
            "bacc": round(bacc * 100, 2) if bacc is not None else None,
            "avg_doc_tokens": round(sub["doc_tokens"].mean(), 1),
        })
    return rows


def build_fine_bins(df: pd.DataFrame, model: str, width: int) -> list[dict]:
    """Use `width`-token bins up to max_tokens."""
    rows = []
    dataset = df["dataset"].iloc[0]
    max_tok = int(df["doc_tokens"].max())

    edges = list(range(0, max_tok + width, width))
    if edges[-1] < max_tok:
        edges.append(max_tok + 1)

    for lo, hi in zip(edges[:-1], edges[1:]):
        hi_display = hi if hi <= max_tok else -1
        label = f"{lo}-{hi}" if hi <= max_tok else f"{lo}+"

        mask = (df["doc_tokens"] >= lo) & (df["doc_tokens"] < hi)
        sub = df[mask]
        n = len(sub)
        if n < MIN_SAMPLES_PER_BIN:
            continue

        bacc = compute_bacc(sub["label"].values, sub["pred_label"].values)
        rows.append({
            "model": model,
            "dataset": dataset,
            "bin_label": label,
            "bin_min": lo,
            "bin_max": hi_display,
            "n": n,
            "bacc": round(bacc * 100, 2) if bacc is not None else None,
            "avg_doc_tokens": round(sub["doc_tokens"].mean(), 1),
        })
    return rows


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load predictions CSV and infer model+dataset from the path."""
    df = pd.read_csv(csv_path)
    stem = csv_path.stem  # e.g. "flan-t5-large_RAGTruth.predictions"
    base = stem.replace(".predictions", "")
    parts = base.split("_", 1)
    model = parts[0]
    dataset = parts[1] if len(parts) > 1 else ""
    # Restore hyphens in dataset name
    dataset = dataset.replace("_", "-")
    # Fix common patterns
    dataset = dataset.replace("TofuEval-", "TofuEval-").replace("-MediaS", "-MediaS")
    df["dataset"] = dataset
    return df, model, dataset


def main():
    # Collect all predictions CSVs
    pred_files = sorted(RESULTS_DIR.glob("*.predictions.csv"))
    if not pred_files:
        print("[rebin] No predictions CSV files found. Run evaluate.py first to generate them.")
        sys.exit(1)

    print(f"[rebin] Found {len(pred_files)} predictions file(s)")

    # Load existing summary_bins.csv to preserve bins for datasets
    # without predictions CSVs
    old_csv = OUTPUT_DIR / "summary_bins.csv"
    old_rows = []
    if old_csv.exists():
        old = pd.read_csv(old_csv)
        old_rows = old.to_dict("records")

    new_rows = []

    for csv_path in pred_files:
        print(f"[rebin] Processing: {csv_path.name}")
        df, model, dataset = load_predictions(csv_path)

        # Determine binning strategy
        max_tok = df["doc_tokens"].max()
        config_width = FINE_BIN_CONFIG.get(dataset)

        if config_width is None and max_tok >= 4000:
            # Use absolute bins
            rows = build_abs_bins(df, model)
            print(f"  -> Absolute bins: {len(rows)} bins, max_tokens={max_tok:.0f}")
        else:
            width = config_width if config_width else DEFAULT_FINE_WIDTH
            rows = build_fine_bins(df, model, width)
            print(f"  -> Fine bins ({width}-token): {len(rows)} bins, max_tokens={max_tok:.0f}")

        new_rows.extend(rows)

        # Mark old rows from this (model, dataset) as superseded
        old_rows = [r for r in old_rows
                    if not (r["model"] == model and r["dataset"] == dataset)]

    # Merge: new (rebinned) rows + preserved old rows
    all_rows = new_rows + old_rows

    # Save
    out_df = pd.DataFrame(all_rows)
    cols = ["model", "dataset", "bin_label", "bin_min", "n", "bacc", "avg_doc_tokens"]
    out_df = out_df[cols].sort_values(["dataset", "model", "bin_min"])
    out_path = OUTPUT_DIR / "summary_bins.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[rebin] Saved {len(out_df)} rows to {out_path}")
    print(f"  Datasets: {out_df['dataset'].unique().tolist()}")


if __name__ == "__main__":
    main()
