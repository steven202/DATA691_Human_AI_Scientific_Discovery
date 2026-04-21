"""
analysis.py
-----------
Aggregate result JSONs from evaluate.py and produce:
  - Summary tables (printed + saved as CSV)
  - Figure 1: BAcc vs. document-length bin (line chart, one line per model)
  - Figure 2: Inference latency (s) per model × dataset (bar chart)

Usage
-----
python long_context_eval/analysis.py --results_dir results/

Outputs (written to results_dir):
  results/summary_overall.csv
  results/summary_bins.csv
  results/bacc_vs_length.png
  results/latency_bar.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------------

def load_results(results_dir: Path) -> List[dict]:
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    if not results:
        print(f"[analysis] No JSON files found in {results_dir}. "
              "Run evaluate.py first.")
        sys.exit(1)
    print(f"[analysis] Loaded {len(results)} result file(s).")
    return results


# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------

def build_overall_table(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "n_samples": r["n_samples"],
            "overall_bacc": r["overall_bacc"],
            "avg_doc_tokens": r["avg_doc_tokens"],
            "inference_time_s": r["inference_time_s"],
            "samples_per_minute": r["samples_per_minute"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "dataset"]).reset_index(drop=True)
    return df


def build_bin_table(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for b in r.get("bins", []):
            rows.append({
                "model": r["model"],
                "dataset": r["dataset"],
                "bin_label": b["bin_label"],
                "bin_min": b["bin_min"],
                "n": b["n"],
                "bacc": b["bacc"],
                "avg_doc_tokens": b["avg_doc_tokens"],
            })
    df = pd.DataFrame(rows)
    # Sort bins in natural order
    bin_order = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]
    df["bin_order"] = df["bin_label"].apply(
        lambda x: bin_order.index(x) if x in bin_order else 99
    )
    df = df.sort_values(["model", "dataset", "bin_order"]).drop(
        columns="bin_order"
    ).reset_index(drop=True)
    return df


def print_tables(overall: pd.DataFrame, bins: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TABLE 1 — Overall Performance")
    print("=" * 70)
    print(
        overall.to_string(
            index=False,
            columns=["model", "dataset", "n_samples", "overall_bacc",
                     "avg_doc_tokens", "inference_time_s", "samples_per_minute"],
        )
    )

    print("\n" + "=" * 70)
    print("TABLE 2 — BAcc by Document-Length Bin")
    print("=" * 70)
    pivot = bins.pivot_table(
        index=["model", "dataset"],
        columns="bin_label",
        values="bacc",
        aggfunc="first",
    )
    bin_order = [b for b in ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]
                 if b in pivot.columns]
    pivot = pivot.reindex(columns=bin_order)
    print(pivot.to_string())


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

BIN_ORDER = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]


def plot_bacc_vs_length(bins: pd.DataFrame, out_path: Path):
    """Line chart: BAcc vs. doc-length bin, one line per (model, dataset)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = bins.groupby(["model", "dataset"])
    for i, ((model, dataset), grp) in enumerate(groups):
        grp = grp.set_index("bin_label").reindex(BIN_ORDER).dropna(subset=["bacc"])
        if grp.empty:
            continue
        ax.plot(
            grp.index,
            grp["bacc"],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=f"{model} / {dataset}",
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Document Token-Length Bin", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_title("MiniCheck Performance vs. Document Length", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(BIN_ORDER, rotation=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_latency_bar(overall: pd.DataFrame, out_path: Path):
    """Grouped bar chart: inference time (s) per model × dataset."""
    models = overall["model"].unique()
    datasets = overall["dataset"].unique()

    x = range(len(datasets))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 5))

    for i, model in enumerate(models):
        sub = overall[overall["model"] == model].set_index("dataset")
        vals = [sub.loc[d, "inference_time_s"] if d in sub.index else 0 for d in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width * 0.9,
            label=model,
            color=COLORS[i % len(COLORS)],
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}s",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Inference Time (seconds)", fontsize=12)
    ax.set_title("Inference Latency by Model and Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_bacc_heatmap(bins: pd.DataFrame, out_path: Path):
    """Heatmap: model + dataset rows × length-bin columns, cell = BAcc."""
    try:
        import seaborn as sns
    except ImportError:
        print("[analysis] seaborn not installed; skipping heatmap.")
        return

    pivot = bins.pivot_table(
        index=["model", "dataset"],
        columns="bin_label",
        values="bacc",
        aggfunc="first",
    )
    bin_ord = [b for b in BIN_ORDER if b in pivot.columns]
    pivot = pivot.reindex(columns=bin_ord)

    fig, ax = plt.subplots(figsize=(max(8, len(bin_ord) * 1.5), max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=40,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "BAcc (%)"},
    )
    ax.set_title("Balanced Accuracy (%) — Model × Document Length", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Document Token-Length Bin", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_spm_vs_length_scatter(overall: pd.DataFrame, out_path: Path):
    """Scatter plot: Samples per minute vs. average document tokens (log-log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = overall["model"].unique()
    for i, model in enumerate(models):
        sub = overall[overall["model"] == model]
        ax.scatter(
            sub["avg_doc_tokens"],
            sub["samples_per_minute"],
            label=model,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            s=120,
            alpha=0.8,
            edgecolors='w'
        )
        # Annotate points with dataset names
        for _, row in sub.iterrows():
            ax.annotate(
                row['dataset'],
                (row['avg_doc_tokens'], row['samples_per_minute']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Average Document Tokens (log scale)", fontsize=12)
    ax.set_ylabel("Samples Per Minute (log scale)", fontsize=12)
    ax.set_title("Inference Efficiency vs. Document Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse MiniCheck long-context evaluation results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result_*.json files from evaluate.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    results = load_results(results_dir)
    overall = build_overall_table(results)
    bins = build_bin_table(results)

    print_tables(overall, bins)

    # Save CSVs
    overall_csv = results_dir / "summary_overall.csv"
    bins_csv = results_dir / "summary_bins.csv"
    overall.to_csv(overall_csv, index=False)
    bins.to_csv(bins_csv, index=False)
    print(f"\n[analysis] Saved: {overall_csv}")
    print(f"[analysis] Saved: {bins_csv}")

    # Save figures
    plot_bacc_vs_length(bins, results_dir / "bacc_vs_length.png")
    plot_latency_bar(overall, results_dir / "latency_bar.png")
    plot_bacc_heatmap(bins, results_dir / "bacc_heatmap.png")
    plot_spm_vs_length_scatter(overall, results_dir / "spm_vs_length.png")


if __name__ == "__main__":
    main()
