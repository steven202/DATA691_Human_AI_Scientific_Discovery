"""
Enhanced visualization of context degradation and "lost in the middle" phenomenon.

This version clearly shows:
1. Full U-shape on ExpertQA (5 length bins with recovery)
2. Partial degradation on RAGTruth (3 bins, no recovery bins available)
3. Clear explanation of why only ExpertQA shows full U-shape

Usage
-----
  python long_context_eval/plot_context_degradation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

MODEL_STYLE = {
    "Bespoke-MiniCheck-7B": {"marker": "o", "color": "#1f77b4", "label": "Bespoke-MiniCheck-7B"},
    "flan-t5-large": {"marker": "s", "color": "#d62728", "label": "flan-t5-large"},
}


def load_data():
    root = Path(__file__).resolve().parents[1]
    bins = pd.read_csv(root / "final_results" / "summary_bins.csv")
    bins = bins.dropna(subset=["bacc"]).copy()
    return bins.sort_values(["dataset", "model", "bin_min"])

def get_bin_order(sub: pd.DataFrame) -> list:
    """Return sorted unique bin labels for this dataset subset, ordered by bin_min."""
    mapping = sub.groupby("bin_label")["bin_min"].first().sort_values()
    return mapping.index.tolist()


def plot_context_degradation_by_dataset(bins, out_path):
    """
    Create the main context degradation plot showing all multi-bin datasets.
    ExpertQA is the only dataset with 5 bins (full U-shape).
    RAGTruth has 3 bins (degradation visible, but no recovery bins).
    """
    # Filter to datasets with multiple bins
    counts = bins.groupby("dataset")["bin_label"].nunique()
    multi_bin_datasets = counts[counts > 1].index
    multi_bins_data = bins[bins["dataset"].isin(multi_bin_datasets)].copy()

    # Order datasets by number of bins (descending) to show most informative first
    dataset_order = ["ExpertQA", "RAGTruth", "Lfqa", "TofuEval-MeetB"]
    dataset_order = [d for d in dataset_order if d in multi_bin_datasets]
    other_datasets = [d for d in multi_bin_datasets if d not in dataset_order]
    dataset_order.extend(other_datasets)

    n_datasets = len(dataset_order)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4.5 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_order):
        sub = multi_bins_data[multi_bins_data["dataset"] == dataset].copy()
        bin_order = get_bin_order(sub)

        for model, group in sub.groupby("model", observed=True):
            style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
            group = group.set_index("bin_label").reindex(bin_order).dropna(subset=["bacc"])

            x_labels = [str(b) for b in group.index]
            x_pos = range(len(x_labels))

            ax.plot(x_pos, group["bacc"].values,
                    label=style["label"],
                    linewidth=2.5, markersize=8,
                    marker=style["marker"], color=style["color"])

            # Annotate with sample counts
            for i, (idx, row) in enumerate(group.iterrows()):
                ax.annotate(f"n={int(row['n'])}",
                            (i, row["bacc"]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha="center", fontsize=8,
                            color=style["color"])

        # Mark the "lost in the middle" trough if this dataset has >= 3 bins
        n_bins = sub["bin_label"].nunique()
        if n_bins >= 3:
            # Find the middle bin(s) — bins 1 to n-2 (exclude first and last)
            trough_start = 0.6
            trough_end = len(bin_order) - 1.6
            if trough_end > trough_start:
                ax.axvspan(trough_start, trough_end, alpha=0.1, color="red")

        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels(bin_order, fontsize=9, rotation=25, ha="right")
        ax.set_xlabel("Token-Length Bin", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if dataset == "ExpertQA":
            ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)

    axes[-1].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_expertqa_u_shape_emphasis(bins, out_path):
    """
    Focus on ExpertQA showing the clear U-shape with annotations.
    """
    expertqa = bins[bins["dataset"] == "ExpertQA"].copy()
    bin_order = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]  # ExpertQA fixed absolute bins

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in expertqa.groupby("model", observed=True):
        style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
        group = group.set_index("bin_label").reindex(bin_order).dropna(subset=["bacc"])

        x_labels = [str(b) for b in group.index]
        x_pos = range(len(x_labels))

        line = ax.plot(x_pos, group["bacc"].values,
                       label=style["label"],
                       linewidth=3, markersize=10,
                       marker=style["marker"], color=style["color"])

        for i, (idx, row) in enumerate(group.iterrows()):
            ax.annotate(f"n={int(row['n'])}\n({row['avg_doc_tokens']:.0f} tok)",
                        (i, row["bacc"]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha="center", fontsize=8,
                        color=style["color"])

    # Highlight the trough
    ax.axvspan(1.6, 2.4, alpha=0.15, color="red", label="Trough: Lost in Middle")
    ax.axvline(x=2, color="red", linestyle="--", alpha=0.5)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel("Document Token-Length Bin", fontsize=13)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=13)
    ax.set_ylim(40, 75)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=11)

    # Compute degradation and recovery from data (not hardcoded)
    bm7b = expertqa[expertqa["model"] == "Bespoke-MiniCheck-7B"].set_index("bin_label").reindex(bin_order)
    t5 = expertqa[expertqa["model"] == "flan-t5-large"].set_index("bin_label").reindex(bin_order)
    if pd.notna(bm7b.loc["0-500", "bacc"]) and pd.notna(bm7b.loc["1000-2000", "bacc"]):
        bm7b_drop = bm7b.loc["1000-2000", "bacc"] - bm7b.loc["0-500", "bacc"]
    else:
        bm7b_drop = 0
    if pd.notna(t5.loc["0-500", "bacc"]) and pd.notna(t5.loc["1000-2000", "bacc"]):
        t5_drop = t5.loc["1000-2000", "bacc"] - t5.loc["0-500", "bacc"]
    else:
        t5_drop = 0
    if pd.notna(bm7b.loc["2000-4000", "bacc"]) and pd.notna(bm7b.loc["1000-2000", "bacc"]):
        bm7b_recovery = bm7b.loc["2000-4000", "bacc"] - bm7b.loc["1000-2000", "bacc"]
    else:
        bm7b_recovery = 0
    if pd.notna(t5.loc["2000-4000", "bacc"]) and pd.notna(t5.loc["1000-2000", "bacc"]):
        t5_recovery = t5.loc["2000-4000", "bacc"] - t5.loc["1000-2000", "bacc"]
    else:
        t5_recovery = 0

    ax.annotate(f"Degradation:\n{bm7b_drop:+.1f}% (7B)\n{t5_drop:+.1f}% (T5)",
                xy=(2, 48), xytext=(0.3, 50),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                color="red")
    ax.annotate(f"Recovery:\n{bm7b_recovery:+.1f}% (7B)\n{t5_recovery:+.1f}% (T5)",
                xy=(3, 62), xytext=(3.5, 70),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                color="green")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "final_results"
    out_dir.mkdir(exist_ok=True)

    bins = load_data()

    # Main plot showing all datasets
    plot_context_degradation_by_dataset(bins, out_dir / "context_degradation_by_dataset.png")

    # Focused ExpertQA U-shape plot
    plot_expertqa_u_shape_emphasis(bins, out_dir / "lost_in_middle_expertqa.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
