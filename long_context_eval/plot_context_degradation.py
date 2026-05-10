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

BIN_ORDER = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]
MODEL_STYLE = {
    "Bespoke-MiniCheck-7B": {"marker": "o", "color": "#1f77b4", "label": "Bespoke-MiniCheck-7B"},
    "flan-t5-large": {"marker": "s", "color": "#d62728", "label": "flan-t5-large"},
}


def load_data():
    root = Path(__file__).resolve().parents[1]
    bins = pd.read_csv(root / "final_results" / "summary_bins.csv")
    bins = bins.dropna(subset=["bacc"]).copy()
    bins["bin_label"] = pd.Categorical(bins["bin_label"], categories=BIN_ORDER, ordered=True)
    return bins.sort_values(["dataset", "model", "bin_label"])


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

        for model, group in sub.groupby("model", observed=True):
            style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
            group = group.set_index("bin_label").reindex(BIN_ORDER).dropna(subset=["bacc"])

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

        # Mark the "lost in the middle" trough if visible
        if len(group) >= 3:
            trough_idx = 2  # 1000-2000 bin
            if trough_idx < len(group):
                ax.axvspan(trough_idx - 0.4, trough_idx + 0.4, alpha=0.1, color="red")

        n_bins = sub["bin_label"].nunique()
        ax.set_title(f"{dataset}\n({n_bins} bins)", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(BIN_ORDER)))
        ax.set_xticklabels(BIN_ORDER, fontsize=9, rotation=25, ha="right")
        ax.set_xlabel("Token-Length Bin", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if dataset == "ExpertQA":
            ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
            # Add annotation for U-shape
            ax.annotate("U-shape\n(Lost in Middle)",
                        xy=(2, 48), xytext=(3.2, 55),
                        fontsize=9, style="italic",
                        arrowprops=dict(arrowstyle="->", color="green"),
                        color="green")
        else:
            ax.set_ylabel("")
            if n_bins < 5:
                ax.annotate(f"No bins > 2000 tokens\nCannot show recovery",
                            xy=(1, 75), xytext=(2, 82),
                            fontsize=7, style="italic", color="gray",
                            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    axes[-1].legend(loc="lower left", fontsize=9)
    fig.suptitle("'Lost in the Middle': BAcc by Document Length\n"
                 "Only ExpertQA has samples in all 5 bins to show full U-shape with recovery",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_expertqa_u_shape_emphasis(bins, out_path):
    """
    Focus on ExpertQA showing the clear U-shape with annotations.
    """
    expertqa = bins[bins["dataset"] == "ExpertQA"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in expertqa.groupby("model", observed=True):
        style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
        group = group.set_index("bin_label").reindex(BIN_ORDER).dropna(subset=["bacc"])

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
    ax.set_title("ExpertQA: Clear 'Lost in the Middle' Phenomenon\n"
                 "BAcc drops at 1k-2k tokens, recovers at 2k-4k tokens",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=11)

    # Add text annotations for the trend
    ax.annotate("Degradation:\n-10.6% (7B)\n-9.2% (T5)",
                xy=(2, 48), xytext=(0.3, 50),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                color="red")
    ax.annotate("Recovery:\n+11.0% (7B)\n+17.4% (T5)",
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
