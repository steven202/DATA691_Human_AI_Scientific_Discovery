"""
plot_lost_in_middle.py
----------------------
Focused visualization of the "lost in the middle" phenomenon.

The phenomenon: BAcc is high at short lengths, drops in the middle,
and recovers at very long lengths (U-shaped curve).

This requires datasets with samples in BOTH short (<1000 tokens) AND
long (>2000 tokens) bins to observe the recovery phase.

Only ExpertQA has sufficient length coverage to show the full U-shape.
Other datasets (RAGTruth, etc.) only have samples in shorter bins.

Usage
-----
  python long_context_eval/plot_lost_in_middle.py
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
    return bins


def plot_expertqa_u_shape(bins, out_path):
    """
    Focus plot on ExpertQA which shows the clearest U-shape.
    ExpertQA has 5 bins spanning 0-500 to 4000+ tokens.
    """
    expertqa = bins[bins["dataset"] == "ExpertQA"].copy()
    expertqa["bin_label"] = pd.Categorical(expertqa["bin_label"], categories=BIN_ORDER, ordered=True)
    expertqa = expertqa.sort_values(["model", "bin_label"])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model, group in expertqa.groupby("model", observed=True):
        style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
        group = group.set_index("bin_label").reindex(BIN_ORDER).dropna(subset=["bacc"])

        x_labels = [str(b) for b in group.index]
        x_pos = range(len(x_labels))
        baccs = group["bacc"].values
        ns = group["n"].values

        line = ax.plot(x_pos, baccs, label=style["label"],
                       linewidth=2.5, markersize=9,
                       marker=style["marker"], color=style["color"])

        # Annotate with sample counts
        for i, (x, (idx, row)) in enumerate(zip(x_pos, group.iterrows())):
            ax.annotate(f"n={int(row['n'])}",
                        (x, row["bacc"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center", fontsize=9,
                        color=style["color"])

    # Mark the "lost in the middle" trough region
    trough_idx = 2  # 1000-2000 token bin
    ax.axvspan(trough_idx - 0.4, trough_idx + 0.4, alpha=0.15, color="red",
               label="Trough: lost in middle")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Document Token-Length Bin", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_ylim(40, 75)
    ax.set_title("ExpertQA: 'Lost in the Middle' Phenomenon\n"
                 "BAcc drops at 1k–2k tokens, recovers at 2k–4k tokens",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Add annotation about the U-shape
    ax.annotate("U-shaped recovery\nat longer lengths",
                xy=(3, 66), xytext=(3.5, 70),
                fontsize=9, style="italic",
                arrowprops=dict(arrowstyle="->", color="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_datasets_comparison(bins, out_path):
    """
    Show all multi-bin datasets to compare U-shape visibility.
    ExpertQA is the only dataset with enough length coverage for full U-shape.
    """
    # Filter to datasets with multiple bins
    counts = bins.groupby("dataset")["bin_label"].nunique()
    multi_bin = counts[counts > 1].index
    multi_bins_data = bins[bins["dataset"].isin(multi_bin)].copy()

    datasets = ["ExpertQA", "RAGTruth"]  # Only these have >1 bin AND meaningful data
    datasets = [d for d in datasets if d in multi_bin]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = multi_bins_data[multi_bins_data["dataset"] == dataset].copy()
        sub["bin_label"] = pd.Categorical(sub["bin_label"], categories=BIN_ORDER, ordered=True)
        sub = sub.sort_values(["model", "bin_label"])

        for model, group in sub.groupby("model", observed=True):
            style = MODEL_STYLE.get(model, {"marker": "o", "color": "#333"})
            group = group.set_index("bin_label").reindex(BIN_ORDER).dropna(subset=["bacc"])

            x_labels = [str(b) for b in group.index]
            x_pos = range(len(x_labels))

            ax.plot(x_pos, group["bacc"].values,
                    label=style["label"],
                    linewidth=2.5, markersize=8,
                    marker=style["marker"], color=style["color"])

            for i, (idx, row) in enumerate(group.iterrows()):
                ax.annotate(f"n={int(row['n'])}",
                            (i, row["bacc"]),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center", fontsize=8,
                            color=style["color"])

        ax.set_xticks(range(len(BIN_ORDER)))
        ax.set_xticklabels(BIN_ORDER, fontsize=9, rotation=20)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xlabel("Token-Length Bin", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Add note about U-shape visibility
        if dataset == "ExpertQA":
            ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
            ax.annotate("U-shape CLEARLY\nvisible here",
                        xy=(2, 50), xytext=(3.2, 52),
                        fontsize=8, style="italic", color="green",
                        arrowprops=dict(arrowstyle="->", color="green"))
        else:
            ax.set_ylabel("")
            ax.annotate("Cannot see U-shape:\nno bins > 2000 tokens",
                        xy=(1, 75), xytext=(2, 78),
                        fontsize=8, style="italic", color="red",
                        arrowprops=dict(arrowstyle="->", color="red"))

    axes[-1].legend(loc="lower left", fontsize=9)
    fig.suptitle("'Lost in the Middle': U-Shape Visibility by Dataset Length Coverage",
                 y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_throughput_tradeoff(bins, out_path):
    """
    Show the efficiency cost: high SPM at short docs, low SPM at long docs.
    """
    from final_analysis import build_overall_table
    import json

    root = Path(__file__).resolve().parents[1]

    # Load overall results to get SPM data
    overall = pd.read_csv(root / "final_results" / "summary_all_models.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in overall.iterrows():
        model = row["model"]
        color = MODEL_STYLE.get(model, {}).get("color", "#333")
        marker = MODEL_STYLE.get(model, {}).get("marker", "o")

        ax.scatter(row["avg_doc_tokens"], row["samples_per_minute"],
                   s=120, color=color, marker=marker,
                   edgecolors="white", linewidth=0.5, zorder=5)

        short_name = model.replace("Bespoke-MiniCheck-7B", "BM-7B").replace("flan-t5-large", "FT5-L")
        ax.annotate(f"{short_name}\n({row['dataset'][:10]})",
                    (row["avg_doc_tokens"], row["samples_per_minute"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, alpha=0.75)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Average Document Tokens (log)", fontsize=11)
    ax.set_ylabel("Samples per Minute (log)", fontsize=11)
    ax.set_title("Efficiency Cost: Throughput vs. Document Length\n"
                 "SummHay (74k tokens) achieves only 3.9 SPM",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Annotate the key finding
    ax.annotate("SummHay: 74k tokens\n3.9 SPM",
                xy=(74488, 3.9), xytext=(10000, 10),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "final_results"
    out_dir.mkdir(exist_ok=True)

    bins = load_data()

    # Generate focused U-shape plot for ExpertQA
    plot_expertqa_u_shape(bins, out_dir / "lost_in_middle_expertqa.png")

    # Generate comparison plot
    plot_all_datasets_comparison(bins, out_dir / "lost_in_middle_comparison.png")

    # Generate throughput tradeoff plot
    plot_throughput_tradeoff(bins, out_dir / "throughput_vs_length.png")

    print("\nDone! Saved U-shape visualizations to final_results/")


if __name__ == "__main__":
    main()