"""
plot_paper_figures.py
--------------------
Clean, publication-ready figures for the MiniCheck long-context evaluation report.

Data source: all results are pre-aggregated in final_results/
  - summary_all_models.csv      -- overall BAcc + SPM per (model, dataset)
  - summary_adversarial.csv     -- adversarial detection rates

All raw JSON results are in:
  - results/                    -- local model evaluations
  - results_adversarial/        -- adversarial injection results
  - results_openrouter/         -- OpenRouter API results

Figures produced:
  fig1_bacc_local_only.pdf      -- Local models (flan-t5-large vs Bespoke-MiniCheck-7B) on all shared datasets
  fig2_bacc_all_models.pdf      -- All models grouped by dataset (only non-empty comparisons)
  fig3_throughput.pdf           -- Scatter: BAcc vs SPM (log-log), local + API models
  fig4_adversarial.pdf          -- Adversarial detection: by type (left) + by position (right)

Usage
-----
  python long_context_eval/plot_paper_figures.py --data_dir final_results --output_dir final_results
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_LOCAL   = ["#1f77b4", "#ff7f0e"]   # flan-t5-large, Bespoke-MiniCheck-7B
C_API     = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
ALL_COLORS = C_LOCAL + C_API

MODEL_COLOR = {
    "flan-t5-large":           "#1f77b4",
    "Bespoke-MiniCheck-7B":    "#ff7f0e",
    "Gemma-4-26B":             "#2ca02c",
    "GPT-OSS-120B":            "#d62728",
    "GPT-OSS-20B":             "#9467bd",
    "Trinity-Large":           "#8c564b",
    "Gemma-4-31B":             "#e377c2",
}

MODEL_HATCH = {
    "flan-t5-large":           None,
    "Bespoke-MiniCheck-7B":    "//",
    "Gemma-4-26B":             None,
    "GPT-OSS-120B":            None,
    "GPT-OSS-20B":             None,
    "Trinity-Large":           None,
    "Gemma-4-31B":             None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    """Load summary CSVs."""
    overall = pd.read_csv(data_dir / "summary_all_models.csv")
    # Clean model names (replace underscores with dashes for consistency)
    overall["model"] = overall["model"].str.replace("_", "-", regex=False)
    adv = pd.read_csv(data_dir / "summary_adversarial.csv")
    adv["model"] = adv["model"].str.replace("_", "-", regex=False)
    return overall, adv


def annotate_bars(ax, bars, fontsize=8, offset=1.0):
    """Add percentage labels above bars."""
    for bar in bars:  # bars is a BarContainer
        try:
            h = bar.get_height()
        except AttributeError:
            continue
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=fontsize)


def legend_outside(ax_or_fig, loc="right", pad=0.01):
    """Move legend outside the axes."""
    if isinstance(ax_or_fig, plt.Figure):
        ax_or_fig.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                         frameon=True, framealpha=0.9)
    else:
        ax_or_fig.legend(loc="best", frameon=True, framealpha=0.9)


# ---------------------------------------------------------------------------
# Figure 1: Local models only — BAcc on shared datasets
# ---------------------------------------------------------------------------
# Datasets where BOTH flan-t5-large AND Bespoke-MiniCheck-7B have results
SHARED_DATASETS = ["ExpertQA", "RAGTruth", "SciFact", "SummHay", "TofuEval-MediaS"]
LOCAL_MODELS   = ["flan-t5-large", "Bespoke-MiniCheck-7B"]

def plot_fig1_local_only(overall, out_path):
    local = overall[overall["model"].isin(LOCAL_MODELS)]
    shared = local[local["dataset"].isin(SHARED_DATASETS)]

    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = SHARED_DATASETS
    n_datasets = len(datasets)
    n_models = len(LOCAL_MODELS)
    bar_width = 0.35
    group_pad = 0.25

    groups = []
    for di, ds in enumerate(datasets):
        group_left = di * (bar_width * n_models + group_pad)
        groups.append(group_left + bar_width / 2)

        for mi, model in enumerate(LOCAL_MODELS):
            row = shared[(shared["model"] == model) & (shared["dataset"] == ds)]
            if not row.empty:
                val = row["overall_bacc"].values[0]
                x = group_left + mi * bar_width
                color = MODEL_COLOR.get(model, "#333333")
                hatch = MODEL_HATCH.get(model)
                bar = ax.bar(x, val, bar_width, label=model, color=color,
                             hatch=hatch, edgecolor="white", linewidth=0.5)
                annotate_bars(ax, bar, fontsize=8, offset=0.5)

    ax.set_xticks(groups)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("Local MiniCheck Models — BAcc by Dataset", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend outside right
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
              frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: All models — grouped by dataset (only non-empty groups)
# ---------------------------------------------------------------------------

def plot_fig2_all_models(overall, out_path):
    """
    Three panels:
      Left:  ExpertQA — all models that have data (local + OpenRouter)
      Middle: SciFact — all models that have data
      Right:  Throughput (SPM) vs BAcc scatter for all models
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Left: ExpertQA ---
    ax = axes[0]
    exp = overall[overall["dataset"] == "ExpertQA"].copy()
    exp = exp.sort_values("model")
    models_exp = exp["model"].tolist()
    bacc_exp   = exp["overall_bacc"].tolist()
    colors_exp = [MODEL_COLOR.get(m, "#333") for m in models_exp]

    n_exp = len(models_exp)
    bar_width = 0.55
    x_pos = np.arange(n_exp)

    bars = ax.bar(x_pos, bacc_exp, bar_width, color=colors_exp, edgecolor="white")
    annotate_bars(ax, bars, fontsize=8, offset=0.5)

    # Star for all-positive models
    for i, (bar, model) in enumerate(zip(bars, models_exp)):
        if model in ("GPT-OSS-120B", "Trinity-Large"):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    "*", ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_exp, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax.set_title("ExpertQA", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.6)
    ax.text(0.02, 0.97, f"n={exp['n_samples'].iloc[0]}",
            transform=ax.transAxes, va="top", fontsize=8, color="gray")

    # --- Middle: SciFact ---
    ax2 = axes[1]
    sci = overall[overall["dataset"] == "SciFact"].copy()
    sci = sci.sort_values("model")
    models_sci = sci["model"].tolist()
    bacc_sci   = sci["overall_bacc"].tolist()
    colors_sci = [MODEL_COLOR.get(m, "#333") for m in models_sci]

    n_sci = len(models_sci)
    x_pos2 = np.arange(n_sci)

    bars2 = ax2.bar(x_pos2, bacc_sci, bar_width, color=colors_sci, edgecolor="white")
    annotate_bars(ax2, bars2, fontsize=8, offset=0.5)

    for i, (bar, model) in enumerate(zip(bars2, models_sci)):
        if model in ("GPT-OSS-20B", "Gemma-4-31B"):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     "*", ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")

    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(models_sci, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax2.set_title("SciFact", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.6)

    # --- Right: Throughput scatter ---
    ax3 = axes[2]
    for _, row in overall.iterrows():
        model = row["model"]
        color = MODEL_COLOR.get(model, "#333")
        ax3.scatter(row["avg_doc_tokens"], row["overall_bacc"],
                    s=120, color=color, zorder=5, edgecolors="white", linewidth=0.5)
        label = model.replace("Bespoke-MiniCheck-7B", "BM-7B").replace("flan-t5-large", "FT5-L")
        ax3.annotate(f"{label}\n({row['dataset'][:8]})",
                      (row["avg_doc_tokens"], row["overall_bacc"]),
                      xytext=(5, 5), textcoords="offset points", fontsize=7, alpha=0.75)

    ax3.set_xscale("log")
    ax3.set_xlabel("Avg Document Tokens (log)", fontsize=10)
    ax3.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax3.set_title("BAcc vs Document Length", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    ax3.set_ylim(40, 105)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Throughput (SPM) scatter
# ---------------------------------------------------------------------------

def plot_fig3_throughput(overall, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in overall.iterrows():
        model = row["model"]
        color = MODEL_COLOR.get(model, "#333")
        ax.scatter(row["overall_bacc"], row["samples_per_minute"],
                   s=150, color=color, zorder=5, edgecolors="white", linewidth=0.5)

        short = model.replace("Bespoke-MiniCheck-7B", "BM-7B").replace("flan-t5-large", "FT5-L")
        ax.annotate(f"{short}",
                    (row["overall_bacc"], row["samples_per_minute"]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, alpha=0.8)

    # Legend patches
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in MODEL_COLOR.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8,
              frameon=True, framealpha=0.9)

    ax.set_xlim(45, 105)
    ax.set_yscale("log")
    ax.set_xlabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_ylabel("Samples per Minute (log scale)", fontsize=11)
    ax.set_title("Accuracy vs Throughput", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Adversarial detection
# ---------------------------------------------------------------------------

def plot_fig4_adversarial(adv, out_path):
    """
    Left panel: detection rate by hallucination type (numeric, entity, contradict)
    Right panel: detection rate by injection position (beginning, middle, end)
    Random chance = 50% (dashed red line)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    model = "flan-t5-large"  # only one model in adversarial results
    row = adv[adv["model"] == model].iloc[0]

    # --- Left: by type ---
    ax = axes[0]
    types = ["numeric", "entity", "contradict"]
    labels_type = ["Numeric", "Entity", "Contradict"]
    rates_type  = [row["numeric_detection"], row["entity_detection"], row["contradict_detection"]]
    colors_type = ["#4c72b0", "#dd8452", "#55a868"]

    x_pos = np.arange(len(types))
    bar_width = 0.45
    bars = ax.bar(x_pos, rates_type, bar_width, color=colors_type, edgecolor="white", linewidth=0.5)
    annotate_bars(ax, bars, fontsize=9, offset=0.5)

    ax.axhline(y=50, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Random chance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_type, fontsize=11)
    ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.set_title("Adversarial Detection by Hallucination Type", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    # Add fooling rate annotation
    overall_detect = row["detection_rate"]
    ax.text(0.97, 0.97, f"Overall fooling rate: {100-overall_detect:.1f}%\n(Detection: {overall_detect:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # --- Right: by position ---
    ax2 = axes[1]
    positions = ["beginning", "middle", "end"]
    labels_pos = ["Beginning", "Middle", "End"]
    rates_pos  = [row["beginning_detection"], row["middle_detection"], row["end_detection"]]
    colors_pos = ["#8172b3", "#c44e52", "#ccb974"]

    x_pos2 = np.arange(len(positions))
    bars2 = ax2.bar(x_pos2, rates_pos, bar_width, color=colors_pos, edgecolor="white", linewidth=0.5)
    annotate_bars(ax2, bars2, fontsize=9, offset=0.5)

    ax2.axhline(y=50, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Random chance")
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(labels_pos, fontsize=11)
    ax2.set_ylabel("Detection Rate (%)", fontsize=11)
    ax2.set_title("Adversarial Detection by Injection Position", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 80)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="final_results",
                        help="Directory containing summary_all_models.csv and summary_adversarial.csv")
    parser.add_argument("--output_dir", default="final_results",
                        help="Directory to write figures")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall, adv = load_data(data_dir)

    # Save cleaned data with proper names back for reference
    overall.to_csv(out_dir / "summary_all_models_clean.csv", index=False)
    print(f"Cleaned data saved to {out_dir / 'summary_all_models_clean.csv'}")

    print(f"\nOverall data ({len(overall)} rows):")
    print(overall[["model","dataset","overall_bacc","samples_per_minute"]].to_string(index=False))

    print(f"\nAdversarial data ({len(adv)} rows):")
    print(adv.to_string(index=False))

    print("\nGenerating figures...")
    plot_fig1_local_only(overall, out_dir / "fig1_bacc_local_only.png")
    plot_fig2_all_models(overall, out_dir / "fig2_bacc_all_models.png")
    plot_fig3_throughput(overall, out_dir / "fig3_throughput.png")
    plot_fig4_adversarial(adv, out_dir / "fig4_adversarial.png")

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
