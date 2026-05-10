"""
final_analysis.py
-----------------
Comprehensive analysis script that aggregates ALL results:
  1. Original long-context evaluation (results/)
  2. Extended dataset evaluation (Lfqa, TofuEval-MeetB)  3. Adversarial injection experiment (results_adversarial/)
  4. OpenRouter API benchmarks (results_openrouter/)

Generates:
  - Complete summary tables (CSV)
  - Unified heatmap across all experiments
  - Adversarial injection analysis figure
  - OpenRouter comparison figure

Usage
-----
  python long_context_eval/final_analysis.py --results_dir results/ \
      --adv_results results_adversarial/ \
      --openrouter_results results_openrouter/ \
      --output_dir final_results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------

def load_json_results(directory: Path, pattern: str = "*.json") -> List[dict]:
    results = []
    for p in sorted(directory.glob(pattern)):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"[analysis] Warning: Could not load {p}: {e}")
    return results


def load_adversarial_results(directory: Path) -> dict:
    results = {}
    for p in sorted(directory.glob("adversarial_*.json")):
        if p.name == "adversarial_dataset.json":
            continue
        with open(p) as f:
            results[p.stem] = json.load(f)
    return results


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_overall_table(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r.get("model_short", r.get("model", "unknown")),
            "dataset": r.get("dataset", "unknown"),
            "n_samples": r.get("n_samples", 0),
            "overall_bacc": r.get("overall_bacc"),
            "avg_doc_tokens": r.get("avg_doc_tokens"),
            "inference_time_s": r.get("inference_time_s"),
            "samples_per_minute": r.get("samples_per_minute"),
            "_has_bins": int(bool(r.get("bins"))),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (
        df.sort_values(["model", "dataset", "_has_bins", "n_samples"])
        .drop_duplicates(["model", "dataset"], keep="last")
        .drop(columns="_has_bins")
    )
    return df.sort_values(["model", "dataset"]).reset_index(drop=True)


def build_bin_table(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for b in r.get("bins", []):
            rows.append({
                "model": r.get("model_short", r.get("model", "unknown")),
                "dataset": r.get("dataset", "unknown"),
                "bin_label": b.get("bin_label", ""),
                "bin_min": b.get("bin_min", 0),
                "n": b.get("n", 0),
                "bacc": b.get("bacc"),
                "avg_doc_tokens": b.get("avg_doc_tokens"),
            })
    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "bin_label", "n", "bacc", "avg_doc_tokens"])
    df = pd.DataFrame(rows)
    bin_order = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]
    df["bin_order"] = df["bin_label"].apply(lambda x: bin_order.index(x) if x in bin_order else 99)
    return df.sort_values(["model", "dataset", "bin_order"]).drop(columns="bin_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "d", "h"]


def plot_unified_bacc_heatmap(bins: pd.DataFrame, out_path: Path):
    """Heatmap of BAcc across all models and datasets by length bin."""
    try:
        import seaborn as sns
    except ImportError:
        print("[analysis] seaborn not installed; skipping heatmap.")
        return

    if bins.empty:
        print("[analysis] No bin data; skipping heatmap.")
        return

    pivot = bins.pivot_table(
        index=["model", "dataset"],
        columns="bin_label",
        values="bacc",
        aggfunc="first",
    )
    bin_ord = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]
    bin_ord = [b for b in bin_ord if b in pivot.columns]
    pivot = pivot.reindex(columns=bin_ord)

    annot = pivot.copy()
    annot = annot.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")

    fig, ax = plt.subplots(figsize=(max(8, len(bin_ord) * 1.8), max(5, len(pivot) * 0.6)))
    ax.set_facecolor("#e6e6e6")
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=40,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        mask=pivot.isna(),
        ax=ax,
        cbar_kws={"label": "BAcc (%)"},
    )
    ax.set_title("Balanced Accuracy (%) by Length Bin (gray = no samples)", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Document Token-Length Bin", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_overall_bacc_matrix(overall: pd.DataFrame, out_path: Path):
    """Overall BAcc matrix across all evaluated model/dataset pairs."""
    try:
        import seaborn as sns
    except ImportError:
        print("[analysis] seaborn not installed; skipping overall matrix.")
        return

    if overall.empty:
        print("[analysis] No overall data; skipping overall matrix.")
        return

    model_order = [
        "Bespoke-MiniCheck-7B",
        "flan-t5-large",
        "Gemma-4-26B",
        "Gemma-4-31B",
        "GPT-OSS-120B",
        "GPT-OSS-20B",
        "Trinity-Large",
    ]
    dataset_order = [
        "ExpertQA",
        "RAGTruth",
        "SciFact",
        "SummHay",
        "TofuEval-MediaS",
        "Lfqa",
        "TofuEval-MeetB",
    ]

    pivot = overall.pivot_table(
        index="dataset",
        columns="model",
        values="overall_bacc",
        aggfunc="first",
    )
    row_order = [d for d in dataset_order if d in pivot.index] + [d for d in pivot.index if d not in dataset_order]
    col_order = [m for m in model_order if m in pivot.columns] + [m for m in pivot.columns if m not in model_order]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    annot = pivot.copy()
    annot = annot.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")

    fig, ax = plt.subplots(figsize=(max(9, len(col_order) * 1.25), max(4.5, len(row_order) * 0.55)))
    ax.set_facecolor("#e6e6e6")
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=40,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        mask=pivot.isna(),
        ax=ax,
        cbar_kws={"label": "BAcc (%)"},
    )
    ax.set_title("Overall Balanced Accuracy Across Evaluated Models (gray = not run)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_model_comparison(overall: pd.DataFrame, out_path: Path):
    """Bar chart comparing all models across datasets."""
    if overall.empty:
        print("[analysis] No overall data; skipping comparison chart.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BAcc comparison
    ax = axes[0]
    models = overall["model"].unique()
    datasets = overall["dataset"].unique()
    x = range(len(datasets))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        sub = overall[overall["model"] == model].set_index("dataset")
        vals = [sub.loc[d, "overall_bacc"] if d in sub.index else 0 for d in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width * 0.9,
                      label=model, color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("BAcc by Model and Dataset", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 110)

    # Speed comparison (SPM, log scale)
    ax2 = axes[1]
    for i, model in enumerate(models):
        sub = overall[overall["model"] == model]
        ax2.scatter(sub["avg_doc_tokens"], sub["samples_per_minute"],
                    label=model, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                    s=100, alpha=0.8, edgecolors='w')
        for _, row in sub.iterrows():
            ax2.annotate(row["dataset"][:10], (row["avg_doc_tokens"], row["samples_per_minute"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.6)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Avg Document Tokens (log)", fontsize=11)
    ax2.set_ylabel("Samples per Minute (log)", fontsize=11)
    ax2.set_title("Throughput vs. Document Length", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_adversarial_results(adv_results: dict, out_path: Path):
    """Plot adversarial injection detection rates by type and position."""
    if not adv_results:
        print("[analysis] No adversarial results; skipping.")
        return

    # Collect all data
    all_type_data = {}
    all_pos_data = {}

    for name, result in adv_results.items():
        model = result.get("model", name)
        by_type = result.get("by_hallucination_type", {})
        by_pos = result.get("by_injection_position", {})

        for htype, stats in by_type.items():
            if htype not in all_type_data:
                all_type_data[htype] = {}
            all_type_data[htype][model] = stats.get("detection_rate", 0)

        for pos, stats in by_pos.items():
            if pos not in all_pos_data:
                all_pos_data[pos] = {}
            all_pos_data[pos][model] = stats.get("detection_rate", 0)

    n_types = len(all_type_data)
    n_pos = len(all_pos_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By hallucination type
    ax = axes[0]
    if all_type_data:
        type_labels = list(all_type_data.keys())
        models = list(next(iter(all_type_data.values())).keys())
        x = range(len(type_labels))
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            vals = [all_type_data[htype].get(model, 0) for htype in type_labels]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], vals, width * 0.9,
                          label=model, color=COLORS[i % len(COLORS)], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random chance")
        ax.set_xlabel("Hallucination Type", fontsize=11)
        ax.set_ylabel("Detection Rate (%)", fontsize=11)
        ax.set_title("Adversarial Detection by Hallucination Type", fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(type_labels, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 80)

    # By injection position
    ax2 = axes[1]
    if all_pos_data:
        pos_labels = list(all_pos_data.keys())
        models = list(next(iter(all_pos_data.values())).keys())
        x = range(len(pos_labels))
        for i, model in enumerate(models):
            vals = [all_pos_data[pos].get(model, 0) for pos in pos_labels]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax2.bar([xi + offset for xi in x], vals, width * 0.9,
                          label=model, color=COLORS[i % len(COLORS)], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

        ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random chance")
        ax2.set_xlabel("Injection Position", fontsize=11)
        ax2.set_ylabel("Detection Rate (%)", fontsize=11)
        ax2.set_title("Adversarial Detection by Injection Position", fontsize=12, fontweight="bold")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(pos_labels)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.set_ylim(0, 80)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def plot_openrouter_comparison(openrouter_results: List[dict], local_results: List[dict], out_path: Path):
    """Compare OpenRouter API models vs local MiniCheck models."""
    if not openrouter_results:
        print("[analysis] No OpenRouter results; skipping comparison.")
        return

    # Get datasets that have both OpenRouter and local results
    openrouter_df = build_overall_table(openrouter_results)
    local_df = build_overall_table(local_results)

    if openrouter_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BAcc comparison
    ax = axes[0]
    all_models = list(set(openrouter_df["model"].tolist() + local_df["model"].tolist()))
    all_datasets = list(set(openrouter_df["dataset"].tolist() + local_df["dataset"].tolist()))

    x = range(len(all_datasets))
    width = 0.8 / max(len(all_models), 1)

    for i, model in enumerate(all_models):
        sub = pd.concat([openrouter_df, local_df])
        sub = sub[sub["model"] == model].set_index("dataset")
        vals = [sub.loc[d, "overall_bacc"] if d in sub.index else 0 for d in all_datasets]
        offset = (i - len(all_models) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width * 0.9,
                      label=model, color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("OpenRouter vs Local Models (BAcc)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_datasets, rotation=20, ha="right")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Speed comparison
    ax2 = axes[1]
    for i, model in enumerate(all_models):
        sub = pd.concat([openrouter_df, local_df])
        sub = sub[sub["model"] == model]
        ax2.scatter(sub["avg_doc_tokens"], sub["samples_per_minute"],
                    label=model, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                    s=100, alpha=0.8, edgecolors='w')

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Avg Document Tokens (log)", fontsize=11)
    ax2.set_ylabel("Samples per Minute (log)", fontsize=11)
    ax2.set_title("Throughput Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")


def print_comprehensive_summary(
    overall: pd.DataFrame,
    bins: pd.DataFrame,
    adv_results: dict,
    openrouter_results: List[dict],
):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("TABLE 1 — Overall Performance (All Models)")
    print("=" * 80)
    if not overall.empty:
        print(overall.to_string(index=False))
    else:
        print("No data available.")

    print("\n" + "=" * 80)
    print("TABLE 2 — BAcc by Document-Length Bin")
    print("=" * 80)
    if not bins.empty:
        pivot = bins.pivot_table(
            index=["model", "dataset"],
            columns="bin_label",
            values="bacc",
            aggfunc="first",
        )
        bin_order = [b for b in ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"] if b in pivot.columns]
        pivot = pivot.reindex(columns=bin_order)
        print(pivot.to_string())
    else:
        print("No bin data available.")

    print("\n" + "=" * 80)
    print("TABLE 3 — Adversarial Injection Detection")
    print("=" * 80)
    if adv_results:
        for name, r in adv_results.items():
            print(f"\nModel: {r.get('model', name)}")
            print(f"  Overall detection rate: {r.get('detection_rate', 'N/A')}%")
            print(f"  By hallucination type:")
            for htype, stats in r.get("by_hallucination_type", {}).items():
                print(f"    {htype}: {stats.get('detection_rate', 'N/A')}% (n={stats.get('n', 0)})")
            print(f"  By injection position:")
            for pos, stats in r.get("by_injection_position", {}).items():
                print(f"    {pos}: {stats.get('detection_rate', 'N/A')}% (n={stats.get('n', 0)})")
    else:
        print("No adversarial results available.")

    print("\n" + "=" * 80)
    print("TABLE 4 — OpenRouter API Benchmarks")
    print("=" * 80)
    if openrouter_results:
        print(f"{'Model':<25} {'Dataset':<15} {'n':>5} {'BAcc':>6} {'Time(s)':>8} {'SPM':>7}")
        print("-" * 70)
        for r in openrouter_results:
            print(
                f"{r.get('model_short', r.get('model', '?')):<25} "
                f"{r.get('dataset', ''):<15} {r.get('n_samples', 0):>5} "
                f"{r.get('overall_bacc', 0):>6.1f} {r.get('inference_time_s', 0):>8.1f} "
                f"{r.get('samples_per_minute', 0):>7.1f}"
            )
    else:
        print("No OpenRouter results available.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive final analysis.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory with original evaluation JSON results.")
    parser.add_argument("--adv_results", type=str, default="results_adversarial",
                        help="Directory with adversarial injection results.")
    parser.add_argument("--openrouter_results", type=str, default="results_openrouter",
                        help="Directory with OpenRouter benchmark results.")
    parser.add_argument("--output_dir", type=str, default="final_results",
                        help="Directory to write final analysis outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    adv_dir = Path(args.adv_results)
    openrouter_dir = Path(args.openrouter_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_results = load_json_results(results_dir)
    adv_results = load_adversarial_results(adv_dir)
    openrouter_results = load_json_results(openrouter_dir)

    print(f"[analysis] Loaded {len(all_results)} original results, "
          f"{len(adv_results)} adversarial results, "
          f"{len(openrouter_results)} OpenRouter results")

    overall = build_overall_table(all_results + openrouter_results)
    bins = build_bin_table(all_results + openrouter_results)

    print_comprehensive_summary(overall, bins, adv_results, openrouter_results)

    # Save summary CSVs
    overall_csv = output_dir / "summary_all_models.csv"
    bins_csv = output_dir / "summary_bins.csv"
    overall.to_csv(overall_csv, index=False)
    bins.to_csv(bins_csv, index=False)
    print(f"\n[analysis] Saved: {overall_csv}")
    print(f"[analysis] Saved: {bins_csv}")

    # Generate figures
    print("\n[analysis] Generating figures...")
    plot_unified_bacc_heatmap(bins, output_dir / "unified_bacc_heatmap.png")
    plot_overall_bacc_matrix(overall, output_dir / "overall_bacc_matrix.png")
    plot_model_comparison(overall, output_dir / "model_comparison.png")
    plot_adversarial_results(adv_results, output_dir / "adversarial_results.png")
    plot_openrouter_comparison(openrouter_results, all_results, output_dir / "openrouter_comparison.png")

    # Adversarial summary
    if adv_results:
        adv_summary = []
        for name, r in adv_results.items():
            adv_summary.append({
                "model": r.get("model", name),
                "detection_rate": r.get("detection_rate"),
                "inference_time_s": r.get("inference_time_s"),
                "samples_per_minute": r.get("samples_per_minute"),
                "numeric_detection": r.get("by_hallucination_type", {}).get("numeric", {}).get("detection_rate"),
                "entity_detection": r.get("by_hallucination_type", {}).get("entity", {}).get("detection_rate"),
                "contradict_detection": r.get("by_hallucination_type", {}).get("contradict", {}).get("detection_rate"),
                "beginning_detection": r.get("by_injection_position", {}).get("beginning", {}).get("detection_rate"),
                "middle_detection": r.get("by_injection_position", {}).get("middle", {}).get("detection_rate"),
                "end_detection": r.get("by_injection_position", {}).get("end", {}).get("detection_rate"),
            })
        adv_df = pd.DataFrame(adv_summary)
        adv_df.to_csv(output_dir / "summary_adversarial.csv", index=False)
        print(f"[analysis] Saved: {output_dir / 'summary_adversarial.csv'}")

    print(f"\n[analysis] All outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()
