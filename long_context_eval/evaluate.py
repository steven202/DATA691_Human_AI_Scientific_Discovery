"""
evaluate.py
-----------
Long-context evaluation pipeline for MiniCheck models.

Usage
-----
python long_context_eval/evaluate.py \\
    --models flan-t5-large \\
    --datasets TofuEval-MediaS RAGTruth ExpertQA SciFact SummHay \\
    --max_samples 300 \\
    --output_dir results/ \\
    --cache_dir ./ckpts

Output
------
One JSON file per (model, dataset) combination in output_dir:
    results/<model>_<dataset>.json

Schema:
{
    "model": str,
    "dataset": str,
    "n_samples": int,
    "overall_bacc": float,
    "inference_time_s": float,
    "samples_per_minute": float,
    "avg_doc_tokens": float,
    "bins": [
        {
            "bin_label": str,       # e.g. "0-500"
            "bin_min": int,
            "bin_max": int,
            "n": int,
            "bacc": float | null,   # null if < 2 classes in bin
            "avg_doc_tokens": float,
            "inference_time_s": float
        },
        ...
    ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Make sure the repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_context_eval.data_loader import (
    LONG_DOC_SUBSETS,
    load_all_datasets,
    load_llm_aggrefact,
    load_scifact,
    load_summhay,
)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# Token-count bin edges (whitespace tokens)
BINS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, int(1e9)),
]
BIN_LABELS = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000+"]

VALID_MODELS = [
    "roberta-large",
    "deberta-v3-large",
    "flan-t5-large",
    "Bespoke-MiniCheck-7B",
]

# Mapping dataset name → LLM-AggreFact subset name OR special loader key
AGGREFACT_SUBSET_NAMES = {s.lower(): s for s in LONG_DOC_SUBSETS}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def compute_bacc(labels: np.ndarray, preds: np.ndarray) -> Optional[float]:
    """Balanced accuracy; returns None if fewer than 2 classes present."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    return balanced_accuracy_score(labels, preds)


def assign_bins(doc_tokens: pd.Series) -> pd.Series:
    """Return bin-label strings for each doc_tokens value."""
    bins_series = pd.Series(index=doc_tokens.index, dtype=str)
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        mask = (doc_tokens >= lo) & (doc_tokens < hi)
        bins_series[mask] = label
    return bins_series


def load_dataset_by_name(
    dataset_name: str,
    max_samples: Optional[int],
    cache_dir: Optional[str],
) -> pd.DataFrame:
    """Resolve dataset name and return a unified DataFrame."""
    key = dataset_name.lower()

    if key == "scifact":
        return load_scifact(max_samples=max_samples, cache_dir=cache_dir)

    if key == "summhay":
        return load_summhay(max_samples=max_samples, cache_dir=cache_dir)

    # Otherwise assume it's an LLM-AggreFact subset
    # Try best-effort case-insensitive match
    matched = AGGREFACT_SUBSET_NAMES.get(key, dataset_name)
    df = load_llm_aggrefact(
        subsets=[matched],
        max_samples_per_dataset=max_samples,
        cache_dir=cache_dir,
    )
    if df.empty:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Valid LLM-AggreFact subsets: {list(LONG_DOC_SUBSETS)}. "
            f"Other options: SciFact, SummHay."
        )
    return df


# -------------------------------------------------------------------------
# Core evaluation function
# -------------------------------------------------------------------------

def evaluate_model_on_dataset(
    scorer,
    model_name: str,
    df: pd.DataFrame,
    chunk_size: Optional[int] = None,
) -> dict:
    """
    Run MiniCheck inference on df and return result dict.

    Parameters
    ----------
    scorer : MiniCheck
        Pre-instantiated MiniCheck scorer to prevent memory leaks.
    model_name : str
        One of VALID_MODELS.
    df : pd.DataFrame
        Unified dataset with columns: doc, claim, label, doc_tokens.
    chunk_size : int or None
        Passed to MiniCheck.score(); None uses model defaults.

    Returns
    -------
    dict matching the output schema described at the top of this file.
    """
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from minicheck.minicheck import MiniCheck

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {df['dataset'].iloc[0]} ({len(df)} samples)")
    print(f"  Doc tokens — mean: {df['doc_tokens'].mean():.0f}, "
          f"max: {df['doc_tokens'].max()}")
    print(f"{'='*60}")

    docs = df["doc"].tolist()
    claims = df["claim"].tolist()
    labels = df["label"].values

    # --- Full dataset inference ---
    t0 = time.time()
    pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=claims, chunk_size=chunk_size)
    elapsed = time.time() - t0

    pred_labels = np.array(pred_labels)
    overall_bacc = compute_bacc(labels, pred_labels)

    print(f"\nOverall BAcc  : {overall_bacc:.4f}" if overall_bacc is not None else "\nOverall BAcc: N/A (single class)")
    print(f"Inference time: {elapsed:.1f}s  ({len(docs)/elapsed*60:.0f} samples/min)")

    # --- Per-length-bin breakdown ---
    df = df.copy()
    df["pred"] = pred_labels
    df["bin"] = assign_bins(df["doc_tokens"])

    bin_results = []
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        sub = df[df["bin"] == label]
        if len(sub) == 0:
            continue
        sub_labels = sub["label"].values
        sub_preds = sub["pred"].values

        # Re-run timing estimate (proportional to sample count for small models;
        # for larger models vLLM batches everything so we approximate)
        bin_bacc = compute_bacc(sub_labels, sub_preds)

        bin_results.append({
            "bin_label": label,
            "bin_min": lo,
            "bin_max": hi if hi < 1e8 else -1,
            "n": len(sub),
            "bacc": round(bin_bacc * 100, 2) if bin_bacc is not None else None,
            "avg_doc_tokens": round(sub["doc_tokens"].mean(), 1),
        })
        bacc_str = f"{bin_bacc*100:.2f}" if bin_bacc is not None else "N/A (1 class)"
        print(f"  Bin {label:12s}: n={len(sub):4d}, avg_tokens={sub['doc_tokens'].mean():.0f}, BAcc={bacc_str}")

    dataset_name = df["dataset"].iloc[0]
    return {
        "model": model_name,
        "dataset": dataset_name,
        "n_samples": len(df),
        "overall_bacc": round(overall_bacc * 100, 2) if overall_bacc is not None else None,
        "inference_time_s": round(elapsed, 2),
        "samples_per_minute": round(len(docs) / elapsed * 60, 1),
        "avg_doc_tokens": round(df["doc_tokens"].mean(), 1),
        "bins": bin_results,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniCheck models on long-context factual consistency datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["flan-t5-large"],
        choices=VALID_MODELS,
        help="MiniCheck model(s) to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["TofuEval-MediaS", "RAGTruth", "ExpertQA", "SciFact"],
        help=(
            "Dataset names to evaluate on. Valid values: "
            "'SciFact', 'SummHay', or any LLM-AggreFact subset name "
            f"(e.g. {list(LONG_DOC_SUBSETS)!s})."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (useful for quick experiments).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to write result JSON files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace model/dataset cache directory.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Override MiniCheck chunk_size for long-doc processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from minicheck.minicheck import MiniCheck

    all_results = []
    
    # Cache datasets so we don't reload them from disk/HF every time
    loaded_datasets = {}
    for dataset_name in args.datasets:
        print(f"\n[evaluate] Loading dataset: {dataset_name}")
        try:
            df = load_dataset_by_name(dataset_name, args.max_samples, args.cache_dir)
            if not df.empty:
                loaded_datasets[dataset_name] = df
            else:
                print(f"[evaluate] WARNING: '{dataset_name}' returned no samples, skipping.")
        except Exception as e:
            print(f"[evaluate] ERROR loading '{dataset_name}': {e}")
    
    for model_name in args.models:
        print(f"\n[evaluate] Instantiating model: {model_name}")
        scorer = None
        
        for dataset_name, df in loaded_datasets.items():
            # Safe filename: replace slashes and spaces
            safe_model = model_name.replace("/", "_").replace(" ", "_")
            safe_dataset = dataset_name.replace("/", "_").replace(" ", "_").replace("-", "_")
            out_path = output_dir / f"{safe_model}_{safe_dataset}.json"

            if out_path.exists():
                print(f"[evaluate] Result already exists: {out_path}, skipping.")
                with open(out_path) as f:
                    all_results.append(json.load(f))
                continue

            # Need to initialize model? Do it lazily if there's actual work
            if scorer is None:
                if model_name in ("Bespoke-MiniCheck-7B",):
                    scorer = MiniCheck(model_name=model_name, cache_dir=args.cache_dir, enable_prefix_caching=False)
                else:
                    scorer = MiniCheck(model_name=model_name, cache_dir=args.cache_dir)

            try:
                result = evaluate_model_on_dataset(
                    scorer=scorer,
                    model_name=model_name,
                    df=df,
                    chunk_size=args.chunk_size,
                )
            except Exception as e:
                print(f"[evaluate] ERROR evaluating '{model_name}' on '{dataset_name}': {e}")
                import traceback; traceback.print_exc()
                continue

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[evaluate] Saved: {out_path}")
            all_results.append(result)
        
        # Free up memory explicitly if multiple models
        if scorer is not None:
            del scorer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary table
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Model':<25} {'Dataset':<20} {'n':>5} {'BAcc':>6} {'Time(s)':>8} {'SPM':>7}")
        print("-" * 60)
        for r in all_results:
            bacc_str = f"{r['overall_bacc']:.1f}" if r["overall_bacc"] is not None else "N/A"
            print(
                f"{r['model']:<25} {r['dataset']:<20} "
                f"{r['n_samples']:>5} {bacc_str:>6} "
                f"{r['inference_time_s']:>8.1f} {r['samples_per_minute']:>7.0f}"
            )


if __name__ == "__main__":
    main()
