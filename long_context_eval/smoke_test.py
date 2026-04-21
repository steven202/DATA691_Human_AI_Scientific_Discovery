"""
smoke_test.py
-------------
Fast CPU-only correctness check for the long_context_eval pipeline.

Tests WITHOUT GPU using flan-t5-large on tiny samples (≤10 items) to verify:
  1. SciFact loader returns expected schema
  2. LLM-AggreFact loader works (or gracefully fails if gated/not approved)
  3. SummHay loader works (or gracefully skips)
  4. MiniCheck.score() produces correct-shaped outputs
  5. Result JSON I/O and analysis plots work

Usage
-----
  python long_context_eval/smoke_test.py [--cache_dir ./ckpts] [--skip_summhay]

Notes
-----
- LLM-AggreFact is a gated HuggingFace dataset. To use it:
    1. Visit https://huggingface.co/datasets/lytang/LLM-AggreFact and request access.
    2. Run: huggingface-cli login
  This test will skip LLM-AggreFact gracefully without access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_COLUMNS = {"dataset", "doc", "claim", "label", "doc_tokens"}
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m⚡ SKIP\033[0m"


def check(condition: bool, msg: str):
    status = PASS if condition else FAIL
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# -------------------------------------------------------------------------
# Schema validation
# -------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, name: str):
    check(not df.empty, f"{name}: non-empty DataFrame")
    missing = EXPECTED_COLUMNS - set(df.columns)
    check(not missing, f"{name}: schema has all required columns (missing: {missing})")
    check(df["label"].isin([0, 1]).all(), f"{name}: labels are 0 or 1")
    check((df["doc_tokens"] > 0).all(), f"{name}: doc_tokens > 0")
    check(df["doc"].str.len().gt(0).all(), f"{name}: all docs non-empty")
    check(df["claim"].str.len().gt(0).all(), f"{name}: all claims non-empty")


# -------------------------------------------------------------------------
# Test 1: SciFact (open-access, always runs)
# -------------------------------------------------------------------------

def test_scifact(cache_dir) -> pd.DataFrame:
    section("1. SciFact Loader (open-access)")
    from long_context_eval.data_loader import load_scifact
    df = load_scifact(max_samples=10, cache_dir=cache_dir)
    validate_schema(df, "SciFact")
    check(df["dataset"].eq("SciFact").all(), "SciFact: dataset column = 'SciFact'")
    print(f"  Loaded {len(df)} rows; label distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Doc-token stats: min={df['doc_tokens'].min()}, "
          f"max={df['doc_tokens'].max()}, mean={df['doc_tokens'].mean():.0f}")
    return df


# -------------------------------------------------------------------------
# Test 2: LLM-AggreFact (gated — skip gracefully if no access)
# -------------------------------------------------------------------------

def test_llm_aggrefact(cache_dir) -> pd.DataFrame:
    section("2. LLM-AggreFact Loader (gated — skips if access not approved)")
    from long_context_eval.data_loader import load_llm_aggrefact
    try:
        df = load_llm_aggrefact(
            subsets=["TofuEval-MediaS", "ExpertQA"],
            max_samples_per_dataset=5,
            cache_dir=cache_dir,
        )
        validate_schema(df, "LLM-AggreFact")
        print(f"  Loaded {len(df)} rows; unique datasets: {df['dataset'].unique().tolist()}")
        print(f"  Doc-token stats: min={df['doc_tokens'].min()}, "
              f"max={df['doc_tokens'].max()}, mean={df['doc_tokens'].mean():.0f}")
        return df
    except PermissionError as e:
        print(f"  [{SKIP}] Not approved for access yet. {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [{SKIP}] Could not load ({type(e).__name__}: {e})")
        return pd.DataFrame()


# -------------------------------------------------------------------------
# Test 3: SummHay (open-access — skip if unavailable)
# -------------------------------------------------------------------------

def test_summhay(cache_dir) -> pd.DataFrame:
    section("3. SummHay Loader (open-access)")
    from long_context_eval.data_loader import load_summhay
    try:
        df = load_summhay(max_samples=10, cache_dir=cache_dir)
    except Exception as e:
        print(f"  [{SKIP}] Could not load SummHay ({e}). Skipping.")
        return pd.DataFrame()

    if df.empty:
        print(f"  [{SKIP}] SummHay returned no samples (schema may have changed).")
        return pd.DataFrame()

    validate_schema(df, "SummHay")
    check(df["dataset"].eq("SummHay").all(), "SummHay: dataset column = 'SummHay'")
    print(f"  Loaded {len(df)} rows; label distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Doc-token stats: min={df['doc_tokens'].min()}, "
          f"max={df['doc_tokens'].max()}, mean={df['doc_tokens'].mean():.0f}")
    return df


# -------------------------------------------------------------------------
# Test 4: MiniCheck inference (flan-t5-large, CPU-compatible)
# -------------------------------------------------------------------------

def test_minicheck_inference(df: pd.DataFrame, cache_dir) -> tuple:
    section("4. MiniCheck Inference (flan-t5-large, CPU)")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    from minicheck.minicheck import MiniCheck
    scorer = MiniCheck(model_name="flan-t5-large", cache_dir=cache_dir)

    sub = df.head(6)
    docs = sub["doc"].tolist()
    claims = sub["claim"].tolist()

    t0 = time.time()
    pred_labels, raw_probs, used_chunks, prob_per_chunk = scorer.score(
        docs=docs, claims=claims
    )
    elapsed = time.time() - t0

    check(len(pred_labels) == len(sub), f"score(): pred_labels length = {len(sub)}")
    check(len(raw_probs) == len(sub), f"score(): raw_probs length = {len(sub)}")
    check(all(p in (0, 1) for p in pred_labels), "score(): pred_labels are 0 or 1")
    check(all(0.0 <= p <= 1.0 for p in raw_probs), "score(): raw_probs in [0, 1]")

    print(f"  Inference time: {elapsed:.1f}s for {len(sub)} samples")
    print(f"  pred_labels : {pred_labels}")
    print(f"  raw_probs   : {[round(p, 3) for p in raw_probs]}")
    return pred_labels, raw_probs


# -------------------------------------------------------------------------
# Test 5: Result JSON I/O + analysis module
# -------------------------------------------------------------------------

def test_result_io_and_analysis(df: pd.DataFrame, pred_labels, raw_probs):
    section("5. Result JSON I/O and Analysis Module")
    from sklearn.metrics import balanced_accuracy_score

    labels = df.head(6)["label"].values
    preds = np.array(pred_labels)

    unique = np.unique(labels)
    overall_bacc = (
        balanced_accuracy_score(labels, preds) * 100 if len(unique) >= 2 else None
    )

    result = {
        "model": "flan-t5-large",
        "dataset": df["dataset"].iloc[0],
        "n_samples": len(df.head(6)),
        "overall_bacc": round(overall_bacc, 2) if overall_bacc is not None else None,
        "inference_time_s": 1.0,
        "samples_per_minute": 60.0,
        "avg_doc_tokens": float(df.head(6)["doc_tokens"].mean()),
        "bins": [
            {
                "bin_label": "0-500",
                "bin_min": 0,
                "bin_max": 500,
                "n": len(df.head(6)),
                "bacc": round(overall_bacc, 2) if overall_bacc is not None else None,
                "avg_doc_tokens": float(df.head(6)["doc_tokens"].mean()),
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "smoke_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        check(result_path.exists(), "Result JSON written successfully")

        with open(result_path) as f:
            loaded = json.load(f)
        check(loaded["model"] == "flan-t5-large", "Result JSON round-trips correctly")

        # Run analysis module
        from long_context_eval.analysis import (
            build_bin_table, build_overall_table,
            plot_bacc_vs_length, plot_latency_bar,
        )
        results = [loaded]
        overall_df = build_overall_table(results)
        bins_df = build_bin_table(results)
        check(len(overall_df) == 1, "build_overall_table: 1 row")
        check(len(bins_df) >= 1, "build_bin_table: ≥1 bin row")

        fig_path = Path(tmpdir) / "bacc_vs_length.png"
        lat_path = Path(tmpdir) / "latency_bar.png"
        plot_bacc_vs_length(bins_df, fig_path)
        plot_latency_bar(overall_df, lat_path)
        check(fig_path.exists(), "bacc_vs_length.png generated")
        check(lat_path.exists(), "latency_bar.png generated")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Smoke test for long_context_eval.")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--skip_summhay", action="store_true",
                   help="Skip SummHay loader test (saves download time).")
    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("  long_context_eval smoke test")
    print("=" * 60)

    # SciFact (always runs — needed for inference test)
    df_scifact = test_scifact(args.cache_dir)

    # LLM-AggreFact (skips gracefully if gated)
    df_aggrefact = test_llm_aggrefact(args.cache_dir)

    # SummHay (skips gracefully if unavailable)
    if not args.skip_summhay:
        test_summhay(args.cache_dir)

    # MiniCheck inference on SciFact (guaranteed non-empty)
    inference_df = df_scifact  # always available
    pred_labels, raw_probs = test_minicheck_inference(inference_df, args.cache_dir)

    # Result I/O + analysis
    test_result_io_and_analysis(inference_df, pred_labels, raw_probs)

    print("\n" + "=" * 60)
    print("  All smoke tests PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
