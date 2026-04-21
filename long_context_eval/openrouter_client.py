"""
openrouter_client.py
-------------------
OpenRouter API client for fact-checking benchmarks.

Provides two interfaces:
1. OpenRouterCheck : a MiniCheck-compatible scorer using OpenRouter models
2. openrouter_benchmark() : run full benchmarks across multiple free models

Free models available on OpenRouter (all have free tier):
  - google/gemma-4-26b-a4b-it:free
  - google/gemma-4-31b-it:free
  - openai/gpt-oss-120b:free
  - openai/gpt-oss-20b:free
  - microsoft/wizardlm-3-7b:free
  - qwen/qwen3-30b-a3b:free
  - deepseek/deepseek-prover-v2:free
  - deepseek/deepseek-r1:free

API Setup:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  # Or set inline in code

Usage
-----
  # Direct benchmark
  python long_context_eval/openrouter_client.py \\
      --models google/gemma-4-26b-a4b-it:free \\
      --datasets ExpertQA \\
      --max_samples 100 \\
      --output_dir results_openrouter/

  # As a module
  from long_context_eval.openrouter_client import OpenRouterCheck, openrouter_benchmark
  scorer = OpenRouterCheck(model_name="google/gemma-4-26b-a4b-it:free")
  pred_labels, probs, _, _ = scorer.score(docs, claims)
"""

from __future__ import annotations

import os
import sys
import time
import json
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# OpenRouter API client
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


@dataclass
class OpenRouterConfig:
    model: str
    api_key: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 256


def get_default_api_key() -> Optional[str]:
    return os.environ.get(OPENROUTER_API_KEY_ENV)


def build_factcheck_prompt(doc: str, claim: str) -> str:
    """
    Build a fact-checking prompt for an LLM.
    The LLM is asked to answer YES/NO whether the claim is supported by the document.
    """
    return f"""You are a fact-checking assistant. Given a document and a claim, determine whether the claim is supported by the document.

Answer only YES if the claim is supported by the document, or NO if the claim is not supported or is contradicted by the document.

---

DOCUMENT:
{doc[:8000] if len(doc) > 8000 else doc}

---

CLAIM: {claim}

---

ANSWER (YES or NO only):"""


def parse_yes_no_response(response_text: str) -> Tuple[int, float]:
    """
    Parse YES/NO from model response.
    Returns (pred_label, confidence) where pred_label is 1 (YES=supported) or 0 (NO=unsupported).
    """
    text = response_text.strip().upper()

    # Look for YES/NO pattern
    yes_match = re.search(r'\bYES\b', text)
    no_match = re.search(r'\bNO\b', text)

    # Count occurrences to decide
    yes_count = len(re.findall(r'\bYES\b', text))
    no_count = len(re.findall(r'\bNO\b', text))

    if yes_count > no_count:
        return 1, yes_count / (yes_count + no_count + 1e-9)
    elif no_count > yes_count:
        return 0, no_count / (yes_count + no_count + 1e-9)
    else:
        # Default: check first occurrence
        if yes_match and (no_match is None or yes_match.start() < no_match.start()):
            return 1, 0.5
        elif no_match:
            return 0, 0.5
        else:
            # Fallback: try to find indicators
            if any(w in text for w in ["SUPPORTED", "CORRECT", "ACCURATE", "TRUE"]):
                return 1, 0.5
            elif any(w in text for w in ["NOT SUPPORTED", "CONTRADICT", "INCORRECT", "FALSE"]):
                return 0, 0.5
            else:
                # Ambiguous — default to 1 (conservative)
                return 1, 0.5


def make_openrouter_request(
    prompt: str,
    config: OpenRouterConfig,
    api_key: Optional[str] = None,
) -> dict:
    """Make a single request to OpenRouter API."""
    import urllib.request
    import urllib.error

    key = api_key or config.api_key or get_default_api_key()
    if not key:
        raise ValueError(
            f"OpenRouter API key not found. Set {OPENROUTER_API_KEY_ENV} environment variable "
            "or pass api_key parameter."
        )

    url = f"{OPENROUTER_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://minicheck.github.io",
        "X-Title": "MiniCheck Long-Context Evaluation",
    }
    data = {
        "model": config.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=config.timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_openrouter_request_batch(
    prompts: List[str],
    config: OpenRouterConfig,
    api_key: Optional[str] = None,
    batch_size: int = 10,
) -> List[dict]:
    """Make batched requests to OpenRouter API."""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Currently sequential; OpenRouter doesn't support true batch in free tier
        for prompt in batch:
            for attempt in range(config.max_retries):
                try:
                    result = make_openrouter_request(prompt, config, api_key)
                    results.append(result)
                    break
                except Exception as e:
                    if attempt == config.max_retries - 1:
                        # Return a dummy error response
                        results.append({"error": str(e), "choices": [{"message": {"content": "ERROR"}}]})
                    time.sleep(1 * (attempt + 1))  # backoff
    return results


class OpenRouterCheck:
    """
    MiniCheck-compatible fact-checker using OpenRouter API models.

    Usage:
        scorer = OpenRouterCheck(
            model_name="google/gemma-4-26b-a4b-it:free",
            api_key="sk-or-v1-...",
        )
        pred_labels, probs, _, _ = scorer.score(docs=docs, claims=claims)
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 256,
        batch_size: int = 5,
        cache_dir: Optional[str] = None,  # kept for API compatibility
        chunk_size: Optional[int] = None,  # not used for API models
    ):
        self.model_name = model_name
        self.api_key = api_key or get_default_api_key()
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError(
                f"OpenRouter API key not found. Set {OPENROUTER_API_KEY_ENV} "
                "environment variable or pass api_key parameter."
            )

        # Validate connection
        try:
            test_config = OpenRouterConfig(model=model_name, api_key=self.api_key, timeout=10)
            make_openrouter_request(
                "Respond with EXACTLY one word: TESTING",
                test_config,
            )
            print(f"[OpenRouterCheck] Connected to {model_name}")
        except Exception as e:
            print(f"[OpenRouterCheck] Warning: API connection test failed: {e}")

    def score(
        self,
        docs: List[str],
        claims: List[str],
        chunk_size: Optional[int] = None,  # Not used for API models
    ) -> Tuple[List[int], List[float], List[List[str]], List[List[float]]]:
        """
        Score document-claim pairs using OpenRouter model.

        Returns
        -------
        pred_labels : list of 0/1 (0=unsupported, 1=supported)
        probs : list of confidence scores [0, 1]
        used_chunks : dummy (empty) for API compatibility
        prob_per_chunk : dummy (empty) for API compatibility
        """
        assert len(docs) == len(claims), "docs and claims must have same length"

        prompts = [
            build_factcheck_prompt(doc, claim)
            for doc, claim in zip(docs, claims)
        ]

        print(f"[OpenRouterCheck] Running {len(prompts)} samples through {self.model_name}...")
        t0 = time.time()

        responses = make_openrouter_request_batch(
            prompts,
            OpenRouterConfig(
                model=self.model_name,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            api_key=self.api_key,
            batch_size=self.batch_size,
        )

        elapsed = time.time() - t0
        print(f"[OpenRouterCheck] Completed in {elapsed:.1f}s "
              f"({len(prompts)/elapsed*60:.1f} samples/min)")

        pred_labels = []
        probs = []
        used_chunks = []
        prob_per_chunk = []

        for resp in responses:
            if "error" in resp:
                pred_labels.append(1)  # Default to supported on error
                probs.append(0.5)
            else:
                content = resp["choices"][0]["message"]["content"]
                pred, prob = parse_yes_no_response(content)
                pred_labels.append(pred)
                probs.append(prob)

        return pred_labels, probs, used_chunks, prob_per_chunk


# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------

FREE_MODELS = {
    # Gemma family
    "google/gemma-4-26b-a4b-it:free": {"name": "Gemma-4-26B", "provider": "Google"},
    "google/gemma-4-31b-it:free": {"name": "Gemma-4-31B", "provider": "Google"},
    "google/gemma-3-27b-it:free": {"name": "Gemma-3-27B", "provider": "Google"},

    # GPT-OSS
    "openai/gpt-oss-120b:free": {"name": "GPT-OSS-120B", "provider": "OpenAI"},
    "openai/gpt-oss-20b:free": {"name": "GPT-OSS-20B", "provider": "OpenAI"},

    # WizardLM
    "microsoft/wizardlm-3-7b:free": {"name": "WizardLM-3-7B", "provider": "Microsoft"},

    # Qwen
    "qwen/qwen3-30b-a3b:free": {"name": "Qwen3-30B-A3B", "provider": "Qwen"},

    # DeepSeek
    "deepseek/deepseek-prover-v2:free": {"name": "DeepSeek-Prover-V2", "provider": "DeepSeek"},
    "deepseek/deepseek-r1:free": {"name": "DeepSeek-R1", "provider": "DeepSeek"},

    # Other notable free models
    "anthropic/claude-3-haiku:free": {"name": "Claude-3-Haiku", "provider": "Anthropic"},
    "meta-llama/llama-4-maverick:free": {"name": "Llama-4-Maverick", "provider": "Meta"},
    "mistralai/mistral-nemo:free": {"name": "Mistral-Nemo", "provider": "Mistral"},
    "nvidia/llama-4-megatron:free": {"name": "Llama-4-Megatron", "provider": "NVIDIA"},
}

# Models that work well for fact-checking
FACTCHECK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "microsoft/wizardlm-3-7b:free",
    "qwen/qwen3-30b-a3b:free",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def openrouter_benchmark(
    model_names: List[str],
    df: pd.DataFrame,
    output_dir: Path,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 120,
) -> List[dict]:
    """
    Run OpenRouter fact-checking benchmarks on a dataset.

    Parameters
    ----------
    model_names : list of OpenRouter model IDs
    df : DataFrame with doc, claim, label columns
    output_dir : directory to save results
    api_key : OpenRouter API key
    max_retries, timeout : request parameters

    Returns
    -------
    List of result dicts
    """
    from sklearn.metrics import balanced_accuracy_score

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in model_names:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        result_path = output_dir / f"openrouter_{safe_name}.json"

        if result_path.exists():
            print(f"\n[benchmark] Result exists: {result_path}, skipping.")
            with open(result_path) as f:
                results.append(json.load(f))
            continue

        print(f"\n{'='*60}")
        print(f"  OpenRouter Model: {model_name}")
        print(f"  Dataset: {df['dataset'].iloc[0]} ({len(df)} samples)")
        print(f"{'='*60}")

        try:
            scorer = OpenRouterCheck(
                model_name=model_name,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
            )

            docs = df["doc"].tolist()
            claims = df["claim"].tolist()
            labels = df["label"].values

            t0 = time.time()
            pred_labels, probs, _, _ = scorer.score(docs=docs, claims=claims)
            elapsed = time.time() - t0

            pred_labels = np.array(pred_labels)

            overall_bacc = balanced_accuracy_score(labels, pred_labels) * 100

            # Per-length-bin BAcc
            bin_results = []
            bins = [
                (0, 500, "0-500"),
                (500, 1000, "500-1000"),
                (1000, 2000, "1000-2000"),
                (2000, 4000, "2000-4000"),
                (4000, int(1e9), "4000+"),
            ]
            df_copy = df.copy()
            df_copy["pred"] = pred_labels
            df_copy["doc_tokens"] = df_copy["doc"].apply(lambda x: len(x.split()))

            for lo, hi, label in bins:
                sub = df_copy[(df_copy["doc_tokens"] >= lo) & (df_copy["doc_tokens"] < hi)]
                if len(sub) == 0:
                    continue
                sub_labels = sub["label"].values
                sub_preds = sub["pred"].values
                if len(np.unique(sub_labels)) < 2 and len(np.unique(sub_preds)) < 2:
                    bacc = None
                else:
                    bacc = balanced_accuracy_score(sub_labels, sub_preds) * 100
                bin_results.append({
                    "bin_label": label,
                    "bin_min": lo,
                    "bin_max": hi if hi < 1e8 else -1,
                    "n": len(sub),
                    "bacc": round(bacc, 2) if bacc is not None else None,
                    "avg_doc_tokens": round(sub["doc_tokens"].mean(), 1),
                })

            result = {
                "model": model_name,
                "model_short": FREE_MODELS.get(model_name, {}).get("name", model_name),
                "provider": FREE_MODELS.get(model_name, {}).get("provider", "Unknown"),
                "dataset": df["dataset"].iloc[0],
                "n_samples": len(df),
                "overall_bacc": round(overall_bacc, 2),
                "inference_time_s": round(elapsed, 2),
                "samples_per_minute": round(len(df) / elapsed * 60, 1),
                "avg_doc_tokens": round(df["doc"].apply(lambda x: len(x.split())).mean(), 1),
                "bins": bin_results,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[benchmark] Saved: {result_path}")
            print(f"  Overall BAcc: {overall_bacc:.1f}%")
            print(f"  Time: {elapsed:.1f}s ({result['samples_per_minute']:.1f} samples/min)")
            results.append(result)

        except Exception as e:
            print(f"[benchmark] ERROR with '{model_name}': {e}")
            import traceback; traceback.print_exc()
            continue

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenRouter free models for fact-checking."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"],
        help="OpenRouter model IDs to benchmark.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ExpertQA", "RAGTruth"],
        help="Datasets to evaluate on.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum samples per dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_openrouter",
        help="Directory to write results.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key. Can also set OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries per request.",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available free models and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable free models on OpenRouter:")
        print(f"{'Model ID':<50} {'Name':<25} {'Provider'}")
        print("-" * 100)
        for model_id, info in FREE_MODELS.items():
            print(f"{model_id:<50} {info['name']:<25} {info['provider']}")
        return

    api_key = args.api_key or get_default_api_key()
    if not api_key:
        print("ERROR: OpenRouter API key not found.")
        print(f"Set the {OPENROUTER_API_KEY_ENV} environment variable or pass --api_key")
        return

    from long_context_eval.data_loader import load_llm_aggrefact, load_scifact

    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        print(f"\n[benchmark] Loading dataset: {ds_name}")
        key = ds_name.lower()
        if key == "scifact":
            df = load_scifact(max_samples=args.max_samples)
        else:
            df = load_llm_aggrefact(
                subsets=[ds_name],
                max_samples_per_dataset=args.max_samples,
            )

        if df.empty:
            print(f"[benchmark] WARNING: '{ds_name}' returned no samples, skipping.")
            continue

        print(f"[benchmark] Loaded {len(df)} samples from {ds_name}")
        results = openrouter_benchmark(
            model_names=args.models,
            df=df,
            output_dir=output_dir / ds_name,
            api_key=api_key,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        all_results.extend(results)

    if all_results:
        print("\n" + "=" * 70)
        print("OPENROUTER BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Model':<30} {'Dataset':<15} {'n':>5} {'BAcc':>6} {'Time(s)':>8} {'SPM':>7}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['model_short']:<30} {r['dataset']:<15} "
                f"{r['n_samples']:>5} {r['overall_bacc']:>6.1f} "
                f"{r['inference_time_s']:>8.1f} {r['samples_per_minute']:>7.1f}"
            )


if __name__ == "__main__":
    main()
