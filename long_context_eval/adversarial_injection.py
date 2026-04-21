"""
adversarial_injection.py
------------------------
Adversarial hallucination injection experiment for stress-testing MiniCheck's
chunking and fact-checking robustness.

Key idea: Inject synthetic hallucinatory strings (clearly false facts) into
specific positions of long documents — especially the MIDDLE chunks where
prior work shows LLMs struggle most with retrieval — and test whether
MiniCheck correctly identifies the claim as UNSUPPORTED / REFUTED.

Injection positions tested:
  - BEGINNING  : chunk 0 (retrieval-friendly, baseline)
  - MIDDLE     : ~50th percentile chunk (hardest retrieval position)
  - END        : last chunk (sometimes also disadvantaged)
  - SCATTERED  : inject into multiple chunks simultaneously

Hallucination types:
  - NUMERIC    : "The study had N participants" (N is wrong)
  - ENTITY     : "X was discovered by Y" (X or Y is fabricated)
  - STATIC     : "It is well known that ..." followed by false claim
  - CONTRADICT : Direct contradiction of claim elements

Usage
-----
  # Run full adversarial evaluation
  python long_context_eval/adversarial_injection.py \\
      --models flan-t5-large \\
      --datasets ExpertQA RAGTruth \\
      --max_samples 200 \\
      --output_dir results_adversarial/

  # Run only injection generation (skip MiniCheck inference)
  python long_context_eval/adversarial_injection.py \\
      --generate_only --max_samples 500 --output_dir results_adversarial/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_context_eval.data_loader import load_llm_aggrefact, load_scifact


# ---------------------------------------------------------------------------
# Hallucination generators
# ---------------------------------------------------------------------------

_FABRICATED_ENTITIES = [
    "Dr. Alexandra Chen", "Prof. Marcus Webb", "the Blackwood Institute",
    "the Journal of Theoretical Advances", "the Global Research Consortium",
    "Dr. Nikola Russo", "Prof. Elena Vasquez", "the Foundation for Open Science",
]
_FABRICATED_NUMBERS = ["12,847", "3.7 million", "94.2%", "approximately 0.001%", "7,234", "0.0001"]
_FABRICATED_YEARS = ["1987", "2054", "1843", "1776", "2001", "1999"]


def fabricated_entity() -> str:
    return random.choice(_FABRICATED_ENTITIES)


def fabricated_number() -> str:
    return random.choice(_FABRICATED_NUMBERS)


def fabricated_year() -> str:
    return random.choice(_FABRICATED_YEARS)


def fabricated_venue() -> str:
    venues = [
        "Nature", "Science", "Cell", "the Lancet", "the New England Journal of Medicine",
        "PNAS", "arXiv", "the Proceedings of the Royal Society",
    ]
    return random.choice(venues)


def fabricated_metric() -> str:
    metrics = [
        "accuracy increased by 47%", "error rate dropped to 0.3%",
        "the model achieved 99.1% F1 score", "latency was reduced by 2.3x",
        "the dataset contained 1.8M examples", "training converged in 1500 steps",
    ]
    return random.choice(metrics)


# --- Hallucination templates ----------------------------------------------------

def numeric_hallucination(doc_text: str, claim: str) -> Tuple[str, str]:
    """
    Inject a wrong numeric value into the document.
    Tries to find a number in the claim and replaces it with a different value.
    """
    numbers_in_claim = re.findall(r'\b\d+(?:\.\d+)?(?:%|million|billion|k|M|B)?\b', claim, re.I)
    if numbers_in_claim:
        orig = random.choice(numbers_in_claim)
        # Pick a different number
        alternatives = [n for n in ["12,847", "3.7 million", "94.2%", "7,234", "0.0001", "47%", "1500", "99.1%"] if n != orig and not (orig.lower() in n.lower() or n.lower() in orig.lower())]
        fake = random.choice(alternatives) if alternatives else "12,847"
        fake_sentence = (
            f" The {random.choice(['study', 'analysis', 'survey', 'report'])} involved "
            f"approximately {fake} {random.choice(['participants', 'patients', 'subjects', 'data points'])}."
        )
    else:
        fake_sentence = (
            f" According to {fabricated_entity()}, the {random.choice(['experiment', 'trial', 'study'])} "
            f"included {fabricated_number()} {random.choice(['participants', 'iterations', 'samples'])}."
        )
    return doc_text + fake_sentence, "numeric"


def entity_hallucination(doc_text: str, claim: str) -> Tuple[str, str]:
    """
    Inject a fabricated entity (fake researcher, institution, or paper)
    into the document.
    """
    templates = [
        f" This finding was later replicated by {fabricated_entity()} "
        f"and colleagues at {fabricated_entity()} (Nature, {fabricated_year()}).",
        f" The {random.choice(['landmark', 'seminal', 'influential'])} paper by "
        f"{fabricated_entity()} ({fabricated_venue()}, {fabricated_year()}) established this result.",
        f" According to {fabricated_entity()} ({fabricated_venue()}, {fabricated_year()}), "
        f"the {random.choice(['phenomenon', 'effect', 'mechanism'])} was first observed in {fabricated_year()}.",
    ]
    fake_sentence = random.choice(templates)
    return doc_text + fake_sentence, "entity"


def static_hallucination(doc_text: str, claim: str) -> Tuple[str, str]:
    """
    Inject a generic-sounding but false claim prefixed with
    'It is well known that...' to test susceptibility to authoritative-sounding text.
    """
    templates = [
        f" It is well known that {fabricated_entity()} demonstrated in {fabricated_year()} "
        f"that the {random.choice(['opposite', 'contrary', 'reverse'])} effect holds.",
        f" Contrary to the claim, {fabricated_metric()}, as shown by {fabricated_entity()} "
        f"in {fabricated_venue()} ({fabricated_year()}).",
        f" Established literature, including the work of {fabricated_entity()} "
        f"({random.choice(['Science', 'Nature', 'Cell', 'PNAS'])}, {fabricated_year()}), "
        f"demonstrates that this effect is {random.choice(['nonexistent', 'opposite', 'negligible'])}.",
    ]
    fake_sentence = random.choice(templates)
    return doc_text + fake_sentence, "static"


def contradict_hallucination(doc_text: str, claim: str) -> Tuple[str, str]:
    """
    Directly contradict a key claim element by inserting a sentence that
    says the opposite.
    """
    contradict_templates = [
        f" However, {fabricated_entity()} ({fabricated_venue()}, {fabricated_year()}) "
        f"found {random.choice(['no significant', 'the opposite', 'no evidence for'])} "
        f"this {random.choice(['relationship', 'effect', 'correlation', 'difference'])}.",
        f" In contrast, {fabricated_metric()}, contradicting the reported findings.",
        f" More recent work by {fabricated_entity()} has {random.choice(['refuted', 'overturned', 'contradicted'])} "
        f"this conclusion ({fabricated_venue()}, {fabricated_year()}).",
    ]
    fake_sentence = random.choice(contradict_templates)
    return doc_text + fake_sentence, "contradict"


HALLUCINATION_GENERATORS = {
    "numeric": numeric_hallucination,
    "entity": entity_hallucination,
    "static": static_hallucination,
    "contradict": contradict_hallucination,
}
HALLUCINATION_TYPES = list(HALLUCINATION_GENERATORS.keys())


# ---------------------------------------------------------------------------
# Chunk splitting helpers (mirrors MiniCheck's logic)
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, chunk_size_words: int = 500) -> List[str]:
    """Split document into word-count based chunks (mirrors flan-t5-large chunking)."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    from nltk.tokenize import sent_tokenize
    blocks = text.split('\n')
    tokenized_blocks = [sent_tokenize(block) for block in blocks]
    sentences = []
    for block in tokenized_blocks:
        sentences.extend(block)
        sentences.append('\n')
    sentences = sentences[:-1]

    chunks = []
    current_chunk = []
    current_word_count = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > chunk_size_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks if chunks else [text]


def inject_into_chunk(chunks: List[str], injection: str, position: str) -> str:
    """
    Inject hallucination string into chunks at the specified position.

    Parameters
    ----------
    chunks : list of document chunks
    injection : hallucination text to inject
    position : 'beginning', 'middle', 'end', or 'scattered'

    Returns
    -------
    Reconstructed document string with injection embedded in a chunk.
    """
    if not chunks:
        return injection

    n = len(chunks)

    if position == "beginning":
        target_idx = 0
    elif position == "middle":
        target_idx = n // 2
    elif position == "end":
        target_idx = n - 1
    elif position == "scattered":
        # Inject into multiple positions: first, middle, last
        positions = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        injection_chunks = injection.split('. ')
        step = max(1, len(injection_chunks) // len(positions))
        result_chunks = []
        inj_idx = 0
        for i, chunk in enumerate(chunks):
            if i in positions and inj_idx < len(injection_chunks):
                # Add hallucination sentences before this chunk
                end_idx = min(inj_idx + step, len(injection_chunks))
                inj_text = '. '.join(injection_chunks[inj_idx:end_idx])
                result_chunks.append(inj_text + '. ' + chunk)
                inj_idx = end_idx
            else:
                result_chunks.append(chunk)
        return '\n\n'.join(result_chunks)

    # Single injection position
    chunks[target_idx] = chunks[target_idx] + ' ' + injection
    return '\n\n'.join(chunks)


# ---------------------------------------------------------------------------
# Adversarial dataset generation
# ---------------------------------------------------------------------------

def generate_adversarial_sample(
    row: pd.Series,
    hallucination_type: str,
    injection_position: str,
    rng: random.Random,
) -> dict:
    """
    Take a positive (label=1) sample and inject a hallucination to flip it to 0.
    The hallucination is placed at the specified position in the document.
    """
    doc = row["doc"]
    claim = row["claim"]

    generator = HALLUCINATION_GENERATORS[hallucination_type]
    injected_doc, h_type = generator(doc, claim)

    # Split and reconstruct to place injection at correct position
    chunks = split_into_chunks(doc)
    injected_doc = inject_into_chunk(chunks, generator(doc, claim)[0].replace(doc, "").strip(), injection_position)

    # Actually, simpler: just append to the right position
    # Re-do properly:
    chunks = split_into_chunks(doc)
    hallucination_sentence, _ = generator(doc, claim)
    hallucination_sentence = hallucination_sentence[len(doc):].strip()

    injected_doc = inject_into_chunk(chunks, hallucination_sentence, injection_position)

    return {
        "dataset": row.get("dataset", "Adversarial"),
        "doc": injected_doc,
        "claim": claim,
        "label": 0,  # Injected document should now refute/support the false claim
        "original_label": 1,
        "doc_tokens": len(injected_doc.split()),
        "hallucination_type": h_type,
        "injection_position": injection_position,
        "original_doc_tokens": row.get("doc_tokens", 0),
    }


def generate_adversarial_dataset(
    source_df: pd.DataFrame,
    hallucination_types: List[str] = None,
    injection_positions: List[str] = None,
    max_per_config: Optional[int] = None,
    min_doc_tokens: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate adversarial test set from an existing dataset.

    Takes only positive (label=1) samples because we want to inject
    hallucinations that make a TRUE claim become FALSE.

    Parameters
    ----------
    source_df : DataFrame with doc, claim, label columns
    hallucination_types : list of hallucination types to test
    injection_positions : list of positions to inject at
    max_per_config : max samples per (type, position) combination
    min_doc_tokens : only use docs with at least this many tokens
    seed : random seed
    """
    if hallucination_types is None:
        hallucination_types = HALLUCINATION_TYPES
    if injection_positions is None:
        injection_positions = ["beginning", "middle", "end", "scattered"]

    # Filter to positive samples with reasonably long documents
    pos_df = source_df[
        (source_df["label"] == 1) &
        (source_df["doc_tokens"] >= min_doc_tokens)
    ].copy()

    if pos_df.empty:
        print(f"[adversarial] WARNING: No positive samples with >= {min_doc_tokens} tokens found.")
        return pd.DataFrame(columns=[
            "dataset", "doc", "claim", "label", "doc_tokens",
            "hallucination_type", "injection_position", "original_label", "original_doc_tokens"
        ])

    rng = random.Random(seed)
    rows = []

    # Balanced: sample roughly equal numbers per (type, position)
    samples_per_config = min(max_per_config or 50, len(pos_df))

    for h_type in hallucination_types:
        for pos in injection_positions:
            subset = pos_df.sample(n=min(samples_per_config, len(pos_df)), random_state=seed)
            for _, row in subset.iterrows():
                try:
                    adv_row = generate_adversarial_sample(row, h_type, pos, rng)
                    rows.append(adv_row)
                except Exception as e:
                    print(f"[adversarial] ERROR generating sample: {e}")
                    continue

    df = pd.DataFrame(rows)
    print(f"[adversarial] Generated {len(df)} adversarial samples across "
          f"{len(hallucination_types)} types × {len(injection_positions)} positions")
    print(f"  Hallucination types: {hallucination_types}")
    print(f"  Injection positions: {injection_positions}")
    return df


# ---------------------------------------------------------------------------
# Evaluation with MiniCheck
# ---------------------------------------------------------------------------

def evaluate_adversarial(
    df: pd.DataFrame,
    model_name: str,
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Run MiniCheck inference on adversarial dataset and compute detection rates.
    """
    from minicheck.minicheck import MiniCheck

    print(f"\n[adversarial] Evaluating {model_name} on {len(df)} adversarial samples...")

    scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)

    docs = df["doc"].tolist()
    claims = df["claim"].tolist()
    labels = df["label"].values

    t0 = time.time()
    pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)
    elapsed = time.time() - t0

    pred_labels = np.array(pred_labels)
    correct = (pred_labels == labels).sum()
    detection_rate = correct / len(labels) * 100 if len(labels) > 0 else 0

    # Per hallucination type breakdown
    type_results = {}
    for h_type in df["hallucination_type"].unique():
        mask = df["hallucination_type"] == h_type
        sub_labels = labels[mask]
        sub_preds = pred_labels[mask]
        if len(np.unique(sub_labels)) >= 2 or len(np.unique(sub_preds)) >= 2:
            from sklearn.metrics import balanced_accuracy_score
            bacc = balanced_accuracy_score(sub_labels, sub_preds) * 100
        else:
            bacc = (sub_labels == sub_preds).mean() * 100
        type_results[h_type] = {
            "n": int(mask.sum()),
            "detection_rate": round((sub_labels == sub_preds).mean() * 100, 2),
            "bacc": round(bacc, 2),
        }

    # Per injection position breakdown
    position_results = {}
    for pos in df["injection_position"].unique():
        mask = df["injection_position"] == pos
        sub_labels = labels[mask]
        sub_preds = pred_labels[mask]
        if len(np.unique(sub_labels)) >= 2 or len(np.unique(sub_preds)) >= 2:
            from sklearn.metrics import balanced_accuracy_score
            bacc = balanced_accuracy_score(sub_labels, sub_preds) * 100
        else:
            bacc = (sub_labels == sub_preds).mean() * 100
        position_results[pos] = {
            "n": int(mask.sum()),
            "detection_rate": round((sub_labels == sub_preds).mean() * 100, 2),
            "bacc": round(bacc, 2),
        }

    return {
        "model": model_name,
        "n_samples": len(df),
        "detection_rate": round(detection_rate, 2),
        "inference_time_s": round(elapsed, 2),
        "samples_per_minute": round(len(df) / elapsed * 60, 1),
        "by_hallucination_type": type_results,
        "by_injection_position": position_results,
        "per_sample": [
            {
                "hallucination_type": df.iloc[i]["hallucination_type"],
                "injection_position": df.iloc[i]["injection_position"],
                "original_label": int(df.iloc[i]["original_label"]),
                "predicted_label": int(pred_labels[i]),
                "correct": bool(pred_labels[i] == labels[i]),
                "prob": round(float(raw_probs[i]), 4),
            }
            for i in range(len(df))
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial hallucination injection for MiniCheck evaluation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["flan-t5-large"],
        choices=["roberta-large", "deberta-v3-large", "flan-t5-large", "Bespoke-MiniCheck-7B"],
        help="MiniCheck model(s) to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ExpertQA", "RAGTruth"],
        help="Source datasets for adversarial sample generation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Max samples per (type, position) config.",
    )
    parser.add_argument(
        "--min_doc_tokens",
        type=int,
        default=200,
        help="Minimum document token length to consider for injection.",
    )
    parser.add_argument(
        "--hallucination_types",
        nargs="+",
        default=["numeric", "entity", "contradict"],
        choices=HALLUCINATION_TYPES,
        help="Types of hallucinations to inject.",
    )
    parser.add_argument(
        "--injection_positions",
        nargs="+",
        default=["beginning", "middle", "end"],
        choices=["beginning", "middle", "end", "scattered"],
        help="Where in the document to inject hallucinations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_adversarial",
        help="Directory to write results.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Model cache directory.",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate adversarial dataset, skip MiniCheck inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source datasets
    source_dfs = []
    for ds_name in args.datasets:
        key = ds_name.lower()
        if key == "scifact":
            df = load_scifact(max_samples=args.max_samples, cache_dir=args.cache_dir)
        else:
            df = load_llm_aggrefact(
                subsets=[ds_name],
                max_samples_per_dataset=args.max_samples,
                cache_dir=args.cache_dir,
            )
        if not df.empty:
            df["source_dataset"] = ds_name
            source_dfs.append(df)
            print(f"[adversarial] Loaded {len(df)} samples from {ds_name}")

    if not source_dfs:
        print("[adversarial] ERROR: No source data loaded. Check dataset names / access.")
        return

    combined_source = pd.concat(source_dfs, ignore_index=True)
    print(f"[adversarial] Combined source: {len(combined_source)} samples")

    # Generate adversarial dataset
    adv_df = generate_adversarial_dataset(
        source_df=combined_source,
        hallucination_types=args.hallucination_types,
        injection_positions=args.injection_positions,
        max_per_config=args.max_samples,
        min_doc_tokens=args.min_doc_tokens,
        seed=args.seed,
    )

    # Save generated adversarial dataset
    adv_path = output_dir / "adversarial_dataset.json"
    adv_df.to_json(adv_path, orient="records", indent=2)
    print(f"[adversarial] Saved adversarial dataset: {adv_path}")

    if args.generate_only:
        print("[adversarial] --generate_only set; skipping inference.")
        return

    # Run inference
    for model_name in args.models:
        result_path = output_dir / f"adversarial_{model_name.replace('/', '_')}.json"
        if result_path.exists():
            print(f"[adversarial] Result exists: {result_path}, skipping.")

        print(f"\n[adversarial] Evaluating {model_name}...")
        try:
            result = evaluate_adversarial(
                df=adv_df,
                model_name=model_name,
                cache_dir=args.cache_dir,
            )
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[adversarial] Saved: {result_path}")
            print(f"  Detection rate: {result['detection_rate']:.1f}%")
            print(f"  By hallucination type: {result['by_hallucination_type']}")
            print(f"  By injection position: {result['by_injection_position']}")
        except Exception as e:
            print(f"[adversarial] ERROR evaluating '{model_name}': {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
