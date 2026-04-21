"""
data_loader.py
--------------
Load and normalize long-context factual-consistency datasets into a unified schema:

    {"dataset": str, "doc": str, "claim": str, "label": int, "doc_tokens": int}

Supported datasets
------------------
- LLM-AggreFact  (lytang/LLM-AggreFact)                — GATED: visit
                   https://huggingface.co/datasets/lytang/LLM-AggreFact to request access.
                   Once approved, run `huggingface-cli login` before loading.
- SciFact        (allenai/scifact via parquet files)    — open access
- SummHay        (Salesforce/summary-of-a-haystack)     — open access
"""

from __future__ import annotations

import random
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate whitespace-based token count (fast, no tokenizer required)."""
    return len(text.split())


def _to_unified(dataset: str, doc: str, claim: str, label: int) -> dict:
    doc = doc or ""
    claim = claim or ""
    return {
        "dataset": dataset,
        "doc": doc,
        "claim": claim,
        "label": int(label),
        "doc_tokens": _count_tokens(doc),
    }


# ---------------------------------------------------------------------------
# LLM-AggreFact  (GATED dataset — requires HuggingFace access approval)
# ---------------------------------------------------------------------------

# Subsets that naturally have longer grounding documents
LONG_DOC_SUBSETS = {
    "TofuEval-MediaS",
    "TofuEval-MeetB",
    "RAGTruth",
    "ExpertQA",
    "Lfqa",
}

_AGGREFACT_ACCESS_MSG = """
[data_loader] GATED DATASET: 'lytang/LLM-AggreFact' requires HuggingFace access approval.
  1. Visit: https://huggingface.co/datasets/lytang/LLM-AggreFact
  2. Click "Access repository" and agree to the terms.
  3. Run: huggingface-cli login
  Then retry.
"""


def load_llm_aggrefact(
    subsets: Optional[List[str]] = None,
    split: str = "test",
    max_samples_per_dataset: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load LLM-AggreFact benchmark (GATED – requires HuggingFace access).

    Parameters
    ----------
    subsets : list of str or None
        Dataset names to include. If None, uses LONG_DOC_SUBSETS.
    split : str
        HuggingFace split to load (default: 'test').
    max_samples_per_dataset : int or None
        If set, sample at most this many examples per subset.
    cache_dir : str or None
        HuggingFace cache directory.

    Returns
    -------
    pd.DataFrame with unified schema.

    Raises
    ------
    PermissionError  if the dataset is not yet approved for this HF token.
    """
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        hf_ds = load_dataset("lytang/LLM-AggreFact", split=split, cache_dir=cache_dir)
    except DatasetNotFoundError as e:
        if "gated" in str(e).lower():
            raise PermissionError(_AGGREFACT_ACCESS_MSG) from e
        raise

    df = pd.DataFrame(hf_ds)

    target_subsets = subsets if subsets is not None else list(LONG_DOC_SUBSETS)

    # Normalise dataset names for case-insensitive matching
    name_map = {name.lower(): name for name in df["dataset"].unique()}
    resolved = []
    for s in target_subsets:
        key = s.lower()
        if key in name_map:
            resolved.append(name_map[key])
        else:
            print(f"[data_loader] WARNING: subset '{s}' not found in LLM-AggreFact. "
                  f"Available: {list(name_map.values())}")

    df = df[df["dataset"].isin(resolved)].copy()

    rows = []
    for dataset_name, grp in df.groupby("dataset"):
        if max_samples_per_dataset is not None:
            grp = grp.sample(
                n=min(max_samples_per_dataset, len(grp)), random_state=42
            )
        for _, row in grp.iterrows():
            rows.append(_to_unified(row["dataset"], row["doc"], row["claim"], row["label"]))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SciFact  (allenai/scifact — downloaded directly from official S3 bucket)
# ---------------------------------------------------------------------------

_SCIFACT_S3_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def load_scifact(
    split: str = "validation",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,  # kept for API compatibility
) -> pd.DataFrame:
    """
    Load the AllenAI SciFact dataset directly from the official AWS S3 bucket.
    Downloads the data.tar.gz file, extracts corpus and claims in memory.

    Maps SUPPORTS→1, REFUTES→0, drops NOT_ENOUGH_INFO rows.
    Grounding doc = evidence rationale sentences (falls back to full abstract).
    """
    import urllib.request, tarfile, io, json as _json

    claims_filename = "data/claims_dev.jsonl" if split == "validation" else "data/claims_train.jsonl"
    
    try:
        req = urllib.request.Request(_SCIFACT_S3_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            tar_bytes = resp.read()
    except Exception as e:
        raise RuntimeError(f"[data_loader] Failed to download SciFact S3 tarball: {e}") from e

    corpus_rows = []
    claims_rows = []

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Skip macOS hidden files
            if member.name.split("/")[-1].startswith("._"):
                continue
            
            if member.name.endswith("corpus.jsonl"):
                f = tar.extractfile(member)
                if f:
                    corpus_rows = [_json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]
            
            elif member.name.endswith(claims_filename):
                f = tar.extractfile(member)
                if f:
                    claims_rows = [_json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]

    if not corpus_rows or not claims_rows:
        raise RuntimeError(f"[data_loader] Failed to find corpus or claims in SciFact tarball.")

    # Build corpus: doc_id (int) → list of sentences
    corpus: dict[int, list[str]] = {}
    for row in corpus_rows:
        doc_id = int(row["doc_id"])
        abstract = row.get("abstract", [])
        if isinstance(abstract, str):
            abstract = [abstract]
        corpus[doc_id] = list(abstract) if abstract else []

    label_map = {
        "SUPPORTS": 1, "REFUTES": 0,
        "SUPPORT": 1, "CONTRADICT": 0
    }
    rows = []

    for row in claims_rows:
        cited_doc_ids = row.get("cited_doc_ids") or []
        evidence      = row.get("evidence") or {}

        label: Optional[int] = None
        evidence_sents: list[str] = []

        if isinstance(evidence, dict):
            for doc_id_str, ev_list in evidence.items():
                if not isinstance(ev_list, list):
                    continue
                for ev in ev_list:
                    if not isinstance(ev, dict):
                        continue
                    ev_label = ev.get("label", "")
                    if ev_label in label_map and label is None:
                        label = label_map[ev_label]
                    for sent_idx in (ev.get("sentences") or []):
                        doc_id = int(doc_id_str)
                        abst = corpus.get(doc_id, [])
                        if sent_idx < len(abst):
                            evidence_sents.append(abst[sent_idx])

        if label is None:
            continue  # NOT_ENOUGH_INFO or unannotated

        if evidence_sents:
            doc_text = " ".join(evidence_sents)
        elif cited_doc_ids:
            abst = corpus.get(int(cited_doc_ids[0]), [])
            doc_text = " ".join(abst)
        else:
            continue

        if not doc_text.strip():
            continue

        rows.append(_to_unified("SciFact", doc_text, str(row["claim"]), label))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def _get_hf_token() -> Optional[str]:
    """Return the stored HuggingFace token, if any."""
    try:
        import huggingface_hub
        return huggingface_hub.get_token()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SummHay  (Salesforce/summary-of-a-haystack — open access)
# ---------------------------------------------------------------------------

def load_summhay(
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Adapt SummHay for binary factual-consistency evaluation.

    Positive pairs  : (haystack, insight) — insight IS in the haystack (label=1)
    Negative pairs  : (other haystack, insight) — cross-topic negatives (label=0)

    Parameters
    ----------
    max_samples : int or None
        Cap on total rows (balanced 50/50 pos/neg).
    cache_dir : str or None

    Returns
    -------
    pd.DataFrame with unified schema.
    """
    from datasets import load_dataset

    try:
        hf_ds = load_dataset(
            "Salesforce/summary-of-a-haystack",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[data_loader] WARNING: Could not load SummHay: {e}")
        return pd.DataFrame(columns=["dataset", "doc", "claim", "label", "doc_tokens"])

    # Build: (topic_id, haystack_text, insight_text)
    triples: list[tuple[str, str, str]] = []
    haystack_texts: dict[str, str] = {}

    for item in hf_ds:
        topic_id = str(item.get("topic_id", item.get("id", id(item))))

        docs_field = item.get("documents") or []
        if isinstance(docs_field, list) and docs_field:
            if isinstance(docs_field[0], dict):
                haystack_text = "\n\n".join(d.get("document_text", "") for d in docs_field)
            else:
                haystack_text = "\n\n".join(str(d) for d in docs_field)
        else:
            haystack_text = str(item.get("topic", ""))

        haystack_texts[topic_id] = haystack_text

        for subtopic in (item.get("subtopics") or []):
            for insight in (subtopic.get("insights") or []):
                insight_text = insight.get("insight", "")
                if insight_text:
                    triples.append((topic_id, haystack_text, insight_text))

    if not triples:
        print("[data_loader] WARNING: SummHay returned no insight triples. "
              "The dataset schema may have changed.")
        return pd.DataFrame(columns=["dataset", "doc", "claim", "label", "doc_tokens"])

    # Positive pairs
    rows = [_to_unified("SummHay", haystack, insight, 1)
            for _, haystack, insight in triples]

    # Cross-topic negative pairs
    all_topic_ids = list(haystack_texts.keys())
    rng = random.Random(42)
    for topic_id, _, insight in triples:
        other_ids = [t for t in all_topic_ids if t != topic_id]
        if not other_ids:
            continue
        neg_id = rng.choice(other_ids)
        rows.append(_to_unified("SummHay", haystack_texts[neg_id], insight, 0))

    df = pd.DataFrame(rows)

    if max_samples is not None and len(df) > max_samples:
        half = max_samples // 2
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        pos = pos.sample(n=min(half, len(pos)), random_state=42)
        neg = neg.sample(n=min(half, len(neg)), random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "LLM-AggreFact": load_llm_aggrefact,
    "SciFact": load_scifact,
    "SummHay": load_summhay,
}


def load_all_datasets(
    include: Optional[List[str]] = None,
    aggrefact_subsets: Optional[List[str]] = None,
    max_samples_per_dataset: Optional[int] = None,
    cache_dir: Optional[str] = None,
    skip_on_error: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate multiple datasets.

    Parameters
    ----------
    include : list of str or None
        Which loaders to call: any combination of
        ['LLM-AggreFact', 'SciFact', 'SummHay'].
        Defaults to all three.
    aggrefact_subsets : list of str or None
        Subset names within LLM-AggreFact. Defaults to LONG_DOC_SUBSETS.
    max_samples_per_dataset : int or None
        Per-dataset sample cap.
    cache_dir : str or None
    skip_on_error : bool
        If True, skip datasets that raise errors (e.g. gated access) and
        print a warning instead of raising.

    Returns
    -------
    pd.DataFrame with unified schema.
    """
    if include is None:
        include = list(DATASET_LOADERS.keys())

    frames = []
    for name in include:
        print(f"[data_loader] Loading {name} ...")
        try:
            if name == "LLM-AggreFact":
                df = load_llm_aggrefact(
                    subsets=aggrefact_subsets,
                    max_samples_per_dataset=max_samples_per_dataset,
                    cache_dir=cache_dir,
                )
            elif name == "SciFact":
                df = load_scifact(
                    max_samples=max_samples_per_dataset,
                    cache_dir=cache_dir,
                )
            elif name == "SummHay":
                df = load_summhay(
                    max_samples=max_samples_per_dataset,
                    cache_dir=cache_dir,
                )
            else:
                print(f"[data_loader] Unknown dataset '{name}', skipping.")
                continue
        except PermissionError as e:
            print(f"[data_loader] SKIPPED '{name}': {e}")
            if not skip_on_error:
                raise
            continue
        except Exception as e:
            print(f"[data_loader] ERROR loading '{name}': {e}")
            if not skip_on_error:
                raise
            continue

        if df.empty:
            print(f"[data_loader]   → (empty, skipping)")
            continue

        print(f"[data_loader]   → {len(df)} samples, "
              f"avg doc tokens: {df['doc_tokens'].mean():.0f}, "
              f"max doc tokens: {df['doc_tokens'].max()}")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["dataset", "doc", "claim", "label", "doc_tokens"])

    combined = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Total samples: {len(combined)}")
    return combined
