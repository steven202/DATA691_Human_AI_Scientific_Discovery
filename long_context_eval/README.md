# Long-Context Evaluation of MiniCheck

This directory contains the experimental pipeline for evaluating the **MiniCheck** model family (`flan-t5-large`, `Bespoke-MiniCheck-7B`) on naturally long-context factual consistency datasets — without any artificial noise injection.

The core question: **does MiniCheck's accuracy degrade as grounding documents get longer?**

---

## Project History

This project has two phases:

| Phase | Description |
|-------|-------------|
| **Midterm** | Original pipeline: 5 datasets, 2 models, context degradation analysis |
| **Final** | Extended: 7 datasets, adversarial injection, OpenRouter API benchmarks |

---

## Datasets

| Name | HuggingFace path | Avg doc tokens |
|---|---|---|
| TofuEval-MediaS | `lytang/LLM-AggreFact` | ~1,500 |
| TofuEval-MeetB | `lytang/LLM-AggreFact` | ~2,000 |
| RAGTruth | `lytang/LLM-AggreFact` | ~600 |
| ExpertQA | `lytang/LLM-AggreFact` | ~800 |
| LFQA | `lytang/LLM-AggreFact` | ~500 |
| SciFact | `allenai/scifact` | ~200 |
| SummHay | `Salesforce/summary-of-a-haystack` | ~5,000+ |

All datasets are normalised to a unified `(dataset, doc, claim, label)` schema.

---

## Setup

```bash
# From the repo root
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
pip install datasets scikit-learn matplotlib seaborn accelerate nltk
```

---

## Phase 1: Midterm — Original Evaluation Pipeline

### 1. Smoke Test (CPU-only, no GPU needed)

Runs in a few minutes and validates the entire pipeline on tiny samples:

```bash
python long_context_eval/smoke_test.py --cache_dir ./ckpts
```

### 2. Full Evaluation

```bash
python long_context_eval/evaluate.py \
    --models flan-t5-large \
    --datasets TofuEval-MediaS TofuEval-MeetB RAGTruth ExpertQA SciFact SummHay \
    --max_samples 300 \
    --output_dir results/ \
    --cache_dir ./ckpts
```

To also evaluate `Bespoke-MiniCheck-7B` (requires GPU + vLLM):

```bash
pip install "minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main"

python long_context_eval/evaluate.py \
    --models flan-t5-large Bespoke-MiniCheck-7B \
    --datasets TofuEval-MediaS TofuEval-MeetB RAGTruth ExpertQA SciFact SummHay \
    --max_samples 300 \
    --output_dir results/ \
    --cache_dir ./ckpts
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--models` | `flan-t5-large` | One or more model names |
| `--datasets` | see above | Dataset names to evaluate |
| `--max_samples` | None (all) | Cap per dataset |
| `--output_dir` | `results/` | Output directory |
| `--cache_dir` | None | HuggingFace cache dir |
| `--chunk_size` | model default | Override MiniCheck chunk size |

**Resumable**: already-computed `results/<model>_<dataset>.json` files are skipped automatically.

### 3. Midterm Analysis

Generate tables and figures from the original evaluation:

```bash
python long_context_eval/analysis.py --results_dir results/
```

Outputs written to `results/`:

| File | Description |
|---|---|
| `summary_overall.csv` | Overall BAcc + latency per (model, dataset) |
| `summary_bins.csv` | BAcc broken down by document-length bin |
| `bacc_vs_length.png` | Line chart: BAcc vs. length bin |
| `latency_bar.png` | Bar chart: inference time per model × dataset |
| `bacc_heatmap.png` | Heatmap of BAcc across all bins (requires seaborn) |

---

## Phase 2: Final — Extended Evaluation

### 4. Adversarial Hallucination Injection (Week 5-6, Novel Experiment)

Stress-test MiniCheck by injecting synthetic hallucinations into document chunks.

```bash
python long_context_eval/adversarial_injection.py \
    --models flan-t5-large \
    --datasets ExpertQA RAGTruth \
    --max_samples 200 \
    --hallucination_types numeric entity contradict \
    --injection_positions beginning middle end \
    --output_dir results_adversarial/
```

Generates 3 types × 3 positions × N samples of adversarial test cases, then evaluates MiniCheck's detection rate. Results show MiniCheck detects only **42.5%** of injected hallucinations (below 50% random chance).

### 5. OpenRouter API Benchmarks (Week 7-8)

Benchmark free-tier OpenRouter models as fact-checkers:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python long_context_eval/openrouter_client.py \
    --models google/gemma-4-26b-a4b-it:free \
    --datasets ExpertQA \
    --max_samples 100 \
    --output_dir results_openrouter/
```

### 6. Final Comprehensive Analysis

Aggregate all experiments (original + extended + adversarial + API) into unified figures:

```bash
python long_context_eval/final_analysis.py \
    --results_dir results/ \
    --adv_results results_adversarial/ \
    --openrouter_results results_openrouter/ \
    --output_dir final_results/
```

Outputs written to `final_results/`:

| File | Description |
|---|---|
| `summary_all_models.csv` | All models and datasets combined |
| `summary_adversarial.csv` | Adversarial detection rates |
| `unified_bacc_heatmap.png` | Heatmap across all experiments |
| `model_comparison.png` | BAcc and throughput comparison |
| `adversarial_results.png` | Adversarial detection by type and position |
| `openrouter_comparison.png` | OpenRouter vs local models |

---

## Result Schema

Each `results/<model>_<dataset>.json` follows:

```json
{
  "model": "flan-t5-large",
  "dataset": "SciFact",
  "n_samples": 300,
  "overall_bacc": 72.4,
  "inference_time_s": 41.3,
  "samples_per_minute": 436.0,
  "avg_doc_tokens": 218.5,
  "bins": [
    {
      "bin_label": "0-500",
      "bin_min": 0,
      "bin_max": 500,
      "n": 240,
      "bacc": 74.1,
      "avg_doc_tokens": 195.3
    },
    ...
  ]
}
```

Length bins: `0-500`, `500-1000`, `1000-2000`, `2000-4000`, `4000+` (whitespace tokens).

---

## Results Summary

### Overall Balanced Accuracy

| Model | Dataset | n | BAcc (%) | Avg Tokens | SPM |
|-------|---------|---|----------|------------|-----|
| Bespoke-MiniCheck-7B | ExpertQA | 3,702 | 58.2 | 433 | 697 |
| Bespoke-MiniCheck-7B | RAGTruth | 16,371 | 84.1 | 412 | 807 |
| Bespoke-MiniCheck-7B | SciFact | 188 | 75.9 | 57 | 3,083 |
| Bespoke-MiniCheck-7B | SummHay | 100 | 100.0 | 74,488 | 3.9 |
| Bespoke-MiniCheck-7B | TofuEval-MediaS | 726 | 75.9 | 778 | 529 |
| flan-t5-large | ExpertQA | 3,702 | 59.0 | 433 | 1,150 |
| flan-t5-large | Lfqa | 100 | 88.3 | 320 | 1,583 |
| flan-t5-large | RAGTruth | 16,371 | 78.0 | 412 | 1,234 |
| flan-t5-large | SciFact | 188 | 71.1 | 57 | 2,037 |
| flan-t5-large | SummHay | 100 | 100.0 | 74,488 | 13.8 |
| flan-t5-large | TofuEval-MediaS | 726 | 73.6 | 778 | 962 |
| flan-t5-large | TofuEval-MeetB | 100 | 79.4 | 792 | 861 |
| Gemma-4-26B (OpenRouter) | ExpertQA | 20 | 55.0 | 556 | 13 |

### Adversarial Injection Detection Rates (flan-t5-large, random chance = 50%)

| Type/Position | Detection Rate |
|--------------|----------------|
| **Overall** | **42.5%** |
| Numeric | 46.7% |
| Entity | 38.3% |
| Contradict | 42.5% |
| Beginning | 42.5% |
| Middle | 43.3% |
| End | 41.7% |

---

## Key Findings

1. **Context Degradation Confirmed**: BAcc drops as document length increases (ExpertQA: 58.7% → 48.1% at 1000-2000 tokens for Bespoke-MiniCheck-7B)

2. **No Positional Rescue**: Beginning/middle/end injection positions are equally vulnerable to adversarial hallucinations

3. **Extreme Context Survival**: 100% BAcc on 74k-token SummHay documents, but throughput drops to ~4 samples/min

4. **Adversarial Vulnerability**: 57.5% of injected hallucinations fool MiniCheck; entity-type (fabricated citations) are most effective (only 38.3% detection)

5. **General LLMs Underperform Specialized Fact-checkers**: Gemma-4-26B (55.0%) < flan-t5-large (59.0%) despite being a much larger general-purpose model

---

## File Structure

```
long_context_eval/
├── __init__.py
├── data_loader.py          # Dataset loading & normalization
├── evaluate.py             # Original evaluation runner (CLI) [Midterm]
├── analysis.py             # Original analysis script [Midterm]
├── final_analysis.py       # Comprehensive analysis [Final]
├── smoke_test.py           # Fast CPU-only correctness check
├── adversarial_injection.py # Adversarial hallucination injection [Final]
├── openrouter_client.py    # OpenRouter API benchmark client [Final]
├── utils.py
├── README.md               # This file

results/                    # Original + extended evaluation results [Midterm + Final]
├── flan_t5_large_*.json
├── Bespoke-MiniCheck-7B_*.json
├── summary_overall.csv
├── summary_bins.csv
├── bacc_vs_length.png
├── latency_bar.png
└── bacc_heatmap.png

results_adversarial/       # Adversarial injection results [Final]
├── adversarial_dataset.json
└── adversarial_flan_t5_large.json

results_openrouter/         # OpenRouter API benchmarks [Final]
└── Gemma_4_26B_ExpertQA.json

final_results/              # Aggregated analysis [Final]
├── summary_all_models.csv
├── summary_adversarial.csv
├── unified_bacc_heatmap.png
├── model_comparison.png
├── adversarial_results.png
└── openrouter_comparison.png
```

## Report

- **Midterm**: `DATA691_Human_AI_Scientific_Discovery_proposal/midterm.tex`
- **Final**: `DATA691_Human_AI_Scientific_Discovery_proposal/final_report.tex`
