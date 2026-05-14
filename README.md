# Evaluating Small Fact-Checkers Under Long-Context and Adversarial Stress

**Course Project: Human + AI Scientific Discovery**  
**Track: Benchmarking & Dataset Paper**  
**Chenan Wang, Spring 2026**

Evaluates MiniCheck-style small fact-checkers (`flan-t5-large`, `Bespoke-MiniCheck-7B`) under two stress conditions: natural long contexts (up to 74k tokens) and adversarial hallucination injection.

## Key Findings

- **"Lost in the middle"**: 9--10% BAcc degradation at 1k--2k tokens on ExpertQA, with recovery at longer lengths; finer 250-token binning on RAGTruth reveals a subtler U-shape
- **Throughput collapse**: 250x slowdown on 74k-token SummHay documents (3.9 SPM vs 1000+ SPM on short docs)
- **Adversarial vulnerability**: 57.5% fooling rate; entity hallucinations achieve 0.0% conditional detection on RAGTruth
- **No positional rescue**: Injection detection is position-invariant

## Repository Structure

```
.
├── src/          # Main evaluation and analysis code
│   ├── data_loader.py          # Dataset loading (LLM-AggreFact, SciFact, SummHay)
│   ├── evaluate.py             # Evaluation pipeline (MiniCheck inference + bin BAcc)
│   ├── rebin_predictions.py    # Adaptive per-dataset rebinning for U-shape analysis
│   ├── adversarial_injection.py
│   ├── final_analysis.py       # Aggregate results and build summary tables
│   ├── plot_context_degradation.py   # Figure: context degradation by dataset
│   ├── plot_lost_in_middle.py        # Figure: lost-in-middle + throughput
│   └── plot_adversarial.py           # Figure: adversarial results
├── results/                    # Evaluation output (JSON + per-sample predictions CSV)
├── final_results/              # Aggregated CSVs and publication figures (PNG)
├── requirements.txt
└── DATA691_Human_AI_Scientific_Discovery_final_report/
    ├── paper.tex               # NeurIPS 2026 final report
    └── references.bib          # BibTeX bibliography
```

## Setup

```bash
# Create environment
conda create -n minicheck-eval python=3.10
conda activate minicheck-eval

# Install dependencies
pip install -r requirements.txt

# Install MiniCheck
git clone https://github.com/Liyan06/MiniCheck.git
pip install -e MiniCheck/
```

## Datasets

Datasets are loaded automatically via HuggingFace `datasets` on first run. A `--cache_dir` flag is available for offline use.

| Dataset | Source | Avg Tokens | Samples |
|---------|--------|-----------|---------|
| ExpertQA | LLM-AggreFact (HuggingFace) | 433 | 3,702 |
| RAGTruth | LLM-AggreFact (HuggingFace) | 412 | 16,371 |
| SciFact | AllenAI (HuggingFace) | 57 | 188 |
| SummHay | Custom loader (HuggingFace) | 74,488 | 100 |
| TofuEval-MediaS | LLM-AggreFact (HuggingFace) | 778 | 726 |
| TofuEval-MeetB | LLM-AggreFact (HuggingFace) | 792 | 100 |
| Lfqa | LLM-AggreFact (HuggingFace) | 320 | 100 |

## Reproducing Results

### 1. Run evaluations

```bash
# Full evaluation on all datasets with flan-t5-large
python src/evaluate.py \
    --models flan-t5-large \
    --datasets TofuEval-MediaS RAGTruth ExpertQA SciFact SummHay TofuEval-MeetB Lfqa \
    --max_samples 300 \
    --output_dir results/ \
    --cache_dir ./ckpts

# Evaluate Bespoke-MiniCheck-7B (requires GPU with sufficient VRAM)
python src/evaluate.py \
    --models Bespoke-MiniCheck-7B \
    --datasets RAGTruth ExpertQA \
    --max_samples 3000 \
    --output_dir results/
```

### 2. Rebin predictions (for U-shape analysis)

```bash
python src/rebin_predictions.py
```

This reads per-sample prediction CSVs from `results/` and produces adaptive bin-level BAcc in `final_results/summary_bins.csv`.

### 3. Generate figures

```bash
python src/plot_context_degradation.py   # Figure: context degradation
python src/plot_lost_in_middle.py         # Figures: lost-in-middle + throughput
python src/final_analysis.py              # Aggregate analysis
```

All figures are saved to `final_results/`.

### 4. Compile the paper

```bash
cd DATA691_Human_AI_Scientific_Discovery_final_report/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Models Evaluated

| Model | Size | Context Window | Chunk Size |
|-------|------|---------------|------------|
| flan-t5-large | 770M | 512 tokens | 500 words |
| Bespoke-MiniCheck-7B | 7B | 32k tokens | 32k tokens |

Additional OpenRouter API models (Gemma-4-26B, Gemma-4-31B, GPT-OSS-120B, GPT-OSS-20B, Trinity-Large) were probed on a subset of datasets.

## Author

Chenan Wang — Spring 2026
