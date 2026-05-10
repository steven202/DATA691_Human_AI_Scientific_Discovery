# Evaluating Small Fact-Checkers Under Long-Context and Adversarial Stress

**Course Project: Human + AI Scientific Discovery**

This project evaluates MiniCheck-style small fact-checkers under two stress conditions:

1. **Long-context natural stress**: Testing on documents ranging from hundreds to 74k tokens
2. **Adversarial injection**: Injecting synthetic hallucinations at beginning, middle, and end positions

## Key Findings

- **"Lost in the middle" phenomenon**: Both models show 9-10% BAcc degradation at 1k-2k tokens on ExpertQA, with recovery at longer lengths
- **Extreme context penalty**: 250x throughput reduction on 74k-token documents (3.9 SPM vs 1000+ SPM)
- **Adversarial vulnerability**: 57.5% fooling rate; entity hallucinations hardest to detect (38.3%)
- **No positional rescue**: Injection detection is position-invariant

## Repository Structure

```
.
├── long_context_eval/      # Main evaluation code
│   ├── data_loader.py      # Dataset loading
│   ├── evaluate.py         # Evaluation pipeline
│   ├── adversarial_injection.py  # Adversarial injection experiments
│   ├── final_analysis.py   # Analysis and aggregation
│   └── plot_*.py          # Visualization scripts
├── final_results/          # Results and figures
│   ├── summary_*.csv      # Aggregated results
│   └── *.png              # Generated figures
└── DATA691_Human_AI_Scientific_Discovery_final_report/
    └── paper.tex          # NeurIPS 2026 format final report
```

## Setup

```bash
conda create -n minicheck-eval python=3.10
conda activate minicheck-eval
pip install pandas matplotlib seaborn numpy
```

## Reproducing Results

```bash
# Run evaluations
cd long_context_eval
python evaluate.py --model flan-t5-large --dataset ExpertQA

# Run adversarial injection experiments
python adversarial_injection.py

# Generate analysis and figures
python final_analysis.py --results_dir ../results \
    --adv_results ../results_adversarial \
    --openrouter_results ../results_openrouter \
    --output_dir ../final_results
```

## Models Evaluated

- **flan-t5-large** (770M): MiniCheck's default lightweight fact-checker
- **Bespoke-MiniCheck-7B** (7B): Larger specialized fact-checker
- OpenRouter API models: Gemma-4-26B, Gemma-4-31B, GPT-OSS-120B, GPT-OSS-20B, Trinity-Large

## Datasets

| Dataset | Avg Tokens | Samples | Type |
|---------|-----------|---------|------|
| ExpertQA | 433 | 3,702 | Health QA |
| RAGTruth | 412 | 16,371 | RAG hallucination |
| SciFact | 57 | 188 | Scientific fact-checking |
| SummHay | 74,488 | 100 | Long summaries |
| TofuEval-MediaS | 778 | 726 | Multi-topic Wikipedia |
| TofuEval-MeetB | 792 | 100 | Meeting summaries |
| Lfqa | 320 | 100 | Long-form QA |

## References

- Tang et al. MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents (EMNLP 2024)
- Laban et al. The Good, The Bad, and The Summary (arXiv 2024)
- Wang et al. AlignScore: Evaluating Factual Consistency (ACL 2023)

## Author

Chenan Wang - Spring 2026
