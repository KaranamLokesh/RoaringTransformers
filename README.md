# Roaring Transformers

The first Transformer implementation that integrates **Roaring Bitmaps** into the training pipeline for memory-efficient sparse attention masks, gradient compression, and weight pruning.

## Architecture

| Component | Vanilla Transformer | Roaring Transformer |
|-----------|-------------------|-------------------|
| Attention Mask | Dense `O(n²)` tensor | Roaring Bitmap → realized on-demand |
| Gradient Sync | Full dense gradients | Top-K indices packed in Roaring Bitmaps |
| Pruning | N/A | Alive-weight indices in Roaring Bitmaps |

## Quick Start

```bash
pip install -r requirements.txt
python experiments/run_comparison.py --seq_lengths 512 1024 2048
```

## Project Structure

```
src/
  __init__.py
  vanilla_transformer.py    # Standard Transformer baseline
  roaring_attention.py       # Roaring Bitmap attention mask manager
  roaring_gradient.py        # Roaring gradient compression
  roaring_pruning.py         # Roaring pruning mask
  roaring_transformer.py     # Full Roaring Transformer
  data.py                    # Data loading utilities
experiments/
  run_comparison.py          # Main experiment runner
```

## Reproducing

The table and figure numbers below follow the current revision of the paper (manuscript ACRT-2026-0062, v6).
Every script writes its results next to itself in `experiments/`. The CPU benchmarks need `zstandard` and
`scipy` in addition to the core requirements (`pip install -r requirements.txt`).

| Paper item | Script | Result file |
|---|---|---|
| Table 1 (dense mask memory) | analytic: n² × 4 bytes | none |
| Table 2 (model configuration) | `experiments/run_revision_experiments.py` | `revision_main.json` (`meta.model_cfg`) |
| Table 3 (hardware, software, seeds) | `experiments/run_v6_measurements.py` (sysctl, library versions) | `v6_results.json` (`meta`), plus `meta` of `v4_results.json` and `v5_results.json` |
| Table 4 (mask representation size), Figure 1 | `experiments/run_v4_measurements.py`; figure: `experiments/generate_v4_figures.py` | `v4_results.json` (`mask_scaling`, `pattern_sweep`) |
| Table 5 (gradient index size), Figure 2 | `experiments/run_v4_measurements.py`; figure: `experiments/generate_v4_figures.py` | `v4_results.json` (`gradient_index`, `gradient_layout`) |
| Tables 6, 7, 8 (composition: structured, unstructured, content-dependence sweep) with min–max over 3 repetitions; Section 5.6 (in-place mutation) | `experiments/run_v5_timings.py` | `v5_results.json` |
| Table 9 (uniform vs clustered content-dependent entries) | `experiments/run_v6_measurements.py` | `v6_results.json` (`sweep_uniform_same_process`, `sweep_clustered`) |
| Table 10 (window-size ablation) | sizes: `experiments/run_v5_timings.py`; composition columns with spread: `experiments/run_v6_measurements.py` | `v5_results.json` (`window_ablation`), `v6_results.json` (`window_ablation`) |
| Table 11 (convergence), Table 13 (hydration), Figures 5 and 6 | `experiments/run_revision_experiments.py --dataset bpe --config wikitext-103-raw-v1 --seeds 5 --steps 2000 --eval_steps 200` on a CUDA GPU, output saved as `revision_main.json`; figures: `experiments/generate_v4_figures.py` | `revision_main.json` |
| Table 11 equivalence test (Welch CI, TOST at ±0.05 nats) | `experiments/run_v6_stats.py` | `v6_stats.json` |
| Table 12 (pruning ablation) | the same command with `--prune`, output saved as `revision_pruned.json` | `revision_pruned.json` |
| Figures 3 and 4 (composition workloads, content-dependence crossover), Figure 7 (mask construction cost) | `experiments/generate_v4_figures.py` | `v4_results.json` |

CPU benchmarks use 25 timed trials after 5 warm-up runs. The whole setup-and-measure cycle is repeated 3 times,
and the median of the per-repetition medians is reported with their min–max. Random operands use NumPy seed 0.
Training uses seeds 0 to 4.
