# Roaring Transformers

Reference implementation and benchmark suite for the paper **"The Roaring Transformer: When Compressed Bitmaps Are the Right Attention-Mask Representation, and When They Are Not"** (Lokesh Karanam, Purna Sreekar Karanam, Sri Ram Pemmasani).

The project integrates [Roaring Bitmaps](https://roaringbitmap.org/) into a Transformer training pipeline to represent its sparse metadata: attention masks, Top-K gradient indices, and pruning masks. It then measures Roaring against purpose-built encodings of the same structure, not just against a dense float32 mask. The result is a decision rule rather than a blanket recommendation.

## Findings

- **Size: Roaring is never the smallest representation.** Across 15 mask configurations a general-purpose byte compressor is smaller; for example, zstd stores a causal sliding-window mask at n=8,192 in 1.1 KB versus 42.1 KB for Roaring. On the diffuse index sets that magnitude Top-K produces, delta-plus-varint coding is smaller than Roaring at every configuration tested. Roaring becomes the smallest encoding only when indices form contiguous runs.
- **Composition depends on one property of the mask: whether every component is identical for every query row.**
  - If all components are determined by position alone (causal window, padding, global sink tokens), the composed mask has a closed form over two integer arrays (start and end of each row). That (start, length) form is up to 12x faster than Roaring. Do not use Roaring here.
  - If any component varies by row (content-dependent, learned, or randomized attention), the closed form and the row broadcast of the dense and bitset baselines are lost. Roaring is then the fastest representation from n=2,048 upward: 6.3x to 16.3x faster than the best alternative and 8x to 18x faster than a dense array. Below that size the dense forms fit in cache; at n=1,024 Roaring is about 1.2x slower.
  - The switch is categorical. At n=8,192 a single content-dependent entry per row moves Roaring from 6.8x slower than the best alternative to 25x faster, with no change in fragmentation. This holds whether the content-dependent entries are scattered or clustered.
- **In-place mutation: do not use Roaring for masks that are updated incrementally.** A dense array is 11.8x (n=16,384) to 52x (n=1,024) faster.
- **Layout matters.** One flat bitmap over the n² key space with `run_optimize()` is 2.85x smaller and 31x faster to compose than one bitmap per query row (n=8,192). `RoaringAttentionMask` uses the flat layout.
- **Training is unaffected.** On WikiText-103 a 17.7M-parameter model trained with 5 seeds reaches a final validation loss equivalent to the dense-mask baseline within ±0.05 nats (difference −0.010 nats, 95% CI −0.035 to +0.015; two one-sided tests, p=0.003).
- **GPU memory is unchanged.** Masks are expanded to dense tensors before they reach the accelerator (peak VRAM 9,517 vs 9,518 MB). The saving is in host memory: at n=512 the mask is 136x smaller than its dense float32 tensor in the per-row layout used for the training run, and 501x smaller (2,093 bytes) in the recommended flat layout.

Validation is at the 17.7M-parameter scale with sequence length 512 on a single GPU. Larger models, multi-node gradient exchange, and a native GPU kernel for Roaring masks were not tested.

## Components

| Component | Vanilla Transformer | Roaring Transformer |
|-----------|---------------------|---------------------|
| Attention mask | Dense `n × n` tensor | Flat, run-optimized Roaring Bitmap on the host, realized to a dense tensor once per sequence length and cached |
| Gradient indices | Not tracked | Top-K index sets encoded as Roaring Bitmaps (with error feedback) to measure payload size |
| Pruning | None | Indices of surviving weights stored as Roaring Bitmaps; pruned weights are zeroed and compute stays dense |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.12, NumPy 2.5, SciPy 1.18, pyroaring 1.1.0 (CRoaring 4.6.1) and zstandard 0.25 on macOS (Apple M2 Pro). The training run used PyTorch on a single NVIDIA GPU with CUDA.

## Usage

Run from the repository root:

```python
from src.roaring_attention import RoaringAttentionMask

mask = RoaringAttentionMask(seq_len=512, window_size=256, causal=True)
print(mask.get_compressed_size_bytes())   # 2093
print(mask.compression_ratio())           # about 501 relative to dense float32

mask.add_global_tokens([0, 1, 2, 3])      # union with global sink tokens
dense = mask.realize_dense(device="cpu")  # (512, 512) float mask: 0.0 allowed, -inf blocked
```

`src.roaring_transformer.RoaringTransformer` is the counterpart of `src.vanilla_transformer.VanillaTransformer` that builds its attention mask this way.

Main benchmarks (CPU, no arguments needed; each writes a JSON file in `experiments/`):

```bash
python experiments/run_v4_measurements.py   # sizes against trivial baselines
python experiments/run_v5_timings.py        # composition timings and in-place mutation
python experiments/run_v6_measurements.py   # clustered vs uniform content-dependence sweep
python experiments/run_v6_stats.py          # equivalence test on the training results
```

The training comparison needs a CUDA GPU:

```bash
python experiments/run_revision_experiments.py --dataset bpe --config wikitext-103-raw-v1 \
    --seeds 5 --steps 2000 --eval_steps 200
```

## Project Structure

```
src/
  roaring_attention.py         # RoaringAttentionMask: flat, run-optimized mask bitmap
  roaring_gradient.py          # Top-K gradient index encoding with Roaring
  roaring_pruning.py           # Pruning masks stored as Roaring Bitmaps
  roaring_transformer.py       # Transformer LM using the three components above
  vanilla_transformer.py       # Dense-mask baseline
  data.py                      # WikiText (BPE and word-level) and synthetic datasets
experiments/
  run_revision_experiments.py  # WikiText-103 training comparison (GPU)
  run_v4_measurements.py       # Size baselines
  run_v5_timings.py            # Composition and mutation timings
  run_v6_measurements.py       # Clustered sweep, window ablation re-measure, machine info
  run_v6_stats.py              # Confidence interval and TOST equivalence test
  generate_v4_figures.py       # Paper figures
  *.json, figures_*/           # Raw results and rendered figures
  run_comparison.py            # Early end-to-end demo on synthetic data (not used for the paper)
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

## Citation

The paper is currently under review. Until it is published, please cite the paper title and this repository:

```bibtex
@misc{karanam2026roaring,
  title        = {The Roaring Transformer: When Compressed Bitmaps Are the Right
                  Attention-Mask Representation, and When They Are Not},
  author       = {Karanam, Lokesh and Karanam, Purna Sreekar and Pemmasani, Sri Ram},
  year         = {2026},
  note         = {Manuscript under review},
  howpublished = {\url{https://github.com/KaranamLokesh/RoaringTransformers}}
}
```
