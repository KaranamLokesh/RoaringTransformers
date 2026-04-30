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
