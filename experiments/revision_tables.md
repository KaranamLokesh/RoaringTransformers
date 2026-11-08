# Regenerated tables (round-2 revision)

Source: `revision_main.json` / `revision_pruned.json` (wikitext-103-raw-v1, 17,735,936 params, 5 seeds, 2000 steps, seq_len=512, window=256, device=cuda).

All CPU measurements below use window=256, matching the training default (was window=128 in the submitted Table 3 — R3.4).


## Table 3 — attention mask representation size (KB)

Held constant: causal sliding-window pattern, window=256, one mask per sequence length. Dense float32 is the worst-case baseline; the remaining columns are the trivial baselines R1 asked for.

| Seq len | Dense f32 | Packed bitset | (start,len)/row | zstd(bitset) | lzma(bitset) | Roaring per-row (published) | Roaring flat+opt |
|---|---|---|---|---|---|---|---|
| 256 | 256.0 | 8.0 | 2.0 | 0.1 | 0.3 | 3.8 | **1.0** |
| 512 | 1,024.0 | 32.0 | 4.0 | 0.2 | 0.3 | 7.5 | **2.0** |
| 1,024 | 4,096.0 | 128.0 | 8.0 | 0.3 | 0.3 | 15.0 | **4.2** |
| 2,048 | 16,384.0 | 512.0 | 16.0 | 0.4 | 0.3 | 30.0 | **8.6** |
| 4,096 | 65,536.0 | 2,048.0 | 32.0 | 0.5 | 0.6 | 60.0 | **18.5** |
| 8,192 | 262,144.0 | 8,192.0 | 64.0 | 1.1 | 1.6 | 120.0 | **42.1** |
| 16,384 | 1,048,576.0 | 32,768.0 | 128.0 | 3.4 | 5.2 | 240.0 | **104.5** |

## Table 4 — Top-K gradient index encoding (bytes)

Adds the 17.7M row to match the trained model (R3.4) and the delta+varint baseline (R1.2). Ratios are index-payload only.

| Params | Sparsity | k | Raw uint32 | Roaring | delta+varint | zstd(bitset) | Roaring vs raw | Best |
|---|---|---|---|---|---|---|---|---|
| 50,000 | 0.1% | 50 | 200 | 116 | 89 | 181 | 1.72x | delta+varint |
| 50,000 | 0.5% | 250 | 1,000 | 516 | 377 | 514 | 1.94x | delta+varint |
| 50,000 | 1% | 500 | 2,000 | 1,016 | 632 | 879 | 1.97x | delta+varint |
| 50,000 | 5% | 2,500 | 10,000 | 5,016 | 2,504 | 2,367 | 1.99x | zstd(bitset) |
| 50,000 | 10% | 5,000 | 20,000 | 8,208 | 5,000 | 3,450 | 2.44x | zstd(bitset) |
| 100,000 | 0.1% | 100 | 400 | 224 | 197 | 293 | 1.79x | delta+varint |
| 100,000 | 0.5% | 500 | 2,000 | 1,024 | 763 | 991 | 1.95x | delta+varint |
| 100,000 | 1% | 1,000 | 4,000 | 2,024 | 1,283 | 1,668 | 1.98x | delta+varint |
| 100,000 | 5% | 5,000 | 20,000 | 10,024 | 5,006 | 4,654 | 2.00x | zstd(bitset) |
| 100,000 | 10% | 10,000 | 40,000 | 15,096 | 10,000 | 6,850 | 2.65x | zstd(bitset) |
| 500,000 | 0.1% | 500 | 2,000 | 1,072 | 944 | 1,330 | 1.87x | delta+varint |
| 500,000 | 0.5% | 2,500 | 10,000 | 5,072 | 3,840 | 4,917 | 1.97x | delta+varint |
| 500,000 | 1% | 5,000 | 20,000 | 10,072 | 6,401 | 8,300 | 1.99x | delta+varint |
| 500,000 | 5% | 25,000 | 100,000 | 50,072 | 25,030 | 23,582 | 2.00x | zstd(bitset) |
| 500,000 | 10% | 50,000 | 200,000 | 65,608 | 50,000 | 34,496 | 3.05x | zstd(bitset) |
| 1,000,000 | 0.1% | 1,000 | 4,000 | 2,136 | 1,889 | 2,624 | 1.87x | delta+varint |
| 1,000,000 | 0.5% | 5,000 | 20,000 | 10,136 | 7,667 | 9,778 | 1.97x | delta+varint |
| 1,000,000 | 1% | 10,000 | 40,000 | 20,136 | 12,858 | 16,563 | 1.99x | delta+varint |
| 1,000,000 | 5% | 50,000 | 200,000 | 100,136 | 50,075 | 47,125 | 2.00x | zstd(bitset) |
| 1,000,000 | 10% | 100,000 | 400,000 | 126,380 | 100,000 | 69,372 | 3.17x | zstd(bitset) |
| 5,000,000 | 0.1% | 5,000 | 20,000 | 10,624 | 9,421 | 12,816 | 1.88x | delta+varint |
| 5,000,000 | 0.5% | 25,000 | 100,000 | 50,624 | 38,216 | 48,450 | 1.98x | delta+varint |
| 5,000,000 | 1% | 50,000 | 200,000 | 100,624 | 64,035 | 81,817 | 1.99x | delta+varint |
| 5,000,000 | 5% | 250,000 | 1,000,000 | 500,624 | 250,371 | 229,640 | 2.00x | zstd(bitset) |
| 5,000,000 | 10% | 500,000 | 2,000,000 | 626,964 | 500,000 | 336,591 | 3.19x | zstd(bitset) |
| 10,000,000 | 0.1% | 10,000 | 40,000 | 21,232 | 18,835 | 25,539 | 1.88x | delta+varint |
| 10,000,000 | 0.5% | 50,000 | 200,000 | 101,232 | 76,342 | 96,824 | 1.98x | delta+varint |
| 10,000,000 | 1% | 100,000 | 400,000 | 201,232 | 127,930 | 163,830 | 1.99x | delta+varint |
| 10,000,000 | 5% | 500,000 | 2,000,000 | 1,001,232 | 500,762 | 460,449 | 2.00x | zstd(bitset) |
| 10,000,000 | 10% | 1,000,000 | 4,000,000 | 1,254,036 | 1,000,002 | 676,213 | 3.19x | zstd(bitset) |
| 17,735,936 | 0.1% | 17,735 | 70,940 | 37,646 | 33,411 | 45,149 | 1.88x | delta+varint |
| 17,735,936 | 0.5% | 88,679 | 354,716 | 179,534 | 135,612 | 171,938 | 1.98x | delta+varint |
| 17,735,936 | 1% | 177,359 | 709,436 | 356,894 | 226,906 | 290,576 | 1.99x | delta+varint |
| 17,735,936 | 5% | 886,796 | 3,547,184 | 1,775,768 | 888,071 | 818,907 | 2.00x | zstd(bitset) |
| 17,735,936 | 10% | 1,773,593 | 7,094,372 | 2,222,012 | 1,773,594 | 1,201,439 | 3.19x | zstd(bitset) |

## Table 5 — training convergence (CORRECTED)

| Metric | Vanilla | Roaring |
|---|---|---|
| Final val. loss (ppl) | 5.16 ± 0.02 (174.6) | 5.15 ± 0.01 (172.8) |
| Avg step time | 172.0 ± 1.02 ms | 168.8 ± 0.06 ms |
| Peak training VRAM | 9,517 MB | 9,518 MB |
| Mask memory @ seq_len=512 | 1,024.0 KB | 7.5 KB |
| Mask compression vs dense f32 | 1.0x | 136.5x |

Equivalence: Welch p=0.364, Mann-Whitney p=0.222 (fail to reject equality).

**Changes from the submitted Table 5:**

1. Mask memory was reported as 256.0 KB / 3.8 KB / 68.2x — those are seq_len=256 values. At the actual seq_len=512 they are 1,024.0 KB / 7.5 KB / 136.5x.
2. The "Grad Compression 1.32x" row came from the synthetic `paper_results.json` sparsity ablation, not this run. Removed pending a real measurement.
3. The "Pruned Ratio 30.0%" and "Prune Mask Compr. 6.0x" rows were from a separate pruning ablation while the text states pruning is disabled here (R1.3). Split out below.
4. Step time now carries its standard deviation. The 3.19 ms gap is statistically significant but is a **baseline artifact**: `VanillaTransformer._create_dense_mask` rebuilds the mask every forward pass with a 512-iteration Python loop, while `RoaringTransformer` caches it. Any dense-mask caching yields the same gain (R3.2). The vanilla sd of 1.02 ms vs Roaring's 0.06 ms is the signature of that per-step rebuild.

## Pruning ablation (separated from Table 5 per R1.3)

| Metric | Vanilla | Roaring + 30% pruning |
|---|---|---|
| Final val. loss (ppl) | 5.16 (174.6) | 5.24 (189.4) |

Welch p=0.000127 — pruning is a genuine quality/sparsity tradeoff, not a free optimization.


## Window-size ablation (representation level, replaces old Table 6)

At seq_len=8,192. The submitted Table 6 claimed the compression ratio is invariant to window size, measured on a synthetic corpus at seq_len=256. That invariance is a property of the representation, not of training, so it is measured directly here.

| Window | Mask density | Roaring flat+opt (KB) | zstd (KB) | vs dense f32 | Compose flat (ms) | Compose dense (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| 32 | 0.4% | 42.1 | 0.9 | 6,222x | 0.15 | 4.68 | 31.1x |
| 64 | 0.8% | 42.1 | 0.9 | 6,222x | 0.15 | 4.57 | 30.5x |
| 128 | 1.6% | 42.1 | 1.0 | 6,222x | 0.15 | 4.70 | 30.7x |
| 256 | 3.1% | 42.1 | 1.1 | 6,222x | 0.15 | 4.56 | 29.5x |
| 512 | 6.1% | 42.1 | 1.4 | 6,222x | 0.16 | 4.61 | 28.4x |
| 1024 | 11.7% | 42.1 | 2.0 | 6,222x | 0.16 | 4.71 | 29.4x |
| full | 50.0% | 42.1 | 9.9 | 6,222x | 0.15 | 4.90 | 31.8x |

## Still outstanding (requires a GPU training run)

- The **training-loss** arm of the window-size ablation (loss is stable across window sizes) and the **gradient-sparsity** ablation both require training runs. `run_revision_experiments.py` implements neither; they need to be added and run on WikiText-103.
