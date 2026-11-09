#!/usr/bin/env python3
"""
Regenerate every table and figure that does NOT require a new GPU training run.

Fixes the round-2 defects:
  * All figures were built from paper_results.json (synthetic corpus, seq_len=256)
    while Table 5 came from revision_main.json (WikiText-103, seq_len=512, CUDA).
    Figure 5 therefore contradicted Table 5 (57.2/56.4 ms vs 172.0/168.8 ms).
  * Table 5's mask-memory row reported seq_len=256 values for a seq_len=512 run.
  * Table 3 used window=128 while the training default is 256.
  * Table 4 stopped at 10M params though the model has 17.7M.
  * No trivial baselines anywhere.

Still requires a GPU run (NOT produced here): Tables 6 and 7, the window-size and
gradient-sparsity ablations, which need training and do not exist in
run_revision_experiments.py at all.

Usage:
    python experiments/generate_revision_outputs.py
"""

import json
import lzma
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import zstandard as zstd
from pyroaring import BitMap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from run_baseline_comparison import build_flat, delta_varint_size, ZSTD  # noqa: E402

OUT_FIG = os.path.join(HERE, 'figures_revision')
OUT_TAB = os.path.join(HERE, 'revision_tables.md')
ZSTD = zstd.ZstdCompressor(level=6)
WINDOW = 256          # matches the training default
SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384]

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.3})


def causal_rows(n, w=WINDOW):
    return [range(max(0, i - w), i + 1) for i in range(n)]


def log_xticks(ax, ticks):
    """Label a log x-axis with exactly `ticks`, suppressing the minor decades
    that otherwise overprint the labels."""
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# --------------------------------------------------------------------------- #
# Table 3 / Figures 1, 2, 8 -- mask representation (CPU only)
# --------------------------------------------------------------------------- #
def mask_scaling():
    rows_out = []
    for n in SEQ_LENS:
        rows = causal_rows(n)

        t0 = time.perf_counter()
        per_row = [BitMap(r) for r in rows]
        t_roaring = (time.perf_counter() - t0) * 1000
        per_row_bytes = sum(len(b.serialize()) for b in per_row)

        t0 = time.perf_counter()
        flat = build_flat(rows, n, True)
        t_flat = (time.perf_counter() - t0) * 1000
        flat_bytes = len(flat.serialize())

        # dense construction, exactly as VanillaTransformer._create_dense_mask does it
        t0 = time.perf_counter()
        dense = torch.full((n, n), float('-inf'))
        for i, r in enumerate(rows):
            dense[i, r.start:r.stop] = 0.0
        t_dense = (time.perf_counter() - t0) * 1000

        bits = np.zeros((n, n), dtype=bool)
        for i, r in enumerate(rows):
            bits[i, r.start:r.stop] = True
        packed = np.packbits(bits).tobytes()

        rows_out.append({
            'seq_len': n,
            'dense_f32_bytes': n * n * 4,
            'packed_bitset_bytes': len(packed),
            'startlen_bytes': 8 * n,
            'zstd_bytes': len(ZSTD.compress(packed)),
            'lzma_bytes': len(lzma.compress(packed, preset=6)),
            'roaring_per_row_bytes': per_row_bytes,
            'roaring_flat_opt_bytes': flat_bytes,
            'dense_build_ms': t_dense,
            'roaring_per_row_build_ms': t_roaring,
            'roaring_flat_build_ms': t_flat,
        })
    return rows_out


def fig_mask_memory(ms):
    n = [r['seq_len'] for r in ms]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n, [r['dense_f32_bytes'] / 1024 for r in ms], 'o-', color='#c0392b',
            label='Dense float32 (O(n²))')
    ax.plot(n, [r['packed_bitset_bytes'] / 1024 for r in ms], 's-', color='#e67e22',
            label='Packed bitset (1 bit/elem)')
    ax.plot(n, [r['startlen_bytes'] / 1024 for r in ms], '^-', color='#7f8c8d',
            label='(start, length) per row')
    ax.plot(n, [r['roaring_per_row_bytes'] / 1024 for r in ms], 'v-', color='#8e44ad',
            label='Roaring, per-row (as published)')
    ax.plot(n, [r['roaring_flat_opt_bytes'] / 1024 for r in ms], 'D-', color='#2980b9',
            label='Roaring, flat + run_optimize')
    ax.plot(n, [r['zstd_bytes'] / 1024 for r in ms], 'x--', color='#27ae60',
            label='zstd(bitset)')
    ax.set(xscale='log', yscale='log', xlabel='Sequence length',
           ylabel='Mask memory (KB, log scale)')
    log_xticks(ax, n)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title(f'Attention mask representation size (causal, window={WINDOW})')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig1_mask_memory.png'))
    plt.close(fig)


def fig_creation_time(ms):
    n = [r['seq_len'] for r in ms]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n, [r['dense_build_ms'] for r in ms], 'o-', color='#c0392b', label='Dense')
    ax.plot(n, [r['roaring_per_row_build_ms'] for r in ms], 'v-', color='#8e44ad',
            label='Roaring, per-row')
    ax.plot(n, [r['roaring_flat_build_ms'] for r in ms], 'D-', color='#2980b9',
            label='Roaring, flat + run_optimize')
    ax.set(xscale='log', yscale='log', xlabel='Sequence length',
           ylabel='Mask construction time (ms, log scale)')
    log_xticks(ax, n)
    ax.legend(fontsize=8); ax.set_title('Mask construction cost')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig8_mask_creation_time.png'))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Window-size ablation -- representation-level, CPU only
#
# Replaces the substantive claim of the submitted Table 6 ("compression ratio is
# invariant to window size"), which was measured on a synthetic corpus at
# seq_len=256. Whether that invariance holds is a property of the representation,
# not of training, so it is measurable here without a GPU run.
# --------------------------------------------------------------------------- #
WINDOWS = [32, 64, 128, 256, 512, 1024, None]


def window_ablation(n=8192, repeats=5):
    import statistics
    from run_baseline_comparison import timeit

    out = []
    for w in WINDOWS:
        rows = [range(max(0, i - w) if w is not None else 0, i + 1) for i in range(n)]
        flat = build_flat(rows, n, True)
        per_row = sum(len(BitMap(r).serialize()) for r in rows)
        bits = np.zeros((n, n), dtype=bool)
        for i, r in enumerate(rows):
            bits[i, r.start:r.stop] = True
        packed = np.packbits(bits).tobytes()

        pad = build_flat([range(0, n - 37)] * n, n, True)
        glob = build_flat([range(0, 8)] * n, n, True)
        A = bits
        P = np.zeros(n, dtype=bool); P[:n - 37] = True
        G = np.zeros(n, dtype=bool); G[:8] = True
        t_flat = timeit(lambda: (flat & pad) | glob, repeats)
        t_np = timeit(lambda: (A & P) | G, repeats)

        out.append({
            'window': w, 'seq_len': n,
            'density': float(bits.mean()),
            'roaring_flat_opt_bytes': len(flat.serialize()),
            'roaring_per_row_bytes': per_row,
            'zstd_bytes': len(ZSTD.compress(packed)),
            'dense_f32_bytes': n * n * 4,
            'compression_vs_dense': n * n * 4 / len(flat.serialize()),
            'compose_flat_ms': t_flat['median_ms'],
            'compose_dense_ms': t_np['median_ms'],
            'speedup': t_np['median_ms'] / t_flat['median_ms'],
        })
    return out


# --------------------------------------------------------------------------- #
# Table 4 / Figure 3 -- gradient index encoding (CPU only)
# --------------------------------------------------------------------------- #
PARAM_COUNTS = [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 17_735_936]
SPARSITIES = [0.001, 0.005, 0.01, 0.05, 0.10]


def gradient_index(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for N in PARAM_COUNTS:
        grad = torch.randn(N, generator=g)
        for sp in SPARSITIES:
            k = max(1, int(sp * N))
            idx = torch.topk(grad.abs(), k).indices.numpy().astype(np.uint32)
            idx.sort()
            bits = np.zeros(N, dtype=bool); bits[idx] = True
            raw = idx.nbytes
            ro = len(BitMap(idx).serialize())
            out.append({
                'n_params': N, 'sparsity': sp, 'k': int(k),
                'raw_u32_bytes': raw,
                'roaring_bytes': ro,
                'delta_varint_bytes': delta_varint_size(idx),
                'zstd_bitset_bytes': len(ZSTD.compress(np.packbits(bits).tobytes())),
                'roaring_vs_raw': raw / ro,
            })
    return out


def fig_gradient(gi):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for sp, c, m in [(0.001, '#27ae60', 'o'), (0.01, '#2980b9', 's'),
                     (0.10, '#c0392b', '^')]:
        sub = [r for r in gi if r['sparsity'] == sp]
        ax.plot([r['n_params'] for r in sub], [r['roaring_vs_raw'] for r in sub],
                m + '-', color=c, label=f'Roaring vs raw uint32, top-{sp*100:g}%')
        ax.plot([r['n_params'] for r in sub],
                [r['raw_u32_bytes'] / r['delta_varint_bytes'] for r in sub],
                m + '--', color=c, alpha=0.55,
                label=f'delta+varint vs raw uint32, top-{sp*100:g}%')
    ax.set(xscale='log', xlabel='Number of parameters',
           ylabel='Index compression ratio (x)')
    log_xticks(ax, PARAM_COUNTS)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.axhline(1.0, color='gray', lw=0.8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Gradient index compression, Roaring vs the trivial baseline')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig3_gradient_compression.png'))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Table 5 / Figures 4, 5 -- from the real WikiText-103 GPU run
# --------------------------------------------------------------------------- #
def load_runs():
    with open(os.path.join(HERE, 'revision_main.json')) as f:
        main = json.load(f)
    with open(os.path.join(HERE, 'revision_pruned.json')) as f:
        pruned = json.load(f)
    return main, pruned


def fig_convergence(main):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, c, lbl in [('vanilla', '#c0392b', 'Vanilla Transformer'),
                        ('roaring', '#2980b9', 'Roaring Transformer')]:
        curves = [r['val_curve'] for r in main['runs'][arm]]
        steps = [p[0] for p in curves[0]]
        vals = np.array([[p[1] for p in c] for c in curves])
        mean, sd = vals.mean(0), vals.std(0, ddof=1)
        ax.plot(steps, mean, '-', color=c, label=lbl)
        ax.fill_between(steps, mean - sd, mean + sd, color=c, alpha=0.2)
    ax.set(xlabel='Training step', ylabel='Validation loss (nats)')
    ax.legend()
    ax.set_title('WikiText-103 validation convergence (mean ± sd, 5 seeds)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig4_convergence.png'))
    plt.close(fig)


def fig_step_time(main):
    """Real CUDA step times. The gap is a BASELINE ARTIFACT, not a Roaring win:
    VanillaTransformer rebuilds the dense mask every forward pass with a
    seq_len-iteration Python loop, while RoaringTransformer caches it."""
    v = [r['avg_step_ms'] for r in main['runs']['vanilla']]
    r = [x['avg_step_ms'] for x in main['runs']['roaring']]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar(['Vanilla\n(rebuilds mask\nevery step)', 'Roaring\n(cached mask)'],
           [np.mean(v), np.mean(r)],
           yerr=[np.std(v, ddof=1), np.std(r, ddof=1)],
           color=['#c0392b', '#2980b9'], capsize=6, width=0.55)
    for i, vals in enumerate([v, r]):
        ax.text(i, np.mean(vals) + 2.5, f'{np.mean(vals):.1f} ms',
                ha='center', fontweight='bold')
    ax.set(ylabel='Average step time (ms)', ylim=(0, max(np.mean(v), np.mean(r)) * 1.25))
    ax.set_title('Per-step training time, WikiText-103 (CUDA, 5 seeds)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig5_step_time.png'))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_tables(ms, gi, main, pruned, wa):
    m = main['meta']
    seq = m['model_cfg']['seq_len']
    row = next(r for r in ms if r['seq_len'] == seq)
    hyd = next(h for h in main['hydration'] if h['seq_len'] == seq)

    def stat(arm, key):
        xs = [x[key] for x in main['runs'][arm]]
        return np.mean(xs), np.std(xs, ddof=1)

    v_loss, v_sd = stat('vanilla', 'final_val_loss')
    r_loss, r_sd = stat('roaring', 'final_val_loss')
    v_ms, v_mssd = stat('vanilla', 'avg_step_ms')
    r_ms, r_mssd = stat('roaring', 'avg_step_ms')
    v_vram, _ = stat('vanilla', 'peak_vram_mb')
    r_vram, _ = stat('roaring', 'peak_vram_mb')
    eq = main['equivalence']
    pe = pruned['equivalence']

    L = []
    L.append('# Regenerated tables (round-2 revision)\n')
    L.append('> Historical snapshot from the second revision round, written by `generate_revision_outputs.py`. Table numbers below are that round\'s. In the current paper (v6) they are Table 4 (mask size), Table 5 (gradient index), Table 11 (convergence), Table 12 (pruning) and Table 10 (window ablation). The window-ablation timings were re-measured with their min-max spread for v6 (`run_v6_measurements.py`), and the convergence comparison gained a TOST equivalence test (`run_v6_stats.py`). See the README "Reproducing" section for the current table-to-script mapping.\n')
    L.append(f"Source: `revision_main.json` / `revision_pruned.json` "
             f"({m['config']}, {m['n_params']:,} params, {m['seeds']} seeds, "
             f"{m['steps']} steps, seq_len={seq}, window={m['model_cfg']['window_size']}, "
             f"device={m['device']}).\n")
    L.append('All CPU measurements below use window=256, matching the training '
             'default (was window=128 in the submitted Table 3).\n')

    L.append('\n## Table 3 — attention mask representation size (KB)\n')
    L.append('Held constant: causal sliding-window pattern, window=256, one mask per '
             'sequence length. Dense float32 is the worst-case baseline; the remaining '
             'columns are trivial purpose-built baselines.\n')
    L.append('| Seq len | Dense f32 | Packed bitset | (start,len)/row | zstd(bitset) | '
             'lzma(bitset) | Roaring per-row (published) | Roaring flat+opt |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in ms:
        L.append(f"| {r['seq_len']:,} | {r['dense_f32_bytes']/1024:,.1f} | "
                 f"{r['packed_bitset_bytes']/1024:,.1f} | {r['startlen_bytes']/1024:,.1f} | "
                 f"{r['zstd_bytes']/1024:,.1f} | {r['lzma_bytes']/1024:,.1f} | "
                 f"{r['roaring_per_row_bytes']/1024:,.1f} | "
                 f"**{r['roaring_flat_opt_bytes']/1024:,.1f}** |")

    L.append('\n## Table 4 — Top-K gradient index encoding (bytes)\n')
    L.append('Adds the 17.7M row to match the trained model and the '
             'delta+varint baseline. Ratios are index-payload only.\n')
    L.append('| Params | Sparsity | k | Raw uint32 | Roaring | delta+varint | '
             'zstd(bitset) | Roaring vs raw | Best |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in gi:
        cand = {'Roaring': r['roaring_bytes'], 'delta+varint': r['delta_varint_bytes'],
                'zstd(bitset)': r['zstd_bitset_bytes'], 'raw uint32': r['raw_u32_bytes']}
        best = min(cand, key=cand.get)
        L.append(f"| {r['n_params']:,} | {r['sparsity']*100:g}% | {r['k']:,} | "
                 f"{r['raw_u32_bytes']:,} | {r['roaring_bytes']:,} | "
                 f"{r['delta_varint_bytes']:,} | {r['zstd_bitset_bytes']:,} | "
                 f"{r['roaring_vs_raw']:.2f}x | {best} |")

    L.append('\n## Table 5 — training convergence (CORRECTED)\n')
    L.append(f"| Metric | Vanilla | Roaring |")
    L.append('|---|---|---|')
    L.append(f"| Final val. loss (ppl) | {v_loss:.2f} ± {v_sd:.2f} ({np.exp(v_loss):.1f}) | "
             f"{r_loss:.2f} ± {r_sd:.2f} ({np.exp(r_loss):.1f}) |")
    L.append(f"| Avg step time | {v_ms:.1f} ± {v_mssd:.2f} ms | {r_ms:.1f} ± {r_mssd:.2f} ms |")
    L.append(f"| Peak training VRAM | {v_vram:,.0f} MB | {r_vram:,.0f} MB |")
    L.append(f"| Mask memory @ seq_len={seq} | {row['dense_f32_bytes']/1024:,.1f} KB | "
             f"{hyd['roaring_cpu_kb']:.1f} KB |")
    L.append(f"| Mask compression vs dense f32 | 1.0x | "
             f"{row['dense_f32_bytes']/1024/hyd['roaring_cpu_kb']:.1f}x |")
    L.append(f"\nEquivalence: Welch p={eq['welch_p']:.3f}, "
             f"Mann-Whitney p={eq['mannwhitney_p']:.3f} (fail to reject equality). "
             f"The v6 paper adds a TOST test against a +/-0.05 nat margin "
             f"(`run_v6_stats.py`, `v6_stats.json`).\n")
    L.append('**Changes from the submitted Table 5:**\n')
    L.append(f"1. Mask memory was reported as 256.0 KB / 3.8 KB / 68.2x — those are "
             f"seq_len=256 values. At the actual seq_len={seq} they are "
             f"{row['dense_f32_bytes']/1024:,.1f} KB / {hyd['roaring_cpu_kb']:.1f} KB / "
             f"{row['dense_f32_bytes']/1024/hyd['roaring_cpu_kb']:.1f}x.")
    L.append('2. The "Grad Compression 1.32x" row came from the synthetic '
             '`paper_results.json` sparsity ablation, not this run. Removed pending '
             'a real measurement.')
    L.append('3. The "Pruned Ratio 30.0%" and "Prune Mask Compr. 6.0x" rows were from '
             'a separate pruning ablation while the text states pruning is disabled '
             'here. Split out below.')
    L.append(f"4. Step time now carries its standard deviation. The "
             f"{v_ms - r_ms:.2f} ms gap is statistically significant but is a "
             f"**baseline artifact**: `VanillaTransformer._create_dense_mask` rebuilds "
             f"the mask every forward pass with a {seq}-iteration Python loop, while "
             f"`RoaringTransformer` caches it. Any dense-mask caching yields the same "
             f"gain. The vanilla sd of {v_mssd:.2f} ms vs Roaring's {r_mssd:.2f} ms "
             f"is the signature of that per-step rebuild.")

    L.append('\n## Pruning ablation (separated from Table 5)\n')
    L.append('| Metric | Vanilla | Roaring + 30% pruning |')
    L.append('|---|---|---|')
    L.append(f"| Final val. loss (ppl) | {pe['vanilla_mean']:.2f} "
             f"({np.exp(pe['vanilla_mean']):.1f}) | {pe['roaring_mean']:.2f} "
             f"({np.exp(pe['roaring_mean']):.1f}) |")
    L.append(f"\nWelch p={pe['welch_p']:.6f} — pruning is a genuine quality/sparsity "
             f"tradeoff, not a free optimization.\n")

    L.append('\n## Window-size ablation (representation level, replaces old Table 6)\n')
    L.append(f'At seq_len={wa[0]["seq_len"]:,}. The submitted Table 6 claimed the '
             'compression ratio is invariant to window size, measured on a synthetic '
             'corpus at seq_len=256. That invariance is a property of the '
             'representation, not of training, so it is measured directly here.\n')
    L.append('| Window | Mask density | Roaring flat+opt (KB) | zstd (KB) | '
             'vs dense f32 | Compose flat (ms) | Compose dense (ms) | Speedup |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in wa:
        L.append(f"| {r['window'] if r['window'] is not None else 'full'} | "
                 f"{r['density']*100:.1f}% | {r['roaring_flat_opt_bytes']/1024:,.1f} | "
                 f"{r['zstd_bytes']/1024:,.1f} | {r['compression_vs_dense']:,.0f}x | "
                 f"{r['compose_flat_ms']:.2f} | {r['compose_dense_ms']:.2f} | "
                 f"{r['speedup']:.1f}x |")

    L.append('\n## Not measured (would require a GPU training run)\n')
    L.append('- The **training-loss** arm of the window-size ablation and a **gradient-sparsity** '
             'training ablation. `run_revision_experiments.py` implements neither, and the '
             'paper does not report them; both ablations are reported at the '
             'representation level only.')
    with open(OUT_TAB, 'w') as f:
        f.write('\n'.join(L) + '\n')


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    print('Mask representation scaling (CPU)...')
    ms = mask_scaling()
    print('Gradient index encoding (CPU)...')
    gi = gradient_index()
    print('Window-size ablation (CPU)...')
    wa = window_ablation()
    print('Loading GPU run results...')
    main_r, pruned_r = load_runs()

    fig_mask_memory(ms)
    fig_creation_time(ms)
    fig_gradient(gi)
    fig_convergence(main_r)
    fig_step_time(main_r)
    write_tables(ms, gi, main_r, pruned_r, wa)

    with open(os.path.join(HERE, 'revision_outputs.json'), 'w') as f:
        json.dump({'mask_scaling': ms, 'gradient_index': gi,
                   'window_ablation': wa}, f, indent=2)
    print(f'\nFigures -> {OUT_FIG}')
    print(f'Tables   -> {OUT_TAB}')


if __name__ == '__main__':
    main()
