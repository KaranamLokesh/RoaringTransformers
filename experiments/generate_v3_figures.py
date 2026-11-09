#!/usr/bin/env python3
"""
Figures for the round-3 revision, all generated from committed result files.

  fig1_mask_memory.png       every encoding in Table 3, including zlib and lzma
                             (the round-2 figure omitted both while its caption
                             said "all encodings")
  fig2_gradient_compression  Roaring vs the delta+varint baseline
  fig3_mask_creation_time    construction cost, excluded from the composition timings
  fig4_convergence           WikiText-103 validation loss, 5 seeds
  fig5_step_time             per-step time, labelled as a baseline artifact

Usage:
    python experiments/generate_v3_figures.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_v3')

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.3})


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def log_xticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_mask_memory(ms, window):
    n = [r['seq_len'] for r in ms]
    series = [
        ('dense_f32', 'Dense float32 (O(n²))', 'o-', '#c0392b', 1.0),
        ('packed_bitset', 'Packed bitset (1 bit/elem)', 's-', '#e67e22', 1.0),
        ('startlen', '(start, length) per row', '^-', '#7f8c8d', 1.0),
        ('roaring_per_row', 'Roaring, per-row (as submitted)', 'v-', '#8e44ad', 1.0),
        ('roaring_flat_opt', 'Roaring, flat + run_optimize', 'D-', '#2980b9', 1.0),
        ('zlib', 'zlib(bitset)', 'P--', '#16a085', 0.8),
        ('lzma', 'lzma(bitset)', '*--', '#8e6c3a', 0.8),
        ('zstd', 'zstd(bitset)', 'x--', '#27ae60', 1.0),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for key, label, style, color, alpha in series:
        ax.plot(n, [r['sizes_bytes'][key] / 1024 for r in ms], style,
                color=color, label=label, alpha=alpha, markersize=5)
    ax.set(xscale='log', yscale='log', xlabel='Sequence length',
           ylabel='Mask memory (KB, log scale)')
    log_xticks(ax, n)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_title(f'Attention mask representation size (causal, window={window})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_mask_memory.png'))
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
    ax.legend(fontsize=8)
    ax.set_title('Mask construction cost')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_mask_creation_time.png'))
    plt.close(fig)


def fig_gradient(gi):
    params = sorted({r['n_params'] for r in gi})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for sp, c, m in [(0.001, '#27ae60', 'o'), (0.01, '#2980b9', 's'),
                     (0.10, '#c0392b', '^')]:
        sub = sorted((r for r in gi if r['sparsity'] == sp),
                     key=lambda r: r['n_params'])
        ax.plot([r['n_params'] for r in sub], [r['roaring_vs_raw'] for r in sub],
                m + '-', color=c, label=f'Roaring vs raw uint32, top-{sp*100:g}%')
        ax.plot([r['n_params'] for r in sub], [r['delta_vs_raw'] for r in sub],
                m + '--', color=c, alpha=0.55,
                label=f'delta+varint vs raw uint32, top-{sp*100:g}%')
    ax.set(xscale='log', xlabel='Number of parameters',
           ylabel='Index compression ratio (x)')
    log_xticks(ax, params)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.axhline(1.0, color='gray', lw=0.8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Gradient index compression, Roaring vs the trivial baseline')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_gradient_compression.png'))
    plt.close(fig)


def fig_convergence(main):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for a, c, lbl in [('vanilla', '#c0392b', 'Vanilla Transformer'),
                      ('roaring', '#2980b9', 'Roaring Transformer')]:
        curves = [r['val_curve'] for r in main['runs'][a]]
        steps = [p[0] for p in curves[0]]
        vals = np.array([[p[1] for p in c] for c in curves])
        mean, sd = vals.mean(0), vals.std(0, ddof=1)
        ax.plot(steps, mean, '-', color=c, label=lbl)
        ax.fill_between(steps, mean - sd, mean + sd, color=c, alpha=0.2)
    ax.set(xlabel='Training step', ylabel='Validation loss (nats)')
    ax.legend()
    ax.set_title(f'WikiText-103 validation convergence '
                 f'(mean ± sd, {main["meta"]["seeds"]} seeds)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_convergence.png'))
    plt.close(fig)


def fig_step_time(main):
    """The gap is a BASELINE ARTIFACT: the vanilla model rebuilds its dense mask
    every forward pass, while the Roaring path caches the realized tensor."""
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
    ax.set(ylabel='Average step time (ms)',
           ylim=(0, max(np.mean(v), np.mean(r)) * 1.25))
    ax.set_title(f'Per-step training time, WikiText-103 '
                 f'(CUDA, {main["meta"]["seeds"]} seeds)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_step_time.png'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    v3 = load('v3_results.json')
    run = load('revision_main.json')
    fig_mask_memory(v3['mask_scaling'], v3['meta']['window'])
    fig_creation_time(v3['mask_scaling'])
    fig_gradient(v3['gradient_index'])
    fig_convergence(run)
    fig_step_time(run)
    print(f'figures -> {OUT}')


if __name__ == '__main__':
    main()
