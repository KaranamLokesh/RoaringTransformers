#!/usr/bin/env python3
"""
Figures for the round-4 revision.

  fig1_mask_memory            every encoding of Table 3, incl. zlib and lzma
  fig2_gradient_compression   Roaring vs delta+varint
  fig3_mask_creation_time     construction cost (excluded from composition timings)
  fig4_convergence            WikiText-103 validation loss, 5 seeds
  fig5_step_time              per-step time, labelled as a baseline artifact
  fig6_structure_crossover    NEW -- composition cost vs run structure. This is the
                              figure that answers "does the generality earn its
                              overhead": the purpose-built run encoding wins while
                              the mask stays run-structured and disappears once it
                              does not, at which point Roaring is the fastest
                              representation left.
  fig7_composition_workloads  NEW -- structured vs unstructured composition scaling

Usage:
    python experiments/generate_v4_figures.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_v4')

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.3})

ROAR = '#2980b9'
DENSE = '#c0392b'
BITSET = '#e67e22'
CSR = '#8e44ad'
RUNS = '#27ae60'


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def log_xticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_mask_memory(ms, window):
    n = [r['seq_len'] for r in ms]
    series = [('dense_f32', 'Dense float32 (O(n²))', 'o-', DENSE, 1.0),
              ('packed_bitset', 'Packed bitset (1 bit/elem)', 's-', BITSET, 1.0),
              ('startlen', '(start, length) per row', '^-', '#7f8c8d', 1.0),
              ('roaring_per_row', 'Roaring, per-row (as submitted)', 'v-', CSR, 1.0),
              ('roaring_flat_opt', 'Roaring, flat + run_optimize', 'D-', ROAR, 1.0),
              ('zlib', 'zlib(bitset)', 'P--', '#16a085', 0.8),
              ('lzma', 'lzma(bitset)', '*--', '#8e6c3a', 0.8),
              ('zstd', 'zstd(bitset)', 'x--', RUNS, 1.0)]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for key, label, style, color, alpha in series:
        ax.plot(n, [r['sizes_bytes'][key] / 1024 for r in ms], style,
                color=color, label=label, alpha=alpha, markersize=5)
    ax.set(xscale='log', yscale='log', xlabel='Sequence length',
           ylabel='Mask memory (KB, log scale)')
    log_xticks(ax, n)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_title(f'Attention mask representation size (causal, window={window})')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_mask_memory.png'))
    plt.close(fig)


def fig_creation_time(ms):
    n = [r['seq_len'] for r in ms]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(n, [r['dense_build_ms'] for r in ms], 'o-', color=DENSE, label='Dense')
    ax.plot(n, [r['roaring_per_row_build_ms'] for r in ms], 'v-', color=CSR,
            label='Roaring, per-row')
    ax.plot(n, [r['roaring_flat_build_ms'] for r in ms], 'D-', color=ROAR,
            label='Roaring, flat + run_optimize')
    ax.set(xscale='log', yscale='log', xlabel='Sequence length',
           ylabel='Mask construction time (ms, log scale)')
    log_xticks(ax, n)
    ax.legend(fontsize=8); ax.set_title('Mask construction cost')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_mask_creation_time.png'))
    plt.close(fig)


def fig_gradient(gi):
    params = sorted({r['n_params'] for r in gi})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for spx, c, m in [(0.001, RUNS, 'o'), (0.01, ROAR, 's'), (0.10, DENSE, '^')]:
        sub = sorted((r for r in gi if r['sparsity'] == spx), key=lambda r: r['n_params'])
        ax.plot([r['n_params'] for r in sub], [r['roaring_vs_raw'] for r in sub],
                m + '-', color=c, label=f'Roaring vs raw uint32, top-{spx*100:g}%')
        ax.plot([r['n_params'] for r in sub], [r['delta_vs_raw'] for r in sub],
                m + '--', color=c, alpha=0.55,
                label=f'delta+varint vs raw uint32, top-{spx*100:g}%')
    ax.set(xscale='log', xlabel='Number of parameters',
           ylabel='Index compression ratio (x)')
    log_xticks(ax, params)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.axhline(1.0, color='gray', lw=0.8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Gradient index compression, Roaring vs the trivial baseline')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_gradient_compression.png'))
    plt.close(fig)


def fig_convergence(main):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for a, c, lbl in [('vanilla', DENSE, 'Vanilla Transformer'),
                      ('roaring', ROAR, 'Roaring Transformer')]:
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
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_convergence.png'))
    plt.close(fig)


def fig_step_time(main):
    v = [r['avg_step_ms'] for r in main['runs']['vanilla']]
    r = [x['avg_step_ms'] for x in main['runs']['roaring']]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar(['Vanilla\n(rebuilds mask\nevery step)', 'Roaring\n(cached mask)'],
           [np.mean(v), np.mean(r)], yerr=[np.std(v, ddof=1), np.std(r, ddof=1)],
           color=[DENSE, ROAR], capsize=6, width=0.55)
    for i, vals in enumerate([v, r]):
        ax.text(i, np.mean(vals) + 2.5, f'{np.mean(vals):.1f} ms',
                ha='center', fontweight='bold')
    ax.set(ylabel='Average step time (ms)',
           ylim=(0, max(np.mean(v), np.mean(r)) * 1.25))
    ax.set_title(f'Per-step training time, WikiText-103 '
                 f'(CUDA, {main["meta"]["seeds"]} seeds)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_step_time.png'))
    plt.close(fig)


def fig_crossover(sw, n):
    """The paper's central figure.

    Plotted against content-dependent entries per row, NOT against runs per row:
    the row-invariant case and the first row-dependent case carry the same
    fragmentation (~2 runs/row) yet fall on opposite sides of the result, so run
    count is not the explanatory variable and must not be drawn as if it were.
    """
    x = np.arange(len(sw))
    labels = [str(s['random_per_row']) for s in sw]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, [s['roaring_flat_ms'] for s in sw], 'D-', color=ROAR, lw=2.2,
            label='Roaring, flat + run_optimize')
    ax.plot(x, [s['dense_ms'] for s in sw], 'o-', color=DENSE, label='Dense bool array')
    ax.plot(x, [s['packed_bitset_ms'] for s in sw], 's-', color=BITSET,
            label='Packed bitset (numpy bitwise)')
    ax.plot(x, [s['csr_ms'] for s in sw], 'v-', color=CSR, label='Sparse CSR (scipy)')
    runs = [(xi, s['run_arrays_ms']) for xi, s in zip(x, sw)
            if s.get('run_arrays_ms') is not None]
    if runs:
        ax.plot([r[0] for r in runs], [r[1] for r in runs], '^', color=RUNS,
                markersize=13, zorder=5,
                label='(start, length) closed form — exists only when row-invariant')

    ax.axvspan(-0.4, 0.5, color='#bdc3c7', alpha=0.28, zorder=0)
    ax.text(0.015, 0.60, 'row-invariant:\nclosed form\nexists', transform=ax.transAxes,
            fontsize=7.5, va='center', color='#555', linespacing=1.35)
    ax.text(0.24, 0.62, 'row-dependent: closed form gone,\n'
            'baselines lose the row broadcast', transform=ax.transAxes,
            fontsize=7.5, va='center', color='#555', linespacing=1.35)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(yscale='log', xlabel='Content-dependent (random) entries per query row',
           ylabel='Composition time (ms, log scale)', xlim=(-0.4, len(sw) - 0.6))
    sec = ax.secondary_xaxis('top')
    sec.set_xticks(x)
    sec.set_xticklabels([f"{s['runs_per_row']:.1f}" for s in sw], fontsize=7.5)
    sec.set_xlabel('mean contiguous runs per row (note: unchanged across the '
                   'transition)', fontsize=8)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.95)
    ax.set_title(f'What decides the representation (n={n:,})')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig6_structure_crossover.png'))
    plt.close(fig)


def fig_workloads(st, un):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, data, title in ((axes[0], st, 'Structured: (causal & pad) | global'),
                            (axes[1], un, 'Unstructured: (causal & pad) | random')):
        n = [c['seq_len'] for c in data]
        ax.plot(n, [c['roaring_flat_ms'] for c in data], 'D-', color=ROAR, lw=2,
                label='Roaring flat+opt')
        ax.plot(n, [c['dense_ms'] for c in data], 'o-', color=DENSE, label='Dense bool')
        ax.plot(n, [c['packed_bitset_ms'] for c in data], 's-', color=BITSET,
                label='Packed bitset')
        ax.plot(n, [c['csr_ms'] for c in data], 'v-', color=CSR, label='Sparse CSR')
        ra = [(c['seq_len'], c['run_arrays_ms']) for c in data
              if c.get('run_arrays_ms') is not None]
        if ra:
            ax.plot([r[0] for r in ra], [r[1] for r in ra], '^-', color=RUNS,
                    label='(start, length) run arrays')
        ax.set(xscale='log', yscale='log', xlabel='Sequence length', title=title)
        log_xticks(ax, n)
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Composition time (ms, log scale)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig7_composition_workloads.png'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    v4 = load('v4_results.json')
    run = load('revision_main.json')
    fig_mask_memory(v4['mask_scaling'], v4['meta']['window'])
    fig_creation_time(v4['mask_scaling'])
    fig_gradient(v4['gradient_index'])
    fig_convergence(run)
    fig_step_time(run)
    fig_crossover(v4['structure_sweep'], v4['structure_sweep'][0]['seq_len'])
    fig_workloads(v4['composition_structured'], v4['composition_unstructured'])
    print(f'figures -> {OUT}')


if __name__ == '__main__':
    main()
