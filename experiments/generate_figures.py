#!/usr/bin/env python3
"""
Generate clean figures for the Roaring Transformer paper.
NO titles inside the plots — captions are in the docx.
Labels placed outside the plot area, never overlapping data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.labelweight': 'bold',
    'legend.fontsize': 13,
    'legend.framealpha': 0.95,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 200,
    'figure.figsize': (8, 5.5),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})

RED    = '#c0392b'
BLUE   = '#2980b9'
GREEN  = '#27ae60'
ORANGE = '#e67e22'
PURPLE = '#8e44ad'
GRAY   = '#7f8c8d'

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), 'paper_results.json')) as f:
    data = json.load(f)


def fig1():
    """Mask Memory: Dense vs Roaring."""
    fig, ax = plt.subplots()
    subset = [r for r in data['mask_scaling'] if r['window_size'] == 128]
    sls      = [r['seq_len'] for r in subset]
    dense_kb = [r['dense_mem_bytes'] / 1024 for r in subset]
    roar_kb  = [r['roaring_mem_bytes'] / 1024 for r in subset]

    ax.plot(sls, dense_kb, 'o-', color=RED,  label='Dense Mask (float32)')
    ax.plot(sls, roar_kb,  's-', color=BLUE, label='Roaring Bitmap Mask')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mask Memory (KB, log scale)')
    ax.legend(loc='upper left')
    fig.savefig(os.path.join(OUT, 'fig1_mask_memory.png'))
    plt.close(fig)
    print('  done fig1')


def fig2():
    """Compression Ratio vs Sequence Length."""
    fig, ax = plt.subplots()
    for ws, style, color, label in [(64,'o-',GREEN,'Window=64'),(128,'s-',BLUE,'Window=128'),(256,'^-',ORANGE,'Window=256')]:
        subset = [r for r in data['mask_scaling'] if r['window_size'] == ws]
        ax.plot([r['seq_len'] for r in subset],
                [r['compression_ratio'] for r in subset],
                style, color=color, label=label)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Compression Ratio (x)')
    ax.legend(loc='upper left')
    fig.savefig(os.path.join(OUT, 'fig2_compression_ratio.png'))
    plt.close(fig)
    print('  done fig2')


def fig3():
    """Training Loss Convergence."""
    fig, ax = plt.subplots()
    ts = data['training_summary']
    v_mean = np.array(ts['vanilla_loss_mean'])
    v_std  = np.array(ts['vanilla_loss_std'])
    r_mean = np.array(ts['roaring_loss_mean'])
    r_std  = np.array(ts['roaring_loss_std'])
    steps  = np.arange(len(v_mean))

    ax.plot(steps, v_mean, color=RED,  label='Vanilla Transformer')
    ax.fill_between(steps, v_mean-v_std, v_mean+v_std, color=RED, alpha=0.12)
    ax.plot(steps, r_mean, color=BLUE, label='Roaring Transformer')
    ax.fill_between(steps, r_mean-r_std, r_mean+r_std, color=BLUE, alpha=0.12)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(OUT, 'fig3_training_loss.png'))
    plt.close(fig)
    print('  done fig3')


def fig4():
    """Per-Step Time bar chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ts = data['training_summary']
    names  = ['Vanilla\nTransformer', 'Roaring\nTransformer']
    means  = [ts['vanilla_time_mean_ms'], ts['roaring_time_mean_ms']]
    stds   = [ts['vanilla_time_std_ms'],  ts['roaring_time_std_ms']]
    colors = [RED, BLUE]

    bars = ax.bar(names, means, yerr=stds, capsize=10, color=colors,
                  edgecolor='black', linewidth=1, width=0.45, alpha=0.9,
                  error_kw={'linewidth': 2, 'capthick': 2})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 1.5,
                f'{m:.1f} ms', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg Step Time (ms)')
    ax.set_ylim(0, max(means)*1.4)
    fig.savefig(os.path.join(OUT, 'fig4_step_time.png'))
    plt.close(fig)
    print('  done fig4')


def fig5():
    """Gradient Index Compression."""
    fig, ax = plt.subplots()
    for sp, style, color, label in [(0.001,'o-',GREEN,'Top-0.1%'),(0.01,'s-',BLUE,'Top-1.0%'),(0.1,'^-',RED,'Top-10%')]:
        subset = [r for r in data['gradient_compression'] if abs(r['sparsity']-sp)<1e-6]
        pcs    = [r['param_count'] for r in subset]
        ratios = [r['index_compression_ratio'] for r in subset]
        ax.plot(pcs, ratios, style, color=color, label=label)
    ax.axhline(y=1.0, color=GRAY, linestyle='-', alpha=0.3)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x>=1e6 else f'{x/1e3:.0f}K'))
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Index Compression Ratio (x)')
    ax.legend(loc='center right')
    fig.savefig(os.path.join(OUT, 'fig5_gradient_compression.png'))
    plt.close(fig)
    print('  done fig5')


def fig6():
    """Window Size Ablation — simple bar chart, loss only."""
    fig, ax = plt.subplots(figsize=(7, 5))
    wa = data['window_ablation']
    labels = [str(r['window_size']) for r in wa]
    losses = [r['final_loss'] for r in wa]
    bars = ax.bar(labels, losses, color=BLUE, edgecolor='black', linewidth=1, width=0.5)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Final Training Loss')
    fig.savefig(os.path.join(OUT, 'fig6_window_ablation.png'))
    plt.close(fig)
    print('  done fig6')


def fig7():
    """Sparsity Ablation — two side-by-side subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sa = data['sparsity_ablation']
    sp_labels = [f'{r["sparsity"]*100:.1f}%' for r in sa]
    losses = [r['final_loss'] for r in sa]
    comps  = [r['comm_compression'] for r in sa]
    x = np.arange(len(sp_labels))

    ax1.bar(x, losses, color=PURPLE, edgecolor='black', linewidth=1, width=0.5)
    for i, val in enumerate(losses):
        ax1.text(i, val+0.0001, f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(sp_labels)
    ax1.set_xlabel('Gradient Sparsity (Top-K %)')
    ax1.set_ylabel('Final Training Loss')

    ax2.bar(x, comps, color=GREEN, edgecolor='black', linewidth=1, width=0.5)
    for i, val in enumerate(comps):
        ax2.text(i, val+0.01, f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(sp_labels)
    ax2.set_xlabel('Gradient Sparsity (Top-K %)')
    ax2.set_ylabel('Communication Compression (x)')

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(OUT, 'fig7_sparsity_ablation.png'))
    plt.close(fig)
    print('  done fig7')


def fig8():
    """Mask Creation Time."""
    fig, ax = plt.subplots()
    subset = [r for r in data['mask_scaling'] if r['window_size'] == 128]
    sls      = [r['seq_len'] for r in subset]
    dense_ms = [r['dense_time_s']*1000 for r in subset]
    roar_ms  = [r['roaring_time_s']*1000 for r in subset]

    ax.plot(sls, dense_ms, 'o-', color=RED,  label='Dense Mask')
    ax.plot(sls, roar_ms,  's-', color=BLUE, label='Roaring Bitmap')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mask Creation Time (ms, log scale)')
    ax.legend(loc='upper left')
    fig.savefig(os.path.join(OUT, 'fig8_mask_creation_time.png'))
    plt.close(fig)
    print('  done fig8')


if __name__ == '__main__':
    print('Generating figures...')
    fig1(); fig2(); fig3(); fig4()
    fig5(); fig6(); fig7(); fig8()
    print(f'All saved to: {OUT}/')
