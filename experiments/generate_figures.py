#!/usr/bin/env python3
"""
Generate clear, individual figures for the Roaring Transformer paper.
Each figure: large fonts, bold labels, no clutter, one idea per chart.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Global Style: BIG and CLEAR ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 17,
    'axes.labelweight': 'bold',
    'legend.fontsize': 14,
    'legend.framealpha': 0.95,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 200,
    'figure.figsize': (10, 7),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'lines.linewidth': 3,
    'lines.markersize': 10,
})

RED   = '#c0392b'
BLUE  = '#2980b9'
GREEN = '#27ae60'
ORANGE = '#e67e22'
PURPLE = '#8e44ad'
GRAY  = '#7f8c8d'

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), 'paper_results.json')) as f:
    data = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Mask Memory: Dense vs Roaring (log scale)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1():
    fig, ax = plt.subplots()

    subset = [r for r in data['mask_scaling'] if r['window_size'] == 128]
    sls       = [r['seq_len'] for r in subset]
    dense_kb  = [r['dense_mem_bytes'] / 1024 for r in subset]
    roar_kb   = [r['roaring_mem_bytes'] / 1024 for r in subset]

    ax.plot(sls, dense_kb, 'o-', color=RED, label='Dense Mask (float32)')
    ax.plot(sls, roar_kb, 's-', color=BLUE, label='Roaring Bitmap Mask')

    # Label ONLY first and last points — no clutter
    for i in [0, len(sls)-1]:
        d_label = f'{dense_kb[i]:,.0f} KB'
        r_label = f'{roar_kb[i]:,.0f} KB'
        ax.annotate(d_label, (sls[i], dense_kb[i]),
                    textcoords='offset points', xytext=(0, 14), ha='center',
                    fontsize=13, fontweight='bold', color=RED)
        ax.annotate(r_label, (sls[i], roar_kb[i]),
                    textcoords='offset points', xytext=(0, -20), ha='center',
                    fontsize=13, fontweight='bold', color=BLUE)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mask Memory (KB)')
    ax.set_title('Figure 1 — Mask Memory: Dense O(n²) vs Roaring O(n)')
    ax.legend(loc='upper left')

    fig.savefig(os.path.join(OUT, 'fig1_mask_memory.png'))
    plt.close(fig)
    print('  ✓ fig1_mask_memory.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Compression Ratio grows linearly with sequence length
# ═══════════════════════════════════════════════════════════════════════════════

def fig2():
    fig, ax = plt.subplots()

    configs = [
        (64,  'o-', GREEN,  'Window = 64'),
        (128, 's-', BLUE,   'Window = 128'),
        (256, '^-', ORANGE, 'Window = 256'),
    ]

    for ws, style, color, label in configs:
        subset = [r for r in data['mask_scaling'] if r['window_size'] == ws]
        sls    = [r['seq_len'] for r in subset]
        ratios = [r['compression_ratio'] for r in subset]
        ax.plot(sls, ratios, style, color=color, label=label)

    # Label the rightmost point once (all overlap)
    last_ratio = [r for r in data['mask_scaling'] if r['window_size'] == 128][-1]['compression_ratio']
    last_sl    = [r for r in data['mask_scaling'] if r['window_size'] == 128][-1]['seq_len']
    ax.annotate(f'{last_ratio:,.0f}x', (last_sl, last_ratio),
                textcoords='offset points', xytext=(-60, 10),
                fontsize=14, fontweight='bold', color='black')

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Compression Ratio (x)')
    ax.set_title('Figure 2 — Compression Ratio Scales Linearly with n')
    ax.legend(loc='upper left')

    fig.savefig(os.path.join(OUT, 'fig2_compression_ratio.png'))
    plt.close(fig)
    print('  ✓ fig2_compression_ratio.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Training Loss: Vanilla vs Roaring (with error bands)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3():
    fig, ax = plt.subplots()

    ts = data['training_summary']
    v_mean = np.array(ts['vanilla_loss_mean'])
    v_std  = np.array(ts['vanilla_loss_std'])
    r_mean = np.array(ts['roaring_loss_mean'])
    r_std  = np.array(ts['roaring_loss_std'])
    steps  = np.arange(len(v_mean))

    ax.plot(steps, v_mean, color=RED, label='Vanilla Transformer')
    ax.fill_between(steps, v_mean - v_std, v_mean + v_std, color=RED, alpha=0.12)

    ax.plot(steps, r_mean, color=BLUE, label='Roaring Transformer')
    ax.fill_between(steps, r_mean - r_std, r_mean + r_std, color=BLUE, alpha=0.12)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Figure 3 — Training Loss (mean ± std, 3 seeds)')
    ax.legend(loc='upper right')

    # Box with final values
    txt = (f"Final Loss:\n"
           f"  Vanilla = {v_mean[-1]:.4f} ± {v_std[-1]:.4f}\n"
           f"  Roaring = {r_mean[-1]:.4f} ± {r_std[-1]:.4f}")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray'))

    fig.savefig(os.path.join(OUT, 'fig3_training_loss.png'))
    plt.close(fig)
    print('  ✓ fig3_training_loss.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Step Time: side-by-side bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig4():
    fig, ax = plt.subplots()

    ts = data['training_summary']
    names  = ['Vanilla\nTransformer', 'Roaring\nTransformer']
    means  = [ts['vanilla_time_mean_ms'], ts['roaring_time_mean_ms']]
    stds   = [ts['vanilla_time_std_ms'],  ts['roaring_time_std_ms']]
    colors = [RED, BLUE]

    bars = ax.bar(names, means, yerr=stds, capsize=10, color=colors,
                  edgecolor='black', linewidth=1.2, width=0.45, alpha=0.9,
                  error_kw={'linewidth': 2.5, 'capthick': 2.5})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 1.5,
                f'{m:.1f} ms', ha='center', va='bottom',
                fontsize=15, fontweight='bold')

    ax.set_ylabel('Avg Step Time (ms)')
    ax.set_title('Figure 4 — Per-Step Training Time')
    ax.set_ylim(0, max(means) * 1.4)
    ax.text(0.97, 0.95, '↓ Lower is better', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, color=GRAY, style='italic')

    fig.savefig(os.path.join(OUT, 'fig4_step_time.png'))
    plt.close(fig)
    print('  ✓ fig4_step_time.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Gradient Index Compression at different sparsity levels
# ═══════════════════════════════════════════════════════════════════════════════

def fig5():
    fig, ax = plt.subplots()

    configs = [
        (0.001, 'o-', GREEN,  'Top-0.1% (very sparse)'),
        (0.01,  's-', BLUE,   'Top-1.0% (typical)'),
        (0.1,   '^-', RED,    'Top-10% (dense)'),
    ]

    for sp, style, color, label in configs:
        subset = [r for r in data['gradient_compression'] if abs(r['sparsity'] - sp) < 1e-6]
        pcs    = [r['param_count'] for r in subset]
        ratios = [r['index_compression_ratio'] for r in subset]
        ax.plot(pcs, ratios, style, color=color, label=label)

        # Label last point
        ax.annotate(f'{ratios[-1]:.2f}x', (pcs[-1], ratios[-1]),
                    textcoords='offset points', xytext=(12, 0),
                    fontsize=13, fontweight='bold', color=color)

    ax.axhline(y=1.0, color=GRAY, linestyle='-', alpha=0.4, linewidth=1.5)
    ax.text(60000, 1.03, 'No compression (1x)', fontsize=11, color=GRAY)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Index Compression Ratio (x)')
    ax.set_title('Figure 5 — Gradient Index Compression: Roaring vs Raw')
    ax.legend(loc='center right')
    ax.text(0.03, 0.97, '↑ Higher is better', transform=ax.transAxes,
            ha='left', va='top', fontsize=12, color=GRAY, style='italic')

    fig.savefig(os.path.join(OUT, 'fig5_gradient_compression.png'))
    plt.close(fig)
    print('  ✓ fig5_gradient_compression.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Window Size Ablation (simple grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6():
    fig, ax = plt.subplots()

    wa = data['window_ablation']
    labels = [str(r['window_size']) for r in wa]
    losses = [r['final_loss'] for r in wa]
    times  = [r['avg_step_ms'] for r in wa]

    x = np.arange(len(labels))
    w = 0.35

    b1 = ax.bar(x - w/2, losses, w, color=BLUE, edgecolor='black', linewidth=1,
                label='Final Loss')
    # Scale times to fit alongside losses for visual comparison
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, times, w, color=ORANGE, edgecolor='black', linewidth=1,
                 label='Step Time (ms)')

    # Value labels
    for bar, val in zip(b1, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=BLUE)
    for bar, val in zip(b2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Final Loss', color=BLUE)
    ax2.set_ylabel('Step Time (ms)', color=ORANGE)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax.set_title('Figure 6 — Window Size Ablation')

    # Combined legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor='black', label='Final Loss'),
                       Patch(facecolor=ORANGE, edgecolor='black', label='Step Time (ms)')],
              loc='upper right')

    fig.savefig(os.path.join(OUT, 'fig6_window_ablation.png'))
    plt.close(fig)
    print('  ✓ fig6_window_ablation.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Gradient Sparsity Ablation (two separate subplots, not dual-axis)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sa = data['sparsity_ablation']
    sp_labels = [f'{r["sparsity"]*100:.1f}%' for r in sa]
    losses      = [r['final_loss'] for r in sa]
    compressions = [r['comm_compression'] for r in sa]
    x = np.arange(len(sp_labels))

    # Left: Loss
    ax1.bar(x, losses, color=PURPLE, edgecolor='black', linewidth=1, width=0.55)
    for i, val in enumerate(losses):
        ax1.text(i, val + 0.0001, f'{val:.4f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sp_labels)
    ax1.set_xlabel('Gradient Sparsity (Top-K %)')
    ax1.set_ylabel('Final Training Loss')
    ax1.set_title('Loss at Each Sparsity Level')
    ax1.grid(True, alpha=0.25, linestyle='--')

    # Right: Compression
    ax2.bar(x, compressions, color=GREEN, edgecolor='black', linewidth=1, width=0.55)
    for i, val in enumerate(compressions):
        ax2.text(i, val + 0.01, f'{val:.2f}x', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sp_labels)
    ax2.set_xlabel('Gradient Sparsity (Top-K %)')
    ax2.set_ylabel('Communication Compression (x)')
    ax2.set_title('Communication Savings')
    ax2.grid(True, alpha=0.25, linestyle='--')

    fig.suptitle('Figure 7 — Gradient Sparsity Ablation', fontsize=20, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_sparsity_ablation.png'))
    plt.close(fig)
    print('  ✓ fig7_sparsity_ablation.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Mask Creation Time: Dense vs Roaring
# ═══════════════════════════════════════════════════════════════════════════════

def fig8():
    fig, ax = plt.subplots()

    subset = [r for r in data['mask_scaling'] if r['window_size'] == 128]
    sls      = [r['seq_len'] for r in subset]
    dense_ms = [r['dense_time_s'] * 1000 for r in subset]
    roar_ms  = [r['roaring_time_s'] * 1000 for r in subset]

    ax.plot(sls, dense_ms, 'o-', color=RED, label='Dense Mask')
    ax.plot(sls, roar_ms, 's-', color=BLUE, label='Roaring Bitmap')

    # Annotate just the last point with speedup
    speedup = dense_ms[-1] / roar_ms[-1]
    ax.annotate(f'{dense_ms[-1]:.1f} ms', (sls[-1], dense_ms[-1]),
                textcoords='offset points', xytext=(-80, 10),
                fontsize=13, fontweight='bold', color=RED)
    ax.annotate(f'{roar_ms[-1]:.1f} ms\n({speedup:.0f}x faster)',
                (sls[-1], roar_ms[-1]),
                textcoords='offset points', xytext=(-110, -15),
                fontsize=13, fontweight='bold', color=BLUE)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mask Creation Time (ms)')
    ax.set_title('Figure 8 — Mask Creation Speed: Dense vs Roaring')
    ax.legend(loc='upper left')

    fig.savefig(os.path.join(OUT, 'fig8_mask_creation_time.png'))
    plt.close(fig)
    print('  ✓ fig8_mask_creation_time.png')


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating clear individual figures...\n')
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    print(f'\nAll 8 figures saved to: {OUT}/')
