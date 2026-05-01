#!/usr/bin/env python3
"""
Comprehensive experiments for the Roaring Transformer research paper.

Generates all data needed for:
  Table 1: Mask memory scaling (seq_len 256 → 8192)
  Table 2: Gradient compression at varying sparsity ratios
  Table 3: Training convergence comparison (200 steps, 3 seeds)
  Table 4: Ablation on window sizes
  Table 5: Mask creation time scaling
  Figure 1: Mask memory scaling (log-log)
  Figure 2: Training loss curves (with error bands)
  Figure 3: Per-step time comparison
  Figure 4: Gradient compression ratio vs param count
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import psutil
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from src.vanilla_transformer import VanillaTransformer
from src.roaring_transformer import RoaringTransformer
from src.roaring_attention import RoaringAttentionMask, benchmark_mask_creation
from src.roaring_gradient import RoaringGradientCompressor, benchmark_gradient_compression
from src.data import get_dataloader


def get_memory_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def train_model(model, dataloader, num_steps, device, label="model", seed=42):
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    metrics = {'losses': [], 'step_times': [], 'memory_mb': []}
    step = 0
    for epoch in range(500):
        for input_ids, targets in dataloader:
            if step >= num_steps:
                break
            input_ids, targets = input_ids.to(device), targets.to(device)

            t0 = time.time()
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            step_time = time.time() - t0

            if hasattr(model, 'prune_step'):
                model.prune_step()

            metrics['losses'].append(loss.item())
            metrics['step_times'].append(step_time)
            metrics['memory_mb'].append(get_memory_mb())
            step += 1
        if step >= num_steps:
            break

    return metrics


# ─── EXPERIMENT 1: Mask Memory Scaling ────────────────────────────────────────

def exp1_mask_scaling():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Mask Memory & Time Scaling")
    print("=" * 70)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    window_sizes = [64, 128, 256]
    results = []

    for ws in window_sizes:
        for sl in seq_lengths:
            if ws >= sl:
                continue
            stats = benchmark_mask_creation(sl, ws)
            stats['window_size'] = ws
            results.append(stats)
            print(f"  seq={sl:>5}, win={ws:>3} | "
                  f"Dense: {stats['dense_mem_bytes']/1024:>10.1f} KB ({stats['dense_time_s']:.4f}s) | "
                  f"Roaring: {stats['roaring_mem_bytes']/1024:>8.1f} KB ({stats['roaring_time_s']:.4f}s) | "
                  f"Ratio: {stats['compression_ratio']:>8.1f}x")

    return results


# ─── EXPERIMENT 2: Gradient Compression Sweep ────────────────────────────────

def exp2_gradient_compression():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Gradient Compression Sweep")
    print("=" * 70)

    param_counts = [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    sparsity_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = []

    for sp in sparsity_levels:
        for pc in param_counts:
            stats = benchmark_gradient_compression(pc, sparsity=sp)
            stats['param_count'] = pc
            stats['sparsity'] = sp
            results.append(stats)
            print(f"  sparsity={sp:.3f}, params={pc:>10,} | "
                  f"Raw: {stats['raw_index_bytes']/1024:>8.1f} KB | "
                  f"Roaring: {stats['roaring_index_bytes']/1024:>8.1f} KB | "
                  f"Ratio: {stats['index_compression_ratio']:.2f}x | "
                  f"Time: {stats['roaring_compress_time_s']*1000:.1f}ms")

    return results


# ─── EXPERIMENT 3: Training Convergence (Multi-seed) ─────────────────────────

def exp3_training_convergence(num_steps=200, device='cpu'):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Training Convergence ({num_steps} steps, 3 seeds)")
    print("=" * 70)

    seeds = [42, 123, 456]
    seq_len = 256
    batch_size = 8
    vocab_size = 5000
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 256
    window_size = 64

    all_vanilla = {'losses': [], 'step_times': [], 'memory_mb': []}
    all_roaring = {'losses': [], 'step_times': [], 'memory_mb': []}

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        dataloader = get_dataloader(
            dataset_type='synthetic', vocab_size=vocab_size, seq_len=seq_len,
            batch_size=batch_size, num_samples=max(1000, num_steps * batch_size)
        )

        # Vanilla
        print(f"  Training Vanilla (seed={seed})...")
        vanilla = VanillaTransformer(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, ff_dim=ff_dim, max_seq_len=seq_len,
            window_size=window_size
        )
        vm = train_model(vanilla, dataloader, num_steps, device, "Vanilla", seed)
        all_vanilla['losses'].append(vm['losses'])
        all_vanilla['step_times'].append(vm['step_times'])
        all_vanilla['memory_mb'].append(vm['memory_mb'])
        del vanilla

        # Roaring
        print(f"  Training Roaring (seed={seed})...")
        roaring = RoaringTransformer(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, ff_dim=ff_dim, max_seq_len=seq_len,
            window_size=window_size, grad_sparsity=0.01,
            prune_ratio=0.3, prune_warmup=num_steps // 2
        )
        rm = train_model(roaring, dataloader, num_steps, device, "Roaring", seed)
        all_roaring['losses'].append(rm['losses'])
        all_roaring['step_times'].append(rm['step_times'])
        all_roaring['memory_mb'].append(rm['memory_mb'])

        # Grad compression stats from final state
        roaring.train()
        dummy = torch.randint(0, vocab_size, (1, seq_len)).to(device)
        logits = roaring(dummy)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                     torch.randint(0, vocab_size, (seq_len,)).to(device))
        loss.backward()
        grad_stats = roaring.compress_gradients()
        prune_stats = roaring.pruning_manager.get_stats()
        mask_info = roaring.get_mask_memory_bytes(seq_len)
        del roaring

    # Compute mean/std across seeds
    v_losses = np.array(all_vanilla['losses'])
    r_losses = np.array(all_roaring['losses'])
    v_times = np.array(all_vanilla['step_times'])
    r_times = np.array(all_roaring['step_times'])

    summary = {
        'vanilla_loss_mean': v_losses.mean(axis=0).tolist(),
        'vanilla_loss_std': v_losses.std(axis=0).tolist(),
        'roaring_loss_mean': r_losses.mean(axis=0).tolist(),
        'roaring_loss_std': r_losses.std(axis=0).tolist(),
        'vanilla_time_mean_ms': float(v_times.mean() * 1000),
        'vanilla_time_std_ms': float(v_times.std() * 1000),
        'roaring_time_mean_ms': float(r_times.mean() * 1000),
        'roaring_time_std_ms': float(r_times.std() * 1000),
        'vanilla_final_loss_mean': float(v_losses[:, -1].mean()),
        'vanilla_final_loss_std': float(v_losses[:, -1].std()),
        'roaring_final_loss_mean': float(r_losses[:, -1].mean()),
        'roaring_final_loss_std': float(r_losses[:, -1].std()),
        'mask_info': mask_info,
        'prune_stats': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                        for k, v in prune_stats.items()},
        'grad_compression': {
            'total_raw_bytes': sum(s['raw_total_bytes'] for s in grad_stats.values()),
            'total_roaring_bytes': sum(s['roaring_total_bytes'] for s in grad_stats.values()),
        },
        'seeds': seeds,
        'num_steps': num_steps,
    }

    print(f"\n  Vanilla  final loss: {summary['vanilla_final_loss_mean']:.4f} ± {summary['vanilla_final_loss_std']:.4f}")
    print(f"  Roaring  final loss: {summary['roaring_final_loss_mean']:.4f} ± {summary['roaring_final_loss_std']:.4f}")
    print(f"  Vanilla  avg step:   {summary['vanilla_time_mean_ms']:.1f} ± {summary['vanilla_time_std_ms']:.1f} ms")
    print(f"  Roaring  avg step:   {summary['roaring_time_mean_ms']:.1f} ± {summary['roaring_time_std_ms']:.1f} ms")

    return summary, all_vanilla, all_roaring


# ─── EXPERIMENT 4: Window Size Ablation ───────────────────────────────────────

def exp4_window_ablation(num_steps=100, device='cpu'):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Window Size Ablation")
    print("=" * 70)

    window_sizes = [32, 64, 128, 256, None]  # None = full causal
    seq_len = 256
    batch_size = 8
    vocab_size = 5000
    results = []

    dataloader = get_dataloader(
        dataset_type='synthetic', vocab_size=vocab_size, seq_len=seq_len,
        batch_size=batch_size, num_samples=max(500, num_steps * batch_size)
    )

    for ws in window_sizes:
        ws_label = str(ws) if ws else "Full"
        print(f"\n  Window size: {ws_label}")

        roaring = RoaringTransformer(
            vocab_size=vocab_size, embed_dim=128, num_heads=4,
            num_layers=2, ff_dim=256, max_seq_len=seq_len,
            window_size=ws if ws else seq_len, grad_sparsity=0.01,
            prune_ratio=0.3, prune_warmup=num_steps // 2
        )
        rm = train_model(roaring, dataloader, num_steps, device, f"Roaring-w{ws_label}", seed=42)

        if ws and ws < seq_len:
            mask_info = roaring.get_mask_memory_bytes(seq_len)
        else:
            mask_info = {'dense_bytes': seq_len * seq_len * 4, 'roaring_bytes': seq_len * seq_len * 4, 'compression_ratio': 1.0}
            try:
                mask_info = roaring.get_mask_memory_bytes(seq_len)
            except:
                pass

        results.append({
            'window_size': ws_label,
            'final_loss': rm['losses'][-1],
            'avg_step_ms': float(np.mean(rm['step_times']) * 1000),
            'mask_compression': mask_info.get('compression_ratio', 1.0),
            'roaring_mask_kb': mask_info.get('roaring_bytes', 0) / 1024,
        })
        print(f"    Final loss: {rm['losses'][-1]:.4f}, "
              f"Step: {np.mean(rm['step_times'])*1000:.1f}ms, "
              f"Mask compression: {mask_info.get('compression_ratio', 1.0):.1f}x")
        del roaring

    return results


# ─── EXPERIMENT 5: Sparsity Ratio Ablation (Gradient) ─────────────────────────

def exp5_sparsity_ablation(num_steps=100, device='cpu'):
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Gradient Sparsity Ratio Ablation")
    print("=" * 70)

    sparsities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    seq_len = 256
    batch_size = 8
    vocab_size = 5000
    results = []

    dataloader = get_dataloader(
        dataset_type='synthetic', vocab_size=vocab_size, seq_len=seq_len,
        batch_size=batch_size, num_samples=max(500, num_steps * batch_size)
    )

    for sp in sparsities:
        print(f"\n  Gradient sparsity: {sp}")
        roaring = RoaringTransformer(
            vocab_size=vocab_size, embed_dim=128, num_heads=4,
            num_layers=2, ff_dim=256, max_seq_len=seq_len,
            window_size=64, grad_sparsity=sp,
            prune_ratio=0.3, prune_warmup=num_steps // 2
        )
        rm = train_model(roaring, dataloader, num_steps, device, f"Roaring-sp{sp}", seed=42)

        # Get grad stats
        roaring.train()
        dummy = torch.randint(0, vocab_size, (1, seq_len)).to(device)
        logits = roaring(dummy)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                     torch.randint(0, vocab_size, (seq_len,)).to(device))
        loss.backward()
        grad_stats = roaring.compress_gradients()
        total_raw = sum(s['raw_total_bytes'] for s in grad_stats.values())
        total_roaring = sum(s['roaring_total_bytes'] for s in grad_stats.values())

        results.append({
            'sparsity': sp,
            'final_loss': rm['losses'][-1],
            'avg_step_ms': float(np.mean(rm['step_times']) * 1000),
            'raw_comm_kb': total_raw / 1024,
            'roaring_comm_kb': total_roaring / 1024,
            'comm_compression': total_raw / max(1, total_roaring),
        })
        print(f"    Final loss: {rm['losses'][-1]:.4f}, "
              f"Comm compression: {total_raw / max(1, total_roaring):.2f}x")
        del roaring

    return results


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def generate_all_plots(mask_results, training_summary, all_vanilla, all_roaring,
                       window_ablation, sparsity_ablation, grad_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle('The Roaring Transformer: Comprehensive Evaluation', fontsize=16, fontweight='bold', y=0.98)

    # ── Figure 1: Mask Memory Scaling (log-log) ──
    ax1 = fig.add_subplot(4, 2, 1)
    for ws in [64, 128, 256]:
        subset = [r for r in mask_results if r['window_size'] == ws]
        if not subset:
            continue
        sls = [r['seq_len'] for r in subset]
        dense = [r['dense_mem_bytes'] / (1024*1024) for r in subset]
        roaring = [r['roaring_mem_bytes'] / (1024*1024) for r in subset]
        if ws == 128:
            ax1.plot(sls, dense, 'o--', color='#d62728', label='Dense (all windows)', linewidth=2, markersize=6)
        ax1.plot(sls, roaring, 's-', label=f'Roaring (w={ws})', linewidth=2, markersize=5)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('(a) Mask Memory Scaling')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    # ── Figure 2: Compression Ratio Scaling ──
    ax2 = fig.add_subplot(4, 2, 2)
    for ws in [64, 128, 256]:
        subset = [r for r in mask_results if r['window_size'] == ws]
        if not subset:
            continue
        sls = [r['seq_len'] for r in subset]
        ratios = [r['compression_ratio'] for r in subset]
        ax2.plot(sls, ratios, 'o-', label=f'w={ws}', linewidth=2, markersize=6)
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Compression Ratio (x)')
    ax2.set_title('(b) Mask Compression Ratio')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ScalarFormatter())

    # ── Figure 3: Training Loss Curves (with error bands) ──
    ax3 = fig.add_subplot(4, 2, 3)
    v_losses = np.array(all_vanilla['losses'])
    r_losses = np.array(all_roaring['losses'])
    steps_x = np.arange(v_losses.shape[1])
    v_mean, v_std = v_losses.mean(axis=0), v_losses.std(axis=0)
    r_mean, r_std = r_losses.mean(axis=0), r_losses.std(axis=0)
    ax3.plot(steps_x, v_mean, label='Vanilla', color='#1f77b4', linewidth=1.5)
    ax3.fill_between(steps_x, v_mean - v_std, v_mean + v_std, alpha=0.2, color='#1f77b4')
    ax3.plot(steps_x, r_mean, label='Roaring', color='#ff7f0e', linewidth=1.5)
    ax3.fill_between(steps_x, r_mean - r_std, r_mean + r_std, alpha=0.2, color='#ff7f0e')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('(c) Training Loss Convergence (3 seeds)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Figure 4: Per-step Time ──
    ax4 = fig.add_subplot(4, 2, 4)
    v_times = np.array(all_vanilla['step_times']) * 1000
    r_times = np.array(all_roaring['step_times']) * 1000
    ax4.plot(steps_x, v_times.mean(axis=0), label='Vanilla', alpha=0.7, linewidth=1)
    ax4.plot(steps_x, r_times.mean(axis=0), label='Roaring', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('(d) Per-Step Wall-Clock Time')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── Figure 5: Gradient Compression vs Param Count ──
    ax5 = fig.add_subplot(4, 2, 5)
    for sp in [0.001, 0.01, 0.1]:
        subset = [r for r in grad_results if abs(r['sparsity'] - sp) < 1e-6]
        if not subset:
            continue
        pcs = [r['param_count'] for r in subset]
        ratios = [r['index_compression_ratio'] for r in subset]
        ax5.plot(pcs, ratios, 'o-', label=f'top-{sp*100:.1f}%', linewidth=2, markersize=5)
    ax5.set_xscale('log')
    ax5.set_xlabel('Parameter Count')
    ax5.set_ylabel('Index Compression Ratio (x)')
    ax5.set_title('(e) Gradient Index Compression')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Figure 6: Window Size Ablation ──
    ax6 = fig.add_subplot(4, 2, 6)
    ws_labels = [r['window_size'] for r in window_ablation]
    ws_losses = [r['final_loss'] for r in window_ablation]
    ws_comp = [r['mask_compression'] for r in window_ablation]
    ax6_twin = ax6.twinx()
    bars = ax6.bar(ws_labels, ws_losses, alpha=0.7, color='#2ca02c', label='Final Loss')
    line = ax6_twin.plot(ws_labels, ws_comp, 'ro-', linewidth=2, markersize=7, label='Compression')
    ax6.set_xlabel('Window Size')
    ax6.set_ylabel('Final Loss', color='#2ca02c')
    ax6_twin.set_ylabel('Compression Ratio (x)', color='red')
    ax6.set_title('(f) Window Size Ablation')
    ax6.grid(True, alpha=0.3, axis='y')

    # ── Figure 7: Sparsity Ablation ──
    ax7 = fig.add_subplot(4, 2, 7)
    sp_vals = [r['sparsity'] for r in sparsity_ablation]
    sp_losses = [r['final_loss'] for r in sparsity_ablation]
    sp_comp = [r['comm_compression'] for r in sparsity_ablation]
    ax7.plot(sp_vals, sp_losses, 'o-', color='#9467bd', linewidth=2, label='Final Loss')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(sp_vals, sp_comp, 's-', color='#d62728', linewidth=2, label='Comm Compression')
    ax7.set_xscale('log')
    ax7.set_xlabel('Gradient Sparsity (top-k fraction)')
    ax7.set_ylabel('Final Loss', color='#9467bd')
    ax7_twin.set_ylabel('Comm Compression (x)', color='#d62728')
    ax7.set_title('(g) Gradient Sparsity Ablation')
    ax7.grid(True, alpha=0.3)

    # ── Figure 8: Mask Creation Time ──
    ax8 = fig.add_subplot(4, 2, 8)
    ws128 = [r for r in mask_results if r['window_size'] == 128]
    sls = [r['seq_len'] for r in ws128]
    dense_t = [r['dense_time_s'] * 1000 for r in ws128]
    roaring_t = [r['roaring_time_s'] * 1000 for r in ws128]
    ax8.plot(sls, dense_t, 'o-', label='Dense', linewidth=2, markersize=6)
    ax8.plot(sls, roaring_t, 's-', label='Roaring', linewidth=2, markersize=6)
    ax8.set_xscale('log', base=2)
    ax8.set_yscale('log')
    ax8.set_xlabel('Sequence Length')
    ax8.set_ylabel('Time (ms)')
    ax8.set_title('(h) Mask Creation Time')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join('experiments', 'paper_figures.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\nAll figures saved to {path}")
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    mask_results = exp1_mask_scaling()
    grad_results = exp2_gradient_compression()
    training_summary, all_vanilla, all_roaring = exp3_training_convergence(num_steps=200, device=device)
    window_ablation = exp4_window_ablation(num_steps=100, device=device)
    sparsity_ablation = exp5_sparsity_ablation(num_steps=100, device=device)

    generate_all_plots(mask_results, training_summary, all_vanilla, all_roaring,
                       window_ablation, sparsity_ablation, grad_results)

    # Save all data
    all_results = {
        'mask_scaling': mask_results,
        'gradient_compression': [{k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                   for k, v in r.items()} for r in grad_results],
        'training_summary': {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                             for k, v in training_summary.items()},
        'window_ablation': window_ablation,
        'sparsity_ablation': sparsity_ablation,
    }
    path = os.path.join('experiments', 'paper_results.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results saved to {path}")


if __name__ == '__main__':
    main()
