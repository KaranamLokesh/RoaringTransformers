#!/usr/bin/env python3
"""
Roaring Transformer vs Vanilla Transformer — Comparison Experiments

Runs side-by-side training of both architectures and compares:
1. Attention mask memory (dense vs Roaring compressed)
2. Training loss convergence
3. Wall-clock time per step
4. Peak memory usage (RAM, optional CUDA)
5. Gradient compression efficiency (Roaring vs raw arrays)
6. Pruning mask compression

Usage:
    python experiments/run_comparison.py [--seq_lengths 512 1024 2048]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import json
import psutil
import torch
import torch.nn as nn
import numpy as np

from src.vanilla_transformer import VanillaTransformer
from src.roaring_transformer import RoaringTransformer
from src.roaring_attention import RoaringAttentionMask, benchmark_mask_creation
from src.roaring_gradient import RoaringGradientCompressor, benchmark_gradient_compression
from src.data import get_dataloader


def get_memory_mb():
    """Get current process RSS memory in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def train_model(model, dataloader, num_steps, device, model_name="model"):
    """Train a model and collect metrics."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        'losses': [],
        'step_times': [],
        'memory_mb': [],
    }

    step = 0
    for epoch in range(100):  # enough epochs to hit num_steps
        for input_ids, targets in dataloader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            mem_before = get_memory_mb()

            t0 = time.time()
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            step_time = time.time() - t0

            # Roaring-specific: pruning step
            if hasattr(model, 'prune_step'):
                model.prune_step()

            metrics['losses'].append(loss.item())
            metrics['step_times'].append(step_time)
            metrics['memory_mb'].append(get_memory_mb())

            if step % 10 == 0:
                print(f"  [{model_name}] step {step:4d} | loss {loss.item():.4f} | "
                      f"time {step_time:.3f}s | mem {get_memory_mb():.0f}MB")

            step += 1

        if step >= num_steps:
            break

    return metrics


def experiment_mask_memory(seq_lengths, window_size=128):
    """Experiment 1: Compare mask memory at various sequence lengths."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Attention Mask Memory Comparison")
    print("=" * 60)

    results = []
    for seq_len in seq_lengths:
        stats = benchmark_mask_creation(seq_len, window_size)
        results.append(stats)
        print(f"\n  seq_len={seq_len}, window={window_size}")
        print(f"    Dense:   {stats['dense_mem_bytes'] / 1024:.1f} KB  ({stats['dense_time_s']:.3f}s)")
        print(f"    Roaring: {stats['roaring_mem_bytes'] / 1024:.1f} KB  ({stats['roaring_time_s']:.3f}s)")
        print(f"    Compression: {stats['compression_ratio']:.1f}x")

    return results


def experiment_gradient_compression():
    """Experiment 2: Gradient compression efficiency."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Gradient Compression (Roaring vs Raw)")
    print("=" * 60)

    param_counts = [100_000, 500_000, 1_000_000, 5_000_000]
    results = []

    for pc in param_counts:
        stats = benchmark_gradient_compression(pc, sparsity=0.01)
        results.append(stats)
        print(f"\n  params={pc:,}")
        print(f"    Raw indices:     {stats['raw_index_bytes'] / 1024:.1f} KB")
        print(f"    Roaring indices: {stats['roaring_index_bytes'] / 1024:.1f} KB")
        print(f"    Index compression: {stats['index_compression_ratio']:.2f}x")
        print(f"    Compress time:   {stats['roaring_compress_time_s']*1000:.1f}ms")
        print(f"    Decompress time: {stats['roaring_decompress_time_s']*1000:.1f}ms")

    return results


def experiment_training_comparison(seq_len=256, num_steps=100, batch_size=8,
                                   vocab_size=5000, device='cpu'):
    """Experiment 3: Side-by-side training comparison."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3: Training Comparison (seq_len={seq_len}, steps={num_steps})")
    print("=" * 60)

    embed_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 256
    window_size = 64

    # Data
    dataloader = get_dataloader(
        dataset_type='synthetic', vocab_size=vocab_size, seq_len=seq_len,
        batch_size=batch_size, num_samples=max(500, num_steps * batch_size)
    )

    # --- Vanilla Transformer ---
    print("\n--- Vanilla Transformer ---")
    vanilla = VanillaTransformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, ff_dim=ff_dim, max_seq_len=seq_len,
        window_size=window_size
    )
    vanilla_param_count = sum(p.numel() for p in vanilla.parameters())
    print(f"  Parameters: {vanilla_param_count:,}")

    mem_before_vanilla = get_memory_mb()
    vanilla_metrics = train_model(vanilla, dataloader, num_steps, device, "Vanilla")
    vanilla_peak_mem = max(vanilla_metrics['memory_mb']) - mem_before_vanilla

    # Free vanilla
    del vanilla
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Roaring Transformer ---
    print("\n--- Roaring Transformer ---")
    roaring = RoaringTransformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, ff_dim=ff_dim, max_seq_len=seq_len,
        window_size=window_size, grad_sparsity=0.01,
        prune_ratio=0.3, prune_warmup=num_steps // 2
    )
    roaring_param_count = sum(p.numel() for p in roaring.parameters())
    print(f"  Parameters: {roaring_param_count:,}")

    mem_before_roaring = get_memory_mb()
    roaring_metrics = train_model(roaring, dataloader, num_steps, device, "Roaring")
    roaring_peak_mem = max(roaring_metrics['memory_mb']) - mem_before_roaring

    # Mask memory comparison
    vanilla_mask_bytes = seq_len * seq_len * 4
    roaring_mask_info = roaring.get_mask_memory_bytes(seq_len)

    # Gradient compression stats (on final gradients)
    roaring.train()
    dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    logits = roaring(dummy_input)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                 torch.randint(0, vocab_size, (seq_len,)).to(device))
    loss.backward()
    grad_stats = roaring.compress_gradients()

    # Summary
    print("\n" + "-" * 60)
    print("TRAINING COMPARISON SUMMARY")
    print("-" * 60)
    print(f"{'Metric':<35} {'Vanilla':>12} {'Roaring':>12}")
    print("-" * 60)
    print(f"{'Final Loss':<35} {vanilla_metrics['losses'][-1]:>12.4f} {roaring_metrics['losses'][-1]:>12.4f}")
    print(f"{'Avg Step Time (ms)':<35} {np.mean(vanilla_metrics['step_times'])*1000:>12.1f} {np.mean(roaring_metrics['step_times'])*1000:>12.1f}")
    print(f"{'Peak Memory Delta (MB)':<35} {vanilla_peak_mem:>12.1f} {roaring_peak_mem:>12.1f}")
    print(f"{'Mask Memory (KB)':<35} {vanilla_mask_bytes/1024:>12.1f} {roaring_mask_info['roaring_bytes']/1024:>12.1f}")
    print(f"{'Mask Compression Ratio':<35} {'1.0x':>12} {roaring_mask_info['compression_ratio']:>11.1f}x")

    # Aggregate gradient compression
    total_raw = sum(s['raw_total_bytes'] for s in grad_stats.values())
    total_roaring = sum(s['roaring_total_bytes'] for s in grad_stats.values())
    if total_roaring > 0:
        print(f"{'Grad Comm (raw, KB)':<35} {total_raw/1024:>12.1f}")
        print(f"{'Grad Comm (Roaring, KB)':<35} {'':>12} {total_roaring/1024:>12.1f}")
        print(f"{'Grad Compression Ratio':<35} {'':>12} {total_raw/total_roaring:>11.2f}x")

    # Pruning stats
    prune_stats = roaring.pruning_manager.get_stats()
    if prune_stats['is_pruned']:
        print(f"{'Pruned Ratio':<35} {'N/A':>12} {prune_stats['pruned_ratio']:>11.1%}")
        print(f"{'Prune Mask (Roaring, KB)':<35} {'N/A':>12} {prune_stats['roaring_mask_bytes']/1024:>12.1f}")
        print(f"{'Prune Mask Compression':<35} {'N/A':>12} {prune_stats['mask_compression_ratio']:>11.1f}x")

    return {
        'vanilla': vanilla_metrics,
        'roaring': roaring_metrics,
        'mask_info': roaring_mask_info,
        'grad_stats': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                           for kk, vv in v.items()} for k, v in grad_stats.items()},
        'prune_stats': prune_stats,
    }


def plot_results(training_results, mask_results, output_dir='experiments'):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Roaring Transformer vs Vanilla Transformer', fontsize=14, fontweight='bold')

    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(training_results['vanilla']['losses'], label='Vanilla', alpha=0.7)
    ax.plot(training_results['roaring']['losses'], label='Roaring', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Step Time
    ax = axes[0, 1]
    ax.plot(training_results['vanilla']['step_times'], label='Vanilla', alpha=0.5)
    ax.plot(training_results['roaring']['step_times'], label='Roaring', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title('Per-Step Wall-Clock Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Memory Usage
    ax = axes[1, 0]
    ax.plot(training_results['vanilla']['memory_mb'], label='Vanilla', alpha=0.7)
    ax.plot(training_results['roaring']['memory_mb'], label='Roaring', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('RSS Memory (MB)')
    ax.set_title('Process Memory During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Mask Memory Scaling
    ax = axes[1, 1]
    seq_lens = [r['seq_len'] for r in mask_results]
    dense_mem = [r['dense_mem_bytes'] / (1024 * 1024) for r in mask_results]
    roaring_mem = [r['roaring_mem_bytes'] / (1024 * 1024) for r in mask_results]
    ax.plot(seq_lens, dense_mem, 'o-', label='Dense Mask', linewidth=2)
    ax.plot(seq_lens, roaring_mem, 's-', label='Roaring Mask', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Mask Memory Scaling (O(n²) vs ~O(n))')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_results.png')
    plt.savefig(path, dpi=150)
    print(f"\nPlots saved to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Roaring vs Vanilla Transformer Comparison')
    parser.add_argument('--seq_lengths', nargs='+', type=int, default=[256, 512, 1024, 2048],
                        help='Sequence lengths for mask memory experiment')
    parser.add_argument('--train_seq_len', type=int, default=256,
                        help='Sequence length for training comparison')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Training steps per model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, auto)')
    parser.add_argument('--window_size', type=int, default=128,
                        help='Sliding window size for sparse attention')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Experiment 1: Mask memory scaling
    mask_results = experiment_mask_memory(args.seq_lengths, args.window_size)

    # Experiment 2: Gradient compression
    grad_results = experiment_gradient_compression()

    # Experiment 3: Training comparison
    training_results = experiment_training_comparison(
        seq_len=args.train_seq_len,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=device,
    )

    # Generate plots
    plot_results(training_results, mask_results)

    # Save raw results
    output_path = os.path.join('experiments', 'results.json')
    serializable = {
        'mask_results': mask_results,
        'training_summary': {
            'vanilla_final_loss': training_results['vanilla']['losses'][-1],
            'roaring_final_loss': training_results['roaring']['losses'][-1],
            'vanilla_avg_step_ms': float(np.mean(training_results['vanilla']['step_times']) * 1000),
            'roaring_avg_step_ms': float(np.mean(training_results['roaring']['step_times']) * 1000),
            'mask_compression_ratio': training_results['mask_info']['compression_ratio'],
        },
        'prune_stats': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                        for k, v in training_results['prune_stats'].items()},
    }
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
