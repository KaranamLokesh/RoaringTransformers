#!/usr/bin/env python3
"""
Revision experiments for "The Roaring Transformer".

Replaces the toy/synthetic/CPU evaluation with a real-corpus, GPU-capable,
statistically-tested benchmark:

  Toy model + synthetic data    -> >=10M-param model on WikiText-103
  CPU-only, no GPU              -> auto-detects and runs on CUDA
  Random-token convergence      -> real BPE corpus, validation perplexity
  Only 3 seeds, no sig. test    -> N seeds + Welch t-test + Mann-Whitney U
  Hydration overhead unmeasured -> per-seq-len cold-hydration profile,
                                 amortized per-step cost, GPU peak VRAM

Usage (cloud GPU, full run):
    python experiments/run_revision_experiments.py --dataset bpe \
        --config wikitext-103-raw-v1 --seeds 5 --steps 2000 --eval_steps 200

Quick smoke test (CPU, tiny):
    python experiments/run_revision_experiments.py --quick
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from src.vanilla_transformer import VanillaTransformer
from src.roaring_transformer import RoaringTransformer
from src.roaring_attention import RoaringAttentionMask
from src.data import get_dataloader


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device, max_batches=None):
    """Return mean cross-entropy (nats) over the eval set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids, targets = input_ids.to(device), targets.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(1, n)


def train_one(model, train_loader, val_loader, num_steps, device, name,
              eval_every, log_every=50):
    """Train a single model, returning loss curve + final validation loss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    losses, val_curve, step_times = [], [], []
    step = 0
    while step < num_steps:
        for input_ids, targets in train_loader:
            if step >= num_steps:
                break
            input_ids, targets = input_ids.to(device), targets.to(device)

            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            if device == 'cuda':
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)

            if hasattr(model, 'prune_step'):
                model.prune_step()

            losses.append(loss.item())
            if step % log_every == 0:
                print(f"  [{name}] step {step:5d} | train_loss {loss.item():.4f} "
                      f"| {step_times[-1]*1000:.1f} ms/step")
            if eval_every and step > 0 and step % eval_every == 0:
                vl = evaluate(model, val_loader, device, max_batches=50)
                val_curve.append((step, vl))
                print(f"  [{name}] step {step:5d} | VAL loss {vl:.4f} "
                      f"| ppl {np.exp(vl):.1f}")
            step += 1

    final_val = evaluate(model, val_loader, device, max_batches=100)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6
                    if device == 'cuda' else None)
    return {
        'name': name,
        'train_losses': losses,
        'val_curve': val_curve,
        'final_val_loss': final_val,
        'final_val_ppl': float(np.exp(final_val)),
        'avg_step_ms': float(np.mean(step_times) * 1000),
        'peak_vram_mb': peak_vram_mb,
    }


# --------------------------------------------------------------------------- #
# Hydration overhead profiling
# --------------------------------------------------------------------------- #
def profile_hydration(seq_lengths, window_size, device, num_forward_passes=2000):
    """Measure the cost of realizing a Roaring mask into a dense GPU tensor.

    Separates: (a) CPU bitmap build, (b) CPU dense fill, (c) host->device
    transfer. Because the Roaring Transformer caches the realized mask per
    sequence length, this whole cost is paid ONCE and amortized over every
    forward pass with that sequence length -- we report that amortized figure.
    """
    print("\n" + "=" * 64)
    print("HYDRATION OVERHEAD PROFILE (CPU Roaring -> dense -> accelerator)")
    print("=" * 64)
    results = []
    for n in seq_lengths:
        # (a) build the per-row Roaring bitmaps on CPU
        t0 = time.time()
        mask = RoaringAttentionMask(n, window_size=window_size, causal=True)
        build_s = time.time() - t0

        # (b) fill the dense CPU tensor
        t0 = time.time()
        dense_cpu = mask.realize_dense('cpu')
        fill_s = time.time() - t0

        # (c) host -> device transfer
        if device == 'cuda':
            torch.cuda.synchronize()
            t0 = time.time()
            _ = dense_cpu.to(device)
            torch.cuda.synchronize()
            transfer_s = time.time() - t0
        else:
            transfer_s = 0.0

        cold_total_ms = (build_s + fill_s + transfer_s) * 1000
        amortized_us = cold_total_ms * 1000 / num_forward_passes
        dense_vram_mb = (n * n * 4) / 1e6
        roaring_kb = mask.get_compressed_size_bytes() / 1024

        row = {
            'seq_len': n,
            'build_ms': build_s * 1000,
            'dense_fill_ms': fill_s * 1000,
            'transfer_ms': transfer_s * 1000,
            'cold_total_ms': cold_total_ms,
            'amortized_per_step_us': amortized_us,
            'dense_realized_vram_mb': dense_vram_mb,
            'roaring_cpu_kb': roaring_kb,
        }
        results.append(row)
        print(f"  n={n:5d} | build {row['build_ms']:7.2f} ms | fill "
              f"{row['dense_fill_ms']:7.2f} ms | xfer {row['transfer_ms']:6.2f} ms "
              f"| cold {cold_total_ms:8.2f} ms | amortized "
              f"{amortized_us:7.2f} us/step | realized VRAM {dense_vram_mb:7.1f} MB")
    return results


# --------------------------------------------------------------------------- #
# Statistics: seeds + significance test for equivalence
# --------------------------------------------------------------------------- #
def equivalence_stats(vanilla_vals, roaring_vals):
    from scipy import stats
    v, r = np.array(vanilla_vals), np.array(roaring_vals)
    out = {
        'vanilla_mean': float(v.mean()), 'vanilla_std': float(v.std(ddof=1)),
        'roaring_mean': float(r.mean()), 'roaring_std': float(r.std(ddof=1)),
        'mean_diff': float(r.mean() - v.mean()),
    }
    if len(v) >= 2 and len(r) >= 2:
        t_stat, t_p = stats.ttest_ind(v, r, equal_var=False)  # Welch
        u_stat, u_p = stats.mannwhitneyu(v, r, alternative='two-sided')
        out.update({
            'welch_t': float(t_stat), 'welch_p': float(t_p),
            'mannwhitney_u': float(u_stat), 'mannwhitney_p': float(u_p),
        })
    return out


# --------------------------------------------------------------------------- #
def build_models(cfg, vocab_size):
    vanilla = VanillaTransformer(
        vocab_size=vocab_size, embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        ff_dim=cfg['ff_dim'], max_seq_len=cfg['seq_len'],
        window_size=cfg['window_size'])
    # When pruning is off, set warmup beyond the run so prune_step never fires;
    # this isolates the LOSSLESS Roaring optimizations for the equivalence claim.
    prune_warmup = cfg['steps'] // 2 if cfg['prune'] else cfg['steps'] + 1
    roaring = RoaringTransformer(
        vocab_size=vocab_size, embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        ff_dim=cfg['ff_dim'], max_seq_len=cfg['seq_len'],
        window_size=cfg['window_size'], grad_sparsity=0.01,
        prune_ratio=0.3, prune_warmup=prune_warmup)
    return vanilla, roaring


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='bpe', choices=['bpe', 'synthetic'])
    p.add_argument('--config', default='wikitext-103-raw-v1',
                   help='HF wikitext config (use wikitext-2-raw-v1 for a fast run)')
    p.add_argument('--max_tokens', type=int, default=None,
                   help='Cap training tokens (budget GPU runs)')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--eval_steps', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--ff_dim', type=int, default=1024)
    p.add_argument('--window_size', type=int, default=256)
    p.add_argument('--prune', action='store_true',
                   help='Enable 30%% weight pruning (LOSSY). Off by default so the '
                        'equivalence run isolates the lossless optimizations.')
    p.add_argument('--device', default='auto')
    p.add_argument('--quick', action='store_true',
                   help='Tiny CPU smoke test (overrides scale args)')
    p.add_argument('--out', default='experiments/revision_results.json')
    args = p.parse_args()

    if args.quick:
        args.dataset = 'synthetic' if args.dataset == 'synthetic' else 'bpe'
        args.config = 'wikitext-2-raw-v1'
        args.seeds, args.steps, args.eval_steps = 2, 30, 15
        args.batch_size, args.seq_len = 4, 128
        args.embed_dim, args.num_layers, args.num_heads, args.ff_dim = 64, 2, 4, 128
        args.max_tokens = 50000

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print(f"Device: {device} | PyTorch {torch.__version__} | dataset={args.dataset}")

    cfg = {k: getattr(args, k) for k in
           ['embed_dim', 'num_heads', 'num_layers', 'ff_dim', 'seq_len',
            'window_size', 'steps', 'prune']}

    # ---- Data ----
    if args.dataset == 'bpe':
        train_loader = get_dataloader('bpe', seq_len=args.seq_len,
                                      batch_size=args.batch_size, split='train',
                                      config=args.config, max_tokens=args.max_tokens)
        val_loader = get_dataloader('bpe', seq_len=args.seq_len,
                                    batch_size=args.batch_size, split='validation',
                                    config=args.config)
        vocab_size = train_loader.dataset.vocab_size
    else:
        vocab_size = 5000
        train_loader = get_dataloader('synthetic', vocab_size=vocab_size,
                                      seq_len=args.seq_len, batch_size=args.batch_size,
                                      num_samples=args.batch_size * args.steps)
        val_loader = get_dataloader('synthetic', vocab_size=vocab_size,
                                    seq_len=args.seq_len, batch_size=args.batch_size,
                                    num_samples=args.batch_size * 100, split='val')
    print(f"vocab_size={vocab_size} | train_batches={len(train_loader)} "
          f"| val_batches={len(val_loader)}")

    # report parameter count once
    probe, _ = build_models(cfg, vocab_size)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Model parameters: {n_params:,}")
    del probe

    # ---- Multi-seed training ----
    runs = {'vanilla': [], 'roaring': []}
    for seed in range(args.seeds):
        print(f"\n{'#'*64}\n# SEED {seed}\n{'#'*64}")
        set_seed(seed)
        vanilla, roaring = build_models(cfg, vocab_size)

        print("\n--- Vanilla ---")
        runs['vanilla'].append(
            train_one(vanilla, train_loader, val_loader, args.steps, device,
                      f"Vanilla-s{seed}", args.eval_steps))
        del vanilla
        if device == 'cuda':
            torch.cuda.empty_cache()

        set_seed(seed)
        print("\n--- Roaring ---")
        runs['roaring'].append(
            train_one(roaring, train_loader, val_loader, args.steps, device,
                      f"Roaring-s{seed}", args.eval_steps))
        del roaring
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---- Statistics ----
    v_finals = [r['final_val_loss'] for r in runs['vanilla']]
    r_finals = [r['final_val_loss'] for r in runs['roaring']]
    stats_out = equivalence_stats(v_finals, r_finals)

    print("\n" + "=" * 64)
    print("CONVERGENCE EQUIVALENCE (final validation loss across seeds)")
    print("=" * 64)
    print(f"  Vanilla: {stats_out['vanilla_mean']:.4f} ± {stats_out['vanilla_std']:.4f}"
          f"  (ppl {np.exp(stats_out['vanilla_mean']):.1f})")
    print(f"  Roaring: {stats_out['roaring_mean']:.4f} ± {stats_out['roaring_std']:.4f}"
          f"  (ppl {np.exp(stats_out['roaring_mean']):.1f})")
    if 'welch_p' in stats_out:
        print(f"  Welch t-test:      p = {stats_out['welch_p']:.4f}")
        print(f"  Mann-Whitney U:    p = {stats_out['mannwhitney_p']:.4f}")
        print("  (high p => fail to reject equality => consistent with no quality penalty)")

    # ---- Hydration profile ----
    hydration = profile_hydration(
        [256, 512, 1024, 2048, 4096], args.window_size, device)

    # ---- GPU peak VRAM comparison ----
    if device == 'cuda':
        vp = np.mean([r['peak_vram_mb'] for r in runs['vanilla']])
        rp = np.mean([r['peak_vram_mb'] for r in runs['roaring']])
        print("\nPeak training VRAM (mean over seeds): "
              f"Vanilla {vp:.1f} MB | Roaring {rp:.1f} MB")

    # ---- Persist ----
    payload = {
        'meta': {
            'device': device, 'dataset': args.dataset, 'config': args.config,
            'n_params': n_params, 'seeds': args.seeds, 'steps': args.steps,
            'model_cfg': cfg, 'vocab_size': vocab_size,
        },
        'runs': runs,
        'equivalence': stats_out,
        'hydration': hydration,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results -> {args.out}")


if __name__ == '__main__':
    main()
