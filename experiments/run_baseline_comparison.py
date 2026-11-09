#!/usr/bin/env python3
"""
Baseline comparison for the round-2 revision of "The Roaring Transformer".

Motivation: benchmarking Roaring only against a dense float32 mask is a weak
test, because nobody would use that for a structured mask. The interesting
question is whether Roaring's generality earns its overhead against a
special-purpose encoding of the known structure.

This script answers that question in three parts:

  A. Mask encoding size    -- Roaring vs (start,len) tuples, packed bitset,
                              zlib/zstd/lzma, across causal / BigBird /
                              block-sparse patterns.
  B. Gradient index size   -- Roaring vs delta+varint, bitset, zstd.
  C. Composition + query   -- the operations Roaring can do that a byte-stream
                              compressor cannot: set algebra without
                              decompression, in-place mutation, membership.

Parts A and B are size-only and Roaring loses most of them; that result is
reported as-is. Part C is where the surviving contribution lives.

Usage:
    python experiments/run_baseline_comparison.py --out experiments/baseline_results.json
"""

import argparse
import json
import lzma
import os
import statistics
import sys
import time
import zlib

import numpy as np
import zstandard as zstd
from pyroaring import BitMap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ZSTD = zstd.ZstdCompressor(level=6)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
def varint_bytes(values):
    """LEB128 varint encoding -- the standard companion to delta encoding."""
    out = bytearray()
    for v in values:
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def delta_varint_size(sorted_idx):
    deltas = np.diff(np.concatenate([[0], np.asarray(sorted_idx, dtype=np.int64)]))
    return len(varint_bytes(deltas.tolist()))


def packbits(rows, n):
    bits = np.zeros((n, n), dtype=bool)
    for i, r in enumerate(rows):
        bits[i, list(r)] = True
    return np.packbits(bits).tobytes()


def build_flat(rows, n, run_optimize):
    """One Roaring bitmap over the flattened n*n key space.

    Contiguous row spans are inserted with add_range so they become Run
    Containers directly; run_optimize() then re-encodes the remaining
    array/bitmap containers. The paper's implementation does neither, which
    costs it ~2.3x on size and ~25x on composition time.
    """
    bm = BitMap()
    for i, r in enumerate(rows):
        cols = list(r)
        if not cols:
            continue
        if cols == list(range(cols[0], cols[-1] + 1)):
            bm.add_range(i * n + cols[0], i * n + cols[-1] + 1)
        else:
            bm.update([i * n + c for c in cols])
    if run_optimize:
        bm.run_optimize()
    return bm


def mask_encoding_sizes(rows, n, contiguous):
    """Size in bytes of every candidate encoding of one attention mask."""
    raw_bits = packbits(rows, n)
    per_row = [BitMap(r) for r in rows]
    sizes = {
        'dense_f32': n * n * 4,
        'packed_bitset': len(raw_bits),
        'zlib_bitset': len(zlib.compress(raw_bits, 9)),
        'zstd_bitset': len(ZSTD.compress(raw_bits)),
        'lzma_bitset': len(lzma.compress(raw_bits, preset=6)),
        'roaring_per_row': sum(len(b.serialize()) for b in per_row),
        'roaring_flat': len(build_flat(rows, n, False).serialize()),
        'roaring_flat_opt': len(build_flat(rows, n, True).serialize()),
    }
    # (start, length) per row is only a valid encoding for contiguous runs.
    sizes['rowwise_start_len_u32'] = 8 * n if contiguous else None
    return sizes


# --------------------------------------------------------------------------- #
# Mask patterns
# --------------------------------------------------------------------------- #
def causal_window(n, w=128):
    return [range(max(0, i - w), i + 1) for i in range(n)]


def bigbird(n, w=128, n_global=8, n_random=6, seed=0):
    """Sliding window + global sinks + random attention (BigBird-like)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = set(range(max(0, i - w), i + 1)) | {g for g in range(n_global) if g <= i}
        if i > 0:
            s |= set(rng.integers(0, i + 1, size=min(n_random, i + 1)).tolist())
        out.append(sorted(s))
    return out


def block_sparse(n, block=64, keep=0.25, seed=0):
    """Content-dependent block sparsity: a random causal subset of blocks."""
    rng = np.random.default_rng(seed)
    out = []
    for bi in range(n // block):
        cand = list(range(bi + 1))
        k = max(1, int(len(cand) * keep))
        chosen = set(rng.choice(cand, size=k, replace=False).tolist()) | {bi}
        cols = sorted(c for bj in chosen for c in range(bj * block, (bj + 1) * block))
        for i in range(bi * block, (bi + 1) * block):
            out.append([c for c in cols if c <= i])
    return out


PATTERNS = {
    'causal_window': (causal_window, True),
    'bigbird': (bigbird, False),
    'block_sparse': (block_sparse, False),
}


# --------------------------------------------------------------------------- #
# Part A -- mask encoding sizes
# --------------------------------------------------------------------------- #
def part_a(seq_lengths, window):
    print("\n" + "=" * 78)
    print("PART A -- attention mask encoding size (bytes)")
    print("=" * 78)
    results = []
    for pat_name, (builder, contiguous) in PATTERNS.items():
        print(f"\n--- {pat_name} ---")
        hdr = (f"{'n':>6} {'dense f32':>11} {'bitset':>9} {'(start,len)':>12} "
               f"{'zstd':>9} {'lzma':>9} {'Roar/row':>10} {'Roar/flat+':>11}  best")
        print(hdr)
        for n in seq_lengths:
            if pat_name == 'block_sparse' and n > 4096:
                continue
            rows = builder(n, window) if pat_name != 'block_sparse' else builder(n)
            s = mask_encoding_sizes(rows, n, contiguous)
            best = min((k for k, v in s.items() if v is not None), key=lambda k: s[k])
            results.append({'pattern': pat_name, 'seq_len': n, 'window': window,
                            'sizes_bytes': s, 'best': best,
                            'roaring_vs_best': s['roaring_per_row'] / s[best]})

            def f(k):
                return '   n/a' if s[k] is None else f"{s[k]/1024:.1f}K"
            print(f"{n:>6} {f('dense_f32'):>11} {f('packed_bitset'):>9} "
                  f"{f('rowwise_start_len_u32'):>12} {f('zstd_bitset'):>9} "
                  f"{f('lzma_bitset'):>9} {f('roaring_per_row'):>10} "
                  f"{f('roaring_flat_opt'):>11}  {best}")
    return results


# --------------------------------------------------------------------------- #
# Part B -- gradient index encoding sizes
# --------------------------------------------------------------------------- #
def part_b(n_params, sparsities, clusterings):
    print("\n" + "=" * 78)
    print("PART B -- Top-K gradient index encoding size (bytes)")
    print("=" * 78)
    print(f"{'N':>11} {'sparsity':>9} {'layout':>12} {'k':>9} {'raw u32':>9} "
          f"{'delta+vint':>11} {'zstd(bs)':>9} {'Roaring':>9}  best")
    results = []
    for N in n_params:
        for sp in sparsities:
            k = max(1, int(N * sp))
            for label, n_runs in clusterings.items():
                idx = synth_indices(N, k, n_runs, seed=0)
                bits = np.zeros(N, dtype=bool)
                bits[idx] = True
                s = {
                    'raw_u32': idx.astype(np.uint32).nbytes,
                    'delta_varint': delta_varint_size(idx),
                    'bitset': (N + 7) // 8,
                    'zstd_bitset': len(ZSTD.compress(np.packbits(bits).tobytes())),
                    'roaring': len(BitMap(idx.astype(np.uint32)).serialize()),
                }
                best = min(s, key=s.get)
                results.append({'n_params': N, 'sparsity': sp, 'layout': label,
                                'k': int(len(idx)), 'sizes_bytes': s, 'best': best,
                                'roaring_vs_best': s['roaring'] / s[best]})
                print(f"{N:>11} {sp:>9} {label:>12} {len(idx):>9} "
                      f"{s['raw_u32']/1024:>8.0f}K {s['delta_varint']/1024:>10.0f}K "
                      f"{s['zstd_bitset']/1024:>8.0f}K {s['roaring']/1024:>8.0f}K  {best}")
    return results


def synth_indices(N, k, n_runs, seed=0):
    """k indices spread over n_runs contiguous runs (n_runs=k => fully diffuse)."""
    rng = np.random.default_rng(seed)
    if n_runs >= k:
        return np.sort(rng.choice(N, size=k, replace=False))
    per = max(1, k // n_runs)
    starts = rng.choice(N - per, size=n_runs, replace=False)
    return np.unique(np.concatenate([np.arange(s, s + per) for s in starts]))


# --------------------------------------------------------------------------- #
# Part C -- composition and query (the surviving contribution)
# --------------------------------------------------------------------------- #
def timeit(fn, repeats=7, warmup=2):
    """Median wall time in ms over `repeats` trials, after `warmup` untimed runs."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {'median_ms': statistics.median(ts),
            'min_ms': min(ts),
            'stdev_ms': statistics.stdev(ts) if len(ts) > 1 else 0.0}


def part_c(seq_lengths, window, repeats):
    """Compose `causal_window AND padding OR global`, with construction excluded.

    Every representation is built ONCE up front; only the composition itself is
    timed. Byte-stream compressors must decompress before they can operate, and
    that decompression is counted against them because it is unavoidable.
    """
    print("\n" + "=" * 78)
    print("PART C -- mask composition, construction excluded (median ms)")
    print("=" * 78)
    print(f"{'n':>6} {'Roar/row':>12} {'Roar/flat':>12} {'numpy dense':>12} "
          f"{'zstd+numpy':>12}  flat vs np")
    results = []
    for n in seq_lengths:
        rows = causal_window(n, window)
        pad_cols = list(range(0, n - 37))          # a ragged padding mask
        glob = list(range(8))

        # --- pre-built representations (NOT timed) ---
        bms = [BitMap(r) for r in rows]
        pad_bm, glob_bm = BitMap(pad_cols), BitMap(glob)

        flat = build_flat(rows, n, True)
        pad_flat = build_flat([range(0, n - 37)] * n, n, True)
        glob_flat = build_flat([range(0, 8)] * n, n, True)

        A = np.zeros((n, n), dtype=bool)
        for i, r in enumerate(rows):
            A[i, r.start:r.stop] = True
        P = np.zeros(n, dtype=bool); P[pad_cols] = True
        G = np.zeros(n, dtype=bool); G[glob] = True
        A_comp = ZSTD.compress(np.packbits(A).tobytes())

        def op_row():
            return [(b & pad_bm) | glob_bm for b in bms]

        def op_flat():
            return (flat & pad_flat) | glob_flat

        def op_numpy():
            return (A & P) | G

        def op_zstd():
            raw = zstd.ZstdDecompressor().decompress(
                A_comp, max_output_size=(n * n + 7) // 8)
            arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n * n]
            return (arr.reshape(n, n).astype(bool) & P) | G

        t_r, t_f, t_n, t_z = (timeit(op_row, repeats), timeit(op_flat, repeats),
                              timeit(op_numpy, repeats), timeit(op_zstd, repeats))

        # correctness: all four must agree on cardinality
        card = sum(len(b) for b in op_row())
        assert card == len(op_flat()) == int(op_numpy().sum()) == int(op_zstd().sum()), \
            f"cardinality mismatch at n={n}"

        results.append({'seq_len': n, 'roaring_per_row': t_r, 'roaring_flat': t_f,
                        'numpy_dense': t_n, 'zstd_then_numpy': t_z,
                        'flat_speedup_vs_numpy': t_n['median_ms'] / t_f['median_ms'],
                        'cardinality': card})
        print(f"{n:>6} {t_r['median_ms']:>12.2f} {t_f['median_ms']:>12.2f} "
              f"{t_n['median_ms']:>12.2f} {t_z['median_ms']:>12.2f}  "
              f"{t_n['median_ms']/t_f['median_ms']:>9.1f}x")

    # --- in-place mutation: add a global sink token to an existing mask ---
    print("\n--- in-place mutation (add one global sink to a built mask) ---")
    print(f"{'n':>6} {'Roaring':>12} {'numpy dense':>12}  speedup")
    mutation = []
    for n in seq_lengths:
        rows = causal_window(n, window)
        bms = [BitMap(r) for r in rows]
        A = np.zeros((n, n), dtype=bool)
        for i, r in enumerate(rows):
            A[i, r.start:r.stop] = True
        new_sink = BitMap([n // 2])

        def mut_roaring():
            for b in bms:
                b |= new_sink

        def mut_numpy():
            A[:, n // 2] = True

        t_r, t_n = timeit(mut_roaring, repeats), timeit(mut_numpy, repeats)
        mutation.append({'seq_len': n, 'roaring': t_r, 'numpy_dense': t_n})
        print(f"{n:>6} {t_r['median_ms']:>11.3f} {t_n['median_ms']:>11.3f}  "
              f"{t_n['median_ms']/max(1e-9, t_r['median_ms']):.2f}x")

    return {'composition': results, 'mutation': mutation}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq_lengths', type=int, nargs='+',
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--window', type=int, default=256,
                   help='sliding window; 256 matches the training default')
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                 'baseline_results.json'))
    args = p.parse_args()

    payload = {
        'meta': {'seq_lengths': args.seq_lengths, 'window': args.window,
                 'repeats': args.repeats, 'numpy': np.__version__},
        'part_a_mask_sizes': part_a(args.seq_lengths, args.window),
        'part_b_gradient_sizes': part_b(
            n_params=[1_000_000, 17_735_936],
            sparsities=[0.001, 0.01, 0.1],
            clusterings={'v.clustered': 20, 'clustered': 500, 'diffuse': 10 ** 9}),
        'part_c_operations': part_c(args.seq_lengths, args.window, args.repeats),
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
