#!/usr/bin/env python3
"""
All CPU measurements for the round-3 revision, produced by a SINGLE process.

Why one script: in the round-2 draft the mask-composition sweep came from
run_baseline_comparison.py while the window-size ablation came from
generate_revision_outputs.py. Both time the identical expression
`(causal & padding) | global`, but they ran in separate processes, so the same
quantity (dense composition at n=8,192, window=256) was reported as 5.27 ms in
one table and 4.56 ms in the other. Two tables disagreeing about one number is
a consistency defect.

Here the intersecting configuration is measured once and shared: the
window-ablation row for window=256 carries the numbers measured in the
composition sweep at n=8,192, so the two tables agree by construction rather
than by luck.

Also added relative to round 2:
  * zlib is measured (round 2 claimed it as a baseline in Section 2.4 and in
    contribution 1, but no table or figure ever showed it);
  * the gradient-index layout sweep is carried through to the manuscript --
    Roaring is smallest for run-structured index sets, which the round-2 draft
    claimed never happened;
  * every headline ratio the manuscript quotes (layout penalty, best-baseline
    margins, "no regime where Roaring is smallest") is derived here and stored,
    so the manuscript generator can read it instead of hard-coding it.

Usage:
    python experiments/run_v3_measurements.py
"""

import json
import lzma
import os
import platform
import sys
import time
import zlib

import numpy as np
import torch
import zstandard as zstd
from pyroaring import BitMap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_baseline_comparison import (  # noqa: E402
    bigbird, block_sparse, build_flat, delta_varint_size, synth_indices, timeit)

OUT = os.path.join(HERE, 'v3_results.json')
ZSTD = zstd.ZstdCompressor(level=6)

WINDOW = 256                                              # training default
SIZE_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384]
COMP_LENS = [1024, 2048, 4096, 8192, 16384]
WINDOWS = [32, 64, 128, 256, 512, 1024, None]             # None = full causal
ABLATION_N = 8192
REPEATS = 5                                               # "median of five trials"

PARAM_COUNTS = [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 17_735_936]
SPARSITIES = [0.001, 0.005, 0.01, 0.05, 0.10]
LAYOUTS = {'runs-20': 20, 'runs-500': 500, 'diffuse': 10 ** 9}

# Encodings that are candidates for "smallest representation of this mask".
# roaring_flat_opt is the layout the paper adopts; roaring_per_row is the one
# the submitted implementation used.
MASK_ENCODINGS = ['dense_f32', 'packed_bitset', 'startlen', 'zlib', 'zstd', 'lzma',
                  'roaring_per_row', 'roaring_flat_opt']


def causal_rows(n, w):
    return [range(max(0, i - w) if w is not None else 0, i + 1) for i in range(n)]


def packed_bits(rows, n):
    bits = np.zeros((n, n), dtype=bool)
    for i, r in enumerate(rows):
        if isinstance(r, range):
            bits[i, r.start:r.stop] = True
        else:
            bits[i, list(r)] = True
    return bits


def mask_sizes(rows, n, contiguous):
    """Serialized size in bytes of every candidate encoding of one mask."""
    bits = packed_bits(rows, n)
    packed = np.packbits(bits).tobytes()
    return bits, {
        'dense_f32': n * n * 4,
        'packed_bitset': len(packed),
        # (start, length) per row is only a valid encoding for contiguous runs
        'startlen': 8 * n if contiguous else None,
        'zlib': len(zlib.compress(packed, 9)),
        'zstd': len(ZSTD.compress(packed)),
        'lzma': len(lzma.compress(packed, preset=6)),
        'roaring_per_row': sum(len(BitMap(r).serialize()) for r in rows),
        'roaring_flat_opt': len(build_flat(rows, n, True).serialize()),
    }


def smallest(sizes):
    return min((k for k in MASK_ENCODINGS if sizes.get(k) is not None),
               key=lambda k: sizes[k])


# --------------------------------------------------------------------------- #
# 1. Mask representation size + construction cost (causal sliding window)
# --------------------------------------------------------------------------- #
def mask_scaling():
    print('\n== mask representation size, causal window=%d ==' % WINDOW)
    out = []
    for n in SIZE_LENS:
        rows = causal_rows(n, WINDOW)

        t0 = time.perf_counter()
        per_row = [BitMap(r) for r in rows]
        t_per_row = (time.perf_counter() - t0) * 1000
        per_row_bytes = sum(len(b.serialize()) for b in per_row)

        t0 = time.perf_counter()
        flat = build_flat(rows, n, True)
        t_flat = (time.perf_counter() - t0) * 1000

        # dense construction exactly as VanillaTransformer._create_dense_mask does it
        t0 = time.perf_counter()
        dense = torch.full((n, n), float('-inf'))
        for i, r in enumerate(rows):
            dense[i, r.start:r.stop] = 0.0
        t_dense = (time.perf_counter() - t0) * 1000
        del dense

        _, sizes = mask_sizes(rows, n, contiguous=True)
        sizes['roaring_per_row'] = per_row_bytes
        sizes['roaring_flat_opt'] = len(flat.serialize())

        rec = {'seq_len': n, 'window': WINDOW, 'sizes_bytes': sizes,
               'smallest': smallest(sizes),
               'dense_build_ms': t_dense,
               'roaring_per_row_build_ms': t_per_row,
               'roaring_flat_build_ms': t_flat}
        out.append(rec)
        print(f"  n={n:>6}  zstd={sizes['zstd']/1024:8.1f}K  "
              f"flat+opt={sizes['roaring_flat_opt']/1024:8.1f}K  "
              f"per-row={sizes['roaring_per_row']/1024:8.1f}K  -> {rec['smallest']}")
    return out


# --------------------------------------------------------------------------- #
# 2. Irregular patterns (BigBird, block-sparse)
# --------------------------------------------------------------------------- #
def pattern_sweep():
    print('\n== irregular mask patterns ==')
    specs = [('bigbird', bigbird, [1024, 2048, 4096, 8192, 16384]),
             ('block_sparse', block_sparse, [1024, 2048, 4096])]
    out = []
    for name, builder, lens in specs:
        for n in lens:
            rows = builder(n, WINDOW) if name == 'bigbird' else builder(n)
            _, sizes = mask_sizes(rows, n, contiguous=False)
            rec = {'pattern': name, 'seq_len': n, 'window': WINDOW,
                   'sizes_bytes': sizes, 'smallest': smallest(sizes)}
            out.append(rec)
            print(f"  {name:>12} n={n:>6}  best-compressor="
                  f"{min(sizes['zlib'], sizes['zstd'], sizes['lzma'])/1024:8.1f}K  "
                  f"flat+opt={sizes['roaring_flat_opt']/1024:8.1f}K  -> {rec['smallest']}")
    return out


# --------------------------------------------------------------------------- #
# 3. Composition (the operation that distinguishes the representations)
# --------------------------------------------------------------------------- #
def compose_once(n, w, repeats=REPEATS):
    """Time `(causal & padding) | global` in four representations.

    Every representation is built before timing starts; only the operation is
    measured. The byte-compressed path is charged for the decompression it
    cannot avoid. Returns (record, bits) so callers can reuse the dense mask.
    """
    rows = causal_rows(n, w)
    bits = packed_bits(rows, n)
    packed = np.packbits(bits).tobytes()

    bms = [BitMap(r) for r in rows]
    flat = build_flat(rows, n, True)
    pad_flat = build_flat([range(0, n - 37)] * n, n, True)
    glob_flat = build_flat([range(0, 8)] * n, n, True)
    pad_bm, glob_bm = BitMap(range(0, n - 37)), BitMap(range(8))

    A = bits
    P = np.zeros(n, dtype=bool); P[:n - 37] = True
    G = np.zeros(n, dtype=bool); G[:8] = True
    A_comp = ZSTD.compress(packed)

    def op_row():
        return [(b & pad_bm) | glob_bm for b in bms]

    def op_flat():
        return (flat & pad_flat) | glob_flat

    def op_dense():
        return (A & P) | G

    def op_zstd():
        raw = zstd.ZstdDecompressor().decompress(
            A_comp, max_output_size=(n * n + 7) // 8)
        arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n * n]
        return (arr.reshape(n, n).astype(bool) & P) | G

    t_row, t_flat = timeit(op_row, repeats), timeit(op_flat, repeats)
    t_dense, t_zstd = timeit(op_dense, repeats), timeit(op_zstd, repeats)

    card = sum(len(b) for b in op_row())
    assert card == len(op_flat()) == int(op_dense().sum()) == int(op_zstd().sum()), \
        f'cardinality mismatch at n={n}, w={w}'

    rec = {'seq_len': n, 'window': w,
           'roaring_per_row_ms': t_row['median_ms'],
           'roaring_flat_ms': t_flat['median_ms'],
           'dense_ms': t_dense['median_ms'],
           'zstd_then_dense_ms': t_zstd['median_ms'],
           'flat_speedup_vs_dense': t_dense['median_ms'] / t_flat['median_ms'],
           'per_row_slowdown_vs_flat': t_row['median_ms'] / t_flat['median_ms'],
           'cardinality': card,
           'sizes_bytes': {'roaring_per_row': sum(len(b.serialize()) for b in bms),
                           'roaring_flat_opt': len(flat.serialize()),
                           'zstd': len(A_comp),
                           'dense_f32': n * n * 4}}
    return rec, bits


def composition_sweep():
    print('\n== mask composition, construction excluded (median of %d) ==' % REPEATS)
    out = []
    for n in COMP_LENS:
        rec, _ = compose_once(n, WINDOW)
        out.append(rec)
        print(f"  n={n:>6}  flat={rec['roaring_flat_ms']:7.3f}  "
              f"dense={rec['dense_ms']:7.3f}  zstd={rec['zstd_then_dense_ms']:7.3f}  "
              f"-> {rec['flat_speedup_vs_dense']:.1f}x")
    return out


def mutation_sweep():
    print('\n== in-place mutation (add one global sink) ==')
    out = []
    for n in COMP_LENS:
        rows = causal_rows(n, WINDOW)
        bms = [BitMap(r) for r in rows]
        A = packed_bits(rows, n)
        sink = BitMap([n // 2])

        def mut_roaring():
            for b in bms:
                b |= sink

        def mut_dense():
            A[:, n // 2] = True

        t_r, t_d = timeit(mut_roaring, REPEATS), timeit(mut_dense, REPEATS)
        out.append({'seq_len': n, 'roaring_ms': t_r['median_ms'],
                    'dense_ms': t_d['median_ms'],
                    'dense_speedup': t_r['median_ms'] / t_d['median_ms']})
        print(f"  n={n:>6}  roaring={t_r['median_ms']:.3f}  dense={t_d['median_ms']:.3f}")
    return out


# --------------------------------------------------------------------------- #
# 4. Window-size ablation -- shares its window=256 row with the composition sweep
# --------------------------------------------------------------------------- #
def window_ablation(comp):
    print('\n== window-size ablation at n=%d ==' % ABLATION_N)
    shared = next(c for c in comp
                  if c['seq_len'] == ABLATION_N and c['window'] == WINDOW)
    out = []
    for w in WINDOWS:
        rows = causal_rows(ABLATION_N, w)
        bits, sizes = mask_sizes(rows, ABLATION_N, contiguous=True)
        if w == WINDOW:
            # Do NOT re-time: reuse the composition measurement so Table 5 and
            # Table 6 report one number for one configuration.
            t_flat, t_dense = shared['roaring_flat_ms'], shared['dense_ms']
            shared_flag = True
        else:
            rec, _ = compose_once(ABLATION_N, w)
            t_flat, t_dense = rec['roaring_flat_ms'], rec['dense_ms']
            shared_flag = False
        out.append({'window': w, 'seq_len': ABLATION_N,
                    'density': float(bits.mean()),
                    'sizes_bytes': sizes,
                    'compose_flat_ms': t_flat, 'compose_dense_ms': t_dense,
                    'speedup': t_dense / t_flat,
                    'shared_with_composition': shared_flag})
        print(f"  w={str(w):>5}  density={bits.mean()*100:5.1f}%  "
              f"flat+opt={sizes['roaring_flat_opt']/1024:7.1f}K  "
              f"zstd={sizes['zstd']/1024:6.1f}K  speedup={t_dense/t_flat:.1f}x"
              f"{'  [shared with Table 5]' if shared_flag else ''}")
    return out


# --------------------------------------------------------------------------- #
# 5. Gradient index encoding
# --------------------------------------------------------------------------- #
def gradient_index(seed=0):
    """Magnitude Top-K over an i.i.d. Gaussian gradient -> diffuse index sets."""
    print('\n== Top-K gradient index encoding (diffuse) ==')
    g = torch.Generator().manual_seed(seed)
    out = []
    for N in PARAM_COUNTS:
        grad = torch.randn(N, generator=g)
        for sp in SPARSITIES:
            k = max(1, int(sp * N))
            idx = torch.topk(grad.abs(), k).indices.numpy().astype(np.uint32)
            idx.sort()
            bits = np.zeros(N, dtype=bool); bits[idx] = True
            sizes = {
                'raw_u32': int(idx.nbytes),
                'roaring': len(BitMap(idx).serialize()),
                'delta_varint': delta_varint_size(idx),
                'zstd_bitset': len(ZSTD.compress(np.packbits(bits).tobytes())),
            }
            out.append({'n_params': N, 'sparsity': sp, 'k': int(k),
                        'sizes_bytes': sizes, 'smallest': min(sizes, key=sizes.get),
                        'roaring_vs_raw': sizes['raw_u32'] / sizes['roaring'],
                        'delta_vs_raw': sizes['raw_u32'] / sizes['delta_varint']})
        print(f"  N={N:>10} done")
    return out


def gradient_layout():
    """Same k, same N -- but the indices are arranged into contiguous runs.

    Round 2 measured this and never reported it, then claimed Roaring is never
    the best choice. It is, for run-structured index sets.
    """
    print('\n== Top-K gradient index encoding, index layout sweep ==')
    out = []
    for N in (1_000_000, 17_735_936):
        for sp in (0.001, 0.01, 0.1):
            k = max(1, int(N * sp))
            for label, n_runs in LAYOUTS.items():
                idx = synth_indices(N, k, n_runs, seed=0)
                bits = np.zeros(N, dtype=bool); bits[idx] = True
                sizes = {
                    'raw_u32': int(idx.astype(np.uint32).nbytes),
                    'roaring': len(BitMap(idx.astype(np.uint32)).serialize()),
                    'delta_varint': delta_varint_size(idx),
                    'zstd_bitset': len(ZSTD.compress(np.packbits(bits).tobytes())),
                }
                out.append({'n_params': N, 'sparsity': sp, 'layout': label,
                            'n_runs': n_runs, 'k': int(len(idx)),
                            'sizes_bytes': sizes,
                            'smallest': min(sizes, key=sizes.get)})
            print(f"  N={N:>10} sp={sp:<6} done")
    return out


# --------------------------------------------------------------------------- #
# 6. Derived claims -- so the manuscript never hard-codes a ratio
# --------------------------------------------------------------------------- #
def derive(ms, ps, comp, gi, gl):
    by_n = {r['seq_len']: r for r in ms}
    comp_by_n = {c['seq_len']: c for c in comp}

    def layout_size_ratio(n):
        s = by_n[n]['sizes_bytes']
        return s['roaring_per_row'] / s['roaring_flat_opt']

    size_ratios = {n: layout_size_ratio(n) for n in by_n}
    speed_ratios = {c['seq_len']: c['per_row_slowdown_vs_flat'] for c in comp}

    # "no mask regime in which Roaring is smallest" -- verified, not asserted
    all_masks = ms + ps
    roaring_wins = [m for m in all_masks
                    if m['smallest'] in ('roaring_flat_opt', 'roaring_per_row')]

    # best byte compressor per mask, and its margin over the best Roaring layout
    def best_compressor(sizes):
        cand = {k: sizes[k] for k in ('zlib', 'zstd', 'lzma')}
        return min(cand, key=cand.get), min(cand.values())

    pattern_margins = []
    for m in all_masks:
        s = m['sizes_bytes']
        name, size = best_compressor(s)
        pattern_margins.append({
            'pattern': m.get('pattern', 'causal_window'), 'seq_len': m['seq_len'],
            'best_compressor': name, 'best_compressor_bytes': size,
            'roaring_flat_opt_bytes': s['roaring_flat_opt'],
            'margin': s['roaring_flat_opt'] / size})

    gi_roaring_wins = [r for r in gi if r['smallest'] == 'roaring']
    gl_roaring_wins = [r for r in gl if r['smallest'] == 'roaring']

    ref_n = 8192
    return {
        'layout_penalty': {
            'reference_seq_len': ref_n,
            'size_ratio_at_ref': size_ratios[ref_n],
            'speed_ratio_at_ref': speed_ratios[ref_n],
            'size_ratio_min': min(size_ratios.values()),
            'size_ratio_max': max(size_ratios.values()),
            'size_ratio_min_at': min(size_ratios, key=size_ratios.get),
            'size_ratio_max_at': max(size_ratios, key=size_ratios.get),
            'speed_ratio_min': min(speed_ratios.values()),
            'speed_ratio_max': max(speed_ratios.values()),
        },
        'mask_size_verdict': {
            'n_configurations': len(all_masks),
            'roaring_ever_smallest': bool(roaring_wins),
            'roaring_smallest_at': [(m.get('pattern', 'causal_window'), m['seq_len'])
                                    for m in roaring_wins],
            'margins': pattern_margins,
        },
        'gradient_verdict': {
            'diffuse_roaring_ever_smallest': bool(gi_roaring_wins),
            'diffuse_roaring_vs_raw_min': min(r['roaring_vs_raw'] for r in gi),
            'diffuse_roaring_vs_raw_max': max(r['roaring_vs_raw'] for r in gi),
            'diffuse_delta_vs_raw_min': min(r['delta_vs_raw'] for r in gi),
            'diffuse_delta_vs_raw_max': max(r['delta_vs_raw'] for r in gi),
            'layout_roaring_smallest_at': [
                {'n_params': r['n_params'], 'sparsity': r['sparsity'],
                 'layout': r['layout'], 'n_runs': r['n_runs'],
                 'sizes_bytes': r['sizes_bytes']} for r in gl_roaring_wins],
        },
        'composition_range': {
            'min_speedup': min(c['flat_speedup_vs_dense'] for c in comp),
            'max_speedup': max(c['flat_speedup_vs_dense'] for c in comp),
            'min_at': min(comp_by_n, key=lambda n: comp_by_n[n]['flat_speedup_vs_dense']),
            'max_at': max(comp_by_n, key=lambda n: comp_by_n[n]['flat_speedup_vs_dense']),
        },
    }


def main():
    t0 = time.time()
    ms = mask_scaling()
    ps = pattern_sweep()
    comp = composition_sweep()
    mut = mutation_sweep()
    wa = window_ablation(comp)
    gi = gradient_index()
    gl = gradient_layout()
    derived = derive(ms, ps, comp, gi, gl)

    payload = {
        'meta': {
            'window': WINDOW, 'repeats': REPEATS,
            'note': ('All timings in this file come from a single process, so the '
                     'composition sweep and the window-size ablation are directly '
                     'comparable; the window=256 row of the ablation is the same '
                     'measurement as the n=8,192 row of the composition sweep.'),
            'numpy': np.__version__, 'torch': torch.__version__,
            'python': platform.python_version(), 'platform': platform.platform(),
            'wall_seconds': round(time.time() - t0, 1),
        },
        'mask_scaling': ms,
        'pattern_sweep': ps,
        'composition': comp,
        'mutation': mut,
        'window_ablation': wa,
        'gradient_index': gi,
        'gradient_layout': gl,
        'derived': derived,
    }
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    d = derived
    print('\n' + '=' * 70)
    print(f"Roaring ever smallest across {d['mask_size_verdict']['n_configurations']} "
          f"mask configurations: {d['mask_size_verdict']['roaring_ever_smallest']}")
    print(f"Roaring smallest for diffuse gradient indices: "
          f"{d['gradient_verdict']['diffuse_roaring_ever_smallest']}")
    print(f"Roaring smallest for run-structured gradient indices at: "
          f"{[(r['n_params'], r['sparsity'], r['layout']) for r in d['gradient_verdict']['layout_roaring_smallest_at']]}")
    print(f"Layout penalty at n={d['layout_penalty']['reference_seq_len']}: "
          f"{d['layout_penalty']['size_ratio_at_ref']:.2f}x size, "
          f"{d['layout_penalty']['speed_ratio_at_ref']:.1f}x compose")
    print(f"\nSaved -> {OUT}  ({payload['meta']['wall_seconds']}s)")


if __name__ == '__main__':
    main()
