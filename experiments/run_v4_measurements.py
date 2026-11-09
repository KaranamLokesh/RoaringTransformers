#!/usr/bin/env python3
"""
Round-4 measurements: the composition claim gets the same baseline treatment the
size claim got in round 3.

Round 3 added trivial baselines for SIZE (Roaring vs zlib/zstd/lzma/
(start,len)/bitset) but left the reframed central claim -- composition speed --
measured only against a dense bool array and zstd->dense. That is the identical
methodological gap one level up. This script adds the next baselines a
practitioner would reach for:

    packed bitset   numpy bitwise ops on n^2/8 bytes, 8x less memory traffic
    scipy CSR       the general sparse representation an ML practitioner uses
    run arrays      the purpose-built (start,len) encoding, vectorised

and measures composition on TWO workloads rather than one:

    structured      (causal & padding) | global      -- one run per row survives
    unstructured    (causal & padding) | random      -- run structure destroyed

Roaring loses the structured workload to the run encoding and wins the
unstructured one. The crossover is measured directly (structure_sweep), which
is what turns "does the generality earn its overhead" into a number.

Usage:
    python experiments/run_v4_measurements.py
"""

import json
import lzma
import os
import platform
import sys
import time
import zlib

import numpy as np
import scipy.sparse as sp
import torch
import zstandard as zstd
from pyroaring import BitMap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_baseline_comparison import (  # noqa: E402
    bigbird, block_sparse, build_flat, delta_varint_size, synth_indices, timeit)

OUT = os.path.join(HERE, 'v4_results.json')
ZSTD = zstd.ZstdCompressor(level=6)

WINDOW = 256
PAD, NGLOB = 37, 8
SIZE_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384]
COMP_LENS = [1024, 2048, 4096, 8192, 16384]
UNSTRUCT_LENS = [1024, 2048, 4096, 8192]
WINDOWS = [32, 64, 128, 256, 512, 1024, None]
ABLATION_N = 8192
SWEEP_N = 8192
SWEEP_RANDOM = [0, 1, 2, 4, 8, 16, 32]
REPEATS = 5

PARAM_COUNTS = [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 17_735_936]
SPARSITIES = [0.001, 0.005, 0.01, 0.05, 0.10]
LAYOUTS = {'runs-20': 20, 'runs-500': 500, 'diffuse': 10 ** 9}

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
    bits = packed_bits(rows, n)
    packed = np.packbits(bits).tobytes()
    return bits, {
        'dense_f32': n * n * 4,
        'packed_bitset': len(packed),
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


def runs_per_row(mask):
    """Mean number of maximal contiguous runs per row -- why (start,len) breaks."""
    n = mask.shape[0]
    pad = np.zeros((n, 1), dtype=np.int8)
    d = np.diff(np.concatenate([pad, mask.astype(np.int8)], axis=1), axis=1)
    return float(np.clip(d, 0, None).sum() / n)


# --------------------------------------------------------------------------- #
# Sizes (unchanged from round 3)
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

        t0 = time.perf_counter()
        dense = torch.full((n, n), float('-inf'))
        for i, r in enumerate(rows):
            dense[i, r.start:r.stop] = 0.0
        t_dense = (time.perf_counter() - t0) * 1000
        del dense

        _, sizes = mask_sizes(rows, n, contiguous=True)
        sizes['roaring_per_row'] = per_row_bytes
        sizes['roaring_flat_opt'] = len(flat.serialize())
        out.append({'seq_len': n, 'window': WINDOW, 'sizes_bytes': sizes,
                    'smallest': smallest(sizes), 'dense_build_ms': t_dense,
                    'roaring_per_row_build_ms': t_per_row,
                    'roaring_flat_build_ms': t_flat})
        print(f"  n={n:>6}  zstd={sizes['zstd']/1024:8.1f}K  "
              f"flat+opt={sizes['roaring_flat_opt']/1024:8.1f}K -> {out[-1]['smallest']}")
    return out


def pattern_sweep():
    print('\n== irregular mask patterns ==')
    specs = [('bigbird', bigbird, [1024, 2048, 4096, 8192, 16384]),
             ('block_sparse', block_sparse, [1024, 2048, 4096])]
    out = []
    for name, builder, lens in specs:
        for n in lens:
            rows = builder(n, WINDOW) if name == 'bigbird' else builder(n)
            _, sizes = mask_sizes(rows, n, contiguous=False)
            out.append({'pattern': name, 'seq_len': n, 'window': WINDOW,
                        'sizes_bytes': sizes, 'smallest': smallest(sizes)})
            print(f"  {name:>12} n={n:>6} -> {out[-1]['smallest']}")
    return out


# --------------------------------------------------------------------------- #
# Composition -- the full baseline set
# --------------------------------------------------------------------------- #
def _pad_vec(n):
    P = np.zeros(n, dtype=bool)
    P[:n - PAD] = True
    return P


def compose(n, second_operand, repeats=REPEATS, window=WINDOW):
    """Time `(causal & padding) | second_operand` in every representation.

    `second_operand` is 'global' (a constant 8-column run, preserving one run per
    row) or an (n, n) boolean array of random attention (destroying run structure).
    Construction is excluded; cardinality is asserted identical everywhere.
    """
    rows = causal_rows(n, window)
    A = packed_bits(rows, n)
    P = _pad_vec(n)
    structured = isinstance(second_operand, str) and second_operand == 'global'

    if structured:
        # The global operand is one row broadcast over all rows. Dense and bitset
        # get that broadcast form (their fastest correct implementation); CSR gets
        # a full (n, n) operand because scipy.sparse cannot broadcast maximum().
        S_vec = np.zeros(n, dtype=bool)
        S_vec[:NGLOB] = True
        S = np.broadcast_to(S_vec, (n, n))
        s_rows = [range(0, NGLOB)] * n
    else:
        S_vec = None
        S = second_operand
        s_rows = [np.flatnonzero(S[i]).tolist() for i in range(n)]

    bms = [BitMap(r) for r in rows]
    s_bms = [BitMap(r) for r in s_rows]
    pad_bm = BitMap(range(0, n - PAD))
    flat = build_flat(rows, n, True)
    pad_flat = build_flat([range(0, n - PAD)] * n, n, True)
    s_flat = build_flat(s_rows, n, True)

    Ap, Pp = np.packbits(A, axis=1), np.packbits(P)
    Sp = np.packbits(S_vec) if structured else np.packbits(S, axis=1)
    S_op = S_vec if structured else S
    A_csr = sp.csr_matrix(A)
    P_csr = sp.csr_matrix(P.reshape(1, n))
    S_csr = sp.csr_matrix(np.ascontiguousarray(S))
    A_comp = ZSTD.compress(np.packbits(A).tobytes())

    start = (np.zeros(n, dtype=int) if window is None
             else np.maximum(0, np.arange(n) - window))
    end = np.arange(n) + 1

    def op_roaring_per_row():
        return [(b & pad_bm) | s for b, s in zip(bms, s_bms)]

    def op_roaring_flat():
        return (flat & pad_flat) | s_flat

    def op_dense():
        return (A & P) | S_op

    def op_bitset():
        return (Ap & Pp) | Sp

    def op_csr():
        return A_csr.multiply(P_csr).maximum(S_csr).tocsr()

    def op_zstd():
        raw = zstd.ZstdDecompressor().decompress(
            A_comp, max_output_size=(n * n + 7) // 8)
        arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n * n]
        return (arr.reshape(n, n).astype(bool) & P) | S

    def op_runs():
        """Structure-aware (start,len): valid only while each row stays one run."""
        e = np.minimum(end, n - PAD)
        s = np.minimum(start, e)
        lo = np.minimum(e, NGLOB)
        ov = np.maximum(0, lo - np.minimum(s, lo))
        return NGLOB + (e - s) - ov

    card = int(op_dense().sum())
    assert card == len(op_roaring_flat()), 'roaring flat cardinality'
    assert card == sum(len(b) for b in op_roaring_per_row()), 'roaring per-row cardinality'
    assert card == int(np.unpackbits(op_bitset(), axis=1)[:, :n].sum()), 'bitset cardinality'
    csr_res = op_csr(); csr_res.eliminate_zeros()
    assert card == csr_res.nnz, 'csr cardinality'
    assert card == int(op_zstd().sum()), 'zstd cardinality'

    times = {
        'roaring_per_row_ms': timeit(op_roaring_per_row, repeats)['median_ms'],
        'roaring_flat_ms': timeit(op_roaring_flat, repeats)['median_ms'],
        'dense_ms': timeit(op_dense, repeats)['median_ms'],
        'packed_bitset_ms': timeit(op_bitset, repeats)['median_ms'],
        'csr_ms': timeit(op_csr, repeats)['median_ms'],
        'zstd_then_dense_ms': timeit(op_zstd, repeats)['median_ms'],
    }
    if structured:
        assert card == int(op_runs().sum()), 'run-array cardinality'
        times['run_arrays_ms'] = timeit(op_runs, repeats)['median_ms']
    else:
        times['run_arrays_ms'] = None       # encoding does not apply

    rivals = {k: v for k, v in times.items()
              if v is not None and not k.startswith('roaring')}
    best = min(rivals, key=rivals.get)
    rec = {'seq_len': n, 'workload': 'structured' if structured else 'unstructured',
           'cardinality': card, 'runs_per_row': runs_per_row(op_dense()),
           'best_rival': best, 'roaring_vs_best_rival': times['roaring_flat_ms'] / rivals[best],
           'roaring_wins': times['roaring_flat_ms'] < rivals[best]}
    rec.update(times)
    return rec


def composition_structured():
    print('\n== STRUCTURED composition: (causal & padding) | global ==')
    print(f"{'n':>7} {'Roaring':>9} {'dense':>9} {'bitset':>9} {'CSR':>9} "
          f"{'runs[]':>9} {'best rival':>14} {'verdict':>9}")
    out = []
    for n in COMP_LENS:
        r = compose(n, 'global')
        out.append(r)
        print(f"{n:>7} {r['roaring_flat_ms']:>9.3f} {r['dense_ms']:>9.3f} "
              f"{r['packed_bitset_ms']:>9.3f} {r['csr_ms']:>9.3f} "
              f"{r['run_arrays_ms']:>9.4f} {r['best_rival']:>14} "
              f"{'WIN' if r['roaring_wins'] else 'lose':>9}")
    return out


def composition_unstructured(seed=0):
    print('\n== UNSTRUCTURED composition: (causal & padding) | random attention ==')
    print(f"{'n':>7} {'Roaring':>9} {'dense':>9} {'bitset':>9} {'CSR':>9} "
          f"{'runs/row':>9} {'best rival':>14} {'verdict':>9}")
    out = []
    for n in UNSTRUCT_LENS:
        rng = np.random.default_rng(seed)
        S = np.zeros((n, n), dtype=bool)
        for i in range(n):
            S[i, rng.integers(0, n, size=6)] = True
        r = compose(n, S)
        out.append(r)
        print(f"{n:>7} {r['roaring_flat_ms']:>9.3f} {r['dense_ms']:>9.3f} "
              f"{r['packed_bitset_ms']:>9.3f} {r['csr_ms']:>9.3f} "
              f"{r['runs_per_row']:>9.1f} {r['best_rival']:>14} "
              f"{'WIN' if r['roaring_wins'] else 'lose':>9}")
    return out


def structure_sweep(seed=0):
    """Where does generality start paying? Sweep run structure at fixed n."""
    print('\n== STRUCTURE SWEEP at n=%d: random entries per row ==' % SWEEP_N)
    print(f"{'rand/row':>9} {'runs/row':>9} {'Roaring':>9} {'dense':>9} "
          f"{'bitset':>9} {'CSR':>9} {'best rival':>14} {'verdict':>9}")
    out = []
    for k in SWEEP_RANDOM:
        if k == 0:
            r = compose(SWEEP_N, 'global')
        else:
            rng = np.random.default_rng(seed)
            S = np.zeros((SWEEP_N, SWEEP_N), dtype=bool)
            for i in range(SWEEP_N):
                S[i, rng.integers(0, SWEEP_N, size=k)] = True
            r = compose(SWEEP_N, S)
        r['random_per_row'] = k
        out.append(r)
        print(f"{k:>9} {r['runs_per_row']:>9.1f} {r['roaring_flat_ms']:>9.3f} "
              f"{r['dense_ms']:>9.3f} {r['packed_bitset_ms']:>9.3f} "
              f"{r['csr_ms']:>9.3f} {r['best_rival']:>14} "
              f"{'WIN' if r['roaring_wins'] else 'lose':>9}")
    return out


def mutation_sweep():
    print('\n== in-place mutation ==')
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
    print('  done')
    return out


def window_ablation(structured):
    print('\n== window-size ablation at n=%d ==' % ABLATION_N)
    shared = next(c for c in structured if c['seq_len'] == ABLATION_N)
    out = []
    for w in WINDOWS:
        rows = causal_rows(ABLATION_N, w)
        bits, sizes = mask_sizes(rows, ABLATION_N, contiguous=True)
        if w == WINDOW:
            t_flat, t_dense = shared['roaring_flat_ms'], shared['dense_ms']
            shared_flag = True
        else:
            r = compose(ABLATION_N, 'global', window=w)
            t_flat, t_dense = r['roaring_flat_ms'], r['dense_ms']
            shared_flag = False
        out.append({'window': w, 'seq_len': ABLATION_N, 'density': float(bits.mean()),
                    'sizes_bytes': sizes, 'compose_flat_ms': t_flat,
                    'compose_dense_ms': t_dense, 'speedup': t_dense / t_flat,
                    'shared_with_composition': shared_flag})
        print(f"  w={str(w):>5} density={bits.mean()*100:5.1f}% "
              f"speedup={t_dense/t_flat:.1f}x{'  [shared]' if shared_flag else ''}")
    return out


# --------------------------------------------------------------------------- #
# Gradient indices (unchanged from round 3)
# --------------------------------------------------------------------------- #
def gradient_index(seed=0):
    print('\n== Top-K gradient index encoding (diffuse) ==')
    g = torch.Generator().manual_seed(seed)
    out = []
    for N in PARAM_COUNTS:
        grad = torch.randn(N, generator=g)
        for spx in SPARSITIES:
            k = max(1, int(spx * N))
            idx = torch.topk(grad.abs(), k).indices.numpy().astype(np.uint32)
            idx.sort()
            bits = np.zeros(N, dtype=bool); bits[idx] = True
            sizes = {'raw_u32': int(idx.nbytes),
                     'roaring': len(BitMap(idx).serialize()),
                     'delta_varint': delta_varint_size(idx),
                     'zstd_bitset': len(ZSTD.compress(np.packbits(bits).tobytes()))}
            out.append({'n_params': N, 'sparsity': spx, 'k': int(k),
                        'sizes_bytes': sizes, 'smallest': min(sizes, key=sizes.get),
                        'roaring_vs_raw': sizes['raw_u32'] / sizes['roaring'],
                        'delta_vs_raw': sizes['raw_u32'] / sizes['delta_varint']})
    print('  done')
    return out


def gradient_layout():
    print('\n== gradient index layout sweep ==')
    out = []
    for N in (1_000_000, 17_735_936):
        for spx in (0.001, 0.01, 0.1):
            k = max(1, int(N * spx))
            for label, n_runs in LAYOUTS.items():
                idx = synth_indices(N, k, n_runs, seed=0)
                bits = np.zeros(N, dtype=bool); bits[idx] = True
                sizes = {'raw_u32': int(idx.astype(np.uint32).nbytes),
                         'roaring': len(BitMap(idx.astype(np.uint32)).serialize()),
                         'delta_varint': delta_varint_size(idx),
                         'zstd_bitset': len(ZSTD.compress(np.packbits(bits).tobytes()))}
                out.append({'n_params': N, 'sparsity': spx, 'layout': label,
                            'n_runs': n_runs, 'k': int(len(idx)),
                            'sizes_bytes': sizes, 'smallest': min(sizes, key=sizes.get)})
    print('  done')
    return out


# --------------------------------------------------------------------------- #
def derive(ms, ps, st, un, sw, gi, gl):
    by_n = {r['seq_len']: r for r in ms}
    st_by_n = {c['seq_len']: c for c in st}

    size_ratios = {n: by_n[n]['sizes_bytes']['roaring_per_row'] /
                   by_n[n]['sizes_bytes']['roaring_flat_opt'] for n in by_n}
    speed_ratios = {c['seq_len']: c['roaring_per_row_ms'] / c['roaring_flat_ms']
                    for c in st}
    all_masks = ms + ps
    roaring_wins = [m for m in all_masks
                    if m['smallest'] in ('roaring_flat_opt', 'roaring_per_row')]

    # crossover: fewest random entries per row at which Roaring beats every rival
    crossing = [s for s in sw if s['roaring_wins']]
    crossover = min((s['random_per_row'] for s in crossing), default=None)
    crossover_runs = next((s['runs_per_row'] for s in sw
                           if s['random_per_row'] == crossover), None)
    ref = 8192
    return {
        'layout_penalty': {
            'reference_seq_len': ref,
            'size_ratio_at_ref': size_ratios[ref],
            'speed_ratio_at_ref': speed_ratios[ref],
            'size_ratio_min': min(size_ratios.values()),
            'size_ratio_max': max(size_ratios.values()),
            'speed_ratio_min': min(speed_ratios.values()),
            'speed_ratio_max': max(speed_ratios.values()),
        },
        'mask_size_verdict': {
            'n_configurations': len(all_masks),
            'roaring_ever_smallest': bool(roaring_wins),
            'roaring_smallest_at': [(m.get('pattern', 'causal_window'), m['seq_len'])
                                    for m in roaring_wins],
        },
        'composition_verdict': {
            'structured_roaring_wins': [c['seq_len'] for c in st if c['roaring_wins']],
            'structured_best_rival': sorted({c['best_rival'] for c in st}),
            'structured_worst_deficit': max(c['roaring_vs_best_rival'] for c in st),
            'structured_deficit_at_max_n': st[-1]['roaring_vs_best_rival'],
            'unstructured_roaring_wins': [c['seq_len'] for c in un if c['roaring_wins']],
            'unstructured_best_rival': sorted({c['best_rival'] for c in un}),
            'unstructured_margin_min': min(1 / c['roaring_vs_best_rival'] for c in un),
            'unstructured_margin_max': max(1 / c['roaring_vs_best_rival'] for c in un),
            'unstructured_vs_dense_max': max(c['dense_ms'] / c['roaring_flat_ms']
                                             for c in un),
            'crossover_random_per_row': crossover,
            'crossover_runs_per_row': crossover_runs,
        },
        'gradient_verdict': {
            'diffuse_roaring_ever_smallest': any(r['smallest'] == 'roaring' for r in gi),
            'layout_roaring_smallest_at': [
                {'n_params': r['n_params'], 'sparsity': r['sparsity'],
                 'layout': r['layout'], 'n_runs': r['n_runs'],
                 'sizes_bytes': r['sizes_bytes']}
                for r in gl if r['smallest'] == 'roaring'],
        },
    }


def main():
    t0 = time.time()
    ms = mask_scaling()
    ps = pattern_sweep()
    st = composition_structured()
    un = composition_unstructured()
    sw = structure_sweep()
    mut = mutation_sweep()
    wa = window_ablation(st)
    gi = gradient_index()
    gl = gradient_layout()
    d = derive(ms, ps, st, un, sw, gi, gl)

    payload = {
        'meta': {'window': WINDOW, 'repeats': REPEATS,
                 'note': ('All timings come from a single process. The window=256 row '
                          'of the ablation is the same measurement as the n=8,192 row '
                          'of the structured composition sweep.'),
                 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
                 'torch': torch.__version__, 'python': platform.python_version(),
                 'platform': platform.platform(),
                 'wall_seconds': round(time.time() - t0, 1)},
        'mask_scaling': ms, 'pattern_sweep': ps,
        'composition_structured': st, 'composition_unstructured': un,
        'structure_sweep': sw, 'mutation': mut, 'window_ablation': wa,
        'gradient_index': gi, 'gradient_layout': gl, 'derived': d,
    }
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    cv = d['composition_verdict']
    print('\n' + '=' * 72)
    print(f"structured   : Roaring wins at {cv['structured_roaring_wins'] or 'NO n'}; "
          f"best rival {cv['structured_best_rival']}; "
          f"worst deficit {cv['structured_worst_deficit']:.1f}x")
    print(f"unstructured : Roaring wins at {cv['unstructured_roaring_wins']}; "
          f"margin {cv['unstructured_margin_min']:.1f}-{cv['unstructured_margin_max']:.1f}x "
          f"over best rival ({cv['unstructured_best_rival']}), "
          f"up to {cv['unstructured_vs_dense_max']:.0f}x over dense")
    print(f"crossover    : {cv['crossover_random_per_row']} random entries/row "
          f"({cv['crossover_runs_per_row']:.1f} runs/row)" if cv['crossover_random_per_row']
          is not None else 'crossover : none found')
    print(f"\nSaved -> {OUT}  ({payload['meta']['wall_seconds']}s)")


if __name__ == '__main__':
    main()
