#!/usr/bin/env python3
"""
Round-5: re-measure every composition timing with enough trials to be stable.

v4 used 5 trials per configuration. At n=1,024 the whole packed bitset is 128 KB
and lives in L2, so the operation takes ~10-50 us and a 5-trial median is not
reproducible: v4 recorded 0.052 ms for the bitset where five independent
repetitions give 0.011-0.012 ms. That turned a real Roaring LOSS at n=1,024 into
a reported 3.0x win, and put a fabricated lower bound on the margin range quoted
in the abstract, Section 5.4 and the conclusion.

This script re-runs only the timing-dependent measurements -- sizes and gradient
index encodings are deterministic and are carried over from v4_results.json
unchanged -- at 25 trials, with the entire setup-and-measure cycle repeated
REPETITIONS times so the between-repetition spread is recorded and can be quoted.

Usage:
    python experiments/run_v5_timings.py
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
import zstandard as zstd
from pyroaring import BitMap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_baseline_comparison import build_flat, timeit  # noqa: E402
from run_v4_measurements import (  # noqa: E402
    ABLATION_N, COMP_LENS, NGLOB, PAD, SWEEP_N, SWEEP_RANDOM, UNSTRUCT_LENS,
    WINDOW, WINDOWS, causal_rows, mask_sizes, packed_bits, runs_per_row)

OUT = os.path.join(HERE, 'v5_results.json')
ZSTD = zstd.ZstdCompressor(level=6)

TRIALS = 25          # per repetition (v4 used 5)
WARMUP = 5
REPETITIONS = 3      # independent setup+measure cycles

REPS = ['roaring_per_row_ms', 'roaring_flat_ms', 'dense_ms', 'packed_bitset_ms',
        'csr_ms', 'zstd_then_dense_ms', 'run_arrays_ms']
RIVALS = [r for r in REPS if not r.startswith('roaring')]


def _pad_vec(n):
    P = np.zeros(n, dtype=bool)
    P[:n - PAD] = True
    return P


def measure_once(n, second_operand, window=WINDOW):
    """One setup + timing cycle. Returns {representation: median_ms} plus meta."""
    rows = causal_rows(n, window)
    A = packed_bits(rows, n)
    P = _pad_vec(n)
    structured = isinstance(second_operand, str) and second_operand == 'global'

    if structured:
        S_vec = np.zeros(n, dtype=bool); S_vec[:NGLOB] = True
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

    def op_row():
        return [(b & pad_bm) | s for b, s in zip(bms, s_bms)]

    def op_flat():
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
        return (arr.reshape(n, n).astype(bool) & P) | S_op

    def op_runs():
        e = np.minimum(end, n - PAD)
        s = np.minimum(start, e)
        lo = np.minimum(e, NGLOB)
        ov = np.maximum(0, lo - np.minimum(s, lo))
        return NGLOB + (e - s) - ov

    card = int(op_dense().sum())
    assert card == len(op_flat()) == sum(len(b) for b in op_row())
    assert card == int(np.unpackbits(op_bitset(), axis=1)[:, :n].sum())
    c = op_csr(); c.eliminate_zeros(); assert card == c.nnz
    assert card == int(op_zstd().sum())
    if structured:
        assert card == int(op_runs().sum())

    t = {'roaring_per_row_ms': timeit(op_row, TRIALS, WARMUP)['median_ms'],
         'roaring_flat_ms': timeit(op_flat, TRIALS, WARMUP)['median_ms'],
         'dense_ms': timeit(op_dense, TRIALS, WARMUP)['median_ms'],
         'packed_bitset_ms': timeit(op_bitset, TRIALS, WARMUP)['median_ms'],
         'csr_ms': timeit(op_csr, TRIALS, WARMUP)['median_ms'],
         'zstd_then_dense_ms': timeit(op_zstd, TRIALS, WARMUP)['median_ms'],
         'run_arrays_ms': timeit(op_runs, TRIALS, WARMUP)['median_ms']
         if structured else None}
    t['_runs_per_row'] = runs_per_row(op_dense())
    t['_cardinality'] = card
    return t


def aggregate(n, second_operand, window=WINDOW, label=''):
    """Repeat the whole cycle and record median-of-medians plus spread."""
    runs = [measure_once(n, second_operand, window) for _ in range(REPETITIONS)]
    rec = {'seq_len': n, 'window': window,
           'runs_per_row': runs[0]['_runs_per_row'],
           'cardinality': runs[0]['_cardinality'],
           'trials': TRIALS, 'repetitions': REPETITIONS}
    for k in REPS:
        vals = [r[k] for r in runs]
        if vals[0] is None:
            rec[k] = None
            rec[k.replace('_ms', '_spread')] = None
            continue
        rec[k] = statistics.median(vals)
        rec[k.replace('_ms', '_spread')] = [min(vals), max(vals)]
    applicable = {k: rec[k] for k in RIVALS if rec[k] is not None}
    best = min(applicable, key=applicable.get)
    rec['best_rival'] = best
    rec['best_rival_ms'] = applicable[best]
    rec['roaring_wins'] = rec['roaring_flat_ms'] < applicable[best]
    rec['margin_over_best_rival'] = applicable[best] / rec['roaring_flat_ms']
    # margin computed at the extremes of the spread, i.e. the worst and best case
    rr = rec['roaring_flat_spread']
    br = rec[best.replace('_ms', '_spread')]
    rec['margin_spread'] = [br[0] / rr[1], br[1] / rr[0]]
    print(f"  {label:>22} n={n:>6} roaring={rec['roaring_flat_ms']:7.4f} "
          f"best rival={best.replace('_ms',''):>16}={rec['best_rival_ms']:7.4f} "
          f"margin={rec['margin_over_best_rival']:6.2f}x "
          f"[{rec['margin_spread'][0]:.2f}-{rec['margin_spread'][1]:.2f}] "
          f"{'WIN' if rec['roaring_wins'] else 'LOSE'}")
    return rec


def random_operand(n, k, seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros((n, n), dtype=bool)
    for i in range(n):
        S[i, rng.integers(0, n, size=k)] = True
    return S


def main():
    t0 = time.time()
    print(f'{TRIALS} trials x {REPETITIONS} repetitions per configuration\n')

    print('STRUCTURED  (causal & padding) | global')
    structured = [aggregate(n, 'global', label='structured') for n in COMP_LENS]

    print('\nUNSTRUCTURED  (causal & padding) | random attention')
    unstructured = [aggregate(n, random_operand(n, 6), label='unstructured')
                    for n in UNSTRUCT_LENS]

    print('\nSTRUCTURE SWEEP at n=%d' % SWEEP_N)
    sweep = []
    for k in SWEEP_RANDOM:
        op = 'global' if k == 0 else random_operand(SWEEP_N, k)
        rec = aggregate(SWEEP_N, op, label=f'rand/row={k}')
        rec['random_per_row'] = k
        sweep.append(rec)

    print('\nWINDOW ABLATION at n=%d' % ABLATION_N)
    shared = next(c for c in structured if c['seq_len'] == ABLATION_N)
    ablation = []
    for w in WINDOWS:
        rows = causal_rows(ABLATION_N, w)
        bits, sizes = mask_sizes(rows, ABLATION_N, contiguous=True)
        if w == WINDOW:
            rec = dict(shared)
            rec['shared_with_composition'] = True
        else:
            rec = aggregate(ABLATION_N, 'global', window=w, label=f'window={w}')
            rec['shared_with_composition'] = False
        ablation.append({'window': w, 'seq_len': ABLATION_N,
                         'density': float(bits.mean()), 'sizes_bytes': sizes,
                         'compose_flat_ms': rec['roaring_flat_ms'],
                         'compose_dense_ms': rec['dense_ms'],
                         'speedup': rec['dense_ms'] / rec['roaring_flat_ms'],
                         'shared_with_composition': rec['shared_with_composition']})

    print('\nMUTATION')
    mutation = []
    for n in COMP_LENS:
        vals_r, vals_d = [], []
        for _ in range(REPETITIONS):
            rows = causal_rows(n, WINDOW)
            bms = [BitMap(r) for r in rows]
            A = packed_bits(rows, n)
            sink = BitMap([n // 2])

            def mut_r():
                for b in bms:
                    b |= sink

            def mut_d():
                A[:, n // 2] = True

            vals_r.append(timeit(mut_r, TRIALS, WARMUP)['median_ms'])
            vals_d.append(timeit(mut_d, TRIALS, WARMUP)['median_ms'])
        mutation.append({'seq_len': n, 'roaring_ms': statistics.median(vals_r),
                         'dense_ms': statistics.median(vals_d),
                         'dense_speedup': statistics.median(vals_r) /
                                          statistics.median(vals_d)})
    print('  done')

    # ---- derived -------------------------------------------------------- #
    un_wins = [c['seq_len'] for c in unstructured if c['roaring_wins']]
    un_loses = [c['seq_len'] for c in unstructured if not c['roaring_wins']]
    winning = [c for c in unstructured if c['roaring_wins']]
    derived = {
        'trials': TRIALS, 'repetitions': REPETITIONS,
        'structured': {
            'roaring_wins_at': [c['seq_len'] for c in structured if c['roaring_wins']],
            'best_rival': sorted({c['best_rival'] for c in structured}),
            'deficit_min': min(1 / c['margin_over_best_rival'] for c in structured),
            'deficit_max': max(1 / c['margin_over_best_rival'] for c in structured),
            'deficit_at_max_n': 1 / structured[-1]['margin_over_best_rival'],
        },
        'unstructured': {
            'roaring_wins_at': un_wins, 'roaring_loses_at': un_loses,
            'threshold_n': min(un_wins) if un_wins else None,
            'best_rival': sorted({c['best_rival'] for c in unstructured}),
            'margin_min_where_winning': min((c['margin_over_best_rival']
                                             for c in winning), default=None),
            'margin_max': max(c['margin_over_best_rival'] for c in unstructured),
            'vs_dense_max': max(c['dense_ms'] / c['roaring_flat_ms']
                                for c in unstructured),
            'vs_dense_where_winning': [c['dense_ms'] / c['roaring_flat_ms']
                                       for c in winning],
            'loss_detail': [{'seq_len': c['seq_len'],
                             'roaring_ms': c['roaring_flat_ms'],
                             'best_rival': c['best_rival'],
                             'best_rival_ms': c['best_rival_ms'],
                             'bitset_bytes': c['seq_len'] ** 2 // 8}
                            for c in unstructured if not c['roaring_wins']],
        },
        'sweep': {
            'row_invariant': {k: sweep[0][k] for k in
                              ('runs_per_row', 'roaring_flat_ms', 'best_rival',
                               'best_rival_ms', 'margin_over_best_rival')},
            'first_row_dependent': {k: sweep[1][k] for k in
                                    ('runs_per_row', 'roaring_flat_ms', 'best_rival',
                                     'best_rival_ms', 'margin_over_best_rival')},
            'roaring_ms_min': sweep[1]['roaring_flat_ms'],
            'roaring_ms_max': sweep[-1]['roaring_flat_ms'],
            'runs_min': sweep[1]['runs_per_row'], 'runs_max': sweep[-1]['runs_per_row'],
            'dense_min': min(s['dense_ms'] for s in sweep[1:]),
            'dense_max': max(s['dense_ms'] for s in sweep[1:]),
        },
    }

    payload = {'meta': {'trials': TRIALS, 'warmup': WARMUP,
                        'repetitions': REPETITIONS, 'window': WINDOW,
                        'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
                        'python': platform.python_version(),
                        'platform': platform.platform(),
                        'processor': platform.processor(),
                        'wall_seconds': round(time.time() - t0, 1),
                        'note': ('Timings only. Sizes and gradient-index encodings are '
                                 'deterministic and carried over from v4_results.json.')},
               'composition_structured': structured,
               'composition_unstructured': unstructured,
               'structure_sweep': sweep,
               'window_ablation': ablation,
               'mutation': mutation,
               'derived': derived}
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    u = derived['unstructured']
    print('\n' + '=' * 72)
    print(f"unstructured: Roaring wins at {u['roaring_wins_at']}, loses at "
          f"{u['roaring_loses_at']}")
    print(f"              margin where winning: "
          f"{u['margin_min_where_winning']:.1f}x to {u['margin_max']:.1f}x "
          f"over {u['best_rival']}")
    print(f"              vs dense where winning: "
          f"{min(u['vs_dense_where_winning']):.0f}x to "
          f"{max(u['vs_dense_where_winning']):.0f}x")
    print(f"structured:   Roaring loses everywhere, by "
          f"{derived['structured']['deficit_min']:.1f}x to "
          f"{derived['structured']['deficit_max']:.1f}x")
    print(f"\nSaved -> {OUT}  ({payload['meta']['wall_seconds']}s)")


if __name__ == '__main__':
    main()
