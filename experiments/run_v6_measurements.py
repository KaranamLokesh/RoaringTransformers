#!/usr/bin/env python3
"""
Round-6 measurements (CPU only), same protocol as run_v5_timings.py:
25 trials after 5 warm-up runs, the whole setup-and-measure cycle repeated
3 times, median of the per-repetition medians reported with the min-max spread.

1. Content-dependence sweep at n=8,192 with a second pattern. Table 7 uses
   uniform-random extra indices per row. Here the k extra indices of each row
   are clustered: one contiguous block of k columns at a random offset per row.
   The question is whether the winner flip between the row-invariant first row
   (k=0) and the first row-dependent row (k=1) holds when the row-dependent
   component is clustered rather than scattered. The uniform pattern is
   re-measured in the same process so the two patterns are directly comparable.

2. Window-size ablation (Table 8) with the between-repetition spread recorded.
   run_v5_timings.py kept only the median for the window ablation rows, so the
   spread cannot be recovered from v5_results.json; this re-measures all seven
   windows in one process.

Usage:
    python experiments/run_v6_measurements.py
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_v4_measurements import ABLATION_N, SWEEP_N, SWEEP_RANDOM, WINDOWS  # noqa: E402
from run_v5_timings import (REPETITIONS, TRIALS, WARMUP, aggregate,  # noqa: E402
                            random_operand)

OUT = os.path.join(HERE, 'v6_results.json')


def clustered_operand(n, k, seed=0):
    """k extra columns per row as one contiguous block at a random offset."""
    rng = np.random.default_rng(seed)
    S = np.zeros((n, n), dtype=bool)
    starts = rng.integers(0, n - k + 1, size=n)
    for i in range(n):
        S[i, starts[i]:starts[i] + k] = True
    return S


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def machine_info():
    keys = ['machdep.cpu.brand_string', 'hw.model', 'hw.ncpu',
            'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu',
            'hw.perflevel0.l1dcachesize', 'hw.perflevel0.l2cachesize',
            'hw.perflevel1.l1dcachesize', 'hw.perflevel1.l2cachesize',
            'hw.memsize', 'hw.cachelinesize']
    info = {k: _cmd(['sysctl', '-n', k]) for k in keys}
    info['os'] = _cmd(['sw_vers', '-productVersion'])
    info['power_source'] = _cmd(['pmset', '-g', 'ps'])
    return info


def sweep(make_operand, label):
    out = []
    for k in SWEEP_RANDOM:
        op = 'global' if k == 0 else make_operand(SWEEP_N, k)
        rec = aggregate(SWEEP_N, op, label=f'{label} k={k}')
        rec['extra_per_row'] = k
        out.append(rec)
    return out


def main():
    import pyroaring
    import scipy
    import zstandard

    t0 = time.time()
    print(f'{TRIALS} trials x {REPETITIONS} repetitions per configuration\n')

    print('CONTENT-DEPENDENCE SWEEP at n=%d, clustered extra indices' % SWEEP_N)
    clustered = sweep(clustered_operand, 'clustered')
    print('\nCONTENT-DEPENDENCE SWEEP at n=%d, uniform extra indices' % SWEEP_N)
    uniform = sweep(random_operand, 'uniform')

    print('\nWINDOW ABLATION at n=%d' % ABLATION_N)
    ablation = []
    for w in WINDOWS:
        rec = aggregate(ABLATION_N, 'global', window=w, label=f'window={w}')
        ablation.append({'window': w, 'seq_len': ABLATION_N,
                         'compose_flat_ms': rec['roaring_flat_ms'],
                         'compose_flat_spread': rec['roaring_flat_spread'],
                         'compose_dense_ms': rec['dense_ms'],
                         'compose_dense_spread': rec['dense_spread'],
                         'speedup': rec['dense_ms'] / rec['roaring_flat_ms']})

    def flip(s):
        return {'k0_fastest': s[0]['best_rival'] if not s[0]['roaring_wins']
                else 'roaring_flat_ms',
                'k0_roaring_deficit': 1 / s[0]['margin_over_best_rival'],
                'k1_roaring_wins': s[1]['roaring_wins'],
                'k1_margin': s[1]['margin_over_best_rival'],
                'k1_margin_spread': s[1]['margin_spread'],
                'roaring_wins_all_k_ge_1': all(r['roaring_wins'] for r in s[1:]),
                'runs_per_row': [r['runs_per_row'] for r in s]}

    payload = {
        'meta': {'trials': TRIALS, 'warmup': WARMUP, 'repetitions': REPETITIONS,
                 'seed': 0, 'seq_len': SWEEP_N,
                 'numpy': np.__version__, 'scipy': scipy.__version__,
                 'pyroaring': pyroaring.__version__,
                 'croaring': getattr(pyroaring, '__croaring_version__', None),
                 'zstandard': zstandard.__version__,
                 'python': platform.python_version(),
                 'platform': platform.platform(),
                 'machine': machine_info(),
                 'wall_seconds': round(time.time() - t0, 1)},
        'sweep_clustered': clustered,
        'sweep_uniform_same_process': uniform,
        'window_ablation': ablation,
        'derived': {'clustered': flip(clustered), 'uniform': flip(uniform)},
    }
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    for name in ('clustered', 'uniform'):
        d = payload['derived'][name]
        print(f"\n{name}: k=0 Roaring {d['k0_roaring_deficit']:.1f}x slower than "
              f"{d['k0_fastest']}; k=1 Roaring wins={d['k1_roaring_wins']} by "
              f"{d['k1_margin']:.1f}x; wins at every k>=1: "
              f"{d['roaring_wins_all_k_ge_1']}")
    print(f"\nSaved -> {OUT}  ({payload['meta']['wall_seconds']}s)")


if __name__ == '__main__':
    main()
