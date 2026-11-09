#!/usr/bin/env python3
"""
Equivalence test for the Table 9 convergence claim, from the existing 5-seed
results in revision_main.json (no new training).

Failing to reject equality (Welch, Mann-Whitney) is not evidence of equivalence,
so this adds:
  * the Welch 95% confidence interval on the difference in final validation loss
    (Roaring minus vanilla), and
  * a two one-sided tests (TOST) procedure against a stated margin, using the
    Welch standard error and Welch-Satterthwaite degrees of freedom. TOST at
    alpha=0.05 is equivalent to the 90% CI lying inside (-margin, +margin).

MARGIN is 0.05 nats of validation cross-entropy, i.e. a factor exp(0.05)=1.051
(about 5%) in perplexity. It was fixed by the authors when the test was added,
after the seed results existed, so it is stated rather than pre-registered; the
smallest margin at which TOST would pass is reported alongside it.

Usage:
    python experiments/run_v6_stats.py
"""

import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'revision_main.json')
OUT = os.path.join(HERE, 'v6_stats.json')
MARGIN = 0.05
ALPHA = 0.05


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return se, df


def main():
    d = json.load(open(SRC))
    van = np.array([r['final_val_loss'] for r in d['runs']['vanilla']])
    roa = np.array([r['final_val_loss'] for r in d['runs']['roaring']])
    diff = roa.mean() - van.mean()
    se, df = welch(roa, van)

    t95 = stats.t.ppf(1 - ALPHA / 2, df)
    t90 = stats.t.ppf(1 - ALPHA, df)
    ci95 = [diff - t95 * se, diff + t95 * se]
    ci90 = [diff - t90 * se, diff + t90 * se]

    # TOST: H0a diff <= -margin, H0b diff >= +margin
    t_lower = (diff + MARGIN) / se
    t_upper = (diff - MARGIN) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    min_margin = max(abs(ci90[0]), abs(ci90[1]))

    welch_p = stats.ttest_ind(roa, van, equal_var=False).pvalue
    mw_p = stats.mannwhitneyu(roa, van, alternative='two-sided').pvalue

    out = {
        'source': 'revision_main.json',
        'seeds': [r['name'] for r in d['runs']['vanilla']] +
                 [r['name'] for r in d['runs']['roaring']],
        'vanilla_final_val_loss': van.tolist(),
        'roaring_final_val_loss': roa.tolist(),
        'vanilla_mean': van.mean(), 'vanilla_sd': van.std(ddof=1),
        'roaring_mean': roa.mean(), 'roaring_sd': roa.std(ddof=1),
        'diff_roaring_minus_vanilla': diff,
        'welch_se': se, 'welch_df': df,
        'ci95': ci95, 'ci90': ci90,
        'margin_nats': MARGIN,
        'margin_ppl_factor': math.exp(MARGIN),
        'tost_p_lower': p_lower, 'tost_p_upper': p_upper, 'tost_p': p_tost,
        'tost_equivalent': bool(p_tost < ALPHA),
        'smallest_margin_passing': min_margin,
        'welch_p_check': welch_p, 'mannwhitney_p_check': mw_p,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"diff (Roaring - vanilla) = {diff:+.4f} nats, Welch df = {df:.2f}")
    print(f"95% CI [{ci95[0]:+.4f}, {ci95[1]:+.4f}]   90% CI [{ci90[0]:+.4f}, {ci90[1]:+.4f}]")
    print(f"TOST margin +/-{MARGIN} nats: p = {p_tost:.4f} -> "
          f"{'equivalent' if out['tost_equivalent'] else 'NOT shown equivalent'}")
    print(f"smallest margin at which TOST passes: {min_margin:.4f} nats")
    print(f"check: Welch p = {welch_p:.3f}, Mann-Whitney p = {mw_p:.3f}")
    print(f"Saved -> {OUT}")


if __name__ == '__main__':
    main()
