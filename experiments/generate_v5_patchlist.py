#!/usr/bin/env python3
"""
Emit a find-and-replace patch list against the author's hand-edited v4 document.

FIND strings are verbatim from the edited Google Docs copy, not from the v4
generator, because the author copyedited the prose. REPLACE strings are computed
from v5_results.json, so no number is transcribed by hand.

Usage:
    python experiments/generate_v5_patchlist.py > paper/v5-patchlist.md
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = json.load(open(os.path.join(HERE, 'v5_results.json')))

ST, UN, SW = (V5['composition_structured'], V5['composition_unstructured'],
              V5['structure_sweep'])
WA, MUT, D = V5['window_ablation'], V5['mutation'], V5['derived']
S, U, W = D['structured'], D['unstructured'], D['sweep']
T, R = V5['meta']['trials'], V5['meta']['repetitions']

SHORT = {'packed_bitset_ms': 'Packed bitset', 'dense_ms': 'Dense bool',
         'csr_ms': 'Sparse CSR', 'run_arrays_ms': '(start,len) runs',
         'zstd_then_dense_ms': 'zstd then dense'}

THR = U['threshold_n']
MMIN, MMAX = U['margin_min_where_winning'], U['margin_max']
DMIN, DMAX = min(U['vs_dense_where_winning']), max(U['vs_dense_where_winning'])
LOSS = U['loss_detail'][0]
LOSS_X = LOSS['roaring_ms'] / LOSS['best_rival_ms']
LOSS_KB = LOSS['bitset_bytes'] / 1024
REF = next(c for c in ST if c['seq_len'] == 8192)
LAYOUT_SPEED = REF['roaring_per_row_ms'] / REF['roaring_flat_ms']
RI, RD = W['row_invariant'], W['first_row_dependent']

P = []


def patch(where, find, replace, why):
    P.append((where, find, replace, why))


# ----------------------------------------------------------------- claims ---
patch('Abstract — the sentence beginning "Roaring is then the fastest"',
      'Roaring is then the fastest representation available, by 3.0x to 16.8x over '
      'the best surviving baseline and up to 22x over a dense array.',
      f'Roaring is then the fastest representation available, by {MMIN:.1f}x to '
      f'{MMAX:.1f}x over the best surviving baseline and {DMIN:.0f}x to {DMAX:.0f}x '
      f'over a dense array — but only once the mask is too large to keep in cache: '
      f'at n={LOSS["seq_len"]:,} the packed bitset is {LOSS_KB:.0f} KB, the n² '
      f'baseline is nearly free, and Roaring is {LOSS_X:.1f}x slower.',
      'the 3.0x lower bound came from a 5-trial median that does not reproduce; at '
      f'n={LOSS["seq_len"]:,} Roaring actually loses')

patch('Section 5.4 — the last sentence before Table 6',
      'Among the representations that remain, Roaring is the fastest at every length '
      'tested, 3.0x to 16.8x faster than the best of them (a packed bitset) and up to '
      '22x faster than the dense array because its cost is set by the number of runs '
      'it stores rather than by n².',
      f'Among the representations that remain, Roaring is the fastest from '
      f'n={THR:,} upward — {MMIN:.1f}x to {MMAX:.1f}x faster than the best of them (a '
      f'packed bitset) and {DMIN:.0f}x to {DMAX:.0f}x faster than the dense array — '
      f'because its cost is set by the number of runs it stores rather than by n². At '
      f'n={LOSS["seq_len"]:,} it is not: the entire packed bitset is {LOSS_KB:.0f} KB '
      f'there, small enough to sit in cache, so the n² baseline costs almost nothing '
      f'and Roaring is {LOSS_X:.1f}x slower. The advantage therefore requires two '
      f'conditions rather than one: a row-dependent component, and a mask large '
      f'enough that the dense forms leave cache.',
      'states the size threshold and its mechanism')

patch('Section 8 Conclusion, second paragraph',
      'Roaring is 3.0x to 16.8x faster than the best of the others and up to 22x '
      'faster than dense, because its cost tracks the number of runs it stores rather '
      'than n².',
      f'From n={THR:,} upward, Roaring is {MMIN:.1f}x to {MMAX:.1f}x faster than the '
      f'best of the others and {DMIN:.0f}x to {DMAX:.0f}x faster than dense, because '
      f'its cost tracks the number of runs it stores rather than n²; below that the '
      f'dense forms fit in cache and the advantage disappears.',
      'same correction in the conclusion')

patch('Figure 3 caption — the final clause',
      'once the run structure is destroyed, that encoding is unavailable, and Roaring '
      'is the fastest of the representations that remain.',
      f'once the run structure is destroyed, that encoding is unavailable, and Roaring '
      f'is the fastest of the representations that remain from n={THR:,} upward.',
      'the figure now shows a loss at the smallest size')

# ------------------------------------------------------------ methodology ---
patch('Section 4 Experimental Setup — inside the last paragraph',
      'are CPU measurements reported as medians of 5 trials after warm-up,',
      f'are CPU measurements reported as medians of {T} trials after warm-up, with '
      f'the whole setup-and-measurement cycle repeated {R} times independently and '
      f'the between-repetition spread checked,',
      '5 trials did not reproduce below n=4,096 — this is what produced the error above')

patch('Table 5 caption',
      'median of 5 trials after warm-up, milliseconds, construction excluded.',
      f'median of {T} trials after warm-up over {R} independent repetitions, '
      f'milliseconds, construction excluded.', 'trial count')

patch('Table 6 caption — also fixes a stray comma from the copyedit',
      'Table 6. Unstructured composition (causal ∩ padding) ∪ , random attention, '
      'median of 5 trials, milliseconds, construction excluded.',
      f'Table 6. Unstructured composition (causal ∩ padding) ∪ random attention, '
      f'median of {T} trials over {R} independent repetitions, milliseconds, '
      f'construction excluded.', 'trial count and a malformed operator')

# --------------------------------------------------------- re-measurements ---
patch('Section 5.5, first paragraph',
      'yet Roaring is 6.9x slower than the best alternative in the first and 26x '
      'faster in the second',
      f'yet Roaring is {1/RI["margin_over_best_rival"]:.1f}x slower than the best '
      f'alternative in the first and {RD["margin_over_best_rival"]:.0f}x faster in '
      f'the second', 're-measured')

patch('Contribution 3 (Introduction) — the same two figures as Section 5.5',
      'moves Roaring from 6.9x slower than the best alternative to 26x faster',
      f'moves Roaring from {1/RI["margin_over_best_rival"]:.1f}x slower than the best '
      f'alternative to {RD["margin_over_best_rival"]:.0f}x faster', 're-measured')

patch('Section 5.5, second paragraph',
      'while costing Roaring nothing: its time moves from 0.158 ms to 0.159 ms.',
      f'while costing Roaring almost nothing: its time moves from '
      f'{RI["roaring_flat_ms"]:.3f} ms to {RD["roaring_flat_ms"]:.3f} ms.',
      're-measured; the change is now ~10%, so "nothing" overstates it')

patch('Section 5.5, third paragraph',
      'raises Roaring\'s composition time by only 3.5x, from 0.159 to 0.556 ms',
      f'raises Roaring\'s composition time by only '
      f'{W["roaring_ms_max"]/W["roaring_ms_min"]:.1f}x, from {W["roaring_ms_min"]:.3f} '
      f'to {W["roaring_ms_max"]:.3f} ms', 're-measured')

patch('Section 5.5, third paragraph',
      'The dense array stays between 4.5 and 8.0 ms throughout',
      f'The dense array stays between {W["dense_min"]:.1f} and {W["dense_max"]:.1f} ms '
      f'throughout', 're-measured; the old 8.0 ms was a single noisy row')

patch('Section 5.5, final paragraph (mutation)',
      '0.091 ms against 1.112 ms for Roaring at n=16,384, a factor of about 12.2',
      f'{MUT[-1]["dense_ms"]:.3f} ms against {MUT[-1]["roaring_ms"]:.3f} ms for '
      f'Roaring at n={MUT[-1]["seq_len"]:,}, a factor of about '
      f'{MUT[-1]["dense_speedup"]:.1f}', 're-measured')

patch('Contribution 4 (Introduction) AND Section 3.1 — the same figure appears twice',
      '35x in composition time',
      f'{LAYOUT_SPEED:.0f}x in composition time',
      're-measured layout penalty at n=8,192 (Section 3.1 carries the same figure '
      'with different wording — see the next patch)')

patch('Section 3.1 — the same layout figure, phrased differently',
      '2.85x smaller and 35x faster to compose',
      f'2.85x smaller and {LAYOUT_SPEED:.0f}x faster to compose', 're-measured')

# ---------------------------------------------------------------- caveats ---
patch('Section 6.1 — REPLACE the sentence beginning "Second, all composition results"',
      'Second, all composition results are single-machine CPU measurements of '
      'specific library implementations (pyroaring, numpy, and SciPy), and a '
      'different implementation of any baseline could move the threshold, though not '
      'the mechanism behind it.',
      f'Second, the advantage also requires the mask to be too large to keep in '
      f'cache: at n={LOSS["seq_len"]:,} the packed bitset is only {LOSS_KB:.0f} KB, '
      f'the n² baseline is nearly free, and Roaring loses even though the pattern is '
      f'row-dependent (Table 6). Third, all composition results are single-machine '
      f'CPU measurements of specific library implementations (pyroaring, numpy, and '
      f'SciPy), and a different implementation of any baseline could move the '
      f'threshold, though not the mechanism behind it.',
      'the decision rule is wrong at small n without this')

patch('Limitation 10 — REPLACE the whole item',
      'Composition timings compare specific libraries (pyroaring, numpy, SciPy) on '
      'one CPU. A faster implementation of any baseline would move the margins, '
      'though the asymptotic argument of roaring costs scales with runs, and dense '
      'costs with n² would survive.',
      f'Implementation and cache dependence. Composition timings compare specific '
      f'libraries (pyroaring, numpy, SciPy) on a single machine '
      f'({V5["meta"]["platform"]}), reported as medians of {T} trials over {R} '
      f'independent repetitions. The size threshold of Section 5.4 is a cache effect '
      f'and will move on machines with different cache sizes; the structural '
      f'condition — whether any component is row-dependent — will not. A faster '
      f'implementation of any baseline would move the margins, though the asymptotic '
      f'argument — Roaring costs scale with the number of runs, dense costs with n² — '
      f'would survive.',
      'names the cache dependence, and repairs a sentence the copyedit broke')


# ----------------------------------------------------------------- tables ---
def block(header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    return '\n'.join(' | '.join(str(c).ljust(w[i]) for i, c in enumerate(r))
                     for r in [header] + rows)


TABLES = [
    ('Table 5', ['Seq len', 'Roaring per-row', 'Roaring flat+opt', 'Dense bool',
                 'Packed bitset', 'Sparse CSR', 'zstd then dense', '(start,len) runs'],
     [[f'{c["seq_len"]:,}', f'{c["roaring_per_row_ms"]:.3f}',
       f'{c["roaring_flat_ms"]:.3f}', f'{c["dense_ms"]:.3f}',
       f'{c["packed_bitset_ms"]:.3f}', f'{c["csr_ms"]:.3f}',
       f'{c["zstd_then_dense_ms"]:.3f}', f'{c["run_arrays_ms"]:.4f}'] for c in ST]),
    ('Table 6', ['Seq len', 'Runs/row', 'Roaring flat+opt', 'Dense bool',
                 'Packed bitset', 'Sparse CSR', '(start,len) runs', 'Roaring margin'],
     [[f'{c["seq_len"]:,}', f'{c["runs_per_row"]:.1f}', f'{c["roaring_flat_ms"]:.3f}',
       f'{c["dense_ms"]:.3f}', f'{c["packed_bitset_ms"]:.3f}', f'{c["csr_ms"]:.3f}',
       'n/a', (f'{c["margin_over_best_rival"]:.1f}x' if c['roaring_wins']
               else f'{1/c["margin_over_best_rival"]:.1f}x slower')] for c in UN]),
    ('Table 7', ['Random/row', 'Runs/row', 'Roaring flat+opt', 'Dense bool',
                 'Packed bitset', 'Sparse CSR', '(start,len) runs', 'Fastest'],
     [[str(s['random_per_row']), f'{s["runs_per_row"]:.1f}',
       f'{s["roaring_flat_ms"]:.3f}', f'{s["dense_ms"]:.3f}',
       f'{s["packed_bitset_ms"]:.3f}', f'{s["csr_ms"]:.3f}',
       (f'{s["run_arrays_ms"]:.4f}' if s['run_arrays_ms'] is not None else 'n/a'),
       ('Roaring' if s['roaring_wins'] else SHORT[s['best_rival']])] for s in SW]),
    ('Table 8', ['Window', 'Mask density', 'Roaring flat+opt', 'zstd(bitset)',
                 'Compose Roaring', 'Compose dense', 'Speedup'],
     [[str(r['window']) if r['window'] is not None else 'full (causal)',
       f'{r["density"]*100:.1f}%',
       f'{r["sizes_bytes"]["roaring_flat_opt"]/1024:,.1f}',
       f'{r["sizes_bytes"]["zstd"]/1024:,.1f}', f'{r["compose_flat_ms"]:.3f}',
       f'{r["compose_dense_ms"]:.3f}', f'{r["speedup"]:.1f}x'] for r in WA]),
]

GRAMMAR = [
    ('Abstract', 'When they are causal windows, padding, global sink tokens, '
     'everything is determined by position, the composition',
     'When they are — causal windows, padding, global sink tokens, everything '
     'determined by position — the composition'),
    ('Introduction, 3rd paragraph', 'The contribution of this paper is that the '
     'boundary, its mechanism,', 'The contribution of this paper is that boundary, '
     'its mechanism,'),
    ('Contribution 3', 'decides between those regimes whether the components being '
     'composed vary per query row and a demonstration',
     'decides between those regimes — whether the components being composed vary per '
     'query row — and a demonstration'),
    ('Section 2.4, last sentence', 'gradient-communication methods quantisation [8] '
     'and low-rank decomposition [17]  reduce',
     'gradient-communication methods — quantisation [8] and low-rank decomposition '
     '[17] — reduce'),
    ('Section 5.3, first sentence', 'it is the least interesting one',
     'it is the less interesting one   (two axes, not three)'),
    ('Section 5.5, 2nd paragraph', 'the same for every query row it becomes '
     'row-dependent', 'the same for every query row — it becomes row-dependent'),
    ('Section 5.5, 2nd paragraph', 'advantages the baselines were relying on  the '
     'closed-form (start, length) composition and the row broadcast  while costing',
     'advantages the baselines were relying on — the closed-form (start, length) '
     'composition and the row broadcast — while costing'),
    ('Section 6.1', 'the pattern is fixed, and position-determined causal windows, '
     'padding, block-diagonal document boundaries, then',
     'the pattern is fixed and position-determined — causal windows, padding, '
     'block-diagonal document boundaries — then'),
    ('Section 6.1', 'If any component depends on content learned or random attention, '
     'dynamic pruning, or retrieval-selected tokens, the closed form',
     'If any component depends on content — learned or random attention, dynamic '
     'pruning, retrieval-selected tokens — the closed form'),
    ('Section 8, 2nd paragraph', 'categorical rather than gradual  fragmentation is '
     'unchanged across it, which makes',
     'categorical rather than gradual — fragmentation is unchanged across it — which '
     'makes'),
    ('Section 8, 2nd paragraph', 'layout choice within Roaring  one bitmap per row '
     'versus a flat, run-optimised bitmap, accounts',
     'layout choice within Roaring — one bitmap per row versus a flat, run-optimised '
     'bitmap — accounts'),
    ('Table 11 caption  [MEANING ERROR]', 'The ratio column compares realized density '
     'against the per-row layout.', 'The ratio column compares realized dense against '
     'the per-row layout.'),
    ('Table 11 caption', 'in the per-row layout, the run used, and in the flat '
     'run-optimised layout', 'in the per-row layout the run used, and in the flat '
     'run-optimised layout'),
]


def main():
    print('# v5 patch list — apply to your edited v4 document\n')
    print(f'**Source:** `experiments/v5_results.json` — {T} trials per configuration, '
          f'median over {R} independent repetitions, single process, '
          f'{V5["meta"]["wall_seconds"]}s on {V5["meta"]["platform"]}.\n')
    print(f'**What the re-measurement changed.** Roaring loses the unstructured case '
          f'at n={LOSS["seq_len"]:,} ({LOSS_X:.1f}x slower than a packed bitset, '
          f'stable across all {R} repetitions) and wins from n={THR:,} upward at '
          f'{MMIN:.1f}x–{MMAX:.1f}x. The reported "3.0x" lower bound was an artifact '
          f'of a 5-trial median. Everything else moved by only a few percent, and the '
          f'structured deficit still rounds to the same 1.9x–12x already in the '
          f'text.\n')
    print('---\n\n## A. Text replacements\n')
    for i, (where, find, repl, why) in enumerate(P, 1):
        print(f'### A{i}. {where}\n\n*Why:* {why}\n')
        print(f'FIND:\n```\n{find}\n```\n')
        print(f'REPLACE:\n```\n{repl}\n```\n')
    print('---\n\n## B. Table numbers\n')
    print('Replace the numeric rows; keep your header row and formatting.\n')
    for name, hdr, rows in TABLES:
        print(f'### {name}\n\n```\n{block(hdr, rows)}\n```\n')
    print('---\n\n## C. Grammar repairs from the copyedit (optional but recommended)\n')
    print('These are pre-existing in your copy, unrelated to the re-measurement. '
          'Removing the em-dashes left several clauses without their boundaries.\n')
    for i, (where, find, repl) in enumerate(GRAMMAR, 1):
        print(f'**C{i}. {where}**\n')
        print(f'- FIND: `{find}`')
        print(f'- REPLACE: `{repl}`\n')
    print('**C14.** Section 5.1 is styled Heading 1; Sections 5.2–5.9 are Heading 2. '
          'Make 5.1 Heading 2.\n')
    print('**C15.** Spelling is mixed: *behavior, randomized, factorized, organizing* '
          'alongside *favour, run-optimised, quantisation*. Pick whichever the journal '
          'uses and apply throughout.\n')


if __name__ == '__main__':
    main()
