#!/usr/bin/env python3
"""
Build the round-3 revised manuscript for ACRT-2026-0062.

Emits into paper/v3-review/:
    Roaring-Transformer-v3-Highlighted.docx           (revisions marked in yellow)
    Roaring-Transformer-v3-Clean.docx                 (same text, no marks)
    Roaring-Transformer-v3-Template-Highlighted.docx  (journal template)
    Roaring-Transformer-v3-Template-Clean.docx

Every number in the text, tables and figures is read from a committed result
file. There are no hand-transcribed ratios: the round-2 draft hard-coded the
layout penalty (2.3x / 33x), the BigBird sizes, the gradient ratio ranges and
the step-time p-value, all of which are now derived here or in
run_v3_measurements.py.

Sources:
    experiments/v3_results.json        all CPU measurements, one process
    experiments/revision_main.json     WikiText-103 GPU run, 5 seeds
    experiments/revision_pruned.json   pruning ablation
"""

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_docx_common import Builder, new_document, template_document  # noqa: E402

TEMPLATE = os.path.join(HERE, '..', 'paper',
                        'Roaring_Transformer-Final_REVISED_clean.docx')
OUT_DIR = os.path.join(HERE, '..', 'paper', 'v3-review')
FIG = os.path.join(HERE, 'figures_v3')


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


V3 = load('v3_results.json')
MAIN = load('revision_main.json')
PRUNED = load('revision_pruned.json')

MS = {r['seq_len']: r for r in V3['mask_scaling']}
PS = V3['pattern_sweep']
COMP = V3['composition']
MUT = V3['mutation']
WA = V3['window_ablation']
GI = V3['gradient_index']
GL = V3['gradient_layout']
D = V3['derived']
REPEATS = V3['meta']['repeats']

CFG = MAIN['meta']['model_cfg']
SEQ = CFG['seq_len']
NP_ = MAIN['meta']['n_params']
REF = D['layout_penalty']['reference_seq_len']          # n=8,192
MAXN = COMP[-1]['seq_len']


def arm(a, key):
    xs = [r[key] for r in MAIN['runs'][a]]
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


V_LOSS, V_SD = arm('vanilla', 'final_val_loss')
R_LOSS, R_SD = arm('roaring', 'final_val_loss')
V_MS, V_MSSD = arm('vanilla', 'avg_step_ms')
R_MS, R_MSSD = arm('roaring', 'avg_step_ms')
V_VRAM, _ = arm('vanilla', 'peak_vram_mb')
R_VRAM, _ = arm('roaring', 'peak_vram_mb')
EQ = MAIN['equivalence']
PE = PRUNED['equivalence']
HYD = {h['seq_len']: h for h in MAIN['hydration']}

# Step-time significance, computed rather than transcribed.
STEP_P = float(stats.ttest_ind([r['avg_step_ms'] for r in MAIN['runs']['vanilla']],
                               [r['avg_step_ms'] for r in MAIN['runs']['roaring']],
                               equal_var=False).pvalue)

# The mask footprint the training run actually carried (per-row layout) and the
# footprint the layout adopted in Section 3.1 gives for the same mask.
RUN_KB = HYD[SEQ]['roaring_cpu_kb']
DENSE_KB = MS[SEQ]['sizes_bytes']['dense_f32'] / 1024
FLAT_KB = MS[SEQ]['sizes_bytes']['roaring_flat_opt'] / 1024

# Number of distinct encodings compared in Table 3.
N_ENCODINGS = len([k for k, v in MS[REF]['sizes_bytes'].items() if v is not None])

# Ratio ranges quoted in Section 5.2 are computed over the rows Table 4 actually
# shows, so a reader recomputing them from the table gets the same endpoints.
GI_SHOWN = [r for r in GI if r['n_params'] in (1_000_000, NP_)]
GI_ROARING_MIN = min(r['roaring_vs_raw'] for r in GI_SHOWN)
GI_ROARING_MAX = max(r['roaring_vs_raw'] for r in GI_SHOWN)
GI_DELTA_MIN = min(r['delta_vs_raw'] for r in GI_SHOWN)
GI_DELTA_MAX = max(r['delta_vs_raw'] for r in GI_SHOWN)
# The abstract's "smaller at every parameter count and sparsity" is verified, not asserted.
DELTA_ALWAYS_SMALLER = all(
    r['sizes_bytes']['delta_varint'] < r['sizes_bytes']['roaring'] for r in GI)
assert DELTA_ALWAYS_SMALLER, 'abstract claims delta+varint always beats Roaring'


def kb(x):
    return f'{x/1024:,.1f}'


def sz(n, key):
    return MS[n]['sizes_bytes'][key]


def pattern_rec(name, n):
    return next(p for p in PS if p['pattern'] == name and p['seq_len'] == n)


def best_compressor(sizes):
    cand = {'zlib': sizes['zlib'], 'zstd': sizes['zstd'], 'lzma': sizes['lzma']}
    name = min(cand, key=cand.get)
    return name, cand[name]


def per_row_crossover():
    """Largest n at which the per-row layout is still slower than dense."""
    slower = [c['seq_len'] for c in COMP if c['roaring_per_row_ms'] > c['dense_ms']]
    return max(slower) if slower else None


def build(highlight, use_template=False):
    if use_template:
        doc, styles = template_document(TEMPLATE)
        b = Builder(doc, highlight=highlight, styles=styles)
    else:
        doc = new_document()
        b = Builder(doc, highlight=highlight)

    lp = D['layout_penalty']
    gv = D['gradient_verdict']
    mv = D['mask_size_verdict']

    # ---------------------------------------------------------------- front
    if use_template:
        b.label('Article Title')
        b.title('The Roaring Transformer: [[Compressed Bitmaps as a Composable '
                'Attention-Mask Representation for Long-Context Transformer '
                'Training]]')
        b.label('Authors and Affiliations')
        b.author('Lokesh Karanam1, Purna Sreekar Karanam1, Sri Ram Pemmasani1')
        b.author('1 Independent Researchers, Austin, TX, USA')
        b.author('*Address all correspondence to:')
        b.author('Lokesh Karanam')
        b.author('lokesh.karanam@rrd.com')
        b.label('Abstract')
    else:
        b.title('The Roaring Transformer: [[Compressed Bitmaps as a Composable '
                'Attention-Mask Representation for Long-Context Transformer '
                'Training]]')
        b.author('Lokesh Karanam, Purna Sreekar Karanam, Sri Ram Pemmasani')
        b.affiliation('Independent Researchers, Austin, TX, USA\n'
                      'Correspondence: lokesh.karanam@rrd.com')
        b.section('Abstract')
    b.body(
        'We study whether Roaring Bitmaps, a hybrid compressed set structure, are a '
        'useful representation for the sparse metadata that surrounds the training of '
        '[[Transformer-based language models]]: attention masks, Top-K gradient '
        'indices, and pruning masks. '
        '[[We evaluate Roaring against special-purpose encodings of the same '
        'structure: per-row (start, length) tuples, packed bitsets, '
        'delta-plus-varint, and general-purpose byte compressors. On size the '
        'result is negative: a generic byte compressor (zstd) stores a causal '
        f'sliding-window mask at n={REF:,} in {kb(sz(REF, "zstd"))} KB against '
        f'{kb(sz(REF, "roaring_flat_opt"))} KB for Roaring, and a '
        'delta-plus-varint encoder is smaller than Roaring on the diffuse Top-K '
        'gradient index sets we test, at every parameter count and sparsity. '
        'Roaring is smallest only when the index set carries contiguous-run '
        'structure, a regime we report but do not observe in training.]] '
        '[[The property that distinguishes Roaring is not footprint but operability: '
        'it is the only representation among those tested that is simultaneously '
        'compact and directly operable. Composing an attention mask from parts '
        '(a causal window intersected with padding, unioned with global sink tokens) '
        f'is {COMP[-1]["flat_speedup_vs_dense"]:.0f}x faster than the equivalent dense '
        f'operation at n={MAXN:,}, while byte-compressed '
        'representations cannot be operated on at all without full decompression and '
        'are the slowest option end-to-end. Layout within Roaring also matters: at '
        f'n={REF:,} a flat, run-optimised bitmap is '
        f'{lp["size_ratio_at_ref"]:.2f}x smaller and {lp["speed_ratio_at_ref"]:.0f}x '
        'faster to compose than one bitmap per query row.]] '
        f'On WikiText-103, a {NP_/1e6:.1f}M-parameter model trained '
        f'over {MAIN["meta"]["steps"]:,} steps and {MAIN["meta"]["seeds"]} seeds shows the '
        'lossless Roaring path leaves validation loss statistically unchanged '
        f'(Welch p={EQ["welch_p"]:.2f}; Mann-Whitney p={EQ["mannwhitney_p"]:.2f}). '
        'Masks are hydrated to dense tensors before the accelerator, so GPU VRAM is '
        f'unchanged ({V_VRAM:,.0f} vs {R_VRAM:,.0f} MB); the benefits are to host memory '
        'and to mask manipulation, not to accelerator memory. '
        f'[[Validation is at the {NP_/1e6:.1f}M-parameter scale and at a sequence '
        f'length of {SEQ}; behaviour at billion-parameter scale is projected, not '
        'demonstrated.]]',
        indent=False)

    kw = ('Roaring Bitmaps, Transformer, attention mask representation, '
          '[[set algebra]], gradient sparsification, [[compression baselines]]')
    if use_template:
        b.label('Keywords')
        b.body(kw, indent=False)
    else:
        b.body('Keywords: ' + kw, indent=False)

    # ---------------------------------------------------------------- intro
    b.section('1. Introduction')
    b.body(
        'As sequence lengths grow, the attention mask of the Transformer [6] — the '
        'matrix encoding which token-to-token relationships are permitted — scales '
        'as O(n²). At n = 100,000 '
        'a single float32 mask occupies roughly 40 GB. This figure motivates '
        'compressing the mask representation and the associated host-memory footprint. '
        'Relieving on-GPU VRAM at such lengths additionally requires native on-device '
        'execution of the compressed mask, which we do not implement here; the present '
        'contribution targets representation, host memory, and mask manipulation '
        'rather than a direct reduction of accelerator VRAM. A parallel concern '
        'arises in distributed training, where All-Reduce traffic is commonly '
        'reduced by transmitting only the largest gradient entries [1], leaving '
        'their indices to be encoded as metadata.')
    b.body(
        '[[Compression ratios for such masks are often quoted against a dense float32 '
        'baseline, but that is not a baseline anyone would use for a structurally '
        'regular mask: a causal sliding-window pattern is a set of contiguous '
        'integer ranges, and almost any purpose-built encoding will compress it '
        'well. The meaningful question is not whether some compression helps — it '
        'obviously does — but whether a general-purpose set structure earns its '
        'overhead against an encoding that exploits the known structure. Section '
        '5.1 answers that question directly, and the answer on size is no.]]')
    b.body(
        '[[Size alone, however, is the wrong figure of merit. A representation used '
        'inside a '
        'training loop is not merely stored; it is intersected with padding masks, '
        'unioned with global-token patterns, queried, and mutated. On that axis the '
        'ranking inverts: byte-stream compressors achieve the smallest footprint but '
        'must decompress in full before any operation, and dense arrays support '
        'operations but at O(n²) cost and footprint. Roaring occupies the only point '
        'on that trade-off that is both compact and directly operable, and the gap '
        'widens with sequence length.]]')

    b.body('Our contributions are:', indent=False)
    b.numbered(1, '[[A baseline study of attention-mask and gradient-index encodings '
                  'comparing Roaring against (start, length) tuples, packed bitsets, '
                  'delta-plus-varint, zlib, zstd and lzma. Roaring does not win on '
                  'size, and we report that negative result in full.]]')
    b.numbered(2, '[[A composition benchmark showing Roaring set algebra is '
                  f'{COMP[0]["flat_speedup_vs_dense"]:.1f}x to '
                  f'{COMP[-1]["flat_speedup_vs_dense"]:.0f}x faster than the dense '
                  f'equivalent for n={COMP[0]["seq_len"]:,} to '
                  f'{MAXN:,}, with construction excluded, and that '
                  'byte-compressed representations are the slowest option once the '
                  'cost of decompression is counted.]]')
    b.numbered(3, '[[A layout finding: storing one bitmap per query row and omitting '
                  f'run_optimize() costs {lp["size_ratio_at_ref"]:.2f}x in size and '
                  f'{lp["speed_ratio_at_ref"]:.0f}x in composition time at n={REF:,} '
                  f'({lp["size_ratio_min"]:.2f}–{lp["size_ratio_max"]:.2f}x and '
                  f'{lp["speed_ratio_min"]:.0f}–{lp["speed_ratio_max"]:.0f}x across '
                  'the range tested) relative to a flat, run-optimised bitmap over '
                  'the same pattern.]]')
    b.numbered(4, 'Convergence equivalence on WikiText-103 across five seeds with '
                  'significance testing, and an explicit measurement showing GPU VRAM '
                  'is unchanged.')
    b.numbered(5, '[[A reference implementation, a benchmark suite, and the raw result '
                  'files from which every table and figure in this paper is generated, '
                  'available as stated in Section 13.]]')
    b.body('[[To the best of our knowledge this is the first published integration of '
           'Roaring Bitmaps into a Transformer training pipeline. A targeted '
           'literature search returned no prior published work, though unpublished '
           'use in production systems cannot be excluded.]]')

    # ---------------------------------------------------------------- background
    b.section('2. Background and Related Work')
    b.subsection('2.1 The Attention Mask Problem')
    b.body('In scaled dot-product attention [6] the additive mask M is lower-triangular '
           'for causal language modeling, with zeros at allowed positions and negative '
           'infinity at blocked positions. For sequence length n, M requires n² '
           'elements.')
    b.table(
        'Table 1. Dense attention mask memory. Figures assume a float32 additive mask '
        '(4 bytes/element), the worst case against which compression ratios are '
        'reported. A boolean mask requires 1 byte/element (or 1 bit if packed), and '
        'some implementations compute the mask on the fly; the corresponding dense '
        'footprints are 4x (or up to 32x) smaller than the values shown.',
        ['Sequence length', 'Mask memory', 'Per-layer VRAM'],
        [['1,024', '4 MB', '4 MB'], ['4,096', '64 MB', '64 MB'],
         ['8,192', '256 MB', '256 MB'], ['32,768', '4 GB', '4 GB'],
         ['100,000', '40 GB', '40 GB']])

    b.subsection('2.2 Sparse Attention Methods')
    b.body('Longformer [2] combines a sliding window with global attention, reducing '
           'complexity to O(n·w). BigBird [3] adds random connections, and Sparse '
           'Transformer [4] uses factorised fixed patterns. FlashAttention [7] fuses '
           'attention into a tiled kernel, with FlashAttention-2 [10] and -3 [11] '
           'extending the approach. Mamba [12] and StreamingLLM [13] sidestep '
           'quadratic attention via selective state-space models and attention-sink '
           'mechanisms respectively, while Ring Attention [14] distributes the '
           'computation across devices. Native Sparse Attention [15] proposes '
           'hardware-aligned, natively trainable sparse attention. '
           '[[These methods represent their masks as dense tensors or as implicit '
           'index arithmetic. Our question is orthogonal to which sparsity pattern is '
           'chosen: given any such pattern, what is the right in-memory '
           'representation for composing and manipulating it? That question does not '
           'appear to have been studied directly, and Section 5 shows the answer is '
           'not obvious.]]')

    b.subsection('2.3 Roaring Bitmaps')
    b.body('Roaring Bitmaps [5] partition the 32-bit key space into 2^16-element chunks, '
           'each stored as an Array Container (sparse chunks, sorted uint16), a Bitmap '
           'Container (dense chunks, 8 KB), or a Run Container (contiguous runs, as '
           '(start, length) pairs). '
           '[[Roaring is reportedly used in systems such as Apache Lucene, Druid, '
           'InfluxDB and Elasticsearch. It supports SIMD-accelerated set operations; '
           'a throughput figure exceeding 10 billion integers per second is reported '
           'in the original Roaring literature and is not a measurement from our '
           'setup.]] '
           '[[Container selection is automatic, but run-length encoding is not applied '
           'unless run_optimize() is called explicitly, a detail that materially '
           'affects both size and operation cost (Sections 5.1 and 5.3). Run-length '
           'encoded containers were introduced as a later enhancement [16] and are '
           'central to the layout we adopt.]]')

    b.subsection('[[2.4 Baseline Encodings]]')
    b.body('[[We compare against four families. (i) Structure-specific encodings: for '
           'a causal sliding-window mask, each row is one contiguous range, so a '
           '(start, length) pair per row suffices, costing 8n bytes. (ii) Packed '
           'bitsets: one bit per element, n²/8 bytes, with no container logic. '
           '(iii) General-purpose byte compressors applied to the packed bitset: zlib, '
           'zstd [20] and lzma, all three reported in Table 3. (iv) For gradient '
           'indices, delta encoding followed by '
           'LEB128 varint, the standard approach for sorted integer sequences. Each is '
           'a plausible choice a practitioner would reach for before adopting a '
           'specialised set structure. Orthogonal families of gradient-communication '
           'methods — quantisation [8] and low-rank decomposition [17] — reduce the '
           'value payload rather than the index metadata, and are complementary to '
           'the encodings compared here.]]')

    # ---------------------------------------------------------------- architecture
    b.section('3. Architecture')
    b.body('The Roaring Transformer keeps the neural architecture of a standard '
           'Transformer unchanged; the modifications are confined to the metadata '
           'management layer, preserving full differentiability.')

    b.subsection('3.1 Attention Mask Manager')
    b.body('Attention masks are integer sets: for query position i with window w, '
           'S_i = {max(0, i−w), …, i}. '
           '[[We store a single bitmap over the flattened n² key space and call '
           'run_optimize() after construction. The natural alternative — one bitmap '
           'per query row — pays a per-bitmap serialization header n times and forces '
           'set operations through an n-iteration loop rather than a single '
           'vectorised call. Sections 5.1 and 5.3 quantify the difference: at '
           f'n={REF:,} the flat, run-optimised layout is '
           f'{lp["size_ratio_at_ref"]:.2f}x smaller and {lp["speed_ratio_at_ref"]:.0f}x '
           'faster to compose. The WikiText-103 run reported in Section 5.5 predates '
           'this change and used the per-row layout; Table 7 states both footprints.]] '
           'Before each forward pass the bitmap is hydrated into a dense tensor and '
           'transferred to the accelerator; realized masks are cached per sequence '
           'length.')

    b.subsection('3.2 Gradient Index Compressor')
    b.body('In Top-K sparsification the standard payload is (indices: uint32[k], '
           'values: float32[k]). We replace the index array with a serialized Roaring '
           'Bitmap. Error feedback [9] ensures no gradient information is lost. '
           '[[Section 5.2 shows this is not competitive for the index sets magnitude '
           'Top-K actually produces: delta-plus-varint encoding is smaller at every '
           'configuration we tested. We retain the description for '
           'completeness and report the negative result rather than the more flattering '
           'ratio against raw uint32 arrays.]]')

    b.subsection('3.3 Pruning Manager')
    b.body('After warm-up, the pruning manager records surviving weight indices as '
           'Roaring Bitmaps. One-shot post-training pruning methods such as SparseGPT '
           '[18] and Wanda [19] produce highly sparse, structured masks that are '
           'particularly amenable to this encoding. At 30–50% pruning, Bitmap '
           'Containers give 1-bit-per-weight storage; at 90%+, Array Containers store '
           'only surviving indices. '
           '[[Pruning is lossy and is reported separately as an ablation, not as part '
           'of the equivalence claim.]]')

    # ---------------------------------------------------------------- setup
    b.section('4. Experimental Setup')
    b.table('Table 2. Model configuration.',
            ['Hyperparameter', 'Value'],
            [['Embedding dim', str(CFG['embed_dim'])],
             ['Attention heads', str(CFG['num_heads'])],
             ['Layers', str(CFG['num_layers'])],
             ['FFN hidden dim', str(CFG['ff_dim'])],
             ['Vocabulary size', f'{MAIN["meta"]["vocab_size"]:,}'],
             ['Sequence length', str(SEQ)],
             ['Sliding window', str(CFG['window_size'])],
             ['Total parameters', f'{NP_:,}']])
    b.body(f'Optimizer AdamW (lr=3e-4, weight decay 0.01), cross-entropy loss, '
           f'WikiText-103 tokenized with GPT-2 byte-level BPE, batch size 16, '
           f'{MAIN["meta"]["steps"]:,} steps, {MAIN["meta"]["seeds"]} seeds, on a single '
           f'NVIDIA GPU (CUDA). For the convergence comparison only the lossless '
           f'optimizations are active; weight pruning is evaluated separately. '
           '[[Encoding-size and composition benchmarks (Sections 5.1, 5.3 and 5.4) are '
           f'CPU measurements reported as medians of {REPEATS} trials after warm-up, '
           'with construction excluded from the composition timings. All of them are '
           'produced by a single benchmark process, so timings are directly comparable '
           'across Tables 3, 5 and 6.]]')

    # ---------------------------------------------------------------- results
    b.section('5. Results')
    b.subsection('[[5.1 Mask Representation Against Trivial Baselines]]')
    zstd_margins = [sz(n, 'roaring_flat_opt') / sz(n, 'zstd') for n in MS]
    compressors_always_win = all(
        max(sz(n, 'zlib'), sz(n, 'zstd'), sz(n, 'lzma')) < sz(n, 'roaring_flat_opt')
        for n in MS)
    assert compressors_always_win, 'Section 5.1 claims all three compressors win'
    b.body('[[Table 3 compares every candidate encoding across sequence length. The '
           'ordering is unambiguous. All three general-purpose byte compressors '
           'applied to a packed bitset are smaller than either Roaring layout at every '
           f'sequence length, and the strongest of them (zstd) by '
           f'{min(zstd_margins):.0f}x to {max(zstd_margins):.0f}x — one to two orders '
           'of magnitude. The (start, length) encoding, which '
           'exploits the contiguity of the causal window directly, beats the per-row '
           'Roaring layout. It does not beat the flat, run-optimised layout, which is '
           'the correct way to use Roaring for this pattern — but that establishes '
           'only that layout choice matters, not that Roaring wins on size.]]')
    rows = []
    for n in sorted(MS):
        s = MS[n]['sizes_bytes']
        rows.append([f'{n:,}', kb(s['dense_f32']), kb(s['packed_bitset']),
                     kb(s['startlen']), kb(s['zlib']), kb(s['zstd']), kb(s['lzma']),
                     kb(s['roaring_per_row']), kb(s['roaring_flat_opt'])])
    b.table('[[Table 3. Attention mask representation size in KB, causal sliding '
            f'window, window={CFG["window_size"]}, matching the training default (the '
            'previous version reported this table at window=128). Held constant: the '
            'causal sliding-window pattern, the window size, one mask per sequence '
            'length, and single precision for the dense baseline. Lower is better.]]',
            ['Seq len', 'Dense f32', 'Packed bitset', '(start,len) per row',
             'zlib(bitset)', 'zstd(bitset)', 'lzma(bitset)', 'Roaring per-row',
             'Roaring flat+opt'], rows, col_pt=7.0)
    b.figure(os.path.join(FIG, 'fig1_mask_memory.png'),
             '[[Figure 1. Mask representation size versus sequence length, log-log, '
             'showing every encoding of Table 3. The byte compressors are the '
             'strongest competitors throughout.]]')

    bb = pattern_rec('bigbird', REF)
    bb_name, bb_bytes = best_compressor(bb['sizes_bytes'])
    bs_n = max(p['seq_len'] for p in PS if p['pattern'] == 'block_sparse')
    bs = pattern_rec('block_sparse', bs_n)
    bs_name, bs_bytes = best_compressor(bs['sizes_bytes'])
    verdict = ('We found no mask regime in which Roaring is the smallest '
               f'representation across the {mv["n_configurations"]} mask '
               'configurations measured.'
               if not mv['roaring_ever_smallest'] else
               'Roaring is smallest only at '
               f'{mv["roaring_smallest_at"]}.')
    b.body('[[The same ordering holds for irregular patterns, which we expected to '
           'favour Roaring. For a BigBird-style [3] mask (sliding window, global sinks '
           f'and random attention) at n={REF:,}, the best byte compressor ({bb_name}) '
           f'stores the pattern in {kb(bb_bytes)} KB against '
           f'{kb(bb["sizes_bytes"]["roaring_flat_opt"])} KB for a flat run-optimised '
           f'Roaring bitmap; for a block-sparse pattern at n={bs_n:,} the figures are '
           f'{kb(bs_bytes)} KB ({bs_name}) against '
           f'{kb(bs["sizes_bytes"]["roaring_flat_opt"])} KB. ' + verdict +
           ' Block-sparse patterns were measured to '
           f'n={bs_n:,} rather than n={MAXN:,} because their construction cost '
           'dominates the benchmark; the ordering is stable across the range '
           'measured.]]')

    b.subsection('[[5.2 Gradient Index Encoding Against Trivial Baselines]]')
    b.body('[[Table 4 extends the gradient-index comparison to the trained model size '
           'and adds the delta-plus-varint baseline. Roaring compresses the index '
           f'payload roughly {GI_ROARING_MIN:.1f}–'
           f'{GI_ROARING_MAX:.1f}x relative to raw uint32 arrays, but '
           f'delta-plus-varint achieves {GI_DELTA_MIN:.1f}–'
           f'{GI_DELTA_MAX:.1f}x at the same '
           'configurations, and zstd over a bitset wins at higher densities. Roaring '
           'is not the best choice at any sparsity we tested. We note additionally that '
           'these are index-payload-only ratios; because indices are roughly half of '
           'the Top-K payload, the effect on total gradient communication is at most '
           'about 25%.]]')
    grows = []
    for r in GI:
        if r['n_params'] not in (1_000_000, NP_):
            continue
        s = r['sizes_bytes']
        label = {'roaring': 'Roaring', 'delta_varint': 'delta+varint',
                 'zstd_bitset': 'zstd(bitset)', 'raw_u32': 'raw uint32'}
        grows.append([f'{r["n_params"]:,}', f'{r["sparsity"]*100:g}%', f'{r["k"]:,}',
                      f'{s["raw_u32"]:,}', f'{s["roaring"]:,}',
                      f'{s["delta_varint"]:,}', f'{s["zstd_bitset"]:,}',
                      label[r['smallest']]])
    b.table('[[Table 4. Top-K gradient index encoding, bytes. Sizes for a 1M-parameter '
            f'reference and for the trained {NP_:,}-parameter '
            'model. Held constant: indices drawn by magnitude Top-K over an i.i.d. '
            'Gaussian gradient vector — a diffuse index set with no run structure — '
            'and a single seed. Table text discusses the run-structured case '
            'separately. Lower is better.]]',
            ['Params', 'Sparsity', 'k', 'Raw uint32', 'Roaring', 'delta+varint',
             'zstd(bitset)', 'Smallest'], grows, col_pt=8.0)
    b.figure(os.path.join(FIG, 'fig2_gradient_compression.png'),
             '[[Figure 2. Index compression ratio versus parameter count. Solid lines '
             'are Roaring, dashed lines the delta-plus-varint baseline at the same '
             'sparsity. The baseline dominates at every point. Ratios flatten beyond '
             'roughly 500K parameters because container occupancy saturates: once each '
             '2^16-key chunk holds a representative number of indices, the per-chunk '
             'encoding no longer changes with scale.]]')

    if gv['layout_roaring_smallest_at']:
        w = gv['layout_roaring_smallest_at'][0]
        ws = w['sizes_bytes']
        b.body('[[One regime does favour Roaring, and we report it because the claim '
               'above would otherwise be stronger than our evidence. Table 4 draws '
               'indices by magnitude Top-K over an i.i.d. Gaussian gradient, which '
               'yields a diffuse index set with essentially no contiguous runs. If the '
               'same k indices are instead arranged into a small number of contiguous '
               f'runs ({w["n_runs"]} runs over {w["n_params"]:,} parameters at '
               f'top-{w["sparsity"]*100:g}%), Roaring becomes the smallest encoding at '
               f'{ws["roaring"]:,} bytes against {ws["zstd_bitset"]:,} for zstd over a '
               f'bitset and {ws["delta_varint"]:,} for delta-plus-varint, because each '
               'run collapses into a single Run Container. Whether the index sets '
               'produced by momentum and error-feedback Top-K in real distributed '
               'training carry that structure is not something we measured, and it '
               'would change the ranking in Table 4 if they do. We flag it as an open '
               'question rather than as a result.]]')

    b.subsection('[[5.3 Composition: The Operation That Distinguishes the '
                 'Representations]]')
    b.body('[[Size is only one axis. Inside a training loop a mask is composed from '
           'parts and queried, not merely stored. Table 5 times a representative '
           'composition — a causal window intersected with a padding mask, then unioned '
           'with global sink tokens — with all representations built in advance so that '
           'only the operation is measured. Cardinality of the result is asserted equal '
           'across all four representations.]]')
    crows = []
    for c in COMP:
        crows.append([f'{c["seq_len"]:,}',
                      f'{c["roaring_per_row_ms"]:.3f}',
                      f'{c["roaring_flat_ms"]:.3f}',
                      f'{c["dense_ms"]:.3f}',
                      f'{c["zstd_then_dense_ms"]:.3f}',
                      f'{c["flat_speedup_vs_dense"]:.1f}x'])
    b.table(f'[[Table 5. Mask composition time, median of {REPEATS} trials after '
            'warm-up, milliseconds, construction excluded. The zstd column includes the '
            'decompression that operating on a byte-compressed mask necessarily '
            f'requires. The n={REF:,} row is the same measurement reported at '
            'window=256 in Table 6. Lower is better.]]',
            ['Seq len', 'Roaring per-row', 'Roaring flat+opt', 'Dense array',
             'zstd then dense', 'Flat vs dense'], crows, col_pt=8.0)
    xover = per_row_crossover()
    third = ('Third, the advantage is contingent on layout: the per-row arrangement '
             f'is slower than dense up to n={xover:,}, so a careless implementation '
             'forfeits the benefit entirely.' if xover else
             'Third, the advantage is contingent on layout: the per-row arrangement '
             'is far slower than the flat one at every length tested, so a careless '
             'implementation forfeits most of the benefit.')
    b.body('[[Three points follow. First, the flat Roaring layout is faster than the '
           'dense equivalent at every length tested, and the advantage grows with n '
           f'({COMP[0]["flat_speedup_vs_dense"]:.1f}x at n={COMP[0]["seq_len"]:,} rising '
           f'to {COMP[-1]["flat_speedup_vs_dense"]:.0f}x at n={MAXN:,}) '
           'because Roaring cost scales with the number of runs while the dense cost '
           'scales as n². Second, the byte-compressed representation is the slowest '
           'option at every length despite being the smallest, because it must be fully '
           'decompressed before any operation. ' + third + ']]')
    b.figure(os.path.join(FIG, 'fig3_mask_creation_time.png'),
             '[[Figure 3. Mask construction cost versus sequence length. Construction '
             'is excluded from the composition timings of Table 5 and is reported here '
             'separately; the flat Roaring layout is also the cheapest to build.]]')
    b.body('[[We report one clear negative. In-place mutation — adding a single global '
           'sink token to an already-built mask — favours the dense representation '
           f'decisively: {MUT[-1]["dense_ms"]:.3f} ms against '
           f'{MUT[-1]["roaring_ms"]:.3f} ms for Roaring at '
           f'n={MUT[-1]["seq_len"]:,}, a factor of about '
           f'{MUT[-1]["dense_speedup"]:.1f}. '
           'A column write into a dense array is a single strided memory operation, '
           'whereas Roaring must touch every affected container. Workloads dominated by '
           'incremental mutation rather than bulk composition should not use this '
           'representation.]]')

    b.subsection('[[5.4 Window-Size Ablation]]')
    b.body('[[The compression ratio is invariant to window size. Because that '
           'invariance is a property of the representation rather than of training, '
           'it can be measured directly without a training run. '
           f'Table 6 varies the window at n={WA[0]["seq_len"]:,}, spanning mask '
           f'densities from {WA[0]["density"]*100:.1f}% to '
           f'{WA[-1]["density"]*100:.0f}%.]]')
    wrows = []
    for r in WA:
        wrows.append([str(r['window']) if r['window'] is not None else 'full (causal)',
                      f'{r["density"]*100:.1f}%',
                      kb(r['sizes_bytes']['roaring_flat_opt']),
                      kb(r['sizes_bytes']['zstd']),
                      f'{r["compose_flat_ms"]:.3f}',
                      f'{r["compose_dense_ms"]:.3f}',
                      f'{r["speedup"]:.1f}x'])
    b.table(f'[[Table 6. Window-size ablation at n={WA[0]["seq_len"]:,}, sizes in KB '
            'and composition times in milliseconds. Held constant: sequence length, '
            'causal structure, and the composition workload of Table 5. The '
            f'window={CFG["window_size"]} row is not a separate measurement: it is the '
            f'n={REF:,} row of Table 5, so the two tables report one number for one '
            'configuration.]]',
            ['Window', 'Mask density', 'Roaring flat+opt', 'zstd(bitset)',
             'Compose Roaring', 'Compose dense', 'Speedup'], wrows, col_pt=8.0)
    b.body('[[The Roaring representation is exactly invariant: '
           f'{kb(WA[0]["sizes_bytes"]["roaring_flat_opt"])} KB at every window, because '
           'each row remains a single run irrespective of its length, and a Run '
           'Container stores a run in constant space. The comparison with zstd is more '
           'informative than the invariance itself. The compressed byte stream grows '
           f'with density, from {kb(WA[0]["sizes_bytes"]["zstd"])} KB at a '
           f'{WA[0]["density"]*100:.1f}% dense mask to '
           f'{kb(WA[-1]["sizes_bytes"]["zstd"])} '
           f'KB at full causal attention, so Roaring\'s size disadvantage narrows from '
           f'roughly {WA[0]["sizes_bytes"]["roaring_flat_opt"]/WA[0]["sizes_bytes"]["zstd"]:.0f}x to '
           f'{WA[-1]["sizes_bytes"]["roaring_flat_opt"]/WA[-1]["sizes_bytes"]["zstd"]:.1f}x as the mask '
           'densifies. The composition advantage is stable across the range at roughly '
           f'{min(r["speedup"] for r in WA):.0f}–{max(r["speedup"] for r in WA):.0f}x.]]')

    b.subsection('5.5 Training Convergence')
    b.table('[[Table 7. Training convergence on WikiText-103 '
            f'({NP_:,} parameters, CUDA, '
            f'{MAIN["meta"]["steps"]:,} steps, {MAIN["meta"]["seeds"]} seeds). Loss is '
            'validation cross-entropy in nats, perplexity in parentheses. Pruning is '
            'disabled throughout this table and reported separately in Table 8.]]',
            ['Metric', 'Vanilla', 'Roaring'],
            [['Final validation loss (ppl)',
              f'{V_LOSS:.2f} ± {V_SD:.2f} ({np.exp(V_LOSS):.1f})',
              f'{R_LOSS:.2f} ± {R_SD:.2f} ({np.exp(R_LOSS):.1f})'],
             ['Average step time',
              f'[[{V_MS:.1f} ± {V_MSSD:.2f} ms]]', f'[[{R_MS:.1f} ± {R_MSSD:.2f} ms]]'],
             ['Peak training VRAM', f'{V_VRAM:,.0f} MB', f'{R_VRAM:,.0f} MB'],
             [f'[[Mask memory at seq_len={SEQ} (per-row layout, as run)]]',
              f'[[{DENSE_KB:,.1f} KB]]', f'[[{RUN_KB:.1f} KB]]'],
             ['[[Mask size vs dense float32]]', '[[1.0x]]',
              f'[[{DENSE_KB/RUN_KB:.1f}x]]']],
            note='[[Note. The mask-memory row reports the layout this training run '
                 'actually used, which is the per-row layout. The flat, run-optimised '
                 f'layout adopted in Section 3.1 stores the same n={SEQ} mask in '
                 f'{sz(SEQ, "roaring_flat_opt"):,} bytes '
                 f'({DENSE_KB/FLAT_KB:.0f}x versus dense float32); '
                 'Table 3 reports both. The run was performed before that change and '
                 'has not been repeated, because the layout affects the host-side '
                 'footprint of the mask and not the realized tensor the model '
                 'consumes.]]')
    b.body(f'Across {MAIN["meta"]["seeds"]} seeds the lossless Roaring optimizations '
           f'leave final validation loss statistically indistinguishable from the '
           f'baseline (Welch t-test p={EQ["welch_p"]:.2f}; Mann-Whitney U '
           f'p={EQ["mannwhitney_p"]:.2f}; both fail to reject equality). Peak training '
           f'VRAM is identical because masks are realized as dense tensors on the '
           f'accelerator: the Roaring representation reduces host memory and stored '
           f'mask footprint, not GPU VRAM.')
    b.body('[[The step-time difference requires explanation. The gap is statistically '
           f'significant (Welch p={STEP_P:.3f}), but it is an artifact of the baseline '
           'rather than a benefit of '
           'Roaring. Our vanilla implementation rebuilds the dense mask on every '
           'forward pass using a seq_len-iteration loop, whereas the Roaring '
           'implementation builds the mask once and caches the realized tensor. '
           'Caching the dense mask yields the same gain without any Roaring '
           'involvement. The '
           f'variance confirms the mechanism: the vanilla standard deviation is '
           f'{V_MSSD:.2f} ms against {R_MSSD:.2f} ms for the cached path. We therefore '
           'claim no step-time advantage, only the absence of a step-time penalty.]]')
    b.figure(os.path.join(FIG, 'fig4_convergence.png'),
             'Figure 4. Validation loss on WikiText-103, mean ± standard deviation '
             f'over {MAIN["meta"]["seeds"]} seeds. The curves overlap throughout.')
    b.figure(os.path.join(FIG, 'fig5_step_time.png'),
             '[[Figure 5. Per-step training time on CUDA with standard-error bars. The '
             'axis labels state the mechanism: the baseline rebuilds its mask each '
             'step while the Roaring path reuses a cached tensor. The difference is '
             'therefore attributable to caching, not to the mask representation.]]')

    b.subsection('5.6 Pruning Ablation')
    b.table('[[Table 8. Weight pruning ablation, reported separately from Table 7 '
            'because pruning is lossy and is disabled in the equivalence '
            'comparison.]]',
            ['Metric', 'Vanilla', 'Roaring + 30% pruning'],
            [['Final validation loss (ppl)',
              f'{PE["vanilla_mean"]:.2f} ({np.exp(PE["vanilla_mean"]):.1f})',
              f'{PE["roaring_mean"]:.2f} ({np.exp(PE["roaring_mean"]):.1f})']])
    b.body(f'Removing 30% of weights raises perplexity from '
           f'{np.exp(PE["vanilla_mean"]):.1f} to {np.exp(PE["roaring_mean"]):.1f} '
           f'(Welch p={PE["welch_p"]:.4f}). This is a quality/sparsity tradeoff, not a '
           f'free optimization.')

    b.subsection('5.7 Hydration Overhead')
    h = HYD[SEQ]
    b.body('We profiled the CPU-to-accelerator realization cost per sequence length, '
           'separating bitmap construction, dense fill, and host-to-device transfer. '
           f'At the training sequence length of {SEQ} the one-time cold cost is '
           f'{h["cold_total_ms"]:.1f} ms. Because realized masks are cached per '
           f'sequence length, this is paid once and amortized: over '
           f'{MAIN["meta"]["steps"]:,} steps it corresponds to '
           f'{h["amortized_per_step_us"]:.1f} microseconds per step, under 0.01% of a '
           f'{R_MS:.0f} ms training step. Hydration is therefore negligible at the '
           'scales tested. This amortization depends on the dense realization fitting '
           'in VRAM; at very long contexts the realized mask dominates accelerator '
           'memory, which is the regime motivating a native on-GPU kernel.')

    # ---------------------------------------------------------------- discussion
    b.section('6. Discussion')
    b.subsection('6.1 What the Representation Is and Is Not Good For')
    b.body('[[The results support a narrow claim. If the workload stores a sparse mask '
           'and never operates on it, a byte compressor is strictly better: one to two '
           'orders of magnitude smaller, at the cost of decompression when the mask is '
           'eventually needed. If the workload composes masks from parts — and any '
           'system combining causal structure with padding, global tokens, or '
           'document boundaries does — then the compressed byte stream must be '
           'decompressed on every composition, and the dense array grows as n². '
           'Roaring is the representation that avoids both, and the margin grows with '
           'sequence length. That is a real but bounded contribution.]]')
    b.body('[[A caveat we state plainly: because realized masks are cached per '
           'sequence length, composition happens rarely in our training setup, so the '
           'composition advantage does not translate into a measurable end-to-end '
           'speedup at the scale we train at. It matters where masks are rebuilt '
           'frequently — content-dependent or dynamic sparsity — or where n is large '
           'enough that the dense representation no longer fits.]]')

    b.subsection('6.2 The CPU–GPU Bridge')
    b.body('The current implementation realizes masks as dense tensors before GPU '
           f'transfer. Peak training VRAM is accordingly identical ({V_VRAM:,.0f} vs '
           f'{R_VRAM:,.0f} MB). Extending the benefit into the VRAM domain requires a '
           'native Roaring-to-CSR GPU kernel consuming the serialized bitmap directly '
           'on device. '
           '[[This is not a straightforward port. Roaring containers are '
           'heterogeneous and variable-length, so a warp processing a contiguous key '
           'range may span array, bitmap and run containers with different decode paths, '
           'producing branch divergence within the warp and irregular, unpredictable '
           'memory access. A performant kernel would likely need a container-type-major '
           'layout so that each warp decodes a single container type.]]')

    b.subsection('6.3 Comparison with FlashAttention')
    b.body('FlashAttention [7] avoids materializing the mask by fusing attention into '
           'tiled kernels. This is complementary: FlashAttention optimizes FLOPs and '
           'VRAM during execution, while the representation studied here concerns how '
           'a mask pattern is stored and combined before execution.')

    # ---------------------------------------------------------------- limitations
    b.section('7. Limitations')
    b.numbered(1, 'GPU realization. Masks are converted to dense tensors before GPU '
                  f'transfer, so peak accelerator VRAM is unchanged ({V_VRAM:,.0f} vs '
                  f'{R_VRAM:,.0f} MB measured). A native GPU kernel is required to '
                  'extend savings into VRAM.')
    b.numbered(2, '[[Size. Roaring is not the smallest representation in any mask '
                  'regime we tested, nor for the diffuse gradient index sets that '
                  'magnitude Top-K produces. Generic byte compressors are one to two '
                  'orders of magnitude smaller for masks, and delta-plus-varint is '
                  'smaller for those gradient indices. The one exception is the '
                  'run-structured index set of Section 5.2, which we did not observe '
                  'in training. The contribution rests on operability, not '
                  'footprint.]]')
    b.numbered(3, '[[Validation sequence length. The motivating figure of 40 GB '
                  'assumes n=100,000, but training ran at n=' + str(SEQ) + ', where a '
                  f'dense float32 mask is only {DENSE_KB:,.1f}'
                  ' KB and there is no memory pressure whatsoever. The scaling results '
                  f'in Tables 3 and 5 are measured up to n={MAXN:,}; everything beyond '
                  'that is extrapolation. The gap between the motivation and the '
                  'training configuration is a limitation of compute budget, not a '
                  'claim about behaviour at length.]]')
    b.numbered(4, 'Model scale. Convergence is validated at '
                  f'{NP_/1e6:.1f}M parameters. Validation at '
                  '100M+ parameters and on multi-node clusters remains future work. '
                  '[[No larger training run was performed for this revision; the '
                  'scale-related claims were withdrawn instead, because the compute '
                  'required was not available to us.]]')
    b.numbered(5, '[[Distributed communication. All training was performed on a single '
                  'GPU, so no All-Reduce traffic existed to compress. Any '
                  'communication-compression figure in this paper is analytic: it is '
                  'computed by logging the Top-K gradient index set produced per step '
                  'and measuring the serialized footprint of each encoding offline. '
                  'End-to-end All-Reduce timing, including serialization and '
                  'deserialization overhead in a real multi-node system, is not '
                  'demonstrated.]]')
    b.numbered(6, '[[Mutation. In-place mutation of an existing mask is substantially '
                  'slower than the dense equivalent (Section 5.3).]]')
    b.numbered(7, '[[Ablation scope. We characterise window size and sparsity at the level of '
                  'the representation — window-size invariance in Table 6, the effect '
                  'of sparsity on index encoding in Table 4 — but we do not report how '
                  'final training loss varies across window sizes or sparsity levels. '
                  'That would require additional training runs beyond our compute '
                  'budget, and we make no claim on it.]]')
    b.numbered(8, '[[Index distribution. The gradient-index results assume indices '
                  'drawn by magnitude Top-K over an i.i.d. Gaussian gradient. Real '
                  'Top-K with momentum and error feedback may produce index sets with '
                  'run structure, which Section 5.2 shows would favour Roaring. We did '
                  'not measure the index distribution of a real distributed run.]]')

    # ---------------------------------------------------------------- conclusion
    b.section('8. Conclusion')
    b.body('[[Measured against baselines that exploit the known structure, Roaring '
           'Bitmaps are not a competitive compression mechanism for Transformer '
           'training metadata: byte compressors store these mask patterns far more '
           'compactly, and a delta-plus-varint encoder is smaller on diffuse gradient '
           'indices at every configuration we measured. Roaring is smallest only for '
           'run-structured index sets, a regime we characterise but do not observe in '
           'training.]]')
    b.body('[[The finding that does hold is about operability rather than size. Among '
           'the representations tested, Roaring is the only one that is both compact '
           'and directly operable: composing a mask from causal, padding and global '
           f'components is {COMP[-1]["flat_speedup_vs_dense"]:.0f}x faster than the '
           f'dense equivalent at n={MAXN:,} and the margin grows with '
           'sequence length, while byte-compressed masks must be decompressed in full '
           'and are the slowest option end-to-end. We also found that layout choice — '
           'one bitmap per row versus a flat, run-optimised bitmap — accounts for much '
           'of the overhead that might otherwise be attributed to the data structure '
           'itself.]]')
    b.body('On real WikiText-103 data the lossless optimizations impose no '
           'statistically measurable penalty on model quality, while weight pruning '
           'trades quality for sparsity. The savings accrue to the mask '
           'representation and host memory, not to accelerator VRAM, which is unchanged '
           'under the dense-realization design used here.')

    b.section('9. Future Work')
    b.numbered(1, 'Roaring-to-CSR GPU kernel: decompress directly into sparse CSR in '
                  'VRAM, addressing the warp-divergence issues outlined in '
                  'Section 6.2.')
    b.numbered(2, '[[Dynamic and content-dependent sparsity, where masks are rebuilt '
                  'frequently rather than cached, and where the composition advantage '
                  'of Section 5.3 would translate into end-to-end gains.]]')
    b.numbered(3, 'Large-scale validation: 100M+ parameter models and multi-node '
                  'clusters with real All-Reduce measurement.')
    b.numbered(4, 'Combined Roaring pattern planning with FlashAttention execution.')
    b.numbered(5, '[[Measurement of the index distribution produced by momentum-based '
                  'Top-K in a real distributed run, which determines whether the '
                  'run-structured regime of Section 5.2 occurs in practice.]]')

    b.section('10. Author Contributions')
    b.body('Lokesh Karanam: conceptualization, methodology, software, formal analysis, '
           'investigation, writing — original draft. Purna Sreekar Karanam: '
           'visualization, validation, writing — review and editing. Sri Ram Pemmasani: '
           'validation, writing — review and editing.', indent=False)

    b.section('11. Funding')
    b.body('This research did not receive external funding from any agencies.',
           indent=False)

    b.section('12. Ethical Statement')
    b.body('Not applicable. This research did not involve human subjects, animal '
           'subjects, or any data requiring ethics board approval.', indent=False)

    b.section('13. Data Availability Statement')
    b.body('[[The reference implementation, the benchmark suite, and the raw result '
           'files from which every table and figure in this paper is generated are '
           'available from the corresponding author on request.]]', indent=False)

    b.section('14. Conflict of Interest')
    b.body('The authors declare no conflict of interest.', indent=False)

    b.section('15. Acknowledgments')
    b.body('The authors used Claude (Anthropic) for language editing and formatting '
           'assistance during manuscript preparation. The authors reviewed and edited '
           'the content as needed and take full responsibility for the manuscript\'s '
           'integrity.', indent=False)

    b.section('16. References')
    for i, ref in enumerate(REFERENCES, 1):
        b.reference(i, ref)
    return doc


REFERENCES = [
    'Lin Y, Han S, Mao H, Wang Y, Dally WJ. Deep gradient compression: reducing the '
    'communication bandwidth for distributed training. In: ICLR; 2018.',
    'Beltagy I, Peters ME, Cohan A. Longformer: the long-document transformer. arXiv; 2020.',
    'Zaheer M, Guruganesh G, Dubey A, Ainslie J, Alberti C, Ontañón S, et al. Big Bird: '
    'transformers for longer sequences. In: NeurIPS; 2020.',
    'Child R, Gray S, Radford A, Sutskever I. Generating long sequences with sparse '
    'transformers. arXiv; 2019.',
    'Chambi S, Lemire D, Kaser O, Godin R. Better bitmap performance with Roaring '
    'Bitmaps. Software: Practice and Experience. 2016;46(5):709–719.',
    'Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention '
    'is all you need. In: NeurIPS; 2017.',
    'Dao T, Fu DY, Ermon S, Rudra A, Ré C. FlashAttention: fast and memory-efficient '
    'exact attention with IO-awareness. In: NeurIPS; 2022.',
    'Alistarh D, Grubic D, Li J, Tomioka R, Vojnovic M. QSGD: communication-efficient '
    'SGD via gradient quantization and encoding. In: NeurIPS; 2017.',
    'Stich SU, Cordonnier JB, Jaggi M. Sparsified SGD with memory. In: NeurIPS; 2018.',
    'Dao T. FlashAttention-2: faster attention with better parallelism and work '
    'partitioning. arXiv; 2023.',
    'Shah J, Bikshandi G, Zhang Y, Thakkar V, Ramani P, Dao T. FlashAttention-3: fast '
    'and accurate attention with asynchrony and low-precision. arXiv; 2024.',
    'Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces. '
    'arXiv; 2023.',
    'Xiao G, Tian Y, Chen B, Han S, Lewis M. Efficient streaming language models with '
    'attention sinks. In: ICLR; 2024.',
    'Liu H, Zaharia M, Abbeel P. Ring attention with blockwise transformers for '
    'near-infinite context. arXiv; 2023.',
    'Yuan J, Gao H, Dai D, Luo J, Zhao L, Zhang Z, et al. Native sparse attention: '
    'hardware-aligned and natively trainable sparse attention. arXiv; 2025.',
    'Lemire D, Ssi-Yan-Kai G, Kaser O. Consistently faster and smaller compressed '
    'bitmaps with Roaring. Software: Practice and Experience. 2016;46(11):1547–1569.',
    'Vogels T, Karimireddy SP, Jaggi M. PowerSGD: practical low-rank gradient '
    'compression for distributed optimization. In: NeurIPS; 2019.',
    'Frantar E, Alistarh D. SparseGPT: massive language models can be accurately '
    'pruned in one-shot. In: ICML; 2023.',
    'Sun M, Liu Z, Bair A, Kolter JZ. A simple and effective pruning approach for '
    'large language models. In: ICLR; 2024.',
    'Collet Y, Kucherawy M. Zstandard compression and the application/zstd media type. '
    'RFC 8878; 2021.',
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = [
        (True, False, 'Roaring-Transformer-v3-Highlighted.docx'),
        (False, False, 'Roaring-Transformer-v3-Clean.docx'),
        (True, True, 'Roaring-Transformer-v3-Template-Highlighted.docx'),
        (False, True, 'Roaring-Transformer-v3-Template-Clean.docx'),
    ]
    for hl, tmpl, name in targets:
        path = os.path.join(OUT_DIR, name)
        build(hl, use_template=tmpl).save(path)
        print(f'wrote {os.path.relpath(path, os.path.join(HERE, ".."))}')


if __name__ == '__main__':
    main()
