#!/usr/bin/env python3
"""
Build the round-4 revised manuscript for ACRT-2026-0062.

What changed from round 3: the composition claim now carries the same baseline
set the size claim got. Roaring loses composition on structured masks to a
purpose-built (start, length) encoding, wins it on unstructured masks where that
encoding does not apply, and the property that decides between them is
identified. The
paper's thesis is that boundary rather than a bare speed number.

Emits into paper/v4-review/:
    Roaring-Transformer-v4-Highlighted.docx
    Roaring-Transformer-v4-Clean.docx
    Roaring-Transformer-v4-Template-Highlighted.docx
    Roaring-Transformer-v4-Template-Clean.docx

Sources (every number is read, none transcribed):
    experiments/v4_results.json        all CPU measurements, one process
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
OUT_DIR = os.path.join(HERE, '..', 'paper', 'v4-review')
FIG = os.path.join(HERE, 'figures_v4')


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


V4 = load('v4_results.json')
MAIN = load('revision_main.json')
PRUNED = load('revision_pruned.json')

MS = {r['seq_len']: r for r in V4['mask_scaling']}
PS = V4['pattern_sweep']
ST = V4['composition_structured']
UN = V4['composition_unstructured']
SW = V4['structure_sweep']
MUT = V4['mutation']
WA = V4['window_ablation']
GI = V4['gradient_index']
GL = V4['gradient_layout']
D = V4['derived']
LP, MV, CV, GV = (D['layout_penalty'], D['mask_size_verdict'],
                  D['composition_verdict'], D['gradient_verdict'])
REPEATS = V4['meta']['repeats']

CFG = MAIN['meta']['model_cfg']
SEQ = CFG['seq_len']
NP_ = MAIN['meta']['n_params']
REF = LP['reference_seq_len']
MAXN = ST[-1]['seq_len']
UNMAXN = UN[-1]['seq_len']

SHORT_LABEL = {'run_arrays_ms': '(start,len) runs',
               'packed_bitset_ms': 'Packed bitset', 'csr_ms': 'Sparse CSR',
               'dense_ms': 'Dense bool', 'zstd_then_dense_ms': 'zstd then dense'}
REP_LABEL = {'run_arrays_ms': 'the (start, length) run encoding',
             'packed_bitset_ms': 'a packed bitset', 'csr_ms': 'sparse CSR',
             'dense_ms': 'a dense bool array',
             'zstd_then_dense_ms': 'zstd then dense'}


def arm(a, key):
    xs = [r[key] for r in MAIN['runs'][a]]
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


V_LOSS, V_SD = arm('vanilla', 'final_val_loss')
R_LOSS, R_SD = arm('roaring', 'final_val_loss')
V_MS, V_MSSD = arm('vanilla', 'avg_step_ms')
R_MS, R_MSSD = arm('roaring', 'avg_step_ms')
V_VRAM, _ = arm('vanilla', 'peak_vram_mb')
R_VRAM, _ = arm('roaring', 'peak_vram_mb')
EQ, PE = MAIN['equivalence'], PRUNED['equivalence']
HYD = {h['seq_len']: h for h in MAIN['hydration']}
STEP_P = float(stats.ttest_ind([r['avg_step_ms'] for r in MAIN['runs']['vanilla']],
                               [r['avg_step_ms'] for r in MAIN['runs']['roaring']],
                               equal_var=False).pvalue)
# The pruning run re-uses the identical seeded vanilla arm, so Tables 9 and 10
# share a baseline by construction rather than by coincidence.
PAIRED_BASELINE = np.allclose([r['final_val_loss'] for r in MAIN['runs']['vanilla']],
                              [r['final_val_loss'] for r in PRUNED['runs']['vanilla']])

RUN_KB = HYD[SEQ]['roaring_cpu_kb']
DENSE_KB = MS[SEQ]['sizes_bytes']['dense_f32'] / 1024
FLAT_KB = MS[SEQ]['sizes_bytes']['roaring_flat_opt'] / 1024

GI_SHOWN = [r for r in GI if r['n_params'] in (1_000_000, NP_)]
GI_ROARING_MIN = min(r['roaring_vs_raw'] for r in GI_SHOWN)
GI_ROARING_MAX = max(r['roaring_vs_raw'] for r in GI_SHOWN)
GI_DELTA_MIN = min(r['delta_vs_raw'] for r in GI_SHOWN)
GI_DELTA_MAX = max(r['delta_vs_raw'] for r in GI_SHOWN)
assert all(r['sizes_bytes']['delta_varint'] < r['sizes_bytes']['roaring'] for r in GI)

RI = CV['row_invariant']            # row-invariant operand: closed form applies
RD = CV['row_dependent_first']      # first row-dependent operand: it does not
DG = CV['degradation']
UN_REF = UN[-1]
UN_BEST_MS = min(v for k, v in UN_REF.items()
                 if k.endswith('_ms') and v is not None and not k.startswith('roaring'))


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


def build(highlight, use_template=False):
    if use_template:
        doc, styles = template_document(TEMPLATE)
        b = Builder(doc, highlight=highlight, styles=styles)
    else:
        doc = new_document()
        b = Builder(doc, highlight=highlight)

    # ---------------------------------------------------------------- front
    if use_template:
        b.label('Article Title')
        b.title('The Roaring Transformer: [[When Compressed Bitmaps Are the Right '
                'Attention-Mask Representation, and When They Are Not]]')
        b.label('Authors and Affiliations')
        b.author('Lokesh Karanam1, Purna Sreekar Karanam1, Sri Ram Pemmasani1')
        b.author('1 Independent Researchers, Austin, TX, USA')
        b.author('*Address all correspondence to:')
        b.author('Lokesh Karanam')
        b.author('lokesh.karanam@rrd.com')
        b.label('Abstract')
    else:
        b.title('The Roaring Transformer: [[When Compressed Bitmaps Are the Right '
                'Attention-Mask Representation, and When They Are Not]]')
        b.author('Lokesh Karanam, Purna Sreekar Karanam, Sri Ram Pemmasani')
        b.affiliation('Independent Researchers, Austin, TX, USA\n'
                      'Correspondence: lokesh.karanam@rrd.com')
        b.section('Abstract')
    b.body(
        'We ask whether Roaring Bitmaps, a hybrid compressed set structure, are a '
        'useful representation for the sparse metadata that surrounds the training of '
        'Transformer-based language models: attention masks, Top-K gradient indices, '
        'and pruning masks. '
        '[[We evaluate them against purpose-built encodings of the same structure '
        'rather than against a dense float32 baseline, and the answer is a boundary '
        'rather than a verdict. On size Roaring loses outright: a generic byte '
        f'compressor stores a causal sliding-window mask at n={REF:,} in '
        f'{kb(sz(REF, "zstd"))} KB against {kb(sz(REF, "roaring_flat_opt"))} KB for '
        f'Roaring, across all {MV["n_configurations"]} mask configurations we measured '
        'there is none in which Roaring is smallest, and delta-plus-varint is smaller '
        'on diffuse Top-K gradient indices at every setting tested. On composition — '
        'intersecting a causal window with padding and unioning further components — '
        'the ranking depends on a single categorical property of the pattern: whether '
        'the components being combined are the same for every query row. When they '
        'are — causal windows, padding, global sink tokens, everything determined by '
        'position — the composition has a closed form over two integer arrays, and '
        f'that form is {CV["structured_deficit_at_max_n"]:.0f}x faster than Roaring at '
        f'n={MAXN:,}; Roaring should not be used. When any component varies per row, '
        'as content-dependent, learned or randomised attention does, the closed form '
        'is not expressible and the dense and bitset baselines lose the row broadcast '
        'they depended on. Roaring is then the fastest representation available, by '
        f'{CV["unstructured_margin_min"]:.1f}x to {CV["unstructured_margin_max"]:.1f}x '
        'over the best surviving baseline and up to '
        f'{CV["unstructured_vs_dense_max"]:.0f}x over a dense array. The transition is '
        'categorical rather than gradual: at equal mask fragmentation, one '
        'content-dependent entry per row is enough to cross it.]] '
        f'On WikiText-103, a {NP_/1e6:.1f}M-parameter model trained over '
        f'{MAIN["meta"]["steps"]:,} steps and {MAIN["meta"]["seeds"]} seeds shows the '
        'lossless Roaring path leaves validation loss statistically unchanged '
        f'(Welch p={EQ["welch_p"]:.2f}; Mann-Whitney p={EQ["mannwhitney_p"]:.2f}). '
        'Masks are hydrated to dense tensors before the accelerator, so GPU VRAM is '
        f'unchanged ({V_VRAM:,.0f} vs {R_VRAM:,.0f} MB); the saving is to host memory, '
        f'where the mask is {DENSE_KB/RUN_KB:.0f}x smaller than the tensor realized '
        'from it. '
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
        'as O(n²). At n = 100,000 a single float32 mask occupies roughly 40 GB. This '
        'figure motivates compressing the mask representation and the associated '
        'host-memory footprint. Relieving on-GPU VRAM at such lengths additionally '
        'requires native on-device execution of the compressed mask, which we do not '
        'implement here; the present contribution targets representation, host memory, '
        'and mask manipulation rather than a direct reduction of accelerator VRAM. A '
        'parallel concern arises in distributed training, where All-Reduce traffic is '
        'commonly reduced by transmitting only the largest gradient entries [1], '
        'leaving their indices to be encoded as metadata.')
    b.body(
        '[[Compression ratios for such masks are often quoted against a dense float32 '
        'baseline, but that is not a baseline anyone would use for a structurally '
        'regular mask: a causal sliding-window pattern is a set of contiguous integer '
        'ranges, and almost any purpose-built encoding will compress it well. The '
        'meaningful question is not whether some compression helps — it obviously '
        'does — but whether a general-purpose set structure earns its overhead against '
        'an encoding that exploits the known structure. We apply that test twice: to '
        'size in Section 5.1, where Roaring loses, and to composition in Sections 5.3 '
        'to 5.5, where the answer turns out to depend on a property of the mask that '
        'can be measured.]]')
    b.body(
        '[[That property is whether the mask is assembled from components that are '
        'identical for every query row. A causal sliding window, a padding prefix and '
        'a block of global sink tokens all are: each is determined by position alone, '
        'so the composed mask has a closed form — two integer arrays holding the start '
        'and end of each row — and combining them is arithmetic on those arrays. No '
        'set structure competes with that, and we show Roaring does not; it is an '
        'order of magnitude slower. The same property is what lets a dense or bitset '
        'baseline broadcast one row over the whole mask instead of touching n² '
        'elements. Patterns that depend on content rather than position break it: once '
        'any component differs from row to row, the closed form is no longer '
        'expressible, the broadcast is no longer available, and the practical choices '
        'collapse to a dense array, a bitset, a general sparse matrix, or a compressed '
        'set. Among those Roaring is the fastest at every size we tested. The '
        'contribution of this paper is that boundary, its mechanism, and the '
        'observation that it is categorical: a single content-dependent entry per row '
        'moves a pattern from one side to the other.]]')

    b.body('Our contributions are:', indent=False)
    b.numbered(1, '[[A baseline study of attention-mask and gradient-index encodings '
                  'comparing Roaring against (start, length) tuples, packed bitsets, '
                  'delta-plus-varint, zlib, zstd and lzma. Roaring does not win on '
                  'size, and we report that negative result in full.]]')
    b.numbered(2, '[[A composition study over two workloads and six representations, '
                  'including Reviewer-suggested purpose-built run arrays and a general '
                  'sparse matrix, each given its fastest correct form. Roaring loses '
                  'whenever every component is row-invariant and wins whenever any '
                  'component is not.]]')
    b.numbered(3, '[[Identification of the property that decides between those '
                  'regimes — whether the components being composed vary per query row '
                  '— and a demonstration that the transition is categorical: at '
                  f'n={CV["sweep_n"]:,} a single content-dependent entry per row moves '
                  f'Roaring from {RI["roaring_deficit"]:.1f}x slower than the best '
                  f'alternative to {RD["roaring_margin"]:.0f}x faster, with the '
                  'composed mask no more fragmented than before.]]')
    b.numbered(4, '[[A layout finding: storing one bitmap per query row and omitting '
                  f'run_optimize() costs {LP["size_ratio_at_ref"]:.2f}x in size and '
                  f'{LP["speed_ratio_at_ref"]:.0f}x in composition time at n={REF:,}.]]')
    b.numbered(5, 'Convergence equivalence on WikiText-103 across five seeds with '
                  'significance testing, an explicit measurement showing GPU VRAM is '
                  '[[unchanged, and the host-memory saving quantified against the '
                  'tensor actually realized on the device]].')
    b.numbered(6, '[[A reference implementation, a benchmark suite, and the raw result '
                  'files from which every table and figure is generated, available as '
                  'stated in Section 13.]]')
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
           'chosen: given any such pattern, what is the right in-memory representation '
           'for composing and manipulating it? The answer depends on whether the '
           'pattern is a function of position alone or of content — Longformer and '
           'Sparse Transformer are the former, BigBird and any learned or '
           'retrieval-driven pattern the latter — which Section 5.5 shows is the '
           'deciding distinction.]]')

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
    b.body('[[We compare against five families. (i) Structure-specific encodings: for '
           'a causal sliding-window mask each row is one contiguous range, so a '
           '(start, length) pair per row suffices, costing 8n bytes; composition in '
           'this form is arithmetic on two length-n integer arrays. (ii) Packed '
           'bitsets: one bit per element, n²/8 bytes, with numpy bitwise operations '
           'and no container logic. (iii) General-purpose byte compressors applied to '
           'the packed bitset: zlib, zstd [20] and lzma, all three reported in Table 3. '
           '(iv) General sparse matrices: compressed sparse row (CSR) as provided by '
           'SciPy, the representation an implementer is most likely to reach for when '
           'the pattern is irregular. (v) For gradient indices, delta encoding '
           'followed by LEB128 varint, the standard approach for sorted integer '
           'sequences. Each is a plausible choice a practitioner would make before '
           'adopting a specialised set structure, and each is given its fastest '
           'correct form: where an operand is a single row broadcast over the mask, '
           'the dense and bitset baselines use that broadcast rather than a '
           'materialised matrix. Orthogonal families of gradient-communication methods '
           '— quantisation [8] and low-rank decomposition [17] — reduce the value '
           'payload rather than the index metadata, and are complementary.]]')

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
           f'vectorised call. At n={REF:,} the flat, run-optimised layout is '
           f'{LP["size_ratio_at_ref"]:.2f}x smaller and {LP["speed_ratio_at_ref"]:.0f}x '
           'faster to compose. The WikiText-103 run reported in Section 5.7 predates '
           'this change and used the per-row layout; Table 9 states both footprints.]] '
           'Before each forward pass the bitmap is hydrated into a dense tensor and '
           'transferred to the accelerator; realized masks are cached per sequence '
           'length.')

    b.subsection('3.2 Gradient Index Compressor')
    b.body('In Top-K sparsification the standard payload is (indices: uint32[k], '
           'values: float32[k]). We replace the index array with a serialized Roaring '
           'Bitmap. Error feedback [9] ensures no gradient information is lost. '
           '[[Section 5.2 shows this is not competitive for the index sets magnitude '
           'Top-K actually produces: delta-plus-varint encoding is smaller at every '
           'configuration we tested. We retain the description for completeness and '
           'report the negative result rather than the more flattering ratio against '
           'raw uint32 arrays.]]')

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
           '[[Encoding-size and composition benchmarks (Sections 5.1 to 5.6) are CPU '
           f'measurements reported as medians of {REPEATS} trials after warm-up, with '
           'construction excluded from the composition timings and the cardinality of '
           'the composed result asserted identical across all six representations. '
           'All of them are produced by a single benchmark process, so timings are '
           'directly comparable across tables; where two tables describe the same '
           'configuration it is measured once and shared.]]')

    # ---------------------------------------------------------------- results
    b.section('5. Results')
    b.subsection('[[5.1 Mask Representation Against Trivial Baselines]]')
    zstd_margins = [sz(n, 'roaring_flat_opt') / sz(n, 'zstd') for n in MS]
    assert all(max(sz(n, 'zlib'), sz(n, 'zstd'), sz(n, 'lzma')) <
               sz(n, 'roaring_flat_opt') for n in MS)
    b.body('[[Table 3 compares every candidate encoding across sequence length. The '
           'ordering is unambiguous. All three general-purpose byte compressors '
           'applied to a packed bitset are smaller than either Roaring layout at every '
           f'sequence length, and the strongest of them (zstd) by '
           f'{min(zstd_margins):.0f}x to {max(zstd_margins):.0f}x — one to two orders '
           'of magnitude. The (start, length) encoding, which exploits the contiguity '
           'of the causal window directly, beats the per-row Roaring layout. It does '
           'not beat the flat, run-optimised layout, which is the correct way to use '
           'Roaring for this pattern — but that establishes only that layout choice '
           'matters, not that Roaring wins on size.]]')
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
    verdict = (f'We found no mask regime in which Roaring is the smallest '
               f'representation across the {MV["n_configurations"]} mask '
               'configurations measured.' if not MV['roaring_ever_smallest'] else
               f'Roaring is smallest only at {MV["roaring_smallest_at"]}.')
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
           'measured. If a workload stores masks and never operates on them, this '
           'section is the whole answer: use a byte compressor.]]')

    b.subsection('[[5.2 Gradient Index Encoding Against Trivial Baselines]]')
    b.body('[[Table 4 extends the gradient-index comparison to the trained model size '
           'and adds the delta-plus-varint baseline. Roaring compresses the index '
           f'payload roughly {GI_ROARING_MIN:.1f}–{GI_ROARING_MAX:.1f}x relative to '
           f'raw uint32 arrays, but delta-plus-varint achieves {GI_DELTA_MIN:.1f}–'
           f'{GI_DELTA_MAX:.1f}x at the same configurations, and zstd over a bitset '
           'wins at higher densities. Roaring is not the best choice at any sparsity '
           'we tested. We note additionally that these are index-payload-only ratios; '
           'because indices are roughly half of the Top-K payload, the effect on total '
           'gradient communication is at most about 25%.]]')
    grows = []
    label = {'roaring': 'Roaring', 'delta_varint': 'delta+varint',
             'zstd_bitset': 'zstd(bitset)', 'raw_u32': 'raw uint32'}
    for r in GI:
        if r['n_params'] not in (1_000_000, NP_):
            continue
        s = r['sizes_bytes']
        grows.append([f'{r["n_params"]:,}', f'{r["sparsity"]*100:g}%', f'{r["k"]:,}',
                      f'{s["raw_u32"]:,}', f'{s["roaring"]:,}',
                      f'{s["delta_varint"]:,}', f'{s["zstd_bitset"]:,}',
                      label[r['smallest']]])
    b.table('[[Table 4. Top-K gradient index encoding, bytes. Sizes for a 1M-parameter '
            f'reference and for the trained {NP_:,}-parameter model. Held constant: '
            'indices drawn by magnitude Top-K over an i.i.d. Gaussian gradient vector '
            '— a diffuse index set with no run structure — and a single seed. The '
            'run-structured case is discussed below. Lower is better.]]',
            ['Params', 'Sparsity', 'k', 'Raw uint32', 'Roaring', 'delta+varint',
             'zstd(bitset)', 'Smallest'], grows, col_pt=8.0)
    b.figure(os.path.join(FIG, 'fig2_gradient_compression.png'),
             '[[Figure 2. Index compression ratio versus parameter count. Solid lines '
             'are Roaring, dashed lines the delta-plus-varint baseline at the same '
             'sparsity. The baseline dominates at every point. Ratios flatten beyond '
             'roughly 500K parameters because container occupancy saturates: once each '
             '2^16-key chunk holds a representative number of indices, the per-chunk '
             'encoding no longer changes with scale.]]')

    if GV['layout_roaring_smallest_at']:
        w = GV['layout_roaring_smallest_at'][0]
        ws = w['sizes_bytes']
        b.body('[[The exception is the same one that governs the composition results, '
               'and it appears here first. Table 4 draws indices by magnitude Top-K '
               'over an i.i.d. Gaussian gradient, which yields a diffuse index set '
               'with essentially no contiguous runs. If the same k indices are instead '
               f'arranged into a small number of contiguous runs ({w["n_runs"]} runs '
               f'over {w["n_params"]:,} parameters at top-{w["sparsity"]*100:g}%), '
               f'Roaring becomes the smallest encoding at {ws["roaring"]:,} bytes '
               f'against {ws["zstd_bitset"]:,} for zstd over a bitset and '
               f'{ws["delta_varint"]:,} for delta-plus-varint, because each run '
               'collapses into a single Run Container. Run structure is what Roaring '
               'is built to exploit, in indices as in masks. Whether the index sets '
               'produced by momentum and error-feedback Top-K in real distributed '
               'training carry that structure is not something we measured, and it '
               'would change the ranking in Table 4 if they do.]]')

    # ---- 5.3 structured composition -------------------------------------- #
    b.subsection('[[5.3 Composition on Structured Masks]]')
    b.body('[[Size is only one axis, and for a mask that is used rather than archived '
           'it is the less interesting one. Inside a training loop a mask is composed '
           'from parts: a causal window intersected with a padding mask, unioned with '
           'global sink tokens. Table 5 times that composition in six representations '
           'with everything built in advance, so that only the operation is measured. '
           'The result contradicts the framing of our previous version, and we lead '
           'with it. Every component here is row-invariant: the causal window, the '
           'padding prefix and the global block are all determined by position, so the '
           'composed mask has a closed form and the (start, length) encoding reduces '
           'the whole operation to arithmetic on two integer arrays of length n. That '
           f'encoding is {ST[0]["roaring_vs_best_rival"]:.1f}x to '
           f'{CV["structured_deficit_at_max_n"]:.0f}x faster than Roaring across the '
           'range, and the margin widens with n. The same property lets the dense and '
           'bitset baselines broadcast a single row rather than touch n² elements, '
           'which is why the bitset is also competitive here. Roaring is not the right '
           'representation for this workload and we do not recommend it: a reader '
           'whose masks are built only from position-determined components can stop '
           'at this table.]]')
    crows = []
    for c in ST:
        crows.append([f'{c["seq_len"]:,}', f'{c["roaring_per_row_ms"]:.3f}',
                      f'{c["roaring_flat_ms"]:.3f}', f'{c["dense_ms"]:.3f}',
                      f'{c["packed_bitset_ms"]:.3f}', f'{c["csr_ms"]:.3f}',
                      f'{c["zstd_then_dense_ms"]:.3f}', f'{c["run_arrays_ms"]:.4f}'])
    b.table(f'[[Table 5. Structured composition (causal ∩ padding) ∪ global, median of '
            f'{REPEATS} trials after warm-up, milliseconds, construction excluded. '
            'Every component is determined by position and is therefore identical for '
            'every query row, so the composed mask has a closed form and the '
            '(start, length) encoding applies and wins. The zstd column includes the '
            'decompression a byte-compressed mask necessarily requires. The '
            f'n={REF:,} row is the same measurement as the window={CFG["window_size"]} '
            'row of Table 8. Lower is better.]]',
            ['Seq len', 'Roaring per-row', 'Roaring flat+opt', 'Dense bool',
             'Packed bitset', 'Sparse CSR', 'zstd then dense',
             '(start,len) runs'], crows, col_pt=7.0)

    # ---- 5.4 unstructured composition ------------------------------------ #
    b.subsection('[[5.4 Composition on Unstructured Masks]]')
    b.body('[[The (start, length) encoding wins Table 5 because the workload never '
           'asks it to represent anything it cannot. That stops being true as soon as '
           'one component of the mask differs between query rows. Table 6 repeats the '
           'composition with the global component replaced by a random-attention '
           'component of the kind BigBird [3] introduces. Two things change at once, '
           'and both follow from the same cause. The closed form disappears: a per-row '
           '(start, length) pair cannot express a row whose permitted columns are not '
           'an interval, so the encoding drops out of the comparison rather than '
           'losing it. And the row broadcast disappears with it: the dense and bitset '
           'baselines can no longer apply one row to the whole mask and must process '
           'all n² elements. Among the representations that remain, Roaring is the '
           f'fastest at every length tested — {CV["unstructured_margin_min"]:.1f}x to '
           f'{CV["unstructured_margin_max"]:.1f}x faster than the best of them '
           f'({REP_LABEL[UN_REF["best_rival"]]}) and up to '
           f'{CV["unstructured_vs_dense_max"]:.0f}x faster than the dense array — '
           'because its cost is set by the number of runs it stores rather than by '
           'n².]]')
    urows = []
    for c in UN:
        urows.append([f'{c["seq_len"]:,}', f'{c["runs_per_row"]:.1f}',
                      f'{c["roaring_flat_ms"]:.3f}', f'{c["dense_ms"]:.3f}',
                      f'{c["packed_bitset_ms"]:.3f}', f'{c["csr_ms"]:.3f}',
                      'n/a', f'{1/c["roaring_vs_best_rival"]:.1f}x'])
    b.table(f'[[Table 6. Unstructured composition (causal ∩ padding) ∪ random '
            f'attention, median of {REPEATS} trials, milliseconds, construction '
            'excluded. The (start, length) column is "n/a" because a single run per '
            'row can no longer represent the composed mask — the encoding is '
            'inapplicable, not merely slower. The final column is Roaring\'s margin '
            'over the fastest representation that still applies. Lower is better.]]',
            ['Seq len', 'Runs/row', 'Roaring flat+opt', 'Dense bool', 'Packed bitset',
             'Sparse CSR', '(start,len) runs', 'Roaring margin'], urows, col_pt=7.5)
    b.figure(os.path.join(FIG, 'fig7_composition_workloads.png'),
             '[[Figure 3. Composition cost against sequence length for the two '
             'workloads. Left: while the mask stays run-structured the (start, length) '
             'encoding dominates every set representation. Right: once run structure '
             'is destroyed that encoding is unavailable and Roaring is the fastest of '
             'the representations that remain.]]')

    # ---- 5.5 what decides between the regimes ----------------------------- #
    b.subsection('[[5.5 What Decides Between the Two Regimes]]')
    b.body('[[Sections 5.3 and 5.4 are two ends of one axis, and the useful question '
           'is what governs the switch. The intuitive answer — that Roaring wins once '
           'the mask becomes sufficiently fragmented — is not what we measure. Table 7 '
           f'sweeps content-dependent attention at n={CV["sweep_n"]:,}, from none to 32 '
           'random entries per row, reporting the mean contiguous runs per row '
           'alongside each representation. The first two rows settle the question: '
           f'they carry {RI["runs_per_row"]:.1f} and {RD["runs_per_row"]:.1f} runs per '
           'row respectively — indistinguishable fragmentation — yet Roaring is '
           f'{RI["roaring_deficit"]:.1f}x slower than the best alternative in the first '
           f'and {RD["roaring_margin"]:.0f}x faster in the second. Run count is '
           'therefore not the deciding variable.]]')
    b.body('[[What changes between those two rows is that the second operand stops '
           'being the same for every query row — it becomes row-dependent. That single '
           'change withdraws both '
           'advantages the baselines were relying on — the closed-form (start, length) '
           'composition and the row broadcast — while costing Roaring nothing: its '
           f'time moves from {RI["roaring_ms"]:.3f} ms to {RD["roaring_ms"]:.3f} ms. '
           'The transition is categorical, not gradual, and it is decidable by '
           'inspecting a sparsity pattern rather than by benchmarking it: if every '
           'component of the mask is a function of position alone, use the closed '
           'form; if any component is a function of content, the closed form does not '
           'exist and Roaring is the best remaining option.]]')
    b.body('[[Within the content-dependent regime Roaring then degrades gracefully '
           'while the alternatives do not improve. Increasing fragmentation '
           f'{DG["runs_growth"]:.0f}-fold, from {DG["runs_min"]:.1f} to '
           f'{DG["runs_max"]:.1f} runs per row, raises Roaring\'s composition time by '
           f'only {DG["roaring_growth"]:.1f}x, from {DG["roaring_ms_min"]:.3f} to '
           f'{DG["roaring_ms_max"]:.3f} ms, because it pays per run. The dense array '
           f'stays between {DG["dense_ms_min"]:.1f} and {DG["dense_ms_max"]:.1f} ms '
           'throughout, because it pays per element regardless of structure. Roaring '
           'is fastest at every point in this regime that we measured.]]')
    srows = []
    for s in SW:
        ra = f'{s["run_arrays_ms"]:.4f}' if s['run_arrays_ms'] is not None else 'n/a'
        srows.append([str(s['random_per_row']), f'{s["runs_per_row"]:.1f}',
                      f'{s["roaring_flat_ms"]:.3f}', f'{s["dense_ms"]:.3f}',
                      f'{s["packed_bitset_ms"]:.3f}', f'{s["csr_ms"]:.3f}', ra,
                      'Roaring' if s['roaring_wins'] else SHORT_LABEL[s['best_rival']]])
    b.table(f'[[Table 7. Composition cost against content-dependence at '
            f'n={CV["sweep_n"]:,}. The first row uses a row-invariant global operand; '
            'every later row replaces it with row-dependent random attention. '
            'The (start, length) encoding exists only in the first row, and the dense '
            'and bitset baselines lose their row broadcast after it. Note that '
            'fragmentation is essentially unchanged between the first two rows, yet '
            'the fastest representation changes. Lower is better.]]',
            ['Random/row', 'Runs/row', 'Roaring flat+opt', 'Dense bool',
             'Packed bitset', 'Sparse CSR', '(start,len) runs', 'Fastest'],
            srows, col_pt=7.5)
    b.figure(os.path.join(FIG, 'fig6_structure_crossover.png'),
             '[[Figure 4. Composition cost versus content-dependence, log scale. The '
             '(start, length) encoding appears as a single point because it exists '
             'only while every component is row-invariant. The upper axis shows that '
             'mask fragmentation is unchanged across the transition, so the reversal '
             'is not explained by run count; what changes is that the second operand '
             'stops being the same for every row. Roaring is the only representation '
             'whose cost is governed by the number of runs it stores rather than by '
             'n², which is why it degrades gracefully thereafter.]]')

    b.body('[[We report one clear negative alongside these. In-place mutation — adding '
           'a single global sink token to an already-built mask — favours the dense '
           f'representation decisively: {MUT[-1]["dense_ms"]:.3f} ms against '
           f'{MUT[-1]["roaring_ms"]:.3f} ms for Roaring at n={MUT[-1]["seq_len"]:,}, a '
           f'factor of about {MUT[-1]["dense_speedup"]:.1f}. A column write into a '
           'dense array is a single strided memory operation, whereas Roaring must '
           'touch every affected container. Workloads dominated by incremental '
           'mutation rather than bulk composition should not use this '
           'representation.]]')

    # ---- 5.6 window ablation --------------------------------------------- #
    b.subsection('5.6 Window-Size Ablation')
    b.body('[[The compression ratio is invariant to window size. Because that '
           'invariance is a property of the representation rather than of training, '
           f'it can be measured directly. Table 8 varies the window at '
           f'n={WA[0]["seq_len"]:,}, spanning mask densities from '
           f'{WA[0]["density"]*100:.1f}% to {WA[-1]["density"]*100:.0f}%.]]')
    wrows = []
    for r in WA:
        wrows.append([str(r['window']) if r['window'] is not None else 'full (causal)',
                      f'{r["density"]*100:.1f}%',
                      kb(r['sizes_bytes']['roaring_flat_opt']),
                      kb(r['sizes_bytes']['zstd']),
                      f'{r["compose_flat_ms"]:.3f}', f'{r["compose_dense_ms"]:.3f}',
                      f'{r["speedup"]:.1f}x'])
    b.table(f'[[Table 8. Window-size ablation at n={WA[0]["seq_len"]:,}, sizes in KB '
            'and composition times in milliseconds. Held constant: sequence length, '
            'causal structure, and the composition workload of Table 5. The '
            f'window={CFG["window_size"]} row is not a separate measurement: it is the '
            f'n={REF:,} row of Table 5.]]',
            ['Window', 'Mask density', 'Roaring flat+opt', 'zstd(bitset)',
             'Compose Roaring', 'Compose dense', 'Speedup'], wrows, col_pt=8.0)
    b.body('[[The Roaring representation is exactly invariant: '
           f'{kb(WA[0]["sizes_bytes"]["roaring_flat_opt"])} KB at every window, because '
           'each row remains a single run irrespective of its length, and a Run '
           'Container stores a run in constant space. The comparison with zstd is more '
           'informative than the invariance itself. The compressed byte stream grows '
           f'with density, from {kb(WA[0]["sizes_bytes"]["zstd"])} KB at a '
           f'{WA[0]["density"]*100:.1f}% dense mask to '
           f'{kb(WA[-1]["sizes_bytes"]["zstd"])} KB at full causal attention, so '
           'Roaring\'s size disadvantage narrows from roughly '
           f'{WA[0]["sizes_bytes"]["roaring_flat_opt"]/WA[0]["sizes_bytes"]["zstd"]:.0f}x '
           f'to {WA[-1]["sizes_bytes"]["roaring_flat_opt"]/WA[-1]["sizes_bytes"]["zstd"]:.1f}x '
           'as the mask densifies — the same direction as the composition result, for '
           'the same reason: Roaring pays for runs, not for elements.]]')

    # ---- 5.7 convergence -------------------------------------------------- #
    b.subsection('5.7 Training Convergence')
    b.table('[[Table 9. Training convergence on WikiText-103 '
            f'({NP_:,} parameters, CUDA, {MAIN["meta"]["steps"]:,} steps, '
            f'{MAIN["meta"]["seeds"]} seeds). Loss is validation cross-entropy in nats, '
            'perplexity in parentheses. Pruning is disabled throughout this table and '
            'reported separately in Table 10.]]',
            ['Metric', 'Vanilla', 'Roaring'],
            [['Final validation loss (ppl)',
              f'{V_LOSS:.2f} ± {V_SD:.2f} ({np.exp(V_LOSS):.1f})',
              f'{R_LOSS:.2f} ± {R_SD:.2f} ({np.exp(R_LOSS):.1f})'],
             ['Average step time',
              f'{V_MS:.1f} ± {V_MSSD:.2f} ms', f'{R_MS:.1f} ± {R_MSSD:.2f} ms'],
             ['Peak training VRAM', f'{V_VRAM:,.0f} MB', f'{R_VRAM:,.0f} MB'],
             [f'[[Mask memory at seq_len={SEQ} (per-row layout, as run)]]',
              f'[[{DENSE_KB:,.1f} KB]]', f'[[{RUN_KB:.1f} KB]]'],
             ['[[Mask size vs dense float32]]', '[[1.0x]]',
              f'[[{DENSE_KB/RUN_KB:.1f}x]]']],
            note='[[Note. The mask-memory row reports the layout this training run '
                 'actually used, which is the per-row layout. The flat, run-optimised '
                 f'layout adopted in Section 3.1 stores the same n={SEQ} mask in '
                 f'{sz(SEQ, "roaring_flat_opt"):,} bytes '
                 f'({DENSE_KB/FLAT_KB:.0f}x versus dense float32); Table 3 reports '
                 'both. The run predates that change and has not been repeated, '
                 'because the layout affects the host-side footprint of the mask and '
                 'not the realized tensor the model consumes.]]')
    b.body(f'Across {MAIN["meta"]["seeds"]} seeds the lossless Roaring optimizations '
           f'leave final validation loss statistically indistinguishable from the '
           f'baseline (Welch t-test p={EQ["welch_p"]:.2f}; Mann-Whitney U '
           f'p={EQ["mannwhitney_p"]:.2f}; both fail to reject equality). Peak training '
           f'VRAM is identical because masks are realized as dense tensors on the '
           f'accelerator: the Roaring representation reduces host memory and stored '
           f'mask footprint, not GPU VRAM.')
    b.body('[[The step-time difference requires explanation. The gap is statistically '
           f'significant (Welch p={STEP_P:.3f}), but it is an artifact of the baseline '
           'rather than a benefit of Roaring. Our vanilla implementation rebuilds the '
           'dense mask on every forward pass using a seq_len-iteration loop, whereas '
           'the Roaring implementation builds the mask once and caches the realized '
           'tensor. Caching the dense mask yields the same gain without any Roaring '
           f'involvement. The variance confirms the mechanism: the vanilla standard '
           f'deviation is {V_MSSD:.2f} ms against {R_MSSD:.2f} ms for the cached path. '
           'We therefore claim no step-time advantage, only the absence of a '
           'step-time penalty.]]')
    b.figure(os.path.join(FIG, 'fig4_convergence.png'),
             'Figure 5. Validation loss on WikiText-103, mean ± standard deviation '
             f'over {MAIN["meta"]["seeds"]} seeds. The curves overlap throughout.')
    b.figure(os.path.join(FIG, 'fig5_step_time.png'),
             '[[Figure 6. Per-step training time on CUDA with standard-error bars. The '
             'axis labels state the mechanism: the baseline rebuilds its mask each '
             'step while the Roaring path reuses a cached tensor. The difference is '
             'therefore attributable to caching, not to the mask representation.]]')

    # ---- 5.8 pruning ------------------------------------------------------ #
    b.subsection('5.8 Pruning Ablation')
    paired = ('[[The vanilla arm is the identical seeded run reported in Table 9, so '
              'the two arms differ only in pruning.]] ' if PAIRED_BASELINE else '')
    b.table('[[Table 10. Weight pruning ablation, reported separately from Table 9 '
            'because pruning is lossy and is disabled in the equivalence '
            'comparison.]] ' + paired,
            ['Metric', 'Vanilla', 'Roaring + 30% pruning'],
            [['Final validation loss (ppl)',
              f'{PE["vanilla_mean"]:.2f} ({np.exp(PE["vanilla_mean"]):.1f})',
              f'{PE["roaring_mean"]:.2f} ({np.exp(PE["roaring_mean"]):.1f})']])
    b.body(f'Removing 30% of weights raises perplexity from '
           f'{np.exp(PE["vanilla_mean"]):.1f} to {np.exp(PE["roaring_mean"]):.1f} '
           f'(Welch p={PE["welch_p"]:.4f}). This is a quality/sparsity tradeoff, not a '
           f'free optimization.')

    # ---- 5.9 hydration and host memory ------------------------------------ #
    b.subsection('[[5.9 Hydration and Host Memory]]')
    b.body('[[Table 11 profiles the cost of realizing a mask on the accelerator and '
           'the host-side footprint it is realized from, measured on the training rig. '
           'The realized tensor is what occupies VRAM and it is unchanged by the '
           'representation; what the representation changes is the host-side cost of '
           f'keeping the mask, which at n=4,096 is '
           f'{HYD_MAXN_RATIO:.0f}x smaller than the tensor produced from it. Because '
           'realized masks are cached per sequence length, the one-time hydration cost '
           f'at the training length of {SEQ} is '
           f'{HYD[SEQ]["cold_total_ms"]:.1f} ms, or '
           f'{HYD[SEQ]["amortized_per_step_us"]:.1f} microseconds per step over '
           f'{MAIN["meta"]["steps"]:,} steps — under 0.01% of a {R_MS:.0f} ms training '
           'step. That amortization is also the reason the composition advantage of '
           'Section 5.4 does not surface end-to-end in our own training run: we cache '
           'the mask, so we compose it once.]]')
    hrows = []
    for n in sorted(HYD):
        h = HYD[n]
        dense_b = n * n * 4
        roar_b = h['roaring_cpu_kb'] * 1024
        flat = f'{sz(n, "roaring_flat_opt")/1024:,.1f}' if n in MS else '—'
        hrows.append([f'{n:,}', f'{dense_b/1024/1024:,.2f}',
                      f'{h["roaring_cpu_kb"]:,.1f}', flat,
                      f'{dense_b/roar_b:,.0f}x', f'{h["cold_total_ms"]:.1f}',
                      f'{h["amortized_per_step_us"]:.1f}'])
    b.table('[[Table 11. Mask hydration measured on the training GPU. "Realized dense" '
            'is the tensor the model consumes on the device; the two Roaring columns '
            'are the host-side representation it is built from, in the per-row layout '
            'the run used and in the flat run-optimised layout of Section 3.1. The '
            'ratio column compares realized dense against the per-row layout. '
            'Hydration is paid once per sequence length and then cached.]]',
            ['Seq len', 'Realized dense (MiB)', 'Roaring host, per-row (KiB)',
             'Roaring host, flat+opt (KiB)', 'Ratio', 'Cold hydration (ms)',
             'Amortized (µs/step)'], hrows, col_pt=7.5)
    b.figure(os.path.join(FIG, 'fig3_mask_creation_time.png'),
             '[[Figure 7. Mask construction cost versus sequence length, excluded from '
             'the composition timings of Tables 5 to 7 and reported here separately; '
             'the flat Roaring layout is also the cheapest to build.]]')

    # ---------------------------------------------------------------- discussion
    b.section('6. Discussion')
    b.subsection('[[6.1 A Decision Rule Rather Than a Verdict]]')
    b.body('[[The results reduce to a rule an implementer can apply without repeating '
           'our experiments. If masks are stored and not operated on, use a byte '
           'compressor: it is one to two orders of magnitude smaller than any '
           'representation here, and decompression is paid only when the mask is '
           'finally needed. If masks are composed but the pattern is fixed and '
           'position-determined — causal windows, padding, block-diagonal document '
           'boundaries — then every component is row-invariant, the composition has a '
           'closed form over two integer arrays, and that form beats every set '
           'structure including Roaring by roughly an order of magnitude. If any '
           'component depends on content — learned or random attention, dynamic '
           'pruning, retrieval-selected tokens — the closed form does not exist, the '
           'dense and bitset baselines lose their row broadcast, and Roaring is the '
           'fastest representation left, by a margin that grows with sequence length. '
           'The distinction is categorical and can be settled by reading the '
           'definition of a sparsity pattern rather than by benchmarking it: ask '
           'whether any component varies from one query row to the next.]]')
    b.body('[[Two caveats bound that rule. First, because realized masks are cached per '
           'sequence length, composition happens rarely in our own training setup, so '
           'the advantage of Section 5.4 does not translate into a measurable '
           'end-to-end speedup at the scale we train at; it matters where masks are '
           'rebuilt frequently or where n is large enough that the dense form no '
           'longer fits. Second, all composition results are single-machine CPU '
           'measurements of specific library implementations — pyroaring, numpy and '
           'SciPy — and a different implementation of any baseline could move the '
           'threshold, though not the mechanism behind it.]]')

    b.subsection('6.2 The CPU–GPU Bridge')
    b.body('The current implementation realizes masks as dense tensors before GPU '
           f'transfer. Peak training VRAM is accordingly identical ({V_VRAM:,.0f} vs '
           f'{R_VRAM:,.0f} MB). Extending the benefit into the VRAM domain requires a '
           'native Roaring-to-CSR GPU kernel consuming the serialized bitmap directly '
           'on device. '
           '[[This is not a straightforward port. Roaring containers are heterogeneous '
           'and variable-length, so a warp processing a contiguous key range may span '
           'array, bitmap and run containers with different decode paths, producing '
           'branch divergence within the warp and irregular, unpredictable memory '
           'access. A performant kernel would likely need a container-type-major '
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
                  'magnitude Top-K produces. The contribution rests on operability '
                  'under irregular structure, not on footprint.]]')
    b.numbered(3, '[[Composition on row-invariant masks. Where every component of the '
                  'mask is determined by position, a purpose-built (start, length) '
                  f'encoding is up to {CV["structured_deficit_at_max_n"]:.0f}x faster '
                  'than Roaring (Table 5). We report this as a negative result and '
                  'recommend against Roaring in that regime.]]')
    b.numbered(4, '[[Validation sequence length. The motivating figure of 40 GB assumes '
                  f'n=100,000, but training ran at n={SEQ}, where a dense float32 mask '
                  f'is only {DENSE_KB:,.1f} KB and there is no memory pressure '
                  f'whatsoever. Scaling results are measured up to n={MAXN:,}; '
                  'everything beyond that is extrapolation.]]')
    b.numbered(5, 'Model scale. Convergence is validated at '
                  f'{NP_/1e6:.1f}M parameters. Validation at 100M+ parameters and on '
                  'multi-node clusters remains future work. '
                  '[[No larger training run was performed for this revision; the '
                  'scale-related claims were withdrawn instead, because the compute '
                  'required was not available to us.]]')
    b.numbered(6, '[[Distributed communication. All training was performed on a single '
                  'GPU, so no All-Reduce traffic existed to compress. Any '
                  'communication-compression figure in this paper is analytic, '
                  'computed by logging the Top-K gradient index set produced per step '
                  'and measuring the serialized footprint of each encoding offline. '
                  'End-to-end All-Reduce timing is not demonstrated.]]')
    b.numbered(7, '[[Mutation. In-place mutation of an existing mask is substantially '
                  'slower than the dense equivalent (Section 5.5).]]')
    b.numbered(8, '[[Ablation scope. We characterise window size and sparsity at the '
                  'level of the representation, but do not report how final training '
                  'loss varies across window sizes or sparsity levels. That would '
                  'require additional training runs beyond our compute budget, and we '
                  'make no claim on it.]]')
    b.numbered(9, '[[Index and pattern distributions. The gradient-index results assume '
                  'magnitude Top-K over an i.i.d. Gaussian gradient, and the '
                  'unstructured composition uses synthetic random attention. Whether '
                  'the index sets and mask patterns of a real large-scale run fall '
                  'above or below the threshold of Section 5.5 is exactly the question '
                  'our rule makes testable, and we have not tested it at scale.]]')
    b.numbered(10, '[[Implementation dependence. Composition timings compare specific '
                   'libraries (pyroaring, numpy, SciPy) on one CPU. A faster '
                   'implementation of any baseline would move the margins, though '
                   'the asymptotic argument — Roaring costs scale with runs, dense '
                   'costs with n² — would survive.]]')

    # ---------------------------------------------------------------- conclusion
    b.section('8. Conclusion')
    b.body('[[Measured against baselines that exploit the known structure, Roaring '
           'Bitmaps are not a compression mechanism for Transformer training '
           'metadata: byte compressors store these mask patterns one to two orders of '
           'magnitude more compactly, and a delta-plus-varint encoder is smaller on '
           'diffuse gradient indices at every configuration we measured. Nor are they '
           'the right structure for composing masks assembled purely from '
           'position-determined components, where the composition has a closed form '
           'and a vectorised (start, length) encoding is up to '
           f'{CV["structured_deficit_at_max_n"]:.0f}x faster.]]')
    b.body('[[What they are is the fastest representation that still works once the '
           'mask stops being a function of position alone. A single content-dependent '
           'component is enough: it makes the per-row (start, length) form '
           'inexpressible and takes the row broadcast away from the dense and bitset '
           'baselines at the same time, leaving a choice between a dense array, a '
           'bitset, a general sparse matrix and a compressed set. Roaring is '
           f'{CV["unstructured_margin_min"]:.1f}x to '
           f'{CV["unstructured_margin_max"]:.1f}x faster than the best of the others '
           f'and up to {CV["unstructured_vs_dense_max"]:.0f}x faster than dense, '
           'because its cost tracks the number of runs it stores rather than n². The '
           'transition between the two regimes is categorical rather than gradual — '
           'fragmentation is unchanged across it — which makes the choice of '
           'representation a property an implementer can read off a sparsity pattern '
           'rather than a matter of preference. We also found that layout choice '
           'within Roaring — one bitmap per row versus a flat, run-optimised bitmap — '
           'accounts for much of the overhead that might otherwise be attributed to '
           'the data structure itself.]]')
    b.body('On real WikiText-103 data the lossless optimizations impose no '
           'statistically measurable penalty on model quality, while weight pruning '
           'trades quality for sparsity. The savings accrue to the mask representation '
           'and host memory, not to accelerator VRAM, which is unchanged under the '
           'dense-realization design used here.')

    b.section('9. Future Work')
    b.numbered(1, 'Roaring-to-CSR GPU kernel: decompress directly into sparse CSR in '
                  'VRAM, addressing the warp-divergence issues outlined in '
                  'Section 6.2.')
    b.numbered(2, '[[Measuring the run structure of the sparsity patterns used by '
                  'production long-context models, which determines on which side of '
                  'the Section 5.5 threshold each falls.]]')
    b.numbered(3, '[[Dynamic and content-dependent sparsity end to end, where masks '
                  'are rebuilt every step rather than cached and the composition '
                  'advantage would surface as wall-clock training time.]]')
    b.numbered(4, 'Large-scale validation: 100M+ parameter models and multi-node '
                  'clusters with real All-Reduce measurement.')
    b.numbered(5, 'Combined Roaring pattern planning with FlashAttention execution.')

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

HYD_MAXN = max(HYD)
HYD_MAXN_RATIO = (HYD_MAXN * HYD_MAXN * 4) / (HYD[HYD_MAXN]['roaring_cpu_kb'] * 1024)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for hl, tmpl, name in [
            (True, False, 'Roaring-Transformer-v4-Highlighted.docx'),
            (False, False, 'Roaring-Transformer-v4-Clean.docx'),
            (True, True, 'Roaring-Transformer-v4-Template-Highlighted.docx'),
            (False, True, 'Roaring-Transformer-v4-Template-Clean.docx')]:
        path = os.path.join(OUT_DIR, name)
        build(hl, use_template=tmpl).save(path)
        print(f'wrote {os.path.relpath(path, os.path.join(HERE, ".."))}')


if __name__ == '__main__':
    main()
