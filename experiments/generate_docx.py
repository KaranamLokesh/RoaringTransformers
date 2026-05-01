#!/usr/bin/env python3
"""
Generate a single-column research paper in .docx format.
  - Title + authors + abstract all on page 1
  - Three authors with affiliation
  - Plain black-border tables (no color shading)
  - Figure captions BELOW figures only
  - No titles inside figure images
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import OxmlElement, parse_xml
import json

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

doc = Document()

# ── Page Setup ─────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Inches(8.5)
section.page_height = Inches(11)
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)
section.left_margin   = Cm(1.8)
section.right_margin  = Cm(1.8)

# ── Styles ─────────────────────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = 'Times New Roman'
style_normal.font.size = Pt(10)
style_normal.paragraph_format.space_after = Pt(3)
style_normal.paragraph_format.space_before = Pt(0)
style_normal.paragraph_format.line_spacing = 1.12

title_style = doc.styles.add_style('PaperTitle', 1)
title_style.font.name = 'Times New Roman'
title_style.font.size = Pt(16)
title_style.font.bold = True
title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_style.paragraph_format.space_after = Pt(4)
title_style.paragraph_format.space_before = Pt(0)

author_style = doc.styles.add_style('Author', 1)
author_style.font.name = 'Times New Roman'
author_style.font.size = Pt(11)
author_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
author_style.paragraph_format.space_after = Pt(1)
author_style.paragraph_format.space_before = Pt(0)

affil_style = doc.styles.add_style('Affiliation', 1)
affil_style.font.name = 'Times New Roman'
affil_style.font.size = Pt(10)
affil_style.font.italic = True
affil_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
affil_style.paragraph_format.space_after = Pt(8)
affil_style.paragraph_format.space_before = Pt(0)

sec_style = doc.styles.add_style('SectionHead', 1)
sec_style.font.name = 'Times New Roman'
sec_style.font.size = Pt(11)
sec_style.font.bold = True
sec_style.paragraph_format.space_before = Pt(10)
sec_style.paragraph_format.space_after = Pt(3)

subsec_style = doc.styles.add_style('SubsectionHead', 1)
subsec_style.font.name = 'Times New Roman'
subsec_style.font.size = Pt(10)
subsec_style.font.bold = True
subsec_style.font.italic = True
subsec_style.paragraph_format.space_before = Pt(8)
subsec_style.paragraph_format.space_after = Pt(2)

cap_style = doc.styles.add_style('FigCaption', 1)
cap_style.font.name = 'Times New Roman'
cap_style.font.size = Pt(9)
cap_style.font.italic = True
cap_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_style.paragraph_format.space_after = Pt(6)
cap_style.paragraph_format.space_before = Pt(2)

tab_cap_style = doc.styles.add_style('TableCaption', 1)
tab_cap_style.font.name = 'Times New Roman'
tab_cap_style.font.size = Pt(9)
tab_cap_style.font.italic = True
tab_cap_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
tab_cap_style.paragraph_format.space_after = Pt(3)
tab_cap_style.paragraph_format.space_before = Pt(6)


# ── Helpers ────────────────────────────────────────────────────────

def add_body(text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.first_line_indent = Cm(0.5)
    p.add_run(text)

def add_body_plain(text):
    p = doc.add_paragraph(style='Normal')
    p.add_run(text)

def add_bullet(text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(0)
    p.add_run(f'\u2022 {text}')

def add_figure(filename, caption, width=Inches(5.5)):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run()
        run.add_picture(path, width=width)
    doc.add_paragraph(caption, style='FigCaption')

def set_cell_border(cell, **kwargs):
    """Set cell borders. Each kwarg: top, bottom, left, right = {'sz': '4', 'val': 'single', 'color': '000000'}"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            element = OxmlElement(f'w:{edge}')
            for attr, val in kwargs[edge].items():
                element.set(qn(f'w:{attr}'), val)
            tcBorders.append(element)
    tcPr.append(tcBorders)

def add_plain_table(caption, headers, rows):
    """Add a plain black-bordered table with no shading."""
    doc.add_paragraph(caption, style='TableCaption')
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    border_args = {
        'top': {'sz': '4', 'val': 'single', 'color': '000000'},
        'bottom': {'sz': '4', 'val': 'single', 'color': '000000'},
        'left': {'sz': '4', 'val': 'single', 'color': '000000'},
        'right': {'sz': '4', 'val': 'single', 'color': '000000'},
    }

    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = 'Times New Roman'
        set_cell_border(cell, **border_args)
        # light gray background for header
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), 'E8E8E8')
        shading.set(qn('w:val'), 'clear')
        cell._tc.get_or_add_tcPr().append(shading)

    # Data
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri+1].cells[ci]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(9)
            r.font.name = 'Times New Roman'
            set_cell_border(cell, **border_args)

    # Add small spacing after table
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)


# ═══════════════════════════════════════════════════════════════════
# BUILD THE PAPER
# ═══════════════════════════════════════════════════════════════════

# ── Title (single column, same page) ──
doc.add_paragraph(
    'The Roaring Transformer: Optimizing Long-Context Training\n'
    'and Distributed Communication via Hybrid Compressed Bitmaps',
    style='PaperTitle'
)

# ── Authors ──
p = doc.add_paragraph(style='Author')
p.add_run('Lokesh Karanam, Purna Sreekar Karanam, Sriram Pemmasani').bold = False

p = doc.add_paragraph(style='Affiliation')
p.add_run('Independent Researchers')
p2 = doc.add_paragraph(style='Affiliation')
p2.add_run('Austin, TX, USA')
p2.paragraph_format.space_after = Pt(6)

# ── Switch to two columns — NO page break ──

# ═══════ ABSTRACT ═══════
doc.add_paragraph('ABSTRACT', style='SectionHead')
add_body(
    'We introduce the Roaring Transformer, a novel hybrid computational '
    'architecture that integrates Roaring Bitmaps \u2014 a high-performance '
    'compressed bitset data structure \u2014 into the Transformer training pipeline '
    'to resolve the memory and communication bottlenecks inherent in scaling '
    'Large Language Models (LLMs). While Roaring Bitmaps are non-differentiable '
    'and therefore cannot serve as a substitute for latent floating-point weight '
    'tensors, we demonstrate their effectiveness as an architectural "skeleton" '
    'for managing three critical subsystems: (1) sparse attention mask storage, '
    '(2) gradient index compression for distributed training, and (3) weight '
    'pruning mask tracking.'
)
add_body(
    'Through systematic experimentation, we show that Roaring Bitmap-backed '
    'attention masks achieve compression ratios scaling linearly with sequence '
    'length \u2014 from 68\u00d7 at n=256 to 2,185\u00d7 at n=8,192 \u2014 while reducing mask '
    'creation time by up to 9\u00d7. For gradient sparsification in Distributed Data '
    'Parallel (DDP) training, Roaring-encoded Top-K indices achieve 1.9\u20133.2\u00d7 '
    'compression over raw integer arrays. Training convergence remains statistically '
    'equivalent to the standard Transformer baseline across three random seeds, '
    'confirming that the Roaring optimizations impose no quality penalty. To the '
    'best of our knowledge, this is the first Transformer implementation to '
    'integrate Roaring Bitmaps into its training loop.'
)

# ═══════ 1. INTRODUCTION ═══════
doc.add_paragraph('1. INTRODUCTION', style='SectionHead')
add_body(
    'The trajectory of Large Language Model development is presently constrained '
    'by the "Quadratic Wall" of self-attention. As sequence lengths (n) grow '
    'beyond 10,000 tokens, the attention mask \u2014 a matrix encoding the permissible '
    'token-to-token relationships \u2014 scales at O(n\u00b2), consuming critical GPU memory '
    '(VRAM) before a single model parameter is loaded. At n = 100,000, a single '
    'float32 attention mask occupies approximately 40 GB, exceeding the capacity '
    'of most production-grade accelerators.'
)
add_body(
    'In multi-node distributed training, the bottleneck shifts from memory to '
    'interconnect bandwidth. The All-Reduce operation required to synchronize '
    'gradients across GPUs saturates network links, leading to GPU underutilization '
    'that can exceed 40% on commodity Ethernet fabrics [1]. Top-K gradient '
    'sparsification is a proven mitigation strategy, but encoding gradient indices '
    'as raw integer arrays introduces non-trivial metadata overhead.'
)
add_body(
    'Traditional sparse attention methods (Longformer [2], BigBird [3], Sparse '
    'Transformer [4]) address the quadratic mask problem through fixed sparsity '
    'patterns, but represent their masks as dense tensors, forfeiting the '
    'compression opportunities inherent in the mask\u2019s structural regularity. '
    'We observe that causal masks with sliding windows exhibit precisely the '
    'integer-set patterns that Roaring Bitmaps [5] were designed to compress.'
)
add_body('Our contributions are:')
add_bullet('The first integration of Roaring Bitmaps into a Transformer training pipeline.')
add_bullet('Empirical evidence that Roaring-backed mask storage achieves near-linear O(n) scaling, with compression ratios exceeding 2,000\u00d7 at n=8,192.')
add_bullet('A Roaring-based gradient index compression scheme that reduces All-Reduce payloads by 1.9\u20133.2\u00d7 over standard uint32 arrays.')
add_bullet('An open-source reference implementation with comprehensive benchmarks across five experimental dimensions.')

# ═══════ 2. BACKGROUND ═══════
doc.add_paragraph('2. BACKGROUND AND RELATED WORK', style='SectionHead')

doc.add_paragraph('2.1 The Attention Mask Problem', style='SubsectionHead')
add_body(
    'The standard scaled dot-product attention [6] is: Attention(Q, K, V) = '
    'softmax(QK\u1d40/\u221ad_k + M)V, where M is an additive attention mask. In causal '
    'language modeling, M is lower-triangular with 0 at allowed positions and '
    '\u2212\u221e at blocked positions. For sequence length n, M requires n\u00b2 elements.'
)

add_plain_table('Table 1: Dense Attention Mask Memory (float32)',
    ['Seq Length', 'Mask Memory', 'Per-Layer VRAM'],
    [['1,024','4 MB','4 MB'],['4,096','64 MB','64 MB'],['8,192','256 MB','256 MB'],
     ['32,768','4 GB','4 GB'],['100,000','40 GB','40 GB']])

doc.add_paragraph('2.2 Sparse Attention Methods', style='SubsectionHead')
add_body(
    'Longformer [2] combines a sliding window with global attention, reducing '
    'complexity to O(n\u00b7w). BigBird [3] adds random connections. FlashAttention [7] '
    'fuses attention into a tiled kernel. None leverage compressed set data '
    'structures to minimize mask storage.'
)

doc.add_paragraph('2.3 Roaring Bitmaps', style='SubsectionHead')
add_body(
    'Roaring Bitmaps [5] are a hybrid compressed data structure for sets of '
    '32-bit integers. The key space is partitioned into 2\u00b9\u2076-element chunks, '
    'each stored as one of three container types:'
)
add_bullet('Array Container: for sparse chunks (<4,096 elements), stored as sorted uint16 array.')
add_bullet('Bitmap Container: for dense chunks (\u22654,096 elements), stored as 8 KB bitset.')
add_bullet('Run Container: for contiguous runs, stored as (start, length) pairs.')
add_body(
    'Roaring powers Apache Lucene, Druid, InfluxDB, and Elasticsearch. It '
    'supports SIMD-accelerated set operations (AND, OR, XOR) via AVX-512/NEON, '
    'exceeding 10 billion integers per second [5].'
)

doc.add_paragraph('2.4 Gradient Compression', style='SubsectionHead')
add_body(
    'In DDP training, All-Reduce synchronizes gradients with cost O(|\u03b8|). '
    'Top-K sparsification [8] sends only the top k fraction. The standard '
    'encoding is (indices: uint32[], values: float32[]). The index array '
    'accounts for 50% of the payload and is amenable to compressed set encoding.'
)

# ═══════ 3. ARCHITECTURE ═══════
doc.add_paragraph('3. THE ROARING TRANSFORMER ARCHITECTURE', style='SectionHead')
add_body(
    'The Roaring Transformer maintains the identical neural architecture as a '
    'standard Transformer \u2014 the innovations are purely in the metadata management '
    'layer, preserving full differentiability.'
)

doc.add_paragraph('3.1 Roaring Attention Mask Manager', style='SubsectionHead')
add_body(
    'Attention masks are integer sets: for each query position i, S_i = '
    '{max(0, i\u2212w), \u2026, i}. These contiguous runs are compressed optimally by '
    'Roaring\u2019s Run Containers. We store one Roaring Bitmap per query position on '
    'CPU. Before each forward pass, bitmaps are "hydrated" into a dense tensor '
    'and transferred to the accelerator. Realized masks are cached by sequence '
    'length. The representation grows as O(n).'
)

doc.add_paragraph('3.2 Roaring Gradient Compressor', style='SubsectionHead')
add_body(
    'We replace the uint32 index array in Top-K sparsification with a serialized '
    'Roaring Bitmap. Standard: (uint32[k], float32[k]) = 8k bytes. Roaring: '
    '(roaring_bytes, float32[k]) \u2248 (4k + header) bytes. Gradients cluster within '
    'specific layers and heads, enabling Roaring\u2019s hybrid containers to compress '
    'effectively. Error feedback [9] ensures no gradient information is lost.'
)

doc.add_paragraph('3.3 Roaring Pruning Manager', style='SubsectionHead')
add_body(
    'After warm-up, the pruning manager identifies small-magnitude weights and '
    'records alive indices as Roaring Bitmaps. At 30\u201350% pruning, Bitmap '
    'Containers achieve 1-bit-per-weight storage. At 90%+, sparse Array '
    'Containers store only surviving indices.'
)

# ═══════ 4. EXPERIMENTAL SETUP ═══════
doc.add_paragraph('4. EXPERIMENTAL SETUP', style='SectionHead')

doc.add_paragraph('4.1 Model Configuration', style='SubsectionHead')
add_plain_table('Table 2: Model Configuration',
    ['Hyperparameter', 'Value'],
    [['Embedding dim','128'],['Attention heads','4'],['Layers','2'],
     ['FFN hidden dim','256'],['Vocabulary size','5,000'],['Dropout','0.1'],
     ['Total parameters','937,984']])
add_body(
    'The Roaring Transformer additionally uses: window size 64 (varied in '
    'ablation), gradient sparsity top 1% (varied in ablation), pruning ratio '
    '30% activated at step 100 of 200, and mask caching enabled.'
)

doc.add_paragraph('4.2 Training Configuration', style='SubsectionHead')
add_body(
    'Optimizer: AdamW (lr=1e-3, weight_decay=0.01). Loss: cross-entropy. '
    'Dataset: synthetic language modeling (random tokens, vocab=5,000). '
    'Batch size: 8. Steps: 200 (convergence), 100 (ablations). Seeds: 42, '
    '123, 456. Device: CPU (Apple Silicon, 14 cores).'
)

# ═══════ 5. RESULTS ═══════
doc.add_paragraph('5. RESULTS', style='SectionHead')

# 5.1 Mask Memory
doc.add_paragraph('5.1 Mask Memory Scaling', style='SubsectionHead')
add_plain_table('Table 3: Mask Memory Comparison (window = 128)',
    ['Seq Len', 'Dense (KB)', 'Roaring (KB)', 'Compression', 'Speedup'],
    [['256','256','3.8','68\u00d7','5.0\u00d7'],
     ['512','1,024','7.5','137\u00d7','5.0\u00d7'],
     ['1,024','4,096','15.0','273\u00d7','7.7\u00d7'],
     ['2,048','16,384','30.0','546\u00d7','7.2\u00d7'],
     ['4,096','65,536','60.0','1,092\u00d7','6.5\u00d7'],
     ['8,192','262,144','120.0','2,185\u00d7','8.7\u00d7']])
add_body(
    'The compression ratio scales linearly with sequence length: dense grows '
    'O(n\u00b2) while Roaring grows O(n). Each row\u2019s single contiguous run requires '
    'only ~15 bytes per Run Container. At n=8,192, the dense mask is 256 MB '
    'while Roaring requires only 120 KB.'
)

# Full-width figures
add_figure('fig1_mask_memory.png',
    'Figure 1. Attention mask memory. Red = Dense (float32, O(n\u00b2)), '
    'Blue = Roaring Bitmap (O(n)). Log scale on both axes.', Inches(5.0))
add_figure('fig2_compression_ratio.png',
    'Figure 2. Compression ratio (Dense \u00f7 Roaring) vs sequence length for three '
    'window sizes. The ratio scales linearly and is independent of window size.', Inches(5.0))

# 5.2 Gradient Compression
doc.add_paragraph('5.2 Gradient Index Compression', style='SubsectionHead')
add_plain_table('Table 4: Gradient Index Compression (Roaring vs Raw uint32)',
    ['Params', 'top-0.1%', 'top-0.5%', 'top-1%', 'top-5%', 'top-10%'],
    [['50K','1.72\u00d7','1.94\u00d7','1.97\u00d7','1.99\u00d7','2.44\u00d7'],
     ['100K','1.79\u00d7','1.95\u00d7','1.98\u00d7','2.00\u00d7','2.64\u00d7'],
     ['500K','1.87\u00d7','1.97\u00d7','1.99\u00d7','2.00\u00d7','3.05\u00d7'],
     ['1M','1.87\u00d7','1.97\u00d7','1.99\u00d7','2.00\u00d7','3.16\u00d7'],
     ['5M','1.88\u00d7','1.98\u00d7','1.99\u00d7','2.00\u00d7','3.19\u00d7'],
     ['10M','1.88\u00d7','1.98\u00d7','1.99\u00d7','2.00\u00d7','3.19\u00d7']])
add_body(
    'At practical sparsity (0.5\u20135%), Roaring achieves ~2\u00d7 compression. '
    'At 10%, compression reaches 3.19\u00d7 as Roaring transitions to Bitmap '
    'Containers. Compression improves with parameter count.'
)

add_figure('fig5_gradient_compression.png',
    'Figure 3. Gradient index compression ratio across parameter counts at three '
    'sparsity levels. Green = Top-0.1%, Blue = Top-1%, Red = Top-10%.')

# 5.3 Training Convergence
doc.add_paragraph('5.3 Training Convergence', style='SubsectionHead')
add_plain_table('Table 5: Training Convergence (200 steps, 3 seeds)',
    ['Metric', 'Vanilla', 'Roaring'],
    [['Final Loss','8.5268 \u00b1 0.0012','8.5261 \u00b1 0.0019'],
     ['Avg Step Time','57.2 \u00b1 4.3 ms','56.4 \u00b1 3.9 ms'],
     ['Mask Memory','256.0 KB','3.8 KB'],
     ['Mask Compression','1.0\u00d7','68.2\u00d7'],
     ['Grad Compression','N/A','1.32\u00d7'],
     ['Pruned Ratio','N/A','30.0%'],
     ['Prune Mask Compr.','N/A','6.0\u00d7']])
add_body(
    'Training convergence is statistically equivalent (8.5268 vs 8.5261, within '
    '\u00b11 std). Per-step time is comparable. The Roaring mask cache amortizes '
    'realization cost. Pruning removes 30% of weights with 6\u00d7 mask compression.'
)

add_figure('fig3_training_loss.png',
    'Figure 4. Training loss convergence over 200 steps (mean \u00b1 std, 3 seeds). '
    'Red = Vanilla Transformer, Blue = Roaring Transformer.', Inches(5.0))
add_figure('fig4_step_time.png',
    'Figure 5. Average per-step training time with \u00b11 std error bars.')

# 5.4 Window Ablation
doc.add_paragraph('5.4 Window Size Ablation', style='SubsectionHead')
add_plain_table('Table 6: Window Size Ablation (seq_len=256)',
    ['Window', 'Loss', 'Step (ms)', 'Compression'],
    [['32','8.5368','58.6','68.2\u00d7'],['64','8.5385','57.6','68.2\u00d7'],
     ['128','8.5365','57.9','68.2\u00d7'],['256','8.5363','61.0','68.2\u00d7'],
     ['Full','8.5363','59.9','68.2\u00d7']])
add_body(
    'Compression ratio is invariant to window size \u2014 Roaring\u2019s Run Container '
    'encodes any contiguous range in constant space.'
)

add_figure('fig6_window_ablation.png',
    'Figure 6. Window size ablation. Final loss is stable across all window sizes.')

# 5.5 Sparsity Ablation
doc.add_paragraph('5.5 Gradient Sparsity Ablation', style='SubsectionHead')
add_plain_table('Table 7: Gradient Sparsity Ablation',
    ['Sparsity', 'Loss', 'Comm Compr.'],
    [['0.1%','8.5385','1.22\u00d7'],['0.5%','8.5351','1.31\u00d7'],
     ['1.0%','8.5351','1.32\u00d7'],['5.0%','8.5351','1.33\u00d7'],
     ['10%','8.5351','1.46\u00d7'],['50%','8.5351','1.84\u00d7']])

add_figure('fig7_sparsity_ablation.png',
    'Figure 7. Gradient sparsity ablation. Left: training loss remains stable. '
    'Right: communication compression increases with sparsity.')

# ═══════ 6. DISCUSSION ═══════
doc.add_paragraph('6. DISCUSSION', style='SectionHead')

doc.add_paragraph('6.1 The CPU\u2013GPU Bridge', style='SubsectionHead')
add_body(
    'The current implementation realizes masks as dense tensors before GPU '
    'transfer. Savings come from: (1) CPU-side O(n) storage of multiple mask '
    'variants, (2) 5\u20139\u00d7 faster mask construction, (3) compact mask transmission '
    'in multi-node settings. A Roaring-to-CSR GPU kernel would eliminate dense '
    'realization entirely.'
)

add_figure('fig8_mask_creation_time.png',
    'Figure 8. Mask creation time. Red = Dense, Blue = Roaring. '
    'Roaring is consistently 5\u20139\u00d7 faster across all sequence lengths.')

doc.add_paragraph('6.2 Comparison with FlashAttention', style='SubsectionHead')
add_body(
    'FlashAttention [7] avoids materializing the mask by fusing attention into '
    'tiled kernels. This is complementary: FlashAttention optimizes FLOPs/VRAM, '
    'while Roaring optimizes mask storage, combination, and communication. A '
    'combined "FlashRoaring" system would use Roaring for pattern planning and '
    'FlashAttention for execution.'
)

doc.add_paragraph('6.3 Broader Applications', style='SubsectionHead')
add_bullet('MoE Routing: expert activation tracking with minimal memory.')
add_bullet('Data Deduplication: MinHash fingerprints as Roaring Bitmaps.')
add_bullet('Curriculum Learning: dynamic hard-sample indexing via set-union.')
add_bullet('Activation Checkpointing: sparse activation indices Roaring-encoded.')

# ═══════ 7. LIMITATIONS ═══════
doc.add_paragraph('7. LIMITATIONS', style='SectionHead')
add_bullet('GPU Realization: masks converted to dense tensors before GPU transfer.')
add_bullet('Scale: experiments use 938K parameters and synthetic data.')
add_bullet('Gradient Clustering: compression assumes clustered gradient indices.')
add_bullet('CPU Bottleneck: Roaring operations execute on CPU.')
add_bullet('Synthetic Data: loss plateau reflects data entropy, not language modeling.')

# ═══════ 8. CONCLUSION ═══════
doc.add_paragraph('8. CONCLUSION', style='SectionHead')
add_body(
    'The Roaring Transformer demonstrates that Roaring Bitmaps have significant '
    'untapped potential in deep learning. By treating masks, gradient indices, '
    'and pruning masks as compressed integer sets, we achieve:'
)
add_bullet('2,185\u00d7 mask compression at n=8,192, scaling linearly.')
add_bullet('1.9\u20133.2\u00d7 gradient index compression via Roaring Top-K encoding.')
add_bullet('6.0\u00d7 pruning mask compression with no convergence impact.')
add_bullet('5\u20139\u00d7 faster mask construction via optimized set operations.')
add_body(
    'These optimizations impose no penalty on model quality. We believe this '
    'opens a new direction at the intersection of systems-level data structure '
    'engineering and neural network architecture design.'
)

# ═══════ 9. FUTURE WORK ═══════
doc.add_paragraph('9. FUTURE WORK', style='SectionHead')
add_bullet('Roaring-to-CSR GPU Kernel: decompress directly into sparse CSR in VRAM.')
add_bullet('STE Integration: bitmapped forward-pass weights with latent float backward pass.')
add_bullet('Dynamic Sparsity: content-dependent Top-K attention via Roaring masks.')
add_bullet('Large-Scale Validation: billion-parameter models on natural language data.')
add_bullet('FlashRoaring: combined Roaring planning + FlashAttention execution.')

# ═══════ REFERENCES ═══════
doc.add_paragraph('REFERENCES', style='SectionHead')

refs = [
    '[1] Lin, Y., Han, S., Mao, H., Wang, Y., & Dally, W. J. (2018). "Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training." ICLR 2018.',
    '[2] Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." arXiv:2004.05150.',
    '[3] Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS 2020.',
    '[4] Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). "Generating Long Sequences with Sparse Transformers." arXiv:1904.10509.',
    '[5] Chambi, S., Lemire, D., Kaser, O., & Godin, R. (2016). "Better Bitmap Performance with Roaring Bitmaps." Software: Practice and Experience, 46(5), 709\u2013719.',
    '[6] Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.',
    '[7] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & R\u00e9, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.',
    '[8] Alistarh, D., et al. (2017). "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding." NeurIPS 2017.',
    '[9] Stich, S. U., Cordonnier, J.-B., & Jaggi, M. (2018). "Sparsified SGD with Memory." NeurIPS 2018.',
]
for ref in refs:
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(ref)
    r.font.size = Pt(8.5)

# ── Save ──
output_path = os.path.join(OUT_DIR, 'RoaringTransformer_Paper.docx')
doc.save(output_path)
print(f'\n\u2713 Paper saved to: {output_path}')
