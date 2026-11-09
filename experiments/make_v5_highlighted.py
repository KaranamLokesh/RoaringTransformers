#!/usr/bin/env python3
"""
Build the highlighted companion to v5-patched.docx.

The clean manuscript the author is submitting is v5-patched.docx (their copyedit
plus the v5 corrections). The highlighted companion must have identical text with
the changed passages marked yellow, so it is built FROM v5-patched, not from any
earlier generated file.

What gets highlighted:
  1. Every passage that was yellow in the v4 highlighted manuscript -- i.e. every
     change since the version the reviewers last saw -- transferred onto the
     author's edited text by matching.
  2. The v5 re-measurement changes (the A-patch replacements and the rewritten
     numeric rows of Tables 5 to 8), so the corrected numbers are marked too.

Highlighting splits runs so only the matched text is coloured; surrounding text
and all character formatting are preserved.

Usage:
    python experiments/make_v5_highlighted.py
"""

import copy
import os
import sys

import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.run import Run

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from apply_v5_patch import iter_paragraphs, norm  # noqa: E402
from generate_v5_patchlist import P, TABLES  # noqa: E402

V5_PATCHED = os.path.join(HERE, '..', 'paper', 'v5-patched.docx')
V4_HL = os.path.join(HERE, '..', 'paper', 'v4-review',
                     'Roaring-Transformer-v4-Template-Highlighted.docx')
OUT = os.path.join(HERE, '..', 'paper', 'v5-review',
                   'Roaring-Transformer-v5-Highlighted.docx')

YELLOW = 'yellow'


def para_runs(p):
    return [Run(r, p) for r in p._p.iter(qn('w:r'))]


def _set_highlight(run_el):
    rpr = run_el.find(qn('w:rPr'))
    if rpr is None:
        rpr = run_el.makeelement(qn('w:rPr'), {})
        run_el.insert(0, rpr)
    hl = rpr.find(qn('w:highlight'))
    if hl is None:
        hl = rpr.makeelement(qn('w:highlight'), {})
        rpr.append(hl)
    hl.set(qn('w:val'), YELLOW)


def _clone_with_text(run_el, text, highlight):
    new = copy.deepcopy(run_el)
    for t in new.findall(qn('w:t')):
        new.remove(t)
    tnode = new.makeelement(qn('w:t'), {})
    tnode.set(qn('xml:space'), 'preserve')
    tnode.text = text
    new.append(tnode)
    if highlight:
        _set_highlight(new)
    return new


def highlight_portion(run, lo, hi):
    """Highlight run.text[lo:hi], splitting the run so the rest keeps its colour."""
    text = run.text
    head, mid, tail = text[:lo], text[lo:hi], text[hi:]
    el = run._r
    parent = el.getparent()
    at = parent.index(el)
    frags = [(head, False), (mid, True), (tail, False)]
    frags = [(t, h) for t, h in frags if t != '']
    if len(frags) == 1 and frags[0][1]:      # whole run highlighted, no split
        _set_highlight(el)
        return
    parent.remove(el)
    for offset, (t, h) in enumerate(frags):
        parent.insert(at + offset, _clone_with_text(el, t, h))


def _skeleton(text):
    """Lowercase alphanumerics only, with a map back to original offsets.

    Matching on this skeleton is immune to the punctuation and whitespace edits
    the author made (em-dash removal, comma swaps, double spaces), which broke a
    verbatim match on 44 of the 79 changed passages.
    """
    skel, idx_map = [], []
    for i, c in enumerate(text):
        if c.isalnum():
            skel.append(c.lower())
            idx_map.append(i)
    return ''.join(skel), idx_map


def _highlight_real_range(offsets, idx, end):
    for r, start, stop in offsets:
        if stop <= idx or start >= end:
            continue
        highlight_portion(r, max(idx, start) - start, min(end, stop) - start)


def highlight_span(doc, span, min_skel=15):
    """Highlight `span` by skeleton match (punctuation-insensitive). Returns bool."""
    sk_span, _ = _skeleton(span)
    if len(sk_span) < min_skel:
        return False
    for p in iter_paragraphs(doc):
        runs = para_runs(p)
        if not runs:
            continue
        offsets, pos = [], 0
        for r in runs:
            offsets.append((r, pos, pos + len(r.text)))
            pos += len(r.text)
        full = ''.join(r.text for r in runs)
        sk_full, idx_map = _skeleton(full)
        j = sk_full.find(sk_span)
        if j < 0:
            continue
        idx = idx_map[j]
        end = idx_map[j + len(sk_span) - 1] + 1
        _highlight_real_range(offsets, idx, end)
        return True
    return False


def highlight_span_by_sentences(doc, span):
    """Fallback: highlight each sentence of a span whose whole-span match failed.

    A single reworded word blocks the whole-span skeleton match, but the other
    sentences of the passage are untouched and still locate cleanly.
    """
    import re
    parts = [s for s in re.split(r'(?<=[.;])\s+', span) if len(_skeleton(s)[0]) >= 15]
    return sum(highlight_span(doc, s) for s in parts)


def v4_spans():
    d = docx.Document(V4_HL)
    spans, cur = [], ''
    for p in d.element.body.iter(qn('w:p')):
        for r in p.iter(qn('w:r')):
            run = Run(r, d)
            if run.font.highlight_color == WD_COLOR_INDEX.YELLOW:
                cur += run.text
            elif cur:
                spans.append(cur)
                cur = ''
        if cur:
            spans.append(cur)
            cur = ''
    # long spans first so a whole sentence is matched before any fragment of it
    return sorted({s.strip() for s in spans if len(s.strip()) >= 20},
                  key=len, reverse=True)


def highlight_table_cells(doc):
    """Mark every data cell of the four rewritten composition tables."""
    from apply_v5_patch import find_table
    n = 0
    for name, header, rows in TABLES:
        t, hdr_idx = find_table(doc, header)
        if t is None:
            print(f'  [warn] {name} not found for highlighting')
            continue
        for row in t.rows[hdr_idx + 1:]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        if r.text.strip():
                            _set_highlight(r._r)
                            n += 1
    return n


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = docx.Document(V5_PATCHED)

    spans = v4_spans()
    hit = partial = 0
    misses = []
    for sp in spans:
        if highlight_span(doc, sp):
            hit += 1
        elif highlight_span_by_sentences(doc, sp):
            partial += 1
        else:
            misses.append(sp)
    print(f'v4 "since round-2" spans: {hit} whole + {partial} by-sentence '
          f'= {hit + partial}/{len(spans)} transferred')

    v5_hits = sum(highlight_span(doc, repl) for _w, _f, repl, _y in P)
    print(f'v5 re-measurement passages: {v5_hits}/{len(P)} highlighted')

    cells = highlight_table_cells(doc)
    print(f'table cells highlighted: {cells}')

    doc.save(OUT)
    print(f'\nwrote {os.path.relpath(OUT, os.path.join(HERE, ".."))}')

    if misses:
        print(f'\n{len(misses)} v4 spans located neither whole nor by sentence '
              f'(likely reworded); check these passages by hand:')
        for sp in misses[:12]:
            print('  -', sp[:90])


if __name__ == '__main__':
    main()
