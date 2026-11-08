#!/usr/bin/env python3
"""
Apply the v5 patch list to the author's hand-edited .docx, in place, preserving
every byte of formatting, styling and embedded imagery that is not being changed.

The patch definitions are imported from generate_v5_patchlist so the document and
the written patch list can never disagree.

Matching is tolerant of the substitutions Google Docs makes silently: curly
quotes, non-breaking spaces and non-breaking hyphens are normalised 1:1 (length
preserving, so run offsets stay valid) before searching. A replacement that
spans several runs is written into the first overlapping run and removed from
the rest, which keeps the paragraph's character formatting intact.

Nothing is applied silently: every patch reports FOUND or NOT FOUND, and the
script refuses to write an output file if any patch failed unless --force.

Usage:
    python experiments/apply_v5_patch.py IN.docx OUT.docx [--grammar] [--dry-run]
"""

import argparse
import os
import sys

import docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_v5_patchlist import GRAMMAR, P, TABLES  # noqa: E402

# 1:1 character normalisation -- never changes string length, so the offsets we
# compute against normalised text remain valid against the original runs.
TRANS = str.maketrans({
    '’': "'", '‘': "'", '“': '"', '”': '"',
    ' ': ' ', '‑': '-', '−': '-', '​': ' ',
})


def norm(s):
    return s.translate(TRANS)


def iter_paragraphs(doc):
    """Every w:p in the document body, at any nesting depth.

    Document.paragraphs yields only direct children of the body. Google Docs
    wraps some paragraphs in w:sdt content controls -- in this manuscript the
    Table 5 and Table 6 captions -- and those are invisible to it. Walking the
    element tree catches paragraphs inside sdt wrappers, table cells and any
    other container.
    """
    for el in doc.element.body.iter(qn('w:p')):
        yield Paragraph(el, doc)


def all_runs(p):
    """Every w:r under the paragraph, including runs nested in w:sdt wrappers.

    Paragraph.runs returns only direct w:r children. Google Docs wraps the
    opening runs of the Table 5 and Table 6 captions in run-level content
    controls, which makes the first half of each caption invisible to both
    Paragraph.text and Paragraph.runs.
    """
    return [Run(r, p) for r in p._p.iter(qn('w:r'))]


def replace_in_paragraph(p, find, repl):
    """Replace `find` with `repl` across run boundaries, keeping run formatting."""
    runs = all_runs(p)
    if not runs:
        return False
    spans, pos = [], 0
    for r in runs:
        spans.append((r, pos, pos + len(r.text)))
        pos += len(r.text)
    full = ''.join(r.text for r in runs)
    idx = norm(full).find(norm(find))
    if idx < 0:
        return False
    end = idx + len(find)

    first = True
    for r, start, stop in spans:          # offsets precomputed before mutation
        if stop <= idx or start >= end:
            continue
        head = r.text[:max(idx, start) - start]
        tail = r.text[min(end, stop) - start:]
        if first:
            r.text = head + repl + tail
            first = False
        else:
            r.text = head + tail
    return True


def apply_text_patch(doc, find, repl):
    for p in iter_paragraphs(doc):
        if replace_in_paragraph(p, find, repl):
            return True
    return False


def set_cell(cell, text):
    """Write text into a cell, preserving the first run's character formatting."""
    para = cell.paragraphs[0]
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r.text = ''
    else:
        para.add_run(text)
    for extra in cell.paragraphs[1:]:
        for r in extra.runs:
            r.text = ''


def find_table(doc, header):
    """Locate a table by its header row, tolerating whitespace differences."""
    want = [norm(h).strip().lower() for h in header]
    for t in doc.tables:
        for hdr_idx in (0, 1):            # some templates carry a blank first row
            if len(t.rows) <= hdr_idx or len(t.rows[hdr_idx].cells) != len(header):
                continue
            got = [norm(c.text).strip().lower() for c in t.rows[hdr_idx].cells]
            if got == want:
                return t, hdr_idx
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--grammar', action='store_true',
                    help='also apply the section C copyedit repairs')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='write the output even if some patches did not match')
    args = ap.parse_args()

    doc = docx.Document(args.infile)
    ok, failed = [], []

    print('== A. text replacements ==')
    for i, (where, find, repl, _why) in enumerate(P, 1):
        hit = apply_text_patch(doc, find, repl)
        (ok if hit else failed).append(f'A{i} {where}')
        print(f'  [{"OK " if hit else "MISS"}] A{i}. {where}')

    print('\n== B. table rows ==')
    for name, header, rows in TABLES:
        t, hdr_idx = find_table(doc, header)
        if t is None:
            failed.append(f'{name} (table not located)')
            print(f'  [MISS] {name}: header row not found')
            continue
        data_rows = t.rows[hdr_idx + 1:]
        if len(data_rows) != len(rows):
            failed.append(f'{name} (row count {len(data_rows)} != {len(rows)})')
            print(f'  [MISS] {name}: {len(data_rows)} data rows, expected {len(rows)}')
            continue
        for row, vals in zip(data_rows, rows):
            for cell, val in zip(row.cells, vals):
                set_cell(cell, val)
        ok.append(name)
        print(f'  [OK ] {name}: {len(rows)} rows written')

    if args.grammar:
        print('\n== C. copyedit repairs ==')
        for i, (where, find, repl) in enumerate(GRAMMAR, 1):
            clean = repl.split('   (')[0]        # strip the inline parenthetical note
            hit = apply_text_patch(doc, find, clean)
            (ok if hit else failed).append(f'C{i} {where}')
            print(f'  [{"OK " if hit else "MISS"}] C{i}. {where}')

    print(f'\napplied {len(ok)}, failed {len(failed)}')
    if failed:
        print('\nNOT APPLIED:')
        for f in failed:
            print('  -', f)
        print('\nA miss is usually a wording difference in the edited document. '
              'Search the FIND string in the patch list, adjust it there, and rerun.')

    if args.dry_run:
        print('\ndry run: nothing written')
        return
    if failed and not args.force:
        print('\nrefusing to write a partially patched file; use --force to override')
        sys.exit(1)
    doc.save(args.outfile)
    print(f'\nwrote {args.outfile}')


if __name__ == '__main__':
    main()
