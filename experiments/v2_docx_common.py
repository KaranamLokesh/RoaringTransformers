"""Shared .docx scaffolding for the round-2 (v2) revision documents.

Text passed to the body helpers may contain [[...]] spans. Anything inside the
double brackets is rendered with a yellow highlight, which is how the journal
asks revised passages to be marked. build_clean() strips the markers instead,
so the highlighted and clean manuscripts are generated from one source of truth.
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

BORDER = {e: {'sz': '4', 'val': 'single', 'color': '000000'}
          for e in ('top', 'left', 'bottom', 'right')}


def new_document():
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin = s.bottom_margin = Cm(2.0)
    s.left_margin = s.right_margin = Cm(1.8)

    n = doc.styles['Normal']
    n.font.name = 'Times New Roman'
    n.font.size = Pt(10)
    n.paragraph_format.space_after = Pt(3)
    n.paragraph_format.space_before = Pt(0)
    n.paragraph_format.line_spacing = 1.12

    def mk(name, size, bold=False, italic=False, align=None,
           before=0, after=3):
        st = doc.styles.add_style(name, 1)
        st.font.name = 'Times New Roman'
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        if align is not None:
            st.paragraph_format.alignment = align
        return st

    mk('PaperTitle', 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    mk('Author', 11, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    mk('Affiliation', 10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    mk('SectionHead', 11, bold=True, before=10)
    mk('SubsectionHead', 10, bold=True, italic=True, before=8, after=2)
    mk('FigCaption', 9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
       before=2, after=6)
    mk('TableCaption', 9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
       before=6, after=3)
    return doc


def template_document(path):
    """Open the journal template and strip its body, keeping styles and page setup.

    Returns (doc, style_map). The template uses the built-in Heading 1/2 styles and
    a lowercase 'normal' body style, so callers must not assume the custom style
    names created by new_document().
    """
    doc = Document(path)
    body = doc.element.body
    for child in list(body):
        if not child.tag.endswith('}sectPr'):
            body.remove(child)
    # Clearing the body leaves the template's own figures behind as orphaned
    # media parts, which would ship inside the submission file. Drop them.
    for rid, rel in list(doc.part.rels.items()):
        if 'image' in rel.reltype:
            doc.part.drop_rel(rid)
    style_map = {
        'body': 'normal', 'section': 'Heading 1', 'subsection': 'Heading 2',
        'title': 'normal', 'author': 'normal', 'affiliation': 'normal',
        'figcaption': 'normal', 'tablecaption': 'normal',
    }
    return doc, style_map


DEFAULT_STYLES = {
    'body': 'Normal', 'section': 'SectionHead', 'subsection': 'SubsectionHead',
    'title': 'PaperTitle', 'author': 'Author', 'affiliation': 'Affiliation',
    'figcaption': 'FigCaption', 'tablecaption': 'TableCaption',
}

# The template sets the font explicitly on every run rather than relying on the
# paragraph style, whose font is undefined. Runs that omit it fall back to the
# document default and render inconsistently, so we reproduce the same policy.
TEMPLATE_FONT = 'Helvetica Neue'
TEMPLATE_SIZES = {
    'title': 16.0, 'label': 11.0, 'body': 10.0, 'section': 11.0,
    'subsection': 10.0, 'author': 10.0, 'affiliation': 10.0,
    'figcaption': 10.0, 'tablecaption': 10.0, 'table': 10.0, 'reference': 10.0,
}


class Builder:
    """Writes paragraphs into `doc`, honouring or stripping [[highlight]] spans.

    `styles` maps logical roles to concrete style names so the same content can be
    rendered either into a standalone document or into the journal template.
    """

    def __init__(self, doc, highlight=True, styles=None):
        self.doc = doc
        self.highlight = highlight
        self.s = dict(DEFAULT_STYLES)
        if styles:
            self.s.update(styles)
        self.template_mode = styles is not None

    # -- runs ------------------------------------------------------------- #
    def _style_run(self, run, bold, size, role):
        run.bold = bold
        if self.template_mode:
            run.font.name = TEMPLATE_FONT
            run.font.size = Pt(size if size else TEMPLATE_SIZES.get(role, 10.0))
        elif size:
            run.font.size = Pt(size)

    def _emit(self, paragraph, text, bold=False, size=None, role='body'):
        """Split on [[...]] and emit each fragment as its own run."""
        for i, frag in enumerate(text.split('[[')):
            if i == 0:
                pre, marked = frag, ''
            elif ']]' in frag:
                marked, pre = frag.split(']]', 1)
            else:
                marked, pre = frag, ''
            if i > 0 and marked:
                r = paragraph.add_run(marked)
                self._style_run(r, bold, size, role)
                if self.highlight:
                    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            if pre:
                r = paragraph.add_run(pre)
                self._style_run(r, bold, size, role)

    # -- blocks ----------------------------------------------------------- #
    def label(self, text):
        """Front-matter label ('Article Title', 'Abstract', …) as the template uses."""
        p = self.doc.add_paragraph(style=self.s['body'])
        self._emit(p, text, bold=True, role='label')

    def title(self, text):
        p = self.doc.add_paragraph(style=self.s['title'])
        self._emit(p, text, bold=self.template_mode, role='title')

    def author(self, text):
        p = self.doc.add_paragraph(style=self.s['author'])
        self._emit(p, text, role='author')

    def affiliation(self, text):
        p = self.doc.add_paragraph(style=self.s['affiliation'])
        self._emit(p, text, role='affiliation')

    def section(self, text):
        self._emit(self.doc.add_paragraph(style=self.s['section']), text,
                   role='section')

    def subsection(self, text):
        self._emit(self.doc.add_paragraph(style=self.s['subsection']), text,
                   role='subsection')

    def body(self, text, indent=True):
        p = self.doc.add_paragraph(style=self.s['body'])
        if indent and not self.template_mode:
            p.paragraph_format.first_line_indent = Cm(0.5)
        self._emit(p, text, role='body')

    def bullet(self, text, level=1):
        p = self.doc.add_paragraph(style=self.s['body'])
        p.paragraph_format.left_indent = Cm(0.6 * level)
        p.paragraph_format.first_line_indent = Cm(0)
        self._emit(p, f'• {text}', role='body')

    def numbered(self, idx, text):
        p = self.doc.add_paragraph(style=self.s['body'])
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.4)
        self._emit(p, f'{idx}. {text}', role='body')

    def reference(self, idx, text):
        p = self.doc.add_paragraph(style=self.s['body'])
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        self._emit(p, f'[{idx}] {text}', size=9 if not self.template_mode else None,
                   role='reference')

    def caption(self, text, style=None):
        role = style or 'tablecaption'
        name = self.s.get(role.lower().replace('caption', 'caption'), None)
        if style == 'FigCaption':
            name = self.s['figcaption']
        elif style in (None, 'TableCaption'):
            name = self.s['tablecaption']
        p = self.doc.add_paragraph(style=name)
        role = 'figcaption' if style == 'FigCaption' else 'tablecaption'
        if self.template_mode:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self._emit(p, text, role=role)
        else:
            self._emit(p, text)

    def figure(self, path, caption, width=Inches(5.4)):
        if os.path.exists(path):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(0)
            p.add_run().add_picture(path, width=width)
        self.caption(caption, style='FigCaption')

    def table(self, caption, headers, rows, note=None, col_pt=None):
        # The template sets table text at 10 pt; narrow tables keep that, wider
        # ones step down so the row fits the A4 text block.
        if col_pt is None:
            col_pt = 10.0 if len(headers) <= 4 else (9.0 if len(headers) <= 6 else 8.0)
        self.caption(caption, 'TableCaption')
        t = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(1)
            self._emit(p, h, bold=True, size=col_pt, role='table')
            _border(cell)
            _shade(cell, 'E8E8E8')
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = t.rows[ri + 1].cells[ci]
                cell.text = ''
                p = cell.paragraphs[0]
                p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                               else WD_ALIGN_PARAGRAPH.CENTER)
                p.paragraph_format.space_after = Pt(1)
                self._emit(p, str(val), size=col_pt, role='table')
                _border(cell)
        if note:
            self.caption(note)

    def pagebreak(self):
        self.doc.add_page_break()


def _border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement('w:tcBorders')
    for edge, attrs in BORDER.items():
        el = OxmlElement(f'w:{edge}')
        for k, v in attrs.items():
            el.set(qn(f'w:{k}'), v)
        b.append(el)
    tcPr.append(b)


def _shade(cell, fill):
    sh = OxmlElement('w:shd')
    sh.set(qn('w:fill'), fill)
    sh.set(qn('w:val'), 'clear')
    cell._tc.get_or_add_tcPr().append(sh)
