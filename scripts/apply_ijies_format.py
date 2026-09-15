# -*- coding: utf-8 -*-
"""
Apply IJIES (Ver. 2025.5.18) template formatting to the revised NRGA-Net manuscript.

Covers the editor's format checklist items that can be fixed programmatically:
  - Title block (title 14pt bold, authors 11pt bold, affiliations/email 10.5pt,
    abstract/keywords 10pt) in a single-column (full-width) continuous section.
  - Body text 11pt Times New Roman, single-spaced (style-level).
  - First-order headings 12pt bold, second-order 11pt bold.
  - Figure captions centered, 10pt, label "Figure. N" -> "Fig. N".
  - Table captions centered, 10pt.
  - Tables 10pt (verified).
  - Equations: 11pt italic math, 5 mm indent, spacing above/below, " * " -> " × ",
    non-italic equation numbers.
  - Table renumbering: "Table 2a" -> "Table 3" with cascade shift of Tables 3-9 to 4-10.
  - References: IJIES style normalization (Vol./No./pp. spacing, curly quotes,
    complete Vol/No/pp data for [2] and [13], "In: Proc. of" for [10]).

Usage: python scripts/apply_ijies_format.py
"""
import copy
import re
import shutil
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
LETTER = r"D:\Dataset\pp3\NRGA-Net_response_letter.docx"

TNR = "Times New Roman"


# ----------------------------------------------------------------------------- helpers
def set_run_font(run, size=None, bold=None, italic=None, name=TNR, color=None):
    if name:
        run.font.name = name
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn('w:rFonts'))
        if rfonts is None:
            rfonts = rpr.makeelement(qn('w:rFonts'), {})
            rpr.insert(0, rfonts)
        for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
            rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def para_full_text(p):
    return ''.join(r.text for r in p.runs)


def replace_in_paragraph(p, pattern, repl, regex=True, color=None):
    """Replace pattern across runs of a paragraph, preserving per-run formatting.

    Returns number of replacements. New text inherits formatting of the first
    affected run; optionally re-colors the receiving run (for highlighted doc).
    """
    runs = list(p.runs)
    text = ''.join(r.text for r in runs)
    if regex:
        matches = list(re.finditer(pattern, text))
    else:
        matches = []
        start = 0
        while True:
            i = text.find(pattern, start)
            if i < 0:
                break
            matches.append(type('M', (), {'span': lambda self=None, i=i, L=len(pattern): (i, i + L),
                                          'group': lambda self=None, i=i, L=len(pattern): text[i:i + L]})())
            start = i + len(pattern)
    if not matches:
        return 0
    for m in reversed(matches):
        s, e = m.span()
        try:
            new_txt = m.expand(repl) if regex else repl
        except Exception:
            new_txt = repl
        pos = 0
        first_done = False
        spans = []
        for r in runs:
            rs, re_ = pos, pos + len(r.text)
            spans.append((rs, re_))
            pos = re_
        for r, (rs, re_) in zip(runs, spans):
            if re_ <= s or rs >= e:
                continue
            a = max(s, rs) - rs
            b = min(e, re_) - rs
            if not first_done:
                r.text = r.text[:a] + new_txt + r.text[b:]
                if color is not None:
                    set_run_font(r, color=color)
                first_done = True
            else:
                r.text = r.text[:a] + r.text[b:]
    return len(matches)


def iter_all_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


# ----------------------------------------------------------------------------- steps
def set_style_defaults(doc):
    st = doc.styles['Normal']
    st.font.name = TNR
    st.font.size = Pt(11)
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = rpar = rpr.makeelement(qn('w:rFonts'), {})
        rpr.insert(0, rfonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rfonts.set(qn(attr), TNR)
    pf = st.paragraph_format
    pf.line_spacing = 1.0


def make_section_single_col_first(doc):
    """Make the first section (title block + abstract + keywords) single-column."""
    for child in doc.element.body:
        if child.tag == qn('w:p') and child.pPr is not None and child.pPr.find(qn('w:sectPr')) is not None:
            sectpr = child.pPr.find(qn('w:sectPr'))
            cols = sectpr.find(qn('w:cols'))
            if cols is not None and cols.get(qn('w:num')):
                del cols.attrib[qn('w:num')]
            typ = sectpr.find(qn('w:type'))
            if typ is None:
                typ = sectpr.makeelement(qn('w:type'), {})
                sectpr.insert(0, typ)
            typ.set(qn('w:val'), 'continuous')
            return True
    return False


def format_title_block(doc, red=False):
    """Title 14pt bold, authors 11pt bold, affiliations/email 10.5pt, abstract+keywords 10pt."""
    done = 0
    for p in doc.paragraphs[:12]:
        t = p.text.strip()
        if not t:
            continue
        if t.startswith('Abstract:'):
            for r in p.runs:
                set_run_font(r, size=10)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            done += 1
        elif t.startswith('Keywords:'):
            for r in p.runs:
                set_run_font(r, size=10)
            done += 1
        elif t.startswith('*Corresponding'):
            for r in p.runs:
                set_run_font(r, size=10.5)
            done += 1
        elif re.match(r'^[12][A-Z]', t):  # affiliations "1University..." "2Computer..."
            for r in p.runs:
                set_run_font(r, size=10.5)
            done += 1
        elif p.runs and p.runs[0].bold and done == 0:
            # main title (first bold centered line)
            for r in p.runs:
                set_run_font(r, size=14, bold=True)
            done += 1
        elif p.runs and p.runs[0].bold:
            # author line
            for r in p.runs:
                set_run_font(r, size=11, bold=True)
            done += 1
    return done


def format_headings(doc):
    n1 = n2 = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t or len(t) > 90:
            continue
        if re.match(r'^\d+\.\s+\S', t) or t in ('References', 'Conflicts of Interest', 'Author Contributions',
                                                'Data and Code Availability'):
            for r in p.runs:
                set_run_font(r, size=12, bold=True)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(3)
            n1 += 1
        elif re.match(r'^\d+\.\d+\s+\S', t):
            for r in p.runs:
                set_run_font(r, size=11, bold=True)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            n2 += 1
    return n1, n2


def format_captions(doc, red=False):
    n = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t or len(t) > 400:
            continue
        is_cap = bool(re.match(r'^(Figure\.\s*\d+|Fig\.\s*\d+|Table\s+\d+[ab]?\.)', t))
        if not is_cap:
            continue
        for r in p.runs:
            set_run_font(r, size=10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        replace_in_paragraph(p, r'Figure\.\s*(\d+)', r'Fig. \1', color=(192, 0, 0) if red else None)
        n += 1
    return n


RENUMBER_STEPS = [
    (r'Tables?\s+9(?![0-9a-zA-Z])', 'Table 10'),  # plural form "Tables 9" won't occur; handled below
    (r'Tables?\s+8(?![0-9a-zA-Z])', 'Table 9'),
    (r'Tables?\s+7(?![0-9a-zA-Z])', 'Table 8'),
    (r'Tables?\s+6(?![0-9a-zA-Z])', 'Table 7'),
    (r'Tables?\s+5(?![0-9a-zA-Z])', 'Table 6'),
    (r'Tables?\s+4(?![0-9a-zA-Z])', 'Table 5'),
    (r'Tables?\s+3(?![0-9a-zA-Z])', 'Table 4'),
]


def renumber_tables(doc, red=False):
    """Cascade: Table 2a -> Table 3; Tables 3-9 -> 4-10. Preserves singular/plural prefix."""
    total = 0
    color = (192, 0, 0) if red else None
    # numeric cascade, descending, preserving the word form (Table/Tables)
    for pat, repl_num in [(r'(Tables?)\s+9(?![0-9a-zA-Z])', r'\1 10'),
                          (r'(Tables?)\s+8(?![0-9a-zA-Z])', r'\1 9'),
                          (r'(Tables?)\s+7(?![0-9a-zA-Z])', r'\1 8'),
                          (r'(Tables?)\s+6(?![0-9a-zA-Z])', r'\1 7'),
                          (r'(Tables?)\s+5(?![0-9a-zA-Z])', r'\1 6'),
                          (r'(Tables?)\s+4(?![0-9a-zA-Z])', r'\1 5'),
                          (r'(Tables?)\s+3(?![0-9a-zA-Z])', r'\1 4')]:
        for p in iter_all_paragraphs(doc):
            total += replace_in_paragraph(p, pat, repl_num, color=color)
    # 2a -> 3 LAST so it is not re-shifted
    for p in iter_all_paragraphs(doc):
        total += replace_in_paragraph(p, r'(Tables?)\s+2a(?![0-9a-zA-Z])', r'\1 3', color=color)
    return total


def format_equations(doc):
    n = 0
    for p in doc.paragraphs:
        full = para_full_text(p)
        if not re.search(r'\t\(\d+\)\s*$', full) or '=' not in full:
            continue
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        replace_in_paragraph(p, r'\s*\*\s*', ' × ')
        # split trailing "\t(n)" into a non-italic run
        m = re.search(r'\t\(\d+\)\s*$', full)
        runs = [r for r in p.runs if r.text]
        if runs:
            last = runs[-1]
            lt = last.text
            cut = None
            mm = re.search(r'\t\(\d+\)\s*$', lt)
            if mm:
                cut = mm.start()
            if cut is not None and cut > 0:
                second = copy.deepcopy(last._r)
                last._r.addnext(second)
                from docx.text.run import Run
                r2 = Run(second, last._parent)
                tail = lt[cut:]
                last.text = lt[:cut]
                r2.text = tail
                for r in [last] + [x for x in p.runs if x is not last]:
                    if r is r2:
                        set_run_font(r, size=11, italic=False)
                    else:
                        set_run_font(r, size=11, italic=True)
            else:
                for r in p.runs:
                    set_run_font(r, size=11, italic=True)
        n += 1
    return n


REF_FIXES = {
    '[2] V. Meo': (
        r'Cartography and Geographic Information Science, 2025\.',
        'Cartography and Geographic Information Science, Vol.53, No.4, pp.465-478, 2025.'),
    '[10] J. Horvath': (
        r'In: Pattern Recognition, ICPR International Workshops and Challenges,',
        'In: Proc. of International Conference on Pattern Recognition (ICPR) International Workshops and Challenges,'),
    '[13] L. Abady': (
        r'Vol\. 11, No\. 1, pp\. 1-56, 2022\.',
        'Vol.12, No.1, e42, 2023.'),
}


def fix_references(doc, red=False):
    n = 0
    in_refs = False
    for p in doc.paragraphs:
        t = p.text.strip()
        if t == 'References':
            in_refs = True
            continue
        if not in_refs:
            continue
        if re.match(r'^\[\d+\]', t):
            for prefix, (pat, repl) in REF_FIXES.items():
                if t.startswith(prefix):
                    if replace_in_paragraph(p, pat, repl, color=(192, 0, 0) if red else None):
                        n += 1
            # normalize Vol./No./pp. spacing to template style
            replace_in_paragraph(p, r'(Vol\.)\s+', r'\1')
            replace_in_paragraph(p, r'(No\.)\s+', r'\1')
            replace_in_paragraph(p, r'(pp\.)\s+', r'\1')
            # straight quotes -> curly quotes, alternating
            txt = para_full_text(p)
            if '"' in txt:
                out, open_q = [], True
                for ch in txt:
                    if ch == '"':
                        out.append('\u201c' if open_q else '\u201d')
                        open_q = not open_q
                    else:
                        out.append(ch)
                new_txt = ''.join(out)
                # put the full corrected text into runs, preserving run boundaries by
                # mapping char ranges: simplest is per-run sequential assignment
                pos = 0
                for r in p.runs:
                    seg = new_txt[pos:pos + len(r.text)]
                    r.text = seg
                    pos += len(r.text)
        elif t and not t.startswith('['):
            continue
    return n


def main():
    # ---- revised manuscript: full formatting
    shutil.copyfile(REVISED, REVISED + '.bak')
    doc = Document(REVISED)
    set_style_defaults(doc)
    print('first section single-col:', make_section_single_col_first(doc))
    print('title block paras formatted:', format_title_block(doc))
    h1, h2 = format_headings(doc)
    print(f'headings: {h1} first-order, {h2} second-order')
    print('captions formatted:', format_captions(doc))
    print('table renumber replacements:', renumber_tables(doc, red=False))
    print('equations formatted:', format_equations(doc))
    print('references fixed:', fix_references(doc, red=False))
    doc.save(REVISED)
    print('saved:', REVISED)

    # ---- highlighted manuscript: same formatting, replacements in red
    shutil.copyfile(HIGHLIGHTED, HIGHLIGHTED + '.bak')
    doc = Document(HIGHLIGHTED)
    set_style_defaults(doc)
    make_section_single_col_first(doc)
    format_title_block(doc)
    format_headings(doc)
    format_captions(doc, red=True)
    renumber_tables(doc, red=True)
    format_equations(doc)
    fix_references(doc, red=True)
    doc.save(HIGHLIGHTED)
    print('saved:', HIGHLIGHTED)

    # ---- response letter: renumber table references in our responses only
    #      (paragraphs starting with "Comment:" are verbatim reviewer quotes - untouched)
    shutil.copyfile(LETTER, LETTER + '.bak')
    doc = Document(LETTER)
    # cascade, applied to our own text only (paragraphs starting with
    # "Comment" are verbatim reviewer quotes and are left untouched)
    total = 0
    steps = [(r'(Tables?)\s+9(?![0-9a-zA-Z])', r'\1 10'),
             (r'(Tables?)\s+8(?![0-9a-zA-Z])', r'\1 9'),
             (r'(Tables?)\s+7(?![0-9a-zA-Z])', r'\1 8'),
             (r'(Tables?)\s+6(?![0-9a-zA-Z])', r'\1 7'),
             (r'(Tables?)\s+5(?![0-9a-zA-Z])', r'\1 6'),
             (r'(Tables?)\s+4(?![0-9a-zA-Z])', r'\1 5'),
             (r'(Tables?)\s+3(?![0-9a-zA-Z])', r'\1 4'),
             (r'(Tables?)\s+2a(?![0-9a-zA-Z])', r'\1 3')]
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith('Comment'):
            continue
        for pat, repl in steps:
            total += replace_in_paragraph(p, pat, repl)
    doc.save(LETTER)
    print('response letter table refs updated:', total)


if __name__ == '__main__':
    main()
