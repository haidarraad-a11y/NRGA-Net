# -*- coding: utf-8 -*-
"""
Fix IJIES layout overflow (v3, schema-order-safe):
  1. Figure/table sections -> single-column (full width).
  2. Tables get fixed layout with explicit proportional widths.
  3. Section break inserted before References (refs stay two-column).
  4. Equation symbols: "->" -> arrow, "_" tokens -> true subscripts, ^{..} -> superscript.

Applies to both the revised and the highlighted manuscript.
"""
import copy
import re
import shutil

from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"

W = qn('w:p').split('}')[0] + '}'
EMU_IN = 914400
TEXT_W_IN = 6.7
COL_W_IN = 3.2

PPR_ORDER = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl',
             'numPr', 'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens',
             'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct', 'autoSpaceDE',
             'autoSpaceDN', 'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind',
             'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc', 'textDirection',
             'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle',
             'rPr', 'sectPr', 'pPrChange']
RPR_ORDER = ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike',
             'dstrike', 'outline', 'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid',
             'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position', 'sz',
             'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText', 'vertAlign',
             'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath']
TBLPR_ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
               'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders',
               'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']
TCPR_ORDER = ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd',
              'noWrap', 'tcMar', 'textDirection', 'tcFitText', 'vAlign', 'hideMark']
SECTPR_ORDER = ['headerReference', 'footerReference', 'footnotePr', 'endnotePr', 'type',
                'pgSz', 'pgMar', 'paperSrc', 'pgBorders', 'lnNumType', 'pgNumType', 'cols',
                'formProt', 'vAlign', 'noEndnote', 'titlePg', 'textDirection', 'bidi',
                'rtlGutter', 'docGrid', 'printerSettings']


def ordered_insert(parent, el, order):
    tag = el.tag.split('}')[-1]
    idx = order.index(tag)
    later = set(order[idx + 1:])
    for child in parent:
        if child.tag.split('}')[-1] in later:
            child.addprevious(el)
            return
    parent.append(el)


def para_text(el):
    return ''.join(t.text or '' for t in el.iter(W + 't'))


def image_widths(el):
    return [int(e.get('cx')) / EMU_IN
            for e in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')]


def set_cols(sectpr, single):
    cols = sectpr.find(W + 'cols')
    if cols is None:
        cols = sectpr.makeelement(W + 'cols', {})
        ordered_insert(sectpr, cols, SECTPR_ORDER)
    if single:
        if cols.get(W + 'num'):
            del cols.attrib[W + 'num']
    else:
        cols.set(W + 'num', '2')
    cols.set(W + 'space', '432')


def ensure_continuous(sectpr):
    typ = sectpr.find(W + 'type')
    if typ is None:
        typ = sectpr.makeelement(W + 'type', {})
        ordered_insert(sectpr, typ, SECTPR_ORDER)
    typ.set(W + 'val', 'continuous')


def is_boundary(child):
    if child.tag != W + 'p':
        return False
    ppr = child.find(W + 'pPr')
    return ppr is not None and ppr.find(W + 'sectPr') is not None


def group_sections(body_el):
    groups, cur = [], []
    for child in body_el:
        if child.tag == W + 'sectPr':
            continue
        if is_boundary(child):
            groups.append({'children': cur,
                           'sectpr': child.find(W + 'pPr').find(W + 'sectPr'),
                           'boundary_p': child})
            cur = []
        else:
            cur.append(child)
    groups.append({'children': cur, 'sectpr': 'BODY', 'boundary_p': None})
    return groups


def tables_in(children):
    return [ch for ch in children if ch.tag == W + 'tbl']


def group_wants_full(children):
    for ch in children:
        if ch.tag == W + 'p':
            if any(w > 3.4 for w in image_widths(ch)):
                return True
        elif ch.tag == W + 'tbl':
            grid = ch.find(W + 'tblGrid')
            if grid is not None and len(grid.findall(W + 'gridCol')) >= 3:
                return True
    return False


def set_table_width(tbl, target_in):
    grid = tbl.find(W + 'tblGrid')
    gridcols = grid.findall(W + 'gridCol')
    ncols = len(gridcols)
    if ncols == 0:
        return
    weights = [5.0] * ncols
    for tr in tbl.findall(W + 'tr'):
        for j, tc in enumerate(tr.findall(W + 'tc')):
            if j >= ncols:
                break
            weights[j] = max(weights[j], min(len(para_text(tc)), 45))
    total_w = sum(weights)
    widths_in = [target_in * wgt / total_w for wgt in weights]

    tblpr = tbl.find(W + 'tblPr')
    if tblpr is None:
        tblpr = tbl.makeelement(W + 'tblPr', {})
        tbl.insert(0, tblpr)
    for tag in ('tblW', 'tblLayout'):
        el = tblpr.find(W + tag)
        if el is not None:
            tblpr.remove(el)
    tblw = tblpr.makeelement(W + 'tblW', {W + 'w': str(int(round(target_in * 1440))),
                                          W + 'type': 'dxa'})
    ordered_insert(tblpr, tblw, TBLPR_ORDER)
    layout = tblpr.makeelement(W + 'tblLayout', {W + 'type': 'fixed'})
    ordered_insert(tblpr, layout, TBLPR_ORDER)
    for gc, w_in in zip(gridcols, widths_in):
        gc.set(W + 'w', str(int(round(w_in * 1440))))
    for tr in tbl.findall(W + 'tr'):
        for j, tc in enumerate(tr.findall(W + 'tc')):
            if j >= ncols:
                break
            tcpr = tc.find(W + 'tcPr')
            if tcpr is None:
                tcpr = tc.makeelement(W + 'tcPr', {})
                tc.insert(0, tcpr)
            tcw = tcpr.find(W + 'tcW')
            if tcw is None:
                tcw = tcpr.makeelement(W + 'tcW', {})
                ordered_insert(tcpr, tcw, TCPR_ORDER)
            tcw.set(W + 'w', str(int(round(widths_in[j] * 1440))))
            tcw.set(W + 'type', 'dxa')


SUB_PAT = r'([A-Za-z])_([A-Za-z0-9]+|\{[^}]+\})'
SUP_PAT = r'\^\{([^}]+)\}'
COMBINED = re.compile('(%s)|(%s)' % (SUB_PAT, SUP_PAT))
# COMBINED groups: 1=sub outer, 2=base char, 3=sub text, 4=sup outer, 5=sup text
VERTALIGN_VAL = {'sub': 'subscript', 'sup': 'superscript'}


def fix_equation(p_el):
    text = para_text(p_el)
    if '=' not in text or not re.search(r'\(\d+\)\s*$', text):
        return False
    for t in p_el.iter(W + 't'):
        if t.text and '->' in t.text:
            t.text = re.sub(r'\s*->\s*', ' \u2192 ', t.text)
    text = para_text(p_el)
    if '_' not in text and '^{' not in text:
        return True
    runs = p_el.findall(W + 'r')
    math_runs, keep_runs = [], []
    for r in runs:
        rtext = ''.join(t.text or '' for t in r.findall(W + 't'))
        if re.fullmatch(r'\t?\(\d+\)\s*', rtext or '') or not rtext:
            keep_runs.append(r)
        else:
            math_runs.append(r)
    if not math_runs:
        return True
    math_text = ''.join(''.join(t.text or '' for t in r.findall(W + 't'))
                        for r in math_runs)
    parts, pos = [], 0
    for m in COMBINED.finditer(math_text):
        if m.start() > pos:
            parts.append((math_text[pos:m.start()], None))
        if m.group(1):
            parts.append((m.group(2), None))
            sub = m.group(3)
            parts.append((sub[1:-1] if sub.startswith('{') else sub, 'sub'))
        else:
            parts.append((m.group(5), 'sup'))
        pos = m.end()
    if pos < len(math_text):
        parts.append((math_text[pos:], None))

    tmpl = None
    for r in math_runs:
        if ''.join(t.text or '' for t in r.findall(W + 't')).strip():
            tmpl = r
            break
    if tmpl is None:
        return True
    anchor = keep_runs[-1] if keep_runs else math_runs[0]
    for ptxt, mode in parts:
        if not ptxt:
            continue
        nr = copy.deepcopy(tmpl)
        for el in list(nr):
            if not el.tag.endswith('}rPr'):
                nr.remove(el)
        t = nr.makeelement(W + 't', {})
        t.text = ptxt
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        nr.append(t)
        if mode is not None:
            rpr = nr.find(W + 'rPr')
            if rpr is None:
                rpr = nr.makeelement(W + 'rPr', {})
                nr.insert(0, rpr)
            for va in rpr.findall(W + 'vertAlign'):
                rpr.remove(va)
            va = rpr.makeelement(W + 'vertAlign',
                                 {W + 'val': VERTALIGN_VAL[mode]})
            ordered_insert(rpr, va, RPR_ORDER)
        anchor.addnext(nr)
        anchor = nr
    for r in math_runs:
        p_el.remove(r)
    return True


def center_image_paragraphs(groups):
    for g in groups:
        for ch in g['children']:
            if ch.tag == W + 'p' and image_widths(ch):
                ppr = ch.find(W + 'pPr')
                if ppr is None:
                    ppr = ch.makeelement(W + 'pPr', {})
                    ch.insert(0, ppr)
                jc = ppr.find(W + 'jc')
                if jc is None:
                    jc = ppr.makeelement(W + 'jc', {})
                    ordered_insert(ppr, jc, PPR_ORDER)
                jc.set(W + 'val', 'center')


def process(path):
    doc = Document(path)
    body = doc.element.body

    for child in body:
        if is_boundary(child):
            ensure_continuous(child.find(W + 'pPr').find(W + 'sectPr'))

    groups = group_sections(body)

    # split last (BODY) group before References so refs stay two-column
    last = groups[-1]
    if last['sectpr'] == 'BODY':
        ref_el = None
        for ch in last['children']:
            if ch.tag == W + 'p' and para_text(ch).strip() == 'References':
                ref_el = ch
                break
        if ref_el is not None:
            tmpl_p = None
            for g in groups[:-1]:
                if g['boundary_p'] is not None:
                    tmpl_p = g['boundary_p']
                    break
            bp = copy.deepcopy(tmpl_p)
            ensure_continuous(bp.find(W + 'pPr').find(W + 'sectPr'))
            set_cols(bp.find(W + 'pPr').find(W + 'sectPr'), single=True)
            ref_el.addprevious(bp)

    groups = group_sections(body)

    n_single = 0
    for g in groups:
        if g['sectpr'] == 'BODY':
            continue
        if group_wants_full(g['children']):
            set_cols(g['sectpr'], single=True)
            n_single += 1
    n_tbl = 0
    for g in groups:
        target = TEXT_W_IN if group_wants_full(g['children']) else COL_W_IN
        for tbl in tables_in(g['children']):
            set_table_width(tbl, target)
            n_tbl += 1
    n_eq = sum(1 for ch in body if ch.tag == W + 'p' and fix_equation(ch))
    center_image_paragraphs(groups)

    doc.save(path)
    print(f'{path}: {n_single} single-col sections, {n_tbl} tables sized, {n_eq} equations cleaned')


if __name__ == '__main__':
    for p in (REVISED, HIGHLIGHTED):
        shutil.copyfile(p + '.bak2', p)  # restore pre-fix state
        process(p)
