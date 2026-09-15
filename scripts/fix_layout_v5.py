# -*- coding: utf-8 -*-
"""
v5 patch:
  1. Resize the 2-column calibration table ("Quantity") to full text width so
     its rows stop wrapping (it kept the 3.2in in-column width from v3).
  2. Keep tables on one page: w:cantSplit on all rows + keepNext on all
     paragraphs inside tables except the last row, and on table captions.
  3. Revised copy only: strip leftover red font (highlight color belongs only
     in the highlighted copy).
"""
import re
import shutil

from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'

TBLPR_ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
               'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders',
               'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']
TCPR_ORDER = ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd',
              'noWrap', 'tcMar', 'textDirection', 'tcFitText', 'vAlign', 'hideMark']
RPR_ORDER = ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike',
             'dstrike', 'outline', 'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid',
             'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position', 'sz',
             'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText', 'vertAlign',
             'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath']
PPR_ORDER = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl',
             'numPr', 'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens',
             'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct', 'autoSpaceDE',
             'autoSpaceDN', 'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind',
             'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc', 'textDirection',
             'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle',
             'rPr', 'sectPr', 'pPrChange']


def ordered_insert(parent, el, order):
    tag = el.tag.split('}')[-1]
    later = set(order[order.index(tag) + 1:])
    for child in parent:
        if child.tag.split('}')[-1] in later:
            child.addprevious(el)
            return
    parent.append(el)


def get_or_add(parent, tag, order):
    el = parent.find(W + tag)
    if el is None:
        el = parent.makeelement(W + tag, {})
        ordered_insert(parent, el, order)
    return el


def set_keep_next(p_el):
    ppr = p_el.find(W + 'pPr')
    if ppr is None:
        ppr = p_el.makeelement(W + 'pPr', {})
        p_el.insert(0, ppr)
    kn = get_or_add(ppr, 'keepNext', PPR_ORDER)
    kn.set(W + 'val', 'true')


def resize_table(tbl, target_in):
    grid = tbl.find(W + 'tblGrid')
    gridcols = grid.findall(W + 'gridCol')
    ncols = len(gridcols)
    weights = [5.0] * ncols
    for tr in tbl.findall(W + 'tr'):
        for j, tc in enumerate(tr.findall(W + 'tc')):
            if j >= ncols:
                break
            L = len(''.join(t.text or '' for t in tc.iter(W + 't')))
            weights[j] = max(weights[j], min(L, 45))
    tot = sum(weights)
    widths = [target_in * wgt / tot for wgt in weights]
    tblpr = tbl.find(W + 'tblPr')
    for tag in ('tblW', 'tblLayout'):
        el = tblpr.find(W + tag)
        if el is not None:
            tblpr.remove(el)
    tblw = tblpr.makeelement(W + 'tblW', {W + 'w': str(int(round(target_in * 1440))),
                                          W + 'type': 'dxa'})
    ordered_insert(tblpr, tblw, TBLPR_ORDER)
    layout = tblpr.makeelement(W + 'tblLayout', {W + 'type': 'fixed'})
    ordered_insert(tblpr, layout, TBLPR_ORDER)
    for gc, w_in in zip(gridcols, widths):
        gc.set(W + 'w', str(int(round(w_in * 1440))))
    for tr in tbl.findall(W + 'tr'):
        for j, tc in enumerate(tr.findall(W + 'tc')):
            if j >= ncols:
                break
            tcpr = tc.find(W + 'tcPr')
            if tcpr is None:
                tcpr = tc.makeelement(W + 'tcPr', {})
                tc.insert(0, tcpr)
            tcw = get_or_add(tcpr, 'tcW', TCPR_ORDER)
            tcw.set(W + 'w', str(int(round(widths[j] * 1440))))
            tcw.set(W + 'type', 'dxa')


def keep_tables_together(doc):
    n_cap = 0
    body = doc.element.body
    children = [ch for ch in body]
    for i, ch in enumerate(children):
        if ch.tag != W + 'tbl':
            continue
        rows = ch.findall(W + 'tr')
        for tr in rows:
            trpr = tr.find(W + 'trPr')
            if trpr is None:
                trpr = tr.makeelement(W + 'trPr', {})
                tr.insert(0, trpr)
            if trpr.find(W + 'cantSplit') is None:
                trpr.append(trpr.makeelement(W + 'cantSplit', {}))
        for tr in rows[:-1]:
            for tc in tr.findall(W + 'tc'):
                for p_el in tc.findall(W + 'p'):
                    set_keep_next(p_el)
        # caption paragraph immediately above the table
        j = i - 1
        while j >= 0:
            prev = children[j]
            if prev.tag != W + 'p':
                break
            set_keep_next(prev)
            if re.match(r'^Table\s+\d+\.', ''.join(
                    t.text or '' for t in prev.iter(W + 't')).strip()):
                n_cap += 1
                break
            j -= 1
    return n_cap


def strip_red(doc):
    n = 0
    for color in doc.element.body.iter(W + 'color'):
        val = (color.get(W + 'val') or '').upper()
        if val in ('FF0000', 'C00000', 'FF0100', 'DC143C', '8B0000'):
            color.set(W + 'val', 'auto')
            n += 1
    return n


def process(path, de_red):
    shutil.copyfile(path, path + '.bak5')
    doc = Document(path)
    if de_red:
        n = strip_red(doc)
        print(f'{path}: red runs cleaned: {n}')
    # resize small tables to full width (calibration "Quantity" table)
    n_resized = 0
    for tbl in doc.tables:
        el = tbl._tbl
        grid = el.find(W + 'tblGrid')
        ncols = len(grid.findall(W + 'gridCol')) if grid is not None else 0
        tblpr = el.find(W + 'tblPr')
        tblw = tblpr.find(W + 'tblW') if tblpr is not None else None
        cur = int(tblw.get(W + 'w')) if tblw is not None and tblw.get(W + 'w') else 0
        if ncols <= 2 and 0 < cur < 5000:  # narrow in-column width in twips
            resize_table(el, 6.7)
            n_resized += 1
    print(f'{path}: small tables resized to full width: {n_resized}')
    n_cap = keep_tables_together(doc)
    print(f'{path}: tables kept together, captions marked: {n_cap}')
    doc.save(path)


if __name__ == '__main__':
    process(REVISED, de_red=True)
    process(HIGHLIGHTED, de_red=False)
