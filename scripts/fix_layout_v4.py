# -*- coding: utf-8 -*-
"""
v4 fixes for the IJIES manuscript:
  1. Header (Received/Revised + page number) and footer (journal line, DOI,
     license) copied from the IJIES template into section 0 (all pages).
  2. INASS logo banner (EMF image) copied from the template to the top of page 1.
  3. Column layout rebuilt per element: title block single-column; figures and
     tables (plus their captions/labels) single-column full width; ALL other
     text two-column. Replaces the coarse per-section logic that left the
     tail text (p.15+) single-column.
  4. Calibration table ("Quantity") repaired: stray "-" rows removed; forced
     full width so it can never split across columns again.

Run on the current (already v3-formatted) documents; idempotent.
"""
import copy
import io
import re
import shutil
import zipfile

from docx import Document
from docx.oxml.ns import qn

import lxml.etree as et

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
TPL = r"D:\Dataset\pp3\(Ver. 2025.5.18) IJIES_Format.docx"

W = qn('w:p').split('}')[0] + '}'
R_EMBED = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
WP_EXTENT = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent'
EMU_IN = 914400


def para_text(el):
    return ''.join(t.text or '' for t in el.iter(W + 't'))


def image_widths(el):
    return [int(e.get('cx')) / EMU_IN
            for e in el.iter(WP_EXTENT)]


def is_boundary(child):
    if child.tag != W + 'p':
        return False
    ppr = child.find(W + 'pPr')
    return ppr is not None and ppr.find(W + 'sectPr') is not None


# ------------------------------------------------------------------ headers
def copy_header_footer(doc):
    z = zipfile.ZipFile(TPL)
    sec0 = doc.sections[0]
    for kind, fname in (('header', 'word/header1.xml'), ('footer', 'word/footer1.xml')):
        obj = sec0.header if kind == 'header' else sec0.footer
        obj.is_linked_to_previous = False
        target = obj._element
        for child in list(target):
            target.remove(child)
        src = et.fromstring(z.read(fname))
        tag = W + kind
        for child in src:
            if child.tag == tag:
                continue
            target.append(copy.deepcopy(child))


# ------------------------------------------------------------------ logo
def copy_logo(doc):
    z = zipfile.ZipFile(TPL)
    tdoc = Document(TPL)
    logo_p = copy.deepcopy(tdoc.paragraphs[0]._p)
    # find which media file rId8 points to
    rels = et.fromstring(z.read('word/_rels/document.xml.rels'))
    rid_map = {}
    for rel in rels:
        rid_map[rel.get('Id')] = rel.get('Target')
    # the blip in paragraph 0
    blip = logo_p.find('.//' + '{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
    old_rid = blip.get(R_EMBED)
    target = rid_map[old_rid]  # e.g. media/image1.emf
    blob = z.read('word/' + target)
    # add part to the revised package
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    partname = PackURI('/word/media/logo_banner.emf')
    try:
        part = Part(partname, 'image/x-emf', package=doc.part.package, blob=blob)
    except TypeError:
        part = Part(partname, 'image/x-emf', blob=blob, package=doc.part.package)
    new_rid = doc.part.relate_to(part, RT.IMAGE)
    blip.set(R_EMBED, new_rid)
    # unique docPr id
    for docpr in logo_p.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr'):
        docpr.set('id', '901')
    body = doc.element.body
    body.insert(0, logo_p)


# ------------------------------------------------------------------ columns
def build_boundary(single):
    """Empty paragraph whose sectPr ends the previous group with given cols."""
    p = et.SubElement(et.Element('root'), W + 'p')  # detached
    ppr = p.makeelement(W + 'pPr', {})
    p.append(ppr)
    sectpr = ppr.makeelement(W + 'sectPr', {})
    ppr.append(sectpr)
    typ = sectpr.makeelement(W + 'type', {W + 'val': 'continuous'})
    sectpr.append(typ)
    sectpr.append(sectpr.makeelement(W + 'pgSz', {W + 'w': '11909', W + 'h': '16834',
                                                  W + 'code': '9'}))
    sectpr.append(sectpr.makeelement(W + 'pgMar',
                                     {W + 'top': '1440', W + 'right': '1080',
                                      W + 'bottom': '1440', W + 'left': '1080',
                                      W + 'header': '850', W + 'footer': '567',
                                      W + 'gutter': '0'}))
    cols_attrs = {W + 'space': '432'}
    if not single:
        cols_attrs[W + 'num'] = '2'
    sectpr.append(sectpr.makeelement(W + 'cols', cols_attrs))
    sectpr.append(sectpr.makeelement(W + 'docGrid', {W + 'linePitch': '360'}))
    return p


CAP_PAT = re.compile(r'^(Fig\.\s*\d+|Table\s+\d+\.)')


def compute_modes(children):
    n = len(children)
    modes = ['two'] * n

    # title block: everything before the Introduction heading
    intro_idx = None
    for i, ch in enumerate(children):
        if ch.tag == W + 'p' and para_text(ch).strip() in ('1. Introduction', '1. Introduction.'):
            intro_idx = i
            break
    if intro_idx is None:
        for i, ch in enumerate(children):
            if ch.tag == W + 'p' and para_text(ch).strip().startswith('1. Introduction'):
                intro_idx = i
                break
    for i in range(intro_idx if intro_idx is not None else 0):
        modes[i] = 'one'

    # figures: image paragraphs wider than 3.4in -> full; narrower -> two (in-column)
    img_mode = {}
    for i, ch in enumerate(children):
        if ch.tag == W + 'p':
            ws = image_widths(ch)
            if ws and max(ws) > 3.4:
                modes[i] = 'full'
                img_mode[i] = 'full'
            elif ws:
                modes[i] = 'two'
                img_mode[i] = 'two'

    # tables: all full width (prevents column splitting)
    for i, ch in enumerate(children):
        if ch.tag == W + 'tbl':
            modes[i] = 'full'

    # figure labels/captions after an image paragraph: same mode as the figure,
    # up to and including the Fig. caption
    for i, m in img_mode.items():
        j = i + 1
        while j < n:
            ch = children[j]
            if ch.tag != W + 'p':
                break
            modes[j] = m
            if CAP_PAT.match(para_text(ch).strip()):
                break
            if para_text(ch).strip() == '':
                break
            j += 1

    # table captions immediately before a table: full
    for i, ch in enumerate(children):
        if ch.tag == W + 'tbl':
            j = i - 1
            while j >= 0 and children[j].tag == W + 'p':
                t = para_text(children[j]).strip()
                if CAP_PAT.match(t) or t == '':
                    modes[j] = 'full'
                    if CAP_PAT.match(t):
                        break
                    j -= 1
                else:
                    break
    return modes


def rebuild_columns(doc):
    body = doc.element.body
    children = [ch for ch in body
                if not (ch.tag == W + 'sectPr') and not is_boundary(ch)]
    modes = compute_modes(children)

    # drop old boundaries
    for ch in list(body):
        if is_boundary(ch):
            body.remove(ch)

    # insert new boundaries on mode changes (boundary carries PREVIOUS mode)
    prev = None
    for ch, m in zip(children, modes):
        if prev is not None and m != prev:
            ch.addprevious(build_boundary(prev == 'one' or prev == 'full'))
        prev = m

    # body-level sectPr governs the final group (text after the last
    # figure/table -> two columns; num="2" means two columns, absent num = one)
    last_sectpr = body.find(W + 'sectPr')
    if last_sectpr is not None:
        cols = last_sectpr.find(W + 'cols')
        if cols is None:
            cols = last_sectpr.makeelement(W + 'cols', {})
            last_sectpr.append(cols)
        if modes and modes[-1] == 'two':
            cols.set(W + 'num', '2')
        elif cols.get(W + 'num'):
            del cols.attrib[W + 'num']
        cols.set(W + 'space', '432')
    return children, modes


def fix_calibration_table(children):
    """Remove stray dash-only rows; returns count."""
    n = 0
    for ch in children:
        if ch.tag != W + 'tbl':
            continue
        for tr in list(ch.findall(W + 'tr')):
            cells = [''.join(t.text or '' for t in tc.iter(W + 't')).strip()
                     for tc in tr.findall(W + 'tc')]
            if cells and all(c in ('-', '') for c in cells):
                ch.remove(tr)
                n += 1
    return n


def process(path):
    shutil.copyfile(path, path + '.bak4')
    doc = Document(path)
    # NOTE: rebuild_columns drops boundary paragraphs; the header/footer
    # references live on the first boundary's sectPr, so rebuild FIRST,
    # then attach header/footer (they re-add the reference themselves).
    children, modes = rebuild_columns(doc)
    n_dash = fix_calibration_table(children)
    copy_header_footer(doc)
    copy_logo(doc)
    from collections import Counter
    print(f'{path}: modes={dict(Counter(modes))}, dash-rows removed={n_dash}')
    doc.save(path)
    print(f'{path}: saved')


if __name__ == '__main__':
    process(REVISED)
    process(HIGHLIGHTED)
