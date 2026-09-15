# -*- coding: utf-8 -*-
"""v6: restore headerReference/footerReference on the first section's sectPr
(lost when v4 rebuilt the boundary paragraphs)."""
from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

HDR_REF = W + 'headerReference'
FTR_REF = W + 'footerReference'


def find_header_footer_rids(doc):
    rid = {}
    for rel in doc.part.rels.values():
        if rel.reltype.endswith('/header'):
            rid['header'] = rel.rId
        elif rel.reltype.endswith('/footer'):
            rid['footer'] = rel.rId
    return rid


def add_refs(sectpr, rid):
    for tag, key in ((HDR_REF, 'header'), (FTR_REF, 'footer')):
        for existing in sectpr.findall(tag):
            sectpr.remove(existing)
        el = sectpr.makeelement(tag, {W + 'type': 'default', R + 'id': rid[key]})
        sectpr.insert(0, el)  # header/footer refs come first in sectPr


def process(path):
    doc = Document(path)
    rid = find_header_footer_rids(doc)
    if 'header' not in rid or 'footer' not in rid:
        print(f'{path}: header/footer parts missing, skip')
        return
    body = doc.element.body
    # first boundary paragraph's sectPr = first section
    for ch in body:
        if ch.tag == W + 'p':
            ppr = ch.find(W + 'pPr')
            if ppr is not None and ppr.find(W + 'sectPr') is not None:
                add_refs(ppr.find(W + 'sectPr'), rid)
                print(f'{path}: refs added to first sectPr ({rid})')
                break
    doc.save(path)


if __name__ == '__main__':
    process(REVISED)
    process(HIGHLIGHTED)
