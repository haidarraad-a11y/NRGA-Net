# -*- coding: utf-8 -*-
"""v7: fix inverted column logic on the body-level sectPr (last section).
Text after the last figure/table group (Discussion, Conclusion, References,
declarations) must flow in TWO columns."""
import shutil

from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'


def process(path):
    shutil.copyfile(path, path + '.bak7')
    doc = Document(path)
    body = doc.element.body
    sectpr = body.find(W + 'sectPr')
    cols = sectpr.find(W + 'cols')
    if cols is None:
        cols = sectpr.makeelement(W + 'cols', {})
        sectpr.append(cols)
    cols.set(W + 'num', '2')
    cols.set(W + 'space', '432')
    doc.save(path)
    print(f'{path}: last section set to two columns')


if __name__ == '__main__':
    process(REVISED)
    process(HIGHLIGHTED)
