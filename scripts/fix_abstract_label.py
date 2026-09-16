# -*- coding: utf-8 -*-
"""Re-add the bold 'Abstract:' label run before the rewritten abstract text."""
import copy

from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'


def para_text(p):
    return ''.join(t.text or '' for t in p.iter(W + 't')).strip()


def process(path, make_red):
    doc = Document(path)
    for p in doc.element.body.findall(W + 'p'):
        if para_text(p).startswith('Modern satellite imagery has become vulnerable'):
            runs = p.findall(W + 'r')
            tmpl = runs[0]
            # label run
            label = copy.deepcopy(tmpl)
            for el in list(label):
                if not el.tag.endswith('}rPr'):
                    label.remove(el)
            rpr = label.find(W + 'rPr')
            if rpr is None:
                rpr = label.makeelement(W + 'rPr', {})
                label.insert(0, rpr)
            # bold, and strip any color
            b = rpr.find(W + 'b')
            if b is None:
                rpr.append(rpr.makeelement(W + 'b', {}))
            for c in rpr.findall(W + 'color'):
                rpr.remove(c)
            t = label.makeelement(W + 't', {})
            t.text = 'Abstract: '
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            label.append(t)
            if make_red:
                rpr.append(rpr.makeelement(W + 'color', {W + 'val': 'C00000'}))
            tmpl.addprevious(label)
            doc.save(path)
            print(f'{path}: Abstract: label restored (red={make_red})')
            return
    print(f'{path}: abstract paragraph not found!')


if __name__ == '__main__':
    process(REVISED, make_red=False)
    process(HIGHLIGHTED, make_red=True)
