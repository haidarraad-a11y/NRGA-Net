# -*- coding: utf-8 -*-
"""
Rewrite the Abstract in the IJIES reference style (long, flowing, single
paragraph: threat motivation -> framework -> evaluation -> fingerprint/LOFO
rationale -> headline numbers -> applicability/future work -> scope caveat).
The phrase "generalize across domains" is removed (flagged by reviewers).
Revised copy: black. Highlighted copy: red.
"""
import copy
import shutil

from docx import Document
from docx.oxml.ns import qn

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'
RED = 'C00000'

ABSTRACT = (
    "Modern satellite imagery has become vulnerable to numerous manipulation "
    "threats due to the rapid progress of generative models, and forensic "
    "analysis is important to detect malicious edits and to secure the critical "
    "decisions that rely on remote-sensing data. Though, the purpose of this "
    "study is driven by the security issues that can be witnessed in "
    "satellite-based decision making, the presented framework is tested as a "
    "unified forgery localization and detection system with a reference to the "
    "benchmark forgery datasets. The suggested model combines a dilated "
    "DenseNet-201 semantic encoder with a dual-domain forensic lane that fuses "
    "a fixed-filter noise-residual stream and a learnable frequency-residual "
    "encoder through asymmetric cross-frequency attention, while a spectral-edge "
    "stream recovers boundaries in the Fourier domain, and the new "
    "Fake-LocalDiff benchmark adds prompt-driven latent-diffusion local "
    "replacements to the existing inpainting benchmarks. The proposed model is "
    "implemented and evaluated as a supervised forgery detector using labeled "
    "data, and it is evaluated on Fake-Vaihingen, Fake-LoveDA, and "
    "Fake-LocalDiff under an immutable train, calibration, and final-test "
    "protocol to analyze its effectiveness in localizing forged and authentic "
    "regions and in detecting manipulated images. Because each generative model "
    "imprints its own characteristics, features, and fingerprint on the forged "
    "images, the model is trained jointly on the three datasets (LaMa "
    "inpainting, RePaint, and latent diffusion), and the leave-one-family-out "
    "study shows that omitting one family from training weakens the model and "
    "prevents it from capturing the artifacts of the unseen generator. The "
    "performance on the independent final-test set is astounding, with a 97.80% "
    "pooled forged-class IoU, a Dice = F1 of 98.89%, a precision of 98.31%, a "
    "recall of 99.48%, and a 99.22% detection accuracy. The applicability of "
    "the proposed model to multispectral imagery and to unseen generator "
    "families is considered indicative, and future work will focus on "
    "contrastive validation of the provenance hash and on improving "
    "cross-generator transfer. The experimental analysis is done based on the "
    "three benchmarks, albeit under the incentive of the satellite-imagery "
    "security issues, thus the findings provide insights into the performance "
    "of the unified supervised framework on the benchmark protocols."
)


def para_text(p):
    return ''.join(t.text or '' for t in p.iter(W + 't')).strip()


def process(path, red):
    shutil.copyfile(path, path + '.bak10')
    doc = Document(path)
    target = None
    for p in doc.element.body.findall(W + 'p'):
        if para_text(p).startswith('Abstract:'):
            target = p
            break
    if target is None:
        raise RuntimeError('Abstract paragraph not found')
    runs = target.findall(W + 'r')
    tmpl_r = runs[0] if runs else None
    for child in list(target):
        if child.tag != W + 'pPr':
            target.remove(child)
    nr = copy.deepcopy(tmpl_r)
    for el in list(nr):
        if not el.tag.endswith('}rPr'):
            nr.remove(el)
    t = nr.makeelement(W + 't', {})
    t.text = ABSTRACT
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    nr.append(t)
    if red:
        rpr = nr.find(W + 'rPr')
        if rpr is None:
            rpr = nr.makeelement(W + 'rPr', {})
            nr.insert(0, rpr)
        for c in rpr.findall(W + 'color'):
            rpr.remove(c)
        rpr.append(rpr.makeelement(W + 'color', {W + 'val': RED}))
    target.append(nr)
    doc.save(path)
    print(f'{path}: abstract rewritten (red={red}), length={len(ABSTRACT)} chars')


if __name__ == '__main__':
    process(REVISED, red=False)
    process(HIGHLIGHTED, red=True)
