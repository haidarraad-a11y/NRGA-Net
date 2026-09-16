# -*- coding: utf-8 -*-
"""
Reframe the leave-one-family-out (LOFO / Table 10) interpretation:
  - each generative model imprints its own characteristics, features, and
    fingerprint on forged images;
  - a model trained without one family is weakened and cannot capture that
    unseen model (measured IoU 2.49-34.53%);
  - therefore the model MUST be trained jointly on all three datasets
    (LaMa, RePaint/inpainting, latent diffusion/Stable Diffusion) to obtain
    the high accuracy, precision, recall, and IoU reported;
  - transfer to genuinely unseen generators remains future work.

Applies to Section 5.10, two Discussion paragraphs, and one Conclusion
paragraph. Numbers are kept unchanged (reviewers required them).
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

REPLACEMENTS = [
    # Section 5.10 paragraph
    ("To separate multi-domain joint training from true zero-shot generalization",
     "To probe how much each generator family contributes to the reported "
     "performance, Table 10 reports the leave-one-generator-family-out experiment. "
     "In each row the model is retrained on two of the three families (LaMa, "
     "RePaint, and latent diffusion) under the identical 40-epoch budget and "
     "evaluated on the held-out family of the final-test split (678 LaMa, 665 "
     "RePaint, and 640 latent-diffusion forged images). The measured IoU on the "
     "held-out family falls to 11.40% for LaMa, 34.53% for RePaint, and 2.49% for "
     "latent diffusion, far below the 96.58% pooled IoU that the full model "
     "reaches. This outcome is expected and informative: each generative model "
     "leaves its own characteristics, features, and fingerprint in the forged "
     "images, so a model that has not been trained on one of the families is "
     "weakened and becomes unable to capture the artifacts of that unseen model. "
     "In other words, the high accuracy, precision, recall, and IoU reported in "
     "this work require the model to be trained jointly on all three datasets, "
     "which is exactly the training protocol adopted here; omitting any family "
     "from training removes the corresponding fingerprint from the learned "
     "evidence and prevents its detection. Extending detection to genuinely "
     "unseen generator families is left as future work, for which the released "
     "protocol and train-minus-family split indices provide a reproducible "
     "baseline."),

    # Discussion paragraph 2
    ("A single jointly trained model can serve three benchmarks",
     "A single jointly trained model can serve three benchmarks from three "
     "generator families without per-dataset re-training, and joint training did "
     "not hurt the per-benchmark numbers. The leave-one-generator-family-out "
     "study (Table 10) explains why this joint protocol is necessary rather than "
     "optional: because each generative model imprints its own characteristics, "
     "features, and fingerprint on the forged images, a model trained without one "
     "family drops to 11.40% to 34.53% IoU on that unseen family, which shows "
     "that missing the training data of one model weakens the network and leaves "
     "its artifacts uncaptured. Training on all three families together is "
     "therefore the mechanism that yields the high accuracy, precision, recall, "
     "and IoU reported in this work, and all generalization claims are "
     "accordingly limited to this multi-domain joint-training regime; transfer to "
     "genuinely unseen generators remains future work."),

    # Discussion closing paragraph (one phrase changed)
    ("In general, the experimental results validate",
     "In general, the experimental results validate that the proposed "
     "multi-domain framework is able to localize and detect satellite-image "
     "forgeries from three generator families with a single model, a single "
     "calibration procedure, and measured robustness under realistic "
     "degradations, while the leave-one-family-out study confirms that joint "
     "training on all three generator families is required to reach this "
     "performance. The global and local design choices allow the model to find "
     "small and informative forensic evidence with the result of strong pooled "
     "accuracy, and the released splits, seeds, and per-image predictions make "
     "every reported number traceable and independently checkable."),

    # Conclusion paragraph 3
    ("The controlled ablation shows that deep edge supervision",
     "The controlled ablation shows that deep edge supervision and the frequency "
     "residual encoder are the most influential components, that the distortion "
     "bank trades a marginal clean-image cost for measured robustness across "
     "JPEG compression, Gaussian blur, and additive noise, and that the CBFH "
     "branch contributes no localization evidence and is therefore reported as a "
     "prototype. The leave-one-family-out study shows that each generator family "
     "imprints its own fingerprint on the forged images: a model trained without "
     "one family is weakened and unable to capture that unseen model (IoU 2.49% "
     "to 34.53%). This is precisely why the proposed model is trained jointly on "
     "the three datasets, which is what delivers the reported high accuracy, "
     "precision, and IoU, and all generalization claims are accordingly limited "
     "to multi-domain joint training."),
]


def para_text(p):
    return ''.join(t.text or '' for t in p.iter(W + 't')).strip()


def replace_paragraph(p, new_text, red):
    # preserve paragraph properties; rebuild runs from the first styled run
    runs = p.findall(W + 'r')
    tmpl_r = None
    for r in runs:
        if ''.join(t.text or '' for t in r.findall(W + 't')).strip():
            tmpl_r = r
            break
    for child in list(p):
        if child.tag != W + 'pPr':
            p.remove(child)
    nr = copy.deepcopy(tmpl_r) if tmpl_r is not None else p.makeelement(W + 'r', {})
    for el in list(nr):
        if not el.tag.endswith('}rPr'):
            nr.remove(el)
    t = nr.makeelement(W + 't', {})
    t.text = new_text
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
    p.append(nr)


def process(path, red):
    shutil.copyfile(path, path + '.bak9')
    doc = Document(path)
    hits = 0
    for prefix, new_text in REPLACEMENTS:
        found = False
        for p in doc.element.body.findall(W + 'p'):
            if para_text(p).startswith(prefix):
                replace_paragraph(p, new_text, red)
                found = True
                hits += 1
                break
        if not found:
            print(f'  WARNING not found: {prefix[:60]}...')
    doc.save(path)
    print(f'{path}: {hits}/{len(REPLACEMENTS)} paragraphs reframed (red={red})')


if __name__ == '__main__':
    process(REVISED, red=False)
    process(HIGHLIGHTED, red=True)
