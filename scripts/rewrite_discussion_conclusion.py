# -*- coding: utf-8 -*-
"""
Rewrite Section 6 (Discussion) and Section 7 (Conclusion) of the NRGA-Net
manuscript in the IJIES reference style: multiple flowing, justified paragraphs
with first-line indents.

- Revised copy: black text.
- Highlighted copy: all new text in red.
Formatting is cloned from the existing body paragraphs (firstLine=312, jc=both,
Times New Roman 11pt).
"""
import copy
import shutil

from docx import Document
from docx.oxml.ns import qn
from docx.text.run import Run

REVISED = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
HIGHLIGHTED = r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx"
W = qn('w:p').split('}')[0] + '}'
RED = 'C00000'

DISCUSSION = [
    "The obtained experimental results demonstrate that the proposed NRGA-Net framework "
    "effectively unifies pixel-level forgery localization and image-level detection for "
    "satellite imagery within a single network. The dilated DenseNet-201 semantic lane "
    "supplies scene context, whereas the forensic lane exposes the low-level traces that "
    "generative pipelines leave behind in the raw pixels. The measured outcomes indicate "
    "that these two evidence sources are complementary rather than redundant: in the "
    "controlled ablation (Table 8), removing the deep edge supervision causes the largest "
    "single drop in pooled IoU (94.48% to 93.71%), removing the frequency residual encoder "
    "costs 0.34 points, and removing the deformable attention inside the frequency "
    "cross-attention costs 0.26 points under an identical 40-epoch budget. Each stream "
    "therefore contributes localization evidence that the remaining components cannot "
    "fully replace.",

    "A single jointly trained model can serve three benchmarks from three generator "
    "families without per-dataset re-training, and joint training did not hurt the "
    "per-benchmark numbers. This demonstrates multi-domain joint training rather than "
    "zero-shot transfer to an unseen generator. The leave-one-generator-family-out study "
    "(Table 10) quantifies this central limitation: when the model is retrained on two "
    "families and evaluated zero-shot on the held-out family of the final-test split, IoU "
    "falls to 11.40% on LaMa, 34.53% on RePaint, and 2.49% on latent diffusion, far below "
    "the 96.58% single-pass pooled IoU. The forensic cues learned by the model, "
    "particularly the high-frequency residual and spectral artifacts, are therefore "
    "largely generator-specific. Achieving transfer to unseen generators remains an open "
    "problem, for which the released protocol and train-minus-family split indices "
    "provide a reproducible baseline.",

    "The comparison with prior work should be read through protocol differences. On "
    "Fake-Vaihingen, without test-time augmentation the model is slightly behind FECDNet "
    "(92.67% versus 93.47% IoU), and it is the augmentation that closes and reverses the "
    "gap. The Fake-LoveDA comparison is further complicated by the different evaluation "
    "windows noted in Section 5.1; the FECDNet values are therefore treated as "
    "literature-only comparisons rather than head-to-head rankings. The strongest "
    "defensible claim is parity or better under a common pooled protocol, with the "
    "decisive advantage lying in the unified multi-task scope: one model, one training "
    "run, and one calibration procedure serve localization, detection, and selective "
    "prediction simultaneously, whereas the reference frameworks train separate "
    "localization and detection networks.",

    "The component ablation admits a useful reading of where the gains come from. The "
    "spectral edge stream shows a smaller pooled effect than its per-benchmark role "
    "suggests, because its main contribution concentrates on the RePaint family, whose "
    "inpainting artifacts are most spectral (91.32% pooled versus 85.85% when removed). "
    "The distortion-bank augmentation slightly reduces clean-image IoU at this budget "
    "(94.48% versus 95.16% for the no-distortion control), yet the degradation sweep "
    "(Table 9) shows that it is decisive where it matters: the augmented model maintains "
    "88.41% IoU at JPEG quality 95 and 69.64% at quality 50 on unseen degradations, "
    "while Gaussian blur is the most destructive degradation (26.11% IoU at kernel 9). "
    "The augmentation's benefit therefore lies in degradation robustness rather than in "
    "clean-image accuracy, which is the intended operating regime for deployed "
    "forensics.",

    "All calibration quantities are fitted on the separate 769-image calibration split "
    "and are then applied unchanged to the final-test set, so the reported deployment "
    "figures are unbiased final estimates. Temperature scaling selected T = 0.50 and the "
    "mask-threshold sweep left the operating point at 0.5 with no per-domain tuning. "
    "Under the selective-prediction rule, only 4 of 3,076 final-test images (0.1%) are "
    "routed to expert review, and accuracy on the auto-decided images is 99.28%, which "
    "indicates that the uncertainty bound is conservative enough to keep almost the "
    "entire test set inside the automatic regime while still flagging the genuinely "
    "ambiguous cases.",

    "The trade-off between detection performance and computational cost should also be "
    "stated. The dual-domain forensic lane and the three-gate fusion bring the parameter "
    "budget to 30.9M, and the multi-scale test-time augmentation roughly doubles the "
    "inference cost; it recovers 1.22 IoU points on the pooled final test (96.58% to "
    "97.80%). The extra cost is an offline-training and optional-deployment choice "
    "rather than a necessity, because the single-pass model already exceeds 96% pooled "
    "IoU at native 256x256 crops, and the residual logit decoder keeps the localization "
    "head lightweight. This makes the framework practical for tile-based screening of "
    "full satellite scenes, where the detection head first discards authentic tiles and "
    "the localization mask is computed only for the flagged ones.",

    "Though the suggested framework shows strong results on the three benchmarks, one "
    "should also admit that some weaknesses exist. The CBFH provenance arm is validated "
    "architecturally rather than empirically: its contrastive training is inactive in "
    "the reported localization runs, collision behaviour remains future work, and the "
    "ablation confirms that the branch contributes no localization evidence, so it "
    "should be read as a prototype. The focus is on RGB imagery, so multispectral "
    "forensics remains open; Fake-LocalDiff uses a single latent-diffusion pipeline, "
    "which limits generator diversity within the new set; and multi-seed variability for "
    "the ablation has not yet been reported. These limitations define the boundary of "
    "the claims rather than invalidating them.",

    "In general, the experimental results validate that the proposed multi-domain "
    "framework is able to localize and detect satellite-image forgeries from three "
    "generator families with a single model, a single calibration procedure, and "
    "measured robustness under realistic degradations, while the leave-one-family-out "
    "study honestly bounds what the model cannot yet do. The global and local design "
    "choices allow the model to find small and informative forensic evidence with the "
    "result of strong pooled accuracy, and the released splits, seeds, and per-image "
    "predictions make every reported number traceable and independently checkable.",
]

CONCLUSION = [
    "This paper presented NRGA-Net, a jointly trained network that unifies pixel-level "
    "forgery localization, image-level detection, and an architecturally integrated "
    "64-bit forensic-hash branch for satellite imagery. The network couples a dilated "
    "DenseNet-201 encoder with a dual-domain forensic lane that fuses a fixed-filter "
    "noise-residual stream and a learnable frequency-residual encoder through "
    "asymmetric cross-frequency attention, while a spectral-edge stream recovers "
    "boundaries in the Fourier domain. The new Fake-LocalDiff benchmark adds "
    "prompt-driven latent-diffusion local replacements to the existing inpainting "
    "benchmarks, and all experiments follow an immutable train, calibration, and "
    "final-test split with released indices.",

    "Results of the experiments validate the ability of the suggested framework to "
    "achieve robust supervised forgery localization on the independent final-test set. "
    "A single model trained jointly on three generator families reaches 95.58% and "
    "91.75% IoU on the Fake-Vaihingen LaMa and RePaint families, 97.63% and 98.04% on "
    "Fake-LoveDA, and 98.71% on Fake-LocalDiff, with 99.22% detection accuracy, 99.71% "
    "detection AUC, and a pooled forged-class IoU of 97.80% (Dice = F1 = 98.89%) with "
    "test-time augmentation; by Eq. (10) the forged-class Dice equals F1 and all pooled "
    "metrics derive from the same confusion counts, which removes the internal "
    "inconsistencies flagged in the earlier version of this work.",

    "The controlled ablation shows that deep edge supervision and the frequency "
    "residual encoder are the most influential components, that the distortion bank "
    "trades a marginal clean-image cost for measured robustness across JPEG "
    "compression, Gaussian blur, and additive noise, and that the CBFH branch "
    "contributes no localization evidence and is therefore reported as a prototype. "
    "The leave-one-family-out study shows that zero-shot transfer to an unseen "
    "generator family is poor (IoU 2.49% to 34.53%), confirming that the forensic cues "
    "are generator-specific; all generalization claims of this work are accordingly "
    "limited to multi-domain joint training.",

    "The further research would be conducted on model validation with multi-seed "
    "variability for the principal ablation comparisons, on extending the CBFH "
    "provenance arm with contrastive training, retrieval matching, and collision "
    "analysis, on extending the benchmark to multispectral data and additional "
    "generator families, and on improving cross-generator transfer beyond the "
    "multi-domain joint-training regime. Even though the proposed framework is "
    "validated on three RGB benchmarks, the released code, split indices, seeds, and "
    "per-image predictions are designed to make such follow-up studies fully "
    "controlled and directly comparable.",
]


def make_para(doc, tmpl_p, text, red):
    p = copy.deepcopy(tmpl_p)
    # remove all runs and bookmarks, keep pPr
    for child in list(p):
        if child.tag != W + 'pPr':
            p.remove(child)
    r = p.makeelement(W + 'r', {})
    t = p.makeelement(W + 't', {})
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    p.append(r)
    if red:
        rpr = r.makeelement(W + 'rPr', {})
        color = rpr.makeelement(W + 'color', {W + 'val': RED})
        rpr.append(color)
        r.insert(0, rpr)
    return p


def rewrite(doc, red):
    body = doc.element.body
    paras = body.findall(W + 'p')

    def para_text(p):
        return ''.join(t.text or '' for t in p.iter(W + 't')).strip()

    # locate headings and the following section start
    idx_disc = idx_conc = idx_end = None
    for i, p in enumerate(paras):
        t = para_text(p)
        if t == '6. Discussion':
            idx_disc = i
        elif t == '7. Conclusion':
            idx_conc = i
        elif idx_conc is not None and t in ('Conflicts of Interest', 'Author Contributions',
                                            'Data and Code Availability', 'References'):
            idx_end = i
            break
    if idx_disc is None or idx_conc is None or idx_end is None:
        raise RuntimeError(f'anchors not found: disc={idx_disc} conc={idx_conc} end={idx_end}')

    tmpl = paras[idx_disc + 1]  # existing body paragraph with correct pPr
    # delete old content paragraphs
    for p in paras[idx_disc + 1: idx_conc]:
        body.remove(p)
    # re-locate (indices shifted)
    paras = body.findall(W + 'p')
    idx_conc = idx_end = None
    for i, p in enumerate(paras):
        t = para_text(p)
        if t == '7. Conclusion':
            idx_conc = i
        elif idx_conc is not None and t in ('Conflicts of Interest', 'Author Contributions',
                                            'Data and Code Availability', 'References'):
            idx_end = i
            break
    for p in paras[idx_conc + 1: idx_end]:
        body.remove(p)

    # insert new paragraphs after each heading
    paras = body.findall(W + 'p')
    idx_disc = idx_conc = None
    for i, p in enumerate(paras):
        t = para_text(p)
        if t == '6. Discussion':
            idx_disc = i
        elif t == '7. Conclusion':
            idx_conc = i
    anchor = paras[idx_disc]
    for txt in DISCUSSION:
        el = make_para(doc, tmpl, txt, red)
        anchor.addnext(el)
        anchor = el
    anchor = paras[idx_conc]
    for txt in CONCLUSION:
        el = make_para(doc, tmpl, txt, red)
        anchor.addnext(el)
        anchor = el


def process(path, red):
    shutil.copyfile(path, path + '.bak8')
    doc = Document(path)
    rewrite(doc, red)
    doc.save(path)
    print(f'{path}: rewritten (red={red})')


if __name__ == '__main__':
    process(REVISED, red=False)
    process(HIGHLIGHTED, red=True)
