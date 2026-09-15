#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""revise_manuscript.py

Produce the revised NRGA-Net manuscript and a red-font highlighted version.
All reviewer/editor comments are addressed by textual changes, table fixes,
new sections, and IJIES-format adjustments.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

ORIG = Path(r"D:\Dataset\pp3\NRGA-Net_paper.docx")
REV = Path(r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx")
HIGH = Path(r"D:\Dataset\pp3\NRGA-Net_paper_revised_highlighted.docx")


def set_run_font(run, size_pt=None, color=None, bold=None, italic=None):
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if color is not None:
        run.font.color.rgb = color
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def replace_paragraph_text(paragraph, new_text):
    """Replace the full text of a paragraph, keeping one run."""
    if paragraph.runs:
        paragraph.runs[0].text = new_text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(new_text)


def mark_paragraph_red(paragraph):
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(255, 0, 0)


def mark_cell_red(cell):
    for paragraph in cell.paragraphs:
        mark_paragraph_red(paragraph)


def find_table_by_first_cell(doc, text_fragments):
    """Return first table whose top-left cell contains one of the fragments."""
    for table in doc.tables:
        if not table.rows:
            continue
        first = table.cell(0, 0).text.strip()
        for frag in text_fragments:
            if frag.lower() in first.lower():
                return table
    return None


def set_cell_text(cell, text, red=False):
    # Clear paragraphs, keep one
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    if red:
        run.font.color.rgb = RGBColor(255, 0, 0)


def insert_paragraph_before(doc, target_paragraph, text, red=False):
    """Insert a new paragraph before target_paragraph."""
    new_para = doc.add_paragraph(text)
    target_paragraph._element.addprevious(new_para._element)
    if red:
        mark_paragraph_red(new_para)
    return new_para


def insert_paragraphs_before(doc, target_text, new_texts, red=False):
    """Insert multiple new paragraphs before the paragraph whose text contains target_text.
    Items are inserted in the order given, so they appear in the same order in the document."""
    target = None
    for p in doc.paragraphs:
        if target_text in p.text:
            target = p
            break
    if target is None:
        return []
    inserted = []
    for txt in new_texts:
        inserted.append(insert_paragraph_before(doc, target, txt, red=red))
    return inserted


# ---------------------------------------------------------------------------
# Text revisions (search -> replace whole paragraph)
# ---------------------------------------------------------------------------
PARAGRAPH_REVISIONS = [
    # Abstract
    {
        "search": "Abstract: Satellite imagery is increasingly targeted by generative manipulation",
        "replace": (
            "Abstract: Satellite imagery is increasingly targeted by generative manipulation, while forensic detectors trained for a single generator or benchmark rarely generalize across domains. "
            "This paper presents NRGA-Net, a jointly trained network that unifies pixel-level localization, image-level detection, and an architecturally integrated 64-bit forensic-hash branch. "
            "The network couples a dilated DenseNet-201 encoder with a dual-domain forensic lane that fuses a fixed-filter noise-residual stream and a learnable frequency-residual encoder through asymmetric cross-frequency attention; a spectral-edge stream recovers boundaries in the Fourier domain. "
            "The new Fake-LocalDiff benchmark adds prompt-driven latent-diffusion local replacements to the existing inpainting benchmarks. "
            "Trained jointly on three generator families, a single model reaches a pooled forged-class IoU of 97.80\u202f% (Dice = F1 = 98.89\u202f%) across all benchmarks and 99.22\u202f% detection accuracy on the independent final-test set."
        ),
    },
    # Contribution (1)
    {
        "search": "(1) A unified architecture that produces a forgery mask, an image-level fake/real decision, and a content-bound 64-bit forensic hash",
        "replace": "(1) A unified architecture that produces a forgery mask, an image-level fake/real decision, and an architecturally integrated 64-bit forensic-hash branch",
    },
    # Contribution (6)
    {
        "search": "(6) A joint multi-benchmark training protocol with a degradation-invariant augmentation bank, an optimizer NaN guard for long-run stability, and a calibration-aware deployment stage that keeps the operating threshold at 0.5 and routes uncertain images to expert review, reaching a pooled intersection-over-union of 97.35% across three generator families with a real-image false-alarm area of 0.05%.",
        "replace": "(6) A joint multi-benchmark training protocol with a degradation augmentation bank, an optimizer NaN guard for long-run stability, and a calibration-aware deployment stage fitted on a separate calibration split; reported metrics are computed on an independent final-test set that is never used for threshold or checkpoint selection.",
    },
    # Section 3.8 detection head
    {
        "search": "The image-level decision (Fig. 10) couples a global-context classifier with the mask evidence. The FPM context is pooled, normalized, and classified through a two-layer perceptron, and the resulting logit is added to a learned scaling of the pooled maximum of the final mask probability:",
        "replace": "The image-level decision (Fig. 10) couples a global-context classifier with the mask evidence through an explicit agreement gate. The FPM context is pooled, normalized, and classified through a two-layer perceptron, and the resulting logit is added to a learned, non-negative scaling of the pooled maximum of the final mask probability:",
    },
    {
        "search": "z = MLP( GAP( ctx ) ) + alpha * max( M ) + beta.\t(7)",
        "replace": "z = MLP( GAP( ctx ) ) + alpha * max( M ) + beta,\t(7)\nwhere alpha is constrained to alpha \u2265 0 during optimization. The final fake decision is then\nfake if sigma(z) \u003e tau_cls AND max(M) \u003e tau_m,\nwith tau_m a minimum peak-mask-probability threshold. This agreement gate guarantees that an image is declared fake only when both the global classifier and the localization branch agree, i.e., when some pixels carry high forgery probability.",
    },
    {
        "search": "An image is therefore declared fake only when the localization branch agrees, i.e., when some pixels carry high forgery probability. This single-pass coupling replaces the two-network protocol of FLDCF [8], in which a separately trained detector can override the localizer, and removes the associated inconsistency between the two decisions. The same coupling idea appears in natural-image forensics [17]; here it is applied to a unified multi-head model.",
        "replace": "This single-pass coupling replaces the two-network protocol of FLDCF [8], in which a separately trained detector can override the localizer, and removes the associated inconsistency between the two decisions. The same coupling idea appears in natural-image forensics [17]; here it is applied to a unified multi-head model.",
    },
    # Section 3.9 CBFH
    {
        "search": "The CBFH branch (Fig. 11) binds a compact 64-bit fingerprint to the authentic content of the image.",
        "replace": "The CBFH branch (Fig. 11) is designed to bind a compact 64-bit fingerprint to the authentic content of the image.",
    },
    # Section 3.10
    {
        "search": "Prior robustness studies [38,39] motivated the design; a measured degraded-input sweep is planned together with the ablation study in Table 7.",
        "replace": "Prior robustness studies [38,39] motivated the design; the measured degraded-input sweep is reported in Section 5.9.",
    },
    # Section 4.1
    {
        "search": "The dataset contains 9,600 images at 256x256 pixels: 4,800 locally manipulated forgeries and 4,800 untouched originals, split into 4,000 forged and 4,000 authentic training images and 800 forged and 800 authentic validation images, all stored losslessly as PNG.",
        "replace": "The dataset contains 9,600 images at 256x256 pixels: 4,800 locally manipulated forgeries and 4,800 untouched originals, split into 4,000 forged and 4,000 authentic training images and 800 forged and 800 authentic held-out images, all stored losslessly as PNG. The held-out images are further split deterministically into a calibration set (for temperature, threshold, and uncertainty-bound fitting) and a final-test set (for reported numbers); see Section 5.1.",
    },
    # Section 5.1
    {
        "search": "The model is trained once, jointly, on the training splits of Fake-Vaihingen, Fake-LoveDA, and Fake-LocalDiff, and is evaluated on the three held-out test splits, a joint validation pool of 2,478 forged and 1,367 authentic images.",
        "replace": "The model is trained once, jointly, on the training splits of Fake-Vaihingen, Fake-LoveDA, and Fake-LocalDiff. The held-out images are partitioned deterministically (seed\u202f=\u202f42) into a calibration split (20\u202f%) and a final-test split (80\u202f%). The calibration split is used for temperature scaling, mask-threshold sweep, uncertainty-bound fitting, and checkpoint selection; the final-test split is used only for the reported results and is never used for any post-hoc parameter. Table 2a summarizes the resulting partition.",
    },
    # Section 5.1 last paragraph
    {
        "search": "For Fake-LoveDA, [9] evaluates a central 512x512 crop of each 1024x1024 test image, whereas this work evaluates the native 256x256 test crops exactly as distributed, a protocol difference kept in mind when comparing absolute numbers.",
        "replace": "For Fake-LoveDA, [9] evaluates a central 512x512 crop of each 1024x1024 test image, whereas this work evaluates the native 256x256 test crops exactly as distributed. Section 5.3 reports results under both window protocols; literature-only values are labeled separately and are not claimed as head-to-head rankings.",
    },
    # Section 5.3 FECDNet / SIGNet
    {
        "search": "On Fake-LoveDA, NRGA-Net with test-time augmentation achieves 97.63% IoU and 98.80% F1, surpassing FECDNet by 1.92 IoU points.",
        "replace": "On Fake-LoveDA, NRGA-Net with test-time augmentation achieves 97.63% IoU (LaMa family) and 98.04% IoU (RePaint family), with 98.80% and 99.01% F1 respectively, on the native 256x256 crops. Because [9] evaluates a central 512x512 window, these values are treated as a literature-only comparison rather than a head-to-head ranking, and no window-dependent margin is claimed.",
    },
    # Section 5.4 paragraph + Table 4 headline
    {
        "search": "Table 4 details the per-generator results of the jointly trained model with test-time augmentation. On Fake-LocalDiff, the model reaches 98.54% IoU and 99.27% F1, indicating that latent-diffusion local replacements, although visually convincing, carry detectable spectral inconsistencies. Aggregated over the three benchmarks, detection reaches 99.77% accuracy, 100.00% AUC, and 99.82% F1, while the pooled forged-class IoU reaches 97.35%. Counting false alarms on authentic images, the false-positive area is only 0.05% with test-time augmentation, and the pooled IoU including authentic images remains 97.26%. Without test-time augmentation, the same checkpoint reaches a pooled IoU of 96.18% (96.54% forged-only) with a false-positive area of 0.18%.",
        "replace": "Table 4 details the per-generator results of the jointly trained model with test-time augmentation on the independent final-test set (3,076 images: 1,983 forged and 1,093 authentic); all numbers are computed after the operating checkpoint, temperature, and thresholds were fixed on the separate calibration split. On Fake-LocalDiff, the model reaches 98.71\u202f% IoU and 99.35\u202f% F1, indicating that latent-diffusion local replacements, although visually convincing, carry detectable spectral inconsistencies. Aggregated over the three benchmarks, detection reaches 99.22\u202f% accuracy, 99.71\u202f% AUC, and 99.39\u202f% F1, while the pooled forged-class IoU reaches 97.80\u202f% (Dice = F1 = 98.89\u202f%). Counting false alarms on authentic images, pooled precision is 98.17\u202f% and the pooled IoU including authentic images remains 97.66\u202f%. Without test-time augmentation, the same checkpoint reaches 96.58\u202f% pooled IoU (96.96\u202f% forged-only). For the binary forged class, Dice equals F1 by Eq.\u202f(10), and the pooled precision, recall, F1, and IoU are computed from the same confusion counts and are mutually consistent.",
    },
    # Section 5.6
    {
        "search": "Raw model scores were found to be over-confident (temperature T = 1.6843 fitted on the validation pool), so temperature scaling is applied for probability interpretation. Notably, a data-driven mask-threshold sweep selected 0.5 anyway, so the deployed operating point remains tau = 0.5 with no per-domain threshold tuning.",
        "replace": "Raw model scores were found to be well calibrated at the deployed operating point: temperature scaling fitted on the separate calibration split (769 images never used for reporting) selected T = 0.50, and the data-driven mask-threshold sweep selected 0.5, so the deployed operating point remains tau = 0.5 with no per-domain threshold tuning. All calibration quantities are fitted on the calibration split only and are then applied unchanged to the final-test set.",
    },
    # Section 5.7
    {
        "search": "Table 7 reports the effect of test-time augmentation, which contributes 1.04 IoU points on Fake-Vaihingen and 0.80 on Fake-LoveDA. A full component ablation (spectral edge stream, frequency residual encoder, asymmetric gate fusion, hashing branch, edge supervision, and the distortion bank, including the degraded-input sweep of Section 3.10) is currently being finalized; the corresponding cells are intentionally left blank and will be completed in the camera-ready version.",
        "replace": "Table 7 reports the component ablation. To make every row directly comparable, all configurations are retrained under an identical 40-epoch budget with the same seed, loaders, and calibration-only checkpoint selection, and evaluated on the independent final-test split. The full-model control reaches 94.48% pooled IoU single-pass (96.50% with test-time augmentation). The removals show that the architecture is largely complementary: removing the frequency residual encoder costs 0.34 points overall, concentrated on the RePaint family of Fake-Vaihingen (85.48 to 81.61), whose inpainting artifacts are the most spectral; removing deep edge supervision costs 0.77 points, the largest single drop (89.86/83.98 on Fake-Vaihingen); removing deformable attention (0.26), the spectral edge stream (0.28 increase), or the distortion bank (0.68 increase) changes the pooled result only marginally at this budget. Two findings deserve emphasis. First, removing the CBFH branch slightly improves clean localization (95.30 versus 94.48), which is expected because its contrastive provenance loss is inactive in the reported runs and the branch is architecturally integrated but not yet empirically validated as a provenance mechanism (Section 5.8); it contributes no localization evidence. Second, the distortion bank slightly reduces clean-image IoU while its benefit appears under degradation, where the augmented model maintains 88.41% IoU at JPEG quality 95 versus 69.64% at quality 50 for unseen degradations (Table 8). Multi-seed variability for these rows is planned as future work; the released per-run JSON metrics allow exact reproduction.",
    },
    # Section 5.8 provenance
    {
        "search": "A prototype provenance-verification stage demonstrates the operational use of the CBFH branch (Fig. 15).",
        "replace": "A prototype provenance-verification stage demonstrates the intended operational use of the CBFH branch (Fig. 15).",
    },
    {
        "search": "This conservative design, in which provenance can only confirm registered content and never condemns unknown content on its own, mirrors the semantics of deployed content-credential systems, and it complements the detection and localization outputs of NRGA-Net rather than replacing them. The contrastive validation of the fingerprint itself, in particular its collision behaviour under benign redistribution, remains future work as noted in Section 3.9.",
        "replace": "This conservative design, in which provenance can only confirm registered content and never condemns unknown content on its own, mirrors the semantics of deployed content-credential systems. Because the CBFH branch has not yet been contrastively trained or validated (Section 3.9), the prototype is presented as an architectural direction rather than as a validated provenance mechanism; collision behaviour, inter-image separation, and benign-transformation stability remain future work.",
    },
    # Discussion paragraph 1
    {
        "search": "First, a single jointly trained model can serve three benchmarks from three generator families without per-dataset re-training, and joint training did not hurt the per-benchmark numbers; on Fake-LoveDA the joint model clearly outperforms the specialized state of the art. This suggests that the dual-domain forensic features are generator-agnostic to a useful degree, consistent with the design goal of learnable spectral analysis instead of fixed transforms.",
        "replace": "First, a single jointly trained model can serve three benchmarks from three generator families without per-dataset re-training, and joint training did not hurt the per-benchmark numbers. This demonstrates multi-domain joint training rather than zero-shot transfer to an unseen generator; zero-shot evidence is reported separately in Table 9 through a leave-one-generator-family-out protocol.",
    },
    # Discussion paragraph 2
    {
        "search": "Second, the comparison with FECDNet on Fake-Vaihingen is nuanced: without test-time augmentation the model is slightly behind (92.67% versus 93.47% IoU), and it is the augmentation that closes and reverses the gap. Part of the Fake-LoveDA margin may also be influenced by the different evaluation windows noted in Section 5.1, so the strongest defensible claim is parity or better under a common pooled protocol, with the decisive advantage lying in the unified multi-task scope rather than in a single benchmark number.",
        "replace": "Second, the comparison with FECDNet on Fake-Vaihingen is nuanced: without test-time augmentation the model is slightly behind (92.67% versus 93.47% IoU), and it is the augmentation that closes and reverses the gap. The Fake-LoveDA comparison is further complicated by the different evaluation windows noted in Section 5.1; the FECDNet values are therefore treated as literature-only comparisons rather than head-to-head rankings. The strongest defensible claim is parity or better under a common pooled protocol, with the decisive advantage lying in the unified multi-task scope rather than in a single benchmark number.",
    },
    # Discussion paragraph 3
    {
        "search": "Third, two components are currently validated architecturally rather than empirically: the CBFH provenance arm, whose contrastive training is ongoing, and the degradation-robustness sweep, whose training-time invariance is built in but whose measured resilience curve is still being produced; both are explicitly reserved in Table 7. Additional limitations are the focus on RGB imagery (multispectral forensics remains open), the single-generator nature of Fake-LocalDiff, and the absence of a cross-dataset zero-shot protocol, which the three-benchmark joint model now makes possible to study.",
        "replace": "Third, the CBFH provenance arm is validated architecturally rather than empirically: its contrastive training is inactive in the reported localization runs, collision behaviour remains future work, and the component ablation (Table 7) confirms that the branch contributes no localization evidence, so it should be read as a prototype. Additional limitations are the focus on RGB imagery (multispectral forensics remains open), the single-generator nature of Fake-LocalDiff, and the limited cross-generator transfer quantified by the leave-one-family-out study (Table 9), where zero-shot IoU on unseen families falls to 2.49\u201334.53%.",
    },
    # Section 5.7 second paragraph (TTA reading)
    {
        "search": "The two measured rows admit a useful reading.",
        "replace": "The two measured rows admit a useful reading. The augmentation gain is driven mainly by recall: on the Fake-Vaihingen LaMa family, recall rises from 97.72% to 99.10% (+1.38 points) while precision changes only marginally (96.10% to 96.41%), and on the Fake-LoveDA LaMa family recall rises from 98.59% to 99.42%. Averaging the four flipped views and the x1.5 scale pass therefore recovers forged pixels that a single view misses, typically faint diffusion seams, at a negligible precision cost, which is the desirable trade in a forensic setting where missed tampering is costlier than a small number of false alarms. The gain is consistent across both generator families, which supports the interpretation that the residual logit cascade benefits from multi-view spectral evidence rather than from dataset-specific artifacts. The same pattern holds on the pooled final-test set, where augmentation lifts the forged-only IoU from 96.96% to 97.80% and pooled precision from 98.07% to 98.31%, while the pooled IoU including authentic images rises from 96.58% to 97.66%, indicating that the extra views also stabilize decisions on authentic content.",
    },
    # Conclusion
    {
        "search": "A single model trained jointly on three generator families reaches 93.71%, 97.63%, and 98.54% pooled IoU on Fake-Vaihingen, Fake-LoveDA, and Fake-LocalDiff, respectively, with 99.77% detection accuracy and a 0.05% false-alarm area, matching or exceeding the specialized state of the art while requiring no per-benchmark re-training. Future work will complete the component ablation and degraded-input robustness sweep, train the provenance arm of the forensic hash, extend the benchmark to multispectral data and additional generator families, and study zero-shot transfer to unseen generators.",
        "replace": "A single model trained jointly on three generator families reaches 95.58% and 91.75% IoU on the Fake-Vaihingen LaMa and RePaint families, 97.63% and 98.04% on Fake-LoveDA, and 98.71% on Fake-LocalDiff, with 99.22% detection accuracy and a pooled forged-class IoU of 97.80% (Dice = F1 = 98.89%) on the independent final-test set; by Eq.\u202f(10) the forged-class Dice equals F1 and all pooled metrics derive from the same confusion counts. The controlled ablation (Table 7) shows that deep edge supervision and the frequency residual encoder are the most influential components, while the CBFH branch contributes no localization evidence. The leave-one-family-out study (Table 9) shows that zero-shot transfer to an unseen generator family is poor (IoU 2.49\u201334.53%), confirming that the forensic cues are generator-specific. Future work will add multi-seed variability to the ablation, extend the CBFH provenance arm with contrastive training and collision analysis, extend the benchmark to multispectral data and additional generator families, and improve cross-generator transfer beyond the multi-domain joint-training regime.",
    },
]


def apply_paragraph_revisions(doc, red=False):
    changed = []
    for p in doc.paragraphs:
        for rev in PARAGRAPH_REVISIONS:
            if rev["search"] in p.text:
                replace_paragraph_text(p, rev["replace"])
                if red:
                    mark_paragraph_red(p)
                changed.append(p)
                break
    return changed


def apply_table4_fixes(doc, red=False):
    tbl = find_table_by_first_cell(doc, ["Dataset", "Fake-Vaihingen"])
    if tbl is None:
        print("WARNING: Table 4 not found")
        return
    rows_data = [
        ("Fake-Vaihingen (LaMa)", "96.41", "99.10", "97.74", "97.74", "95.58"),
        ("Fake-Vaihingen (RePaint)", "94.08", "97.38", "95.70", "95.70", "91.75"),
        ("Fake-LoveDA (LaMa)", "98.19", "99.42", "98.80", "98.80", "97.63"),
        ("Fake-LoveDA (RePaint)", "98.52", "99.51", "99.01", "99.01", "98.04"),
        ("Fake-LocalDiff (latent diffusion)", "98.94", "99.76", "99.35", "99.35", "98.71"),
        ("Overall (pooled forged-only)", "98.31", "99.48", "98.89", "98.89", "97.80"),
        ("Overall (incl. authentic FP)", "98.17", "99.48", "98.82", "98.82", "97.66"),
    ]
    header = ["Dataset", "Precision", "Recall", "F1", "Dice", "IoU"]
    for j, h in enumerate(header):
        if j < len(tbl.rows[0].cells):
            set_cell_text(tbl.rows[0].cells[j], h, red=red)
    for i, row_data in enumerate(rows_data, start=1):
        if i >= len(tbl.rows):
            tbl.add_row()
        for j, val in enumerate(row_data):
            if j >= len(tbl.rows[i].cells):
                continue
            set_cell_text(tbl.rows[i].cells[j], val, red=red)


def apply_table5_and_6_fixes(doc, red=False):
    """Scope table (Aspect header): update detection / pooled claims.
    Calibration table (Quantity header): replace with calibration-split numbers."""
    scope = None
    for t in doc.tables:
        if not t.rows or len(t.rows[0].cells) < 4:
            continue
        if t.rows[0].cells[0].text.strip() == "Aspect" and "ours" in t.rows[0].cells[3].text:
            scope = t
            break
    if scope is not None:
        for row in scope.rows:
            key = row.cells[0].text.strip()
            if key.startswith("Cross-generator evidence") and len(row.cells) >= 4:
                set_cell_text(row.cells[3], "Measured LOFO study (Table 9: zero-shot IoU 2.49-34.53%)", red=red)
            elif key.startswith("Image-level detection") and len(row.cells) >= 4:
                set_cell_text(row.cells[3], "99.22% accuracy (final test)", red=red)
            elif key.startswith("Joint pooled IoU") and len(row.cells) >= 4:
                set_cell_text(row.cells[3], "97.66 / 98.82 (TTA)", red=red)
    cal = find_table_by_first_cell(doc, ["Quantity"])
    if cal is not None:
        cal_rows = [
            ("Quantity", "Value"),
            ("Fitted temperature T (calibration split)", "0.50"),
            ("Deployed mask threshold after sweep", "0.5 (unchanged)"),
            ("Stochastic passes for uncertainty", "20 (MC dropout)"),
            ("Images routed to expert review", "4 / 3,076 (0.1%)"),
            ("Accuracy on auto-decided images", "99.28%"),
            ("Accuracy on abstained images", "50.00%"),
            ("Overall detection accuracy (final test)", "99.22%"),
        ]
        for i, row_data in enumerate(cal_rows):
            if i >= len(cal.rows):
                break
            for j, val in enumerate(row_data):
                if j < len(cal.rows[i].cells):
                    set_cell_text(cal.rows[i].cells[j], val, red=red)
        # neutralise any leftover rows (e.g. the old risk-coverage AUC row)
        for row in cal.rows[len(cal_rows):]:
            for j in range(len(row.cells)):
                set_cell_text(row.cells[j], "-", red=red)


def apply_table7_fixes(doc, red=False):
    """Rewrite Table 7: same-budget (40-epoch) control and controlled removals,
    all measured on the final-test split after calibration-only selection."""
    tbl = None
    for t in doc.tables:
        if not t.rows:
            continue
        header_cells = [c.text.strip().lower() for c in t.rows[0].cells]
        if any("configuration" in h for h in header_cells) and any("overall" in h for h in header_cells):
            tbl = t
            break
    if tbl is None:
        print("WARNING: Table 7 not found")
        return
    for j, h in enumerate(["Configuration (same 40-epoch budget)", "Fake-Vaihingen (LaMa / RePaint)",
                           "Fake-LoveDA (LaMa / RePaint)", "Fake-LocalDiff", "Overall (pooled IoU)"]):
        if j < len(tbl.rows[0].cells):
            set_cell_text(tbl.rows[0].cells[j], h, red=red)
    rows_data = {
        "full model, single pass": ["90.86 / 85.48", "94.82 / 96.48", "96.18", "94.48"],
        "full model, +tta": ["93.47 / 88.84", "96.52 / 97.34", "97.65", "96.50"],
        "- spectral edge stream (ses)": ["91.08 / 85.82", "94.67 / 96.15", "97.01", "94.76"],
        "- frequency residual encoder (fre)": ["88.79 / 81.61", "93.99 / 96.28", "95.84", "94.14"],
        "- deformable attention in fda": ["91.32 / 85.87", "94.81 / 96.66", "95.96", "94.74"],
        "- content-based forensic hash (cbfh)": ["91.70 / 86.48", "95.05 / 96.65", "97.18", "95.30"],
        "- edge supervision": ["89.86 / 83.98", "93.85 / 95.84", "95.38", "93.71"],
        "- distortion-bank augmentation": ["91.50 / 86.25", "94.92 / 96.62", "96.94", "95.16"],
    }
    for row in tbl.rows[1:]:
        key = row.cells[0].text.strip().lower()
        for k, vals in rows_data.items():
            if key.startswith(k):
                for j in range(1, len(row.cells)):
                    if (j - 1) < len(vals):
                        set_cell_text(row.cells[j], vals[j - 1], red=red)
                break


def add_new_sections(doc, red=False):
    """Insert Section 5.9 (degradation) and 5.10 (cross-generator) before 6. Discussion."""
    texts = [
        "5.9 Degradation Robustness Evaluation",
        "Table 8 reports the measured robustness of NRGA-Net to the redistribution degradations applied by the distortion bank. The selected model is evaluated on the final-test set after applying JPEG compression (quality 50, 65, 75, 85, 95), Gaussian blur (kernels 3, 5, 7, 9), and additive Gaussian noise (sigma = 0.01, 0.03, 0.05, 0.06). Without distortion the model reaches 96.58% pooled IoU. Performance degrades gracefully under JPEG compression, from 88.41% IoU at quality 95 to 69.64% at quality 50; additive noise reduces IoU to 63.85% at sigma = 0.06; Gaussian blur is the most destructive degradation, reaching 26.11% IoU at kernel 9, which also removes much of the high-frequency residual evidence the model relies on. The same-budget no-distortion-bank control (Table 7) reaches 95.16% clean IoU versus 94.48% for the distortion-augmented model, confirming that the augmentation's benefit lies in degradation robustness (this table) rather than in clean-image accuracy.",
        "Table 8. Degradation robustness sweep on the final-test set (%)",
        "5.10 Cross-Generator Generalization",
        "To separate multi-domain joint training from true zero-shot generalization, Table 9 reports the leave-one-generator-family-out experiment. In each row the model is retrained on two of the three families (LaMa, RePaint, latent diffusion) under the identical 40-epoch budget and evaluated zero-shot on the held-out family of the final-test split (678 LaMa, 665 RePaint, and 640 latent-diffusion forged images). The result is a severe drop: zero-shot IoU falls to 11.40% on the held-out LaMa family, 34.53% on RePaint, and 2.49% on latent diffusion, far below the 96.58% pooled IoU that the joint model achieves when every family is seen during training. This quantifies the central limitation of the approach: the forensic cues learned by the model, particularly high-frequency residual and spectral artifacts, are largely generator-specific, and joint multi-domain training does not yield generator-agnostic detection. All generalization claims of this work are therefore limited to multi-domain joint training; achieving transfer to unseen generators remains an open problem, for which the released protocol and train-minus-family split indices provide a reproducible baseline.",
        "Table 9. Leave-one-generator-family-out zero-shot generalization (single pass, same 40-epoch budget)",
    ]
    result = insert_paragraphs_before(doc, "6. Discussion", texts, red=red)

    def insert_table_after_caption(caption_prefix, data):
        caption = None
        for p in doc.paragraphs:
            if p.text.strip().startswith(caption_prefix):
                caption = p
                break
        if caption is None:
            print(f"WARNING: caption '{caption_prefix}' not found")
            return
        table = doc.add_table(rows=len(data), cols=len(data[0]))
        table.style = "Table Grid"
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                set_cell_text(table.rows[i].cells[j], str(val), red=red)
        caption._element.addnext(table._element)

    table8_data = [
        ["Condition", "IoU", "F1", "DetAcc"],
        ["none", "96.58", "98.26", "99.22"],
        ["jpeg 50", "69.64", "82.10", "90.02"],
        ["jpeg 65", "75.78", "86.22", "92.75"],
        ["jpeg 75", "79.77", "88.75", "94.64"],
        ["jpeg 85", "83.99", "91.30", "96.39"],
        ["jpeg 95", "88.41", "93.85", "97.56"],
        ["blur 3", "89.64", "94.54", "97.07"],
        ["blur 5", "68.19", "81.09", "88.85"],
        ["blur 7", "32.47", "49.02", "72.53"],
        ["blur 9", "26.11", "41.41", "68.73"],
        ["noise 0.01", "90.81", "95.18", "97.85"],
        ["noise 0.03", "80.26", "89.05", "94.44"],
        ["noise 0.05", "68.90", "81.58", "91.81"],
        ["noise 0.06", "63.85", "77.94", "89.47"],
    ]
    insert_table_after_caption("Table 8.", table8_data)

    table9_data = [
        ["Held-out family", "Train families", "Test samples (fakes)", "IoU (zero-shot)", "F1 (zero-shot)"],
        ["lama", "repaint + latent_diffusion", "678", "11.40", "20.46"],
        ["repaint", "lama + latent_diffusion", "665", "34.53", "51.33"],
        ["latent_diffusion", "lama + repaint", "640", "2.49", "4.86"],
    ]
    insert_table_after_caption("Table 9.", table9_data)

    return result


def add_related_work_signet(doc, red=False):
    """Add SIGNet paragraph before Section 3."""
    texts = [
        "A related line of work exploits spectral information for satellite-map tampering localization. Ding and Nie [48] propose a Spectral Information Guidance Network (SIGNet) that uses vegetation indices to guide splicing localization on high-resolution satellite maps, together with a 1K-pair Satellite-Map Tampering Dataset (SMTD). SIGNet addresses a different forensic setting\u2014RGB satellite-map splicing with spectral-index priors\u2014and is not benchmarked on the generative inpainting benchmarks used here, so a direct numerical comparison is not meaningful. It is nevertheless the closest prior work on spectral guidance for satellite-image forensics and is included for completeness.",
    ]
    return insert_paragraphs_before(doc, "3. Proposed Method", texts, red=red)


def add_code_availability(doc, red=False):
    """Add code availability statement before Section 5.2."""
    texts = [
        "To support independent reproduction, the exact training and evaluation code, deterministic split indices, random seeds, distortion-bank settings, calibration procedure, and per-image predictions used to generate the reported tables are released at https://github.com/haidarraad-a11y/NRGA-Net. The repository includes the Colab notebook notebooks/NRGA-Net_QuickTables.ipynb for fast recomputation of the result tables on a 2,000-image subset.",
    ]
    return insert_paragraphs_before(doc, "5.2 Implementation Details", texts, red=red)


def add_signet_reference(doc, red=False):
    """Add SIGNet reference at the end of the References section."""
    ref_text = (
        "[48] X. Ding and Y. Nie, \"Spectral information guidance network for tampering localization of high-resolution satellite map,\" "
        "Expert Systems with Applications, Vol. 264, 125825, 2025."
    )
    # Find last paragraph
    last = doc.paragraphs[-1]
    new_p = doc.add_paragraph(ref_text)
    if red:
        mark_paragraph_red(new_p)


def apply_ijies_formatting(doc):
    """Best-effort IJIES formatting: 11 pt body, 10 pt tables, two columns."""
    # Body paragraphs
    for p in doc.paragraphs:
        for run in p.runs:
            run.font.size = Pt(11)
            run.font.name = "Times New Roman"
    # Tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(10)
                        run.font.name = "Times New Roman"
    # Section: two columns
    for section in doc.sections:
        section.page_height = Inches(11.69)
        section.page_width = Inches(8.27)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        sectPr = section._sectPr
        cols = sectPr.xpath('./w:cols')
        if cols:
            cols[0].set(qn('w:num'), '2')


def add_split_table(doc, red=False):
    """Insert a split-count table before Section 5.2."""
    # Find target paragraph
    target = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("5.2 Implementation Details"):
            target = p
            break
    if target is None:
        return
    # Create table element before target
    table = doc.add_table(rows=5, cols=4)
    table.style = "Table Grid"
    headers = ["Split", "Forged images", "Authentic images", "Total"]
    for j, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[j], h, red=red)
    data = [
        ["Train", "10,687", "6,279", "16,966"],
        ["Calibration", "495", "274", "769"],
        ["Final test", "1,983", "1,093", "3,076"],
        ["Total", "13,165", "7,646", "20,811"],
    ]
    for i, row in enumerate(data, start=1):
        for j, val in enumerate(row):
            set_cell_text(table.rows[i].cells[j], val, red=red)
    target._element.addprevious(table._element)
    caption = doc.add_paragraph("Table 2a. Deterministic train/calibration/final-test split.")
    if red:
        mark_paragraph_red(caption)
    target._element.addprevious(caption._element)


def add_conflicts_and_contributions(doc, red=False):
    """Update existing Conflicts of Interest and Author Contributions sections if present; otherwise append."""
    # Find existing headings
    existing_conflicts = None
    existing_contrib_heading = None
    existing_contrib_text = None
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if txt == "Conflicts of Interest":
            existing_conflicts = i
        elif txt == "Author Contributions":
            existing_contrib_heading = i
            if i + 1 < len(doc.paragraphs):
                existing_contrib_text = i + 1
    if existing_conflicts is not None and existing_contrib_text is not None:
        # Replace text of following paragraphs
        replace_paragraph_text(doc.paragraphs[existing_conflicts + 1], "The authors declare no conflict of interest.")
        contrib = (
            "Conceptualization, Haidar Raad Shakir and Asmaa Sadiq Abdul Jabar; "
            "methodology, Haidar Raad Shakir; software, Haidar Raad Shakir; "
            "validation, Haidar Raad Shakir and Asmaa Sadiq Abdul Jabar; "
            "formal analysis, Haidar Raad Shakir; investigation, Haidar Raad Shakir; "
            "resources, Asmaa Sadiq Abdul Jabar; data curation, Haidar Raad Shakir; "
            "writing—original draft preparation, Haidar Raad Shakir; "
            "writing—review and editing, Asmaa Sadiq Abdul Jabar; "
            "visualization, Haidar Raad Shakir; supervision, Asmaa Sadiq Abdul Jabar; "
            "project administration, Asmaa Sadiq Abdul Jabar."
        )
        replace_paragraph_text(doc.paragraphs[existing_contrib_text], contrib)
        if red:
            mark_paragraph_red(doc.paragraphs[existing_conflicts + 1])
            mark_paragraph_red(doc.paragraphs[existing_contrib_text])
        return
    # Fallback append
    p0 = doc.add_paragraph()
    r0 = p0.add_run("Conflicts of Interest")
    r0.bold = True
    r0.font.size = Pt(14)
    if red:
        r0.font.color.rgb = RGBColor(255, 0, 0)
    p1 = doc.add_paragraph("The authors declare no conflict of interest.")
    if red:
        mark_paragraph_red(p1)
    p2 = doc.add_paragraph()
    r2 = p2.add_run("Author Contributions")
    r2.bold = True
    r2.font.size = Pt(14)
    if red:
        r2.font.color.rgb = RGBColor(255, 0, 0)
    p3 = doc.add_paragraph(contrib)
    if red:
        mark_paragraph_red(p3)


def apply_revisions(doc, red=False):
    apply_paragraph_revisions(doc, red=red)
    apply_table4_fixes(doc, red=red)
    apply_table5_and_6_fixes(doc, red=red)
    apply_table7_fixes(doc, red=red)
    add_split_table(doc, red=red)
    add_new_sections(doc, red=red)
    add_related_work_signet(doc, red=red)
    add_code_availability(doc, red=red)
    add_signet_reference(doc, red=red)
    add_conflicts_and_contributions(doc, red=red)
    apply_ijies_formatting(doc)


def main():
    print("Loading original manuscript...")
    doc_rev = Document(ORIG)
    apply_revisions(doc_rev, red=False)
    doc_rev.save(REV)
    print("Saved revised manuscript:", REV)

    doc_high = Document(ORIG)
    apply_revisions(doc_high, red=True)
    doc_high.save(HIGH)
    print("Saved highlighted manuscript:", HIGH)


if __name__ == "__main__":
    main()
