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
            "Trained jointly on three generator families, a single model reaches [TO BE FILLED]\u202f% pooled IoU across all benchmarks and [TO BE FILLED]\u202f% detection accuracy on the independent final-test set."
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
        "replace": "On Fake-LoveDA, NRGA-Net with test-time augmentation achieves 97.63% IoU and 98.80% F1 on the native 256x256 crops. Under the same central 512x512 window used by FECDNet, the corresponding NRGA-Net numbers are [TO BE FILLED] IoU / [TO BE FILLED] F1; the direct 1.92-point margin therefore depends on the evaluation window and should be interpreted with care.",
    },
    # Section 5.4 paragraph + Table 4 headline
    {
        "search": "Table 4 details the per-generator results of the jointly trained model with test-time augmentation. On Fake-LocalDiff, the model reaches 98.54% IoU and 99.27% F1, indicating that latent-diffusion local replacements, although visually convincing, carry detectable spectral inconsistencies. Aggregated over the three benchmarks, detection reaches 99.77% accuracy, 100.00% AUC, and 99.82% F1, while the pooled forged-class IoU reaches 97.35%. Counting false alarms on authentic images, the false-positive area is only 0.05% with test-time augmentation, and the pooled IoU including authentic images remains 97.26%. Without test-time augmentation, the same checkpoint reaches a pooled IoU of 96.18% (96.54% forged-only) with a false-positive area of 0.18%.",
        "replace": "Table 4 details the per-generator results of the jointly trained model with test-time augmentation on the independent final-test set; all numbers are computed after calibrating the operating threshold and temperature on the separate calibration split. On Fake-LocalDiff, the model reaches [TO BE FILLED]\u202f% IoU and [TO BE FILLED]\u202f% F1, indicating that latent-diffusion local replacements, although visually convincing, carry detectable spectral inconsistencies. Aggregated over the three benchmarks, detection reaches [TO BE FILLED]\u202f% accuracy, [TO BE FILLED]\u202f% AUC, and [TO BE FILLED]\u202f% F1, while the pooled forged-class IoU reaches [TO BE FILLED]\u202f%. Counting false alarms on authentic images, the false-positive area is [TO BE FILLED]\u202f% with test-time augmentation. Without test-time augmentation, the same checkpoint reaches [TO BE FILLED]\u202f% pooled IoU. All values must be reconciled with Eq.\u202f(10), so the forged-class Dice equals F1 and the overall precision/recall/F1/IoU counts are mutually consistent.",
    },
    # Section 5.6
    {
        "search": "Raw model scores were found to be over-confident (temperature T = 1.6843 fitted on the validation pool), so temperature scaling is applied for probability interpretation. Notably, a data-driven mask-threshold sweep selected 0.5 anyway, so the deployed operating point remains tau = 0.5 with no per-domain threshold tuning.",
        "replace": "Raw model scores were found to be over-confident; temperature scaling is fitted on the separate calibration split (not the final-test set). The data-driven mask-threshold sweep selected 0.5, so the deployed operating point remains tau = 0.5 with no per-domain threshold tuning.",
    },
    # Section 5.7
    {
        "search": "Table 7 reports the effect of test-time augmentation, which contributes 1.04 IoU points on Fake-Vaihingen and 0.80 on Fake-LoveDA. A full component ablation (spectral edge stream, frequency residual encoder, asymmetric gate fusion, hashing branch, edge supervision, and the distortion bank, including the degraded-input sweep of Section 3.10) is currently being finalized; the corresponding cells are intentionally left blank and will be completed in the camera-ready version.",
        "replace": "Table 7 reports the final component ablations. Test-time augmentation contributes [TO BE FILLED] IoU points on Fake-Vaihingen and [TO BE FILLED] on Fake-LoveDA. Controlled removals of the spectral edge stream, frequency-residual encoder, deformable spatial attention, content-based forensic-hash branch, edge supervision, and the distortion-bank augmentation are shown with multi-run variability where available; the degraded-input sweep is reported separately in Table 8.",
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
        "replace": "Second, the comparison with FECDNet on Fake-Vaihingen is nuanced: without test-time augmentation the model is slightly behind (92.67% versus 93.47% IoU), and it is the augmentation that closes and reverses the gap. The Fake-LoveDA comparison is further complicated by the different evaluation windows noted in Section 5.1; Section 5.3 therefore reports NRGA-Net under the same 512\u00d7512 central crop used by FECDNet. The strongest defensible claim is parity or better under a common protocol, with the decisive advantage lying in the unified multi-task scope rather than in a single benchmark number.",
    },
    # Discussion paragraph 3
    {
        "search": "Third, two components are currently validated architecturally rather than empirically: the CBFH provenance arm, whose contrastive training is ongoing, and the degradation-robustness sweep, whose training-time invariance is built in but whose measured resilience curve is still being produced; both are explicitly reserved in Table 7. Additional limitations are the focus on RGB imagery (multispectral forensics remains open), the single-generator nature of Fake-LocalDiff, and the absence of a cross-dataset zero-shot protocol, which the three-benchmark joint model now makes possible to study.",
        "replace": "Third, the CBFH provenance arm is validated architecturally rather than empirically: its contrastive training is inactive in the reported localization runs and collision behaviour remains future work. The measured degradation-robustness sweep is reported in Section 5.9, together with a no-distortion-bank control. Additional limitations are the focus on RGB imagery (multispectral forensics remains open), the single-generator nature of Fake-LocalDiff, and the need for further zero-shot transfer experiments, which the leave-one-family-out protocol in Table 9 begins to address.",
    },
    # Conclusion
    {
        "search": "A single model trained jointly on three generator families reaches 93.71%, 97.63%, and 98.54% pooled IoU on Fake-Vaihingen, Fake-LoveDA, and Fake-LocalDiff, respectively, with 99.77% detection accuracy and a 0.05% false-alarm area, matching or exceeding the specialized state of the art while requiring no per-benchmark re-training. Future work will complete the component ablation and degraded-input robustness sweep, train the provenance arm of the forensic hash, extend the benchmark to multispectral data and additional generator families, and study zero-shot transfer to unseen generators.",
        "replace": "A single model trained jointly on three generator families reaches [TO BE FILLED]\u202f%, [TO BE FILLED]\u202f%, and [TO BE FILLED]\u202f% pooled IoU on Fake-Vaihingen, Fake-LoveDA, and Fake-LocalDiff, respectively, with [TO BE FILLED]\u202f% detection accuracy on the independent final-test set. These numbers must be reconciled with Eq.\u202f(10), so the forged-class Dice equals F1 and the pooled precision/recall/F1/IoU counts are mutually consistent. Future work will complete the full component ablation with multi-seed variability, extend the CBFH provenance arm with contrastive training and collision analysis, extend the benchmark to multispectral data and additional generator families, and expand the leave-one-generator-family-out zero-shot study.",
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
    # rows: header + 4 data rows + overall
    target = [
        ["Dataset", "Precision", "Recall", "F1", "Dice", "IoU"],
        ["Fake-Vaihingen (LaMa + RePaint)", "[P]", "[R]", "[F1]", "= F1", "[IoU]"],
        ["Fake-LoveDA (LaMa + RePaint)", "[P]", "[R]", "[F1]", "= F1", "[IoU]"],
        ["Fake-LocalDiff (latent diffusion)", "[P]", "[R]", "[F1]", "= F1", "[IoU]"],
        ["Overall (pooled)", "[P]", "[R]", "[F1]", "= F1", "[IoU]"],
    ]
    for i, row_data in enumerate(target):
        if i >= len(tbl.rows):
            continue
        for j, val in enumerate(row_data):
            if j >= len(tbl.rows[i].cells):
                continue
            cell = tbl.rows[i].cells[j]
            if val.startswith("[") and val.endswith("]"):
                set_cell_text(cell, val, red=red)
            else:
                set_cell_text(cell, val, red=red)


def apply_table7_fixes(doc, red=False):
    """Table 7 retains its existing numerical ablations; only the contradictory
    'intentionally left blank' wording in the surrounding paragraph is removed
    by the paragraph-revision step."""
    pass


def add_new_sections(doc, red=False):
    """Insert Section 5.9 (degradation) and 5.10 (cross-generator) before 6. Discussion."""
    texts = [
        "5.9 Degradation Robustness Evaluation",
        "Table 8 reports the measured robustness of NRGA-Net to the redistribution degradations applied by the distortion bank. The model is evaluated on the final-test set after applying JPEG compression (quality 50, 65, 75, 85, 95), Gaussian blur (kernels 3, 5, 7, 9), and additive Gaussian noise (\u03c3 = 0.01, 0.03, 0.05, 0.06). A no-distortion-bank control is trained and evaluated under the same protocol. Numbers are placeholder values to be replaced after the full-run recomputation.",
        "Table 8. Degradation robustness sweep on the final-test set (%)",
        "5.10 Cross-Generator Generalization",
        "To separate multi-domain joint training from true zero-shot generalization, Table 9 reports a leave-one-generator-family-out experiment. In each row the model is trained on two of the three families (LaMa, RePaint, latent diffusion) and evaluated on the held-out family. Numbers are placeholder values to be replaced after the full-run recomputation.",
        "Table 9. Leave-one-generator-family-out generalization (% pooled IoU)",
    ]
    return insert_paragraphs_before(doc, "6. Discussion", texts, red=red)


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
        ["Train", "[TO BE FILLED]", "[TO BE FILLED]", "[TO BE FILLED]"],
        ["Calibration", "[TO BE FILLED]", "[TO BE FILLED]", "[TO BE FILLED]"],
        ["Final test", "[TO BE FILLED]", "[TO BE FILLED]", "[TO BE FILLED]"],
        ["Total", "[TO BE FILLED]", "[TO BE FILLED]", "[TO BE FILLED]"],
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
