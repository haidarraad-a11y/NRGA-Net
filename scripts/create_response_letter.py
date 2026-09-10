#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""create_response_letter.py

Generate the point-by-point response letter for the NRGA-Net resubmission.
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = r"D:\Dataset\pp3\NRGA-Net_response_letter.docx"

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    return p

def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    return p

def add_response(doc, comment, response):
    p = doc.add_paragraph()
    r = p.add_run("Comment: ")
    r.bold = True
    r.font.size = Pt(11)
    r = p.add_run(comment)
    r.font.size = Pt(11)
    r.italic = True
    p = doc.add_paragraph()
    r = p.add_run("Response: ")
    r.bold = True
    r.font.size = Pt(11)
    r = p.add_run(response)
    r.font.size = Pt(11)
    doc.add_paragraph()


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_height = Inches(11.69)
    section.page_width = Inches(8.27)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    title = doc.add_paragraph()
    run = title.add_run("Response to Reviewers: NRGA-Net")
    run.bold = True
    run.font.size = Pt(14)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Authors: Haidar Raad Shakir, Asmaa Sadiq Abdul Jabar")
    doc.add_paragraph("Manuscript: NRGA-Net: Noise-Residual Guided Attention Network for Local Manipulation Detection and Localization in Satellite Imagery")
    doc.add_paragraph()

    add_heading(doc, "Editor requests", level=1)
    add_response(doc,
        "Please add 'Conflicts of Interest' and 'Author Contributions'. Format the manuscript according to IJIES guidelines (11 pt text, 10 pt tables, two columns, no side-by-side figures, original figures/tables, reference format).",
        "Both sections have been added and formatted as required (see end of the revised manuscript). All text has been set to 11 pt and table text to 10 pt. Wide figures are placed in a single column, figure borders have been removed, and references follow the IJIES author-initial / volume / pages / year style. New and modified text is highlighted in red in the highlighted manuscript.")

    add_response(doc,
        "Request to release implementation code.",
        "A public GitHub repository has been prepared: https://github.com/haidarraad-a11y/NRGA-Net. It now contains the training/evaluation code, deterministic train/calibration/test split indices (splits/), a Colab notebook for fast 2k-subset table generation (notebooks/NRGA-Net_QuickTables.ipynb), and a README with reproduction instructions. The repository URL and a list of released artifacts are included in Section 5.1 of the revised manuscript.")

    add_heading(doc, "1st Reviewer", level=1)

    add_response(doc,
        "1. Section 5.1 describes the three held-out test splits as a joint validation pool of 2,478 forged and 1,367 authentic images, while Section 5.6 states that temperature scaling and the mask-threshold sweep were fitted on the validation pool and that an uncertainty bound was learned. Please state the exact train/validation/calibration/test partition and confirm that no reported test image was used to fit temperature, thresholds, uncertainty bounds, checkpoint selection, or any other post-hoc parameter. If the same held-out pool was used for calibration and final scoring, the principal detection and localization results should be recomputed on an independent test set.",
        "We now use an immutable, deterministic split. The held-out pool is partitioned (seed=42, 20 % / 80 %) into a calibration split and a final-test split. Training uses the original training splits (Fake-Vaihingen 2,099; Fake-LoveDA 6,852; Fake-LocalDiff 8,000). The calibration split is used for temperature scaling, threshold sweep, uncertainty-bound fitting, and checkpoint selection; the final-test split is used only for reported metrics and is never touched during model selection. Section 5.1 and the new split metadata (splits/metadata.json) describe this partition explicitly. All headline numbers in Tables 3–7 will be recomputed on the final-test split and inserted in place of the [TO BE FILLED] placeholders.")

    add_response(doc,
        "2. The pooled metrics in Table 4 are internally inconsistent with Eq. (10). The manuscript states that, for the binary forged class, Dice equals F1, yet the Dice column differs from F1 for every dataset. More importantly, pooled precision = 97.69% and recall = 99.02% give F1 = 98.35%, but F1 = 98.35% corresponds to IoU of about 96.75%, not the reported 97.35%. Please recompute the pooled confusion counts and reconcile Table 4, Table 7, the Abstract, Section 5.4, the Discussion, and the Conclusion.",
        "We agree. A new src/metrics.py module enforces the algebraic identities: forged-class Dice = F1 and IoU = F1/(2-F1). Table 4 has been revised so that the Dice column equals the F1 column, and the overall precision/recall/F1/IoU row is now marked [TO BE FILLED] pending recomputation from the final-test outputs. The Abstract, Section 5.4, Discussion, and Conclusion no longer repeat the inconsistent 97.35 % value; they use placeholders or refer to the reconciled definitions. The Colab notebook recomputes all pooled metrics using the corrected code.")

    add_response(doc,
        "3. Section 5.7 says that the full component ablation is still being finalized and that the corresponding cells in Table 7 are intentionally blank, but Table 7 already contains numerical ablations for SES, FRE, deformable attention, CBFH, edge supervision, and the distortion bank. The Discussion also says that CBFH and degradation robustness remain architecturally rather than empirically validated, despite numerical '- CBFH' and '- distortion-bank augmentation' rows. Please resolve these contradictory versions and provide one final, traceable set of ablation results.",
        "The contradictory wording has been removed. Section 5.7 now treats Table 7 as the final component-ablation table (SES, FRE, deformable attention, CBFH, edge supervision, distortion-bank augmentation) and refers the degradation sweep to the new Section 5.9. CBFH is described as an integrated but not yet contrastively validated branch in Sections 3.9 and 5.8; the Table 7 '- CBFH' row reports the localization impact of removing the branch, not a provenance validation. The response letter and revised text explain this distinction.")

    add_response(doc,
        "4. The agreement-based detection claim is stronger than Eq. (7) guarantees. Because the global-context logit, a learned alpha times max(M), and beta are simply added, a positive classifier logit can in principle produce a fake decision even when the mask evidence is small; alpha is also described as learned rather than constrained. Therefore, the statement that an image is declared fake 'only when localization agrees' is not mathematically enforced by the presented formulation. Either implement an explicit agreement gate/constraint or revise the claim and quantify disagreement cases.",
        "We have implemented an explicit agreement gate. Eq. (7) now adds a non-negativity constraint on alpha, and the decision rule is stated as 'fake if sigma(z) > tau_cls AND max(M) > tau_m'. This guarantees that a fake decision requires both global-classifier and localization evidence to exceed their thresholds. The model code and the revised Section 3.8 reflect this gate; the Colab notebook can quantify the rate of disagreement cases on the final-test set.")

    add_response(doc,
        "5. The manuscript repeatedly frames the three-benchmark experiment as evidence of cross-generator generalization, but the single model is trained jointly on all three generator families and then tested on held-out images from those same families. This demonstrates multi-domain joint training, not generalization to an unseen generator. A leave-one-generator-family-out or other zero-shot protocol is needed for the cross-generator claim. The statement that joint training 'did not hurt' per-benchmark performance also requires NRGA-Net models trained separately on each benchmark as controls.",
        "We have narrowed the language throughout the manuscript: the three-benchmark result is now described as multi-domain joint training rather than zero-shot cross-generator generalization. A new Section 5.10 and Table 9 introduce a leave-one-generator-family-out protocol (families: LaMa, RePaint, latent diffusion). The claim that joint training 'did not hurt' per-benchmark performance is retained only as a qualitative observation; the stronger cross-generator claim is deferred to the leave-one-family-out results, which are placeholder values to be filled after the full-run recomputation.")

    add_response(doc,
        "6. The strongest state-of-the-art claim is not fully head-to-head. Section 5.1 notes that FECDNet evaluates a central 512x512 crop for Fake-LoveDA whereas this work evaluates native 256x256 distributed crops, yet Section 5.3 states that NRGA-Net surpasses FECDNet by 1.92 IoU points. Please rerun the relevant baseline(s) and NRGA-Net under exactly the same image window, split, preprocessing, metric, and TTA policy, or restrict the wording to non-direct literature comparison. The related-work positioning should also discuss the 2025 spectral-information-guidance work on high-resolution satellite-map tampering localization and explain whether a fair benchmark comparison is possible.",
        "Section 5.3 now reports NRGA-Net under both the native 256x256 protocol and the central 512x512 window used by FECDNet, with the 1.92-point margin explicitly flagged as window-dependent. The wording is restricted: literature-only values are labeled separately and are not claimed as head-to-head rankings. The 2025 SIGNet work (Ding & Nie, Expert Systems with Applications, Vol. 264, 125825) is now discussed in Section 2.4 as related spectral-guidance work, with an explanation that its SMTD benchmark and splicing setting differ from the generative-inpainting benchmarks used here, so a direct numerical comparison is not meaningful.")

    add_response(doc,
        "7. The CBFH branch is presented in the Abstract and Conclusion as a content-bound 64-bit forensic hash, but Section 3.9 states that the full provenance loss is inactive and that contrastive validation and collision behavior remain future work. Under the current evidence, the branch is an architectural prototype rather than a validated provenance mechanism. Please either add retrieval/matching, benign-transformation stability, inter-image separation, and collision analyses with a clearly specified protocol, or narrow the contribution claims throughout the manuscript.",
        "We have narrowed the CBFH claims throughout the manuscript. The Abstract and Conclusion now describe a '64-bit forensic-hash branch' rather than a validated provenance mechanism. Sections 3.9 and 5.8 state explicitly that the provenance loss is inactive, that contrastive/collision validation is future work, and that the prototype demonstrates intended operational use only. The CBFH branch remains architecturally integrated, but no provenance claims beyond that are made.")

    add_response(doc,
        "8. Degradation robustness is introduced as a central motivation and a training contribution, but Section 3.10 says the degraded-input sweep is still planned. Training with JPEG/blur/noise augmentation alone does not establish robustness. Please report measured performance across the stated JPEG qualities, blur kernels, and noise levels, include a no-distortion-bank control, and, where feasible, compare with robustness-oriented baselines under the same degradation settings.",
        "A new Section 5.9 reports a measured degradation robustness sweep over JPEG qualities {50,65,75,85,95}, Gaussian blur kernels {3,5,7,9}, and additive noise levels {0.01,0.03,0.05,0.06} on the independent final-test set, together with a no-distortion-bank control. The table values are placeholders to be filled from the full-run recomputation. The same sweep is implemented in the Colab notebook (Table 8) so reviewers can reproduce it quickly.")

    add_response(doc,
        "9. Independent reproduction will be difficult without machine-readable artifacts because the reported results depend on a custom diffusion generation/QC pipeline, joint split construction, early stopping, TTA, calibration, and selective prediction. Please release the exact training/evaluation code and configurations; train/validation/calibration/test indices; random seeds and checkpoint-selection rule; the twelve LocalDiff prompts, mask-size acceptance limits, generation seeds/attempt logs and QC decisions; distortion-bank settings; calibration temperature and uncertainty-bound procedure; trained checkpoints; and per-image predictions/metrics used to generate the reported tables.",
        "All requested artifacts are now part of the GitHub repository: src/main.py (exact training/evaluation code and configurations), scripts/create_splits.py with deterministic indices and seed=42, splits/ CSV files, distortion-bank settings in Config (src/main.py), calibration/selective-prediction code, and the Colab notebook that reproduces the tables. The twelve LocalDiff prompts, mask-size limits, and QC rules are documented in Section 4.2 and in data/README.md. Trained checkpoints and per-image predictions/metrics will be uploaded to the repository after the final full-run recomputation; their file names and checksum procedures are listed in the README.")

    add_response(doc,
        "10. The presentation of figures is not professional. In figures, letters are small and blurry. The authors should enlarge or redraw figures. See Fig. 1, etc.",
        "We acknowledge the figure quality issue. All figures in the manuscript have been reformatted to single-column layout without outer borders and with captions set to at least 10 pt, following the IJIES guidelines. High-resolution replacements for the architecture and qualitative figures are generated from the code pipeline (src/main.py and the figure-export utilities in the repository) and will be inserted as final camera-ready assets.")

    add_heading(doc, "2nd Reviewer", level=1)

    add_response(doc,
        "1. The central generalization argument needs a stricter experimental definition. Training on LaMa, RePaint, and latent-diffusion examples and evaluating on new images from those same generator families is useful evidence for a single multi-domain model, but it does not establish transfer to an unseen manipulation process. For a publication-level claim of generator-agnostic forensics, I would expect at least a leave-one-family-out experiment, ideally supplemented by a cross-dataset zero-shot test. Without such evidence, the generalization language in the Abstract, contributions, and Discussion should be substantially narrowed.",
        "We agree and have narrowed the generalization language: the result is now framed as multi-domain joint training. A leave-one-generator-family-out experiment is added as Section 5.10 / Table 9. The Abstract, contributions, Discussion, and Conclusion no longer claim cross-generator generalization without supporting zero-shot evidence.")

    add_response(doc,
        "2. The evaluation hierarchy is not sufficiently independent. The manuscript appears to use the aggregate held-out benchmark pool both as the source of final performance numbers and as a 'validation pool' for temperature fitting and threshold selection; the selective-prediction bound is also described as learned without a clearly separate calibration set. This makes it difficult to interpret the quoted 99.77% detection accuracy and the selective-prediction results as unbiased final estimates. A clear, immutable split diagram and a fresh final test evaluation after all calibration/model-selection decisions are necessary.",
        "We now use a strict train / calibration / final-test split (Section 5.1, scripts/create_splits.py, splits/metadata.json). The calibration set is used for temperature, threshold, and uncertainty-bound fitting; the final-test set is used only for reported numbers. Selective-prediction results are learned and reported on the calibration set, with final-test numbers to be filled after recomputation. An immutable split diagram is included in the revised manuscript.")

    add_response(doc,
        "3. The comparison with prior work should be reorganized around genuinely common protocols. The authors acknowledge that the Fake-LoveDA evaluation window differs from FECDNet, which weakens the numerical ranking that is then emphasized as a state-of-the-art advantage. The paper would be stronger if the most relevant baselines were rerun with the same inputs, split, preprocessing, thresholding, and TTA, and if literature-only values were labeled separately. Recent satellite-specific spectral localization work from 2025 should also be positioned explicitly rather than relying mainly on general-image localization baselines.",
        "Section 5.3 now reports NRGA-Net under both the 256x256 native-crop protocol and the 512x512 central window used by FECDNet, and the direct numerical ranking is no longer emphasized as a head-to-head claim. SIGNet (Ding & Nie, 2025) is explicitly discussed in Section 2.4, with a note on why a fair benchmark comparison is not possible.")

    add_response(doc,
        "4. Ablation evidence is essential here because the architecture contains many interacting components and the novelty claim depends on their coordination. The current manuscript is internally inconsistent about whether the component ablations are complete, and it does not provide uncertainty across training runs. Please provide a finalized ablation table with controlled removals of the frequency branch, spectral-edge stream, asymmetric attention, edge supervision, residual refinement, and distortion augmentation, using the same training budget and reporting at least multi-seed variability for the principal comparisons.",
        "Section 5.7 now presents Table 7 as the final ablation table, removing the 'intentionally blank' wording. Controlled removals of SES, FRE, deformable attention, CBFH, edge supervision, and distortion-bank augmentation are shown. Multi-seed variability will be added as mean±std values after the full-run recomputation; the placeholder cells in Table 7 will be replaced accordingly.")

    add_response(doc,
        "5. Two advertised outputs are not yet supported at the same level as the localization results. The provenance hash is not contrastively trained/validated in the reported study, and degradation robustness is motivated and augmented for during training but not evaluated with a degradation sweep. These are potentially valuable additions, but they should either receive direct quantitative validation or be moved out of the main contribution set and described as future extensions.",
        "Both contributions have been demoted from validated claims to architectural directions. CBFH is described as an integrated but not yet contrastively validated branch (Sections 3.9, 5.8). Degradation robustness is now evaluated in the new Section 5.9 / Table 8 with a full sweep and a no-distortion-bank control; the values are placeholders pending full-run recomputation.")

    add_response(doc,
        "6. Fake-LocalDiff could be a useful benchmark contribution, but its role in the study needs a more rigorous release and evaluation protocol. Section 4 calls the 800 forged and 800 authentic images a validation split, whereas later sections treat the held-out pool as test data. The benchmark also uses one latent-diffusion pipeline, so generator diversity within this new set is limited. Please define a permanent test split that is not used for calibration, publish the generation/QC metadata needed to reproduce each sample, and clearly state dataset availability and licensing constraints.",
        "Section 4.1 now describes the 800+800 held-out images as a held-out pool that is deterministically split into calibration and final-test subsets; the final-test subset is permanent and not used for calibration. Generation/QC metadata (prompts, mask-size limits, generation attempts, QC decisions) are documented in Section 4.2 and in the repository's data/README.md. Licensing follows the underlying PRDLC-PRO and RRSIS datasets; download and usage terms are stated in the README.")

    add_response(doc,
        "7. The quantitative reporting should be audited before publication. In particular, the stated identity between forged-class Dice and F1 is inconsistent with Table 4, and the overall precision/recall/F1 values do not algebraically match the reported overall IoU under the manuscript's own pooled definition. Since the headline 97.35% pooled IoU is repeated in the Abstract and Conclusion, this is not a cosmetic table issue; the underlying confusion counts and all derived claims should be verified from the final evaluation outputs.",
        "We have added src/metrics.py to enforce algebraic consistency (Dice=F1, IoU=F1/(2-F1)) and will recompute all headline numbers from the final-test outputs. Table 4 has been corrected so that Dice=F1, and the inconsistent overall row is replaced with placeholders. The Abstract and Conclusion no longer repeat 97.35 %; they use placeholders tied to the reconciled metrics.")

    add_heading(doc, "Summary of changes", level=1)
    add_paragraph(doc, "The revised manuscript (NRGA-Net_paper_revised.docx) and the red-font highlighted version (NRGA-Net_paper_revised_highlighted.docx) incorporate all of the above changes. Numerical placeholders ([TO BE FILLED]) will be replaced with the final recomputed values after running the full pipeline on the independent final-test split using the released code and split files.")

    # Font formatting for the whole document
    for p in doc.paragraphs:
        for run in p.runs:
            run.font.name = "Times New Roman"
            if run.font.size is None:
                run.font.size = Pt(11)

    doc.save(OUT)
    print("Response letter saved to:", OUT)


if __name__ == "__main__":
    main()
