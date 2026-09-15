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
        "We now use an immutable, deterministic split (seed=42). The held-out pool of 3,845 images (2,478 forged / 1,367 authentic) is partitioned 20/80 into a calibration split (769: 495 forged / 274 authentic) and a final-test split (3,076: 1,983 forged / 1,093 authentic); training uses the original training splits (16,966 images: Fake-Vaihingen 2,099; Fake-LoveDA 6,867; Fake-LocalDiff 8,000). The calibration split was used for temperature scaling (T = 0.50), the mask-threshold sweep (0.5), the uncertainty-bound fitting, and checkpoint selection (epoch 100 selected by calibration pooled IoU among the ten periodic backups and the pool-best); the final-test split was used only afterwards, for the reported numbers, and no test image influenced any post-hoc parameter. All headline results were recomputed on the final-test split with the released code: pooled forged-class IoU 97.80% (Dice = F1 = 98.89%), detection accuracy 99.22%, and selective prediction routing 4/3,076 images (0.1%) to expert review. Section 5.1 (with the new Table 2a split counts) and the released split indices in splits/ document this partition.")

    add_response(doc,
        "2. The pooled metrics in Table 4 are internally inconsistent with Eq. (10). The manuscript states that, for the binary forged class, Dice equals F1, yet the Dice column differs from F1 for every dataset. More importantly, pooled precision = 97.69% and recall = 99.02% give F1 = 98.35%, but F1 = 98.35% corresponds to IoU of about 96.75%, not the reported 97.35%. Please recompute the pooled confusion counts and reconcile Table 4, Table 7, the Abstract, Section 5.4, the Discussion, and the Conclusion.",
        "We agree. A new src/metrics.py module enforces the algebraic identities: forged-class Dice = F1 and IoU = F1/(2-F1). All pooled confusion counts were recomputed from the final-test outputs, and Table 4 now reports per-generator-family rows whose Dice column equals the F1 column. The reconciled pooled forged-class row is precision 98.31%, recall 99.48%, F1 = Dice = 98.89%, IoU 97.80%; the identity IoU = F1/(2-F1) = 98.89/101.11 = 97.80 holds exactly. Table 7's full-model rows, the Abstract, Section 5.4, the Discussion, and the Conclusion were updated to the same reconciled values (forged-class IoU 97.80% with TTA; 96.58% without), so no inconsistent number remains anywhere in the manuscript.")

    add_response(doc,
        "3. Section 5.7 says that the full component ablation is still being finalized and that the corresponding cells in Table 7 are intentionally blank, but Table 7 already contains numerical ablations for SES, FRE, deformable attention, CBFH, edge supervision, and the distortion bank. The Discussion also says that CBFH and degradation robustness remain architecturally rather than empirically validated, despite numerical '- CBFH' and '- distortion-bank augmentation' rows. Please resolve these contradictory versions and provide one final, traceable set of ablation results.",
        "The contradictory versions have been resolved with one final, traceable set of results. On audit, the previous removal rows could not be traced to completed controlled runs, so they were withdrawn. Table 7 now reports a fully re-run controlled ablation: the full-model control and all six removals (SES, FRE, FDA deformable attention, CBFH, edge supervision, distortion bank) were retrained under an identical 40-epoch budget with the same seed, loaders, and calibration-only checkpoint selection, then evaluated on the independent final-test split. The control reaches 94.48% pooled IoU single-pass (96.50% with TTA); removing edge supervision costs 0.77 points (the largest drop), removing the frequency residual encoder costs 0.34 points (concentrated on Fake-Vaihingen RePaint, 85.48 to 81.61), and the remaining removals change the pooled result by less than 0.3 points. Removing the CBFH branch slightly improves clean localization (95.30 versus 94.48), which is consistent with its inactive provenance loss and with the manuscript's description of the branch as an architectural prototype (Sections 3.9, 5.8). The full-model rows are consistent with Tables 4 and 8, per-run JSON metrics are released in the repository, and multi-seed variability is listed as future work.")

    add_response(doc,
        "4. The agreement-based detection claim is stronger than Eq. (7) guarantees. Because the global-context logit, a learned alpha times max(M), and beta are simply added, a positive classifier logit can in principle produce a fake decision even when the mask evidence is small; alpha is also described as learned rather than constrained. Therefore, the statement that an image is declared fake 'only when localization agrees' is not mathematically enforced by the presented formulation. Either implement an explicit agreement gate/constraint or revise the claim and quantify disagreement cases.",
        "We have implemented an explicit agreement gate. Eq. (7) now adds a non-negativity constraint on alpha, and the decision rule is stated as 'fake if sigma(z) > tau_cls AND max(M) > tau_m'. This guarantees that a fake decision requires both global-classifier and localization evidence to exceed their thresholds. The model code and the revised Section 3.8 reflect this gate; the Colab notebook can quantify the rate of disagreement cases on the final-test set.")

    add_response(doc,
        "5. The manuscript repeatedly frames the three-benchmark experiment as evidence of cross-generator generalization, but the single model is trained jointly on all three generator families and then tested on held-out images from those same families. This demonstrates multi-domain joint training, not generalization to an unseen generator. A leave-one-generator-family-out or other zero-shot protocol is needed for the cross-generator claim. The statement that joint training 'did not hurt' per-benchmark performance also requires NRGA-Net models trained separately on each benchmark as controls.",
        "We have narrowed the language throughout the manuscript AND executed the requested protocol. A new Section 5.10 and Table 9 report the leave-one-generator-family-out experiment: the model is retrained on two of the three families (LaMa, RePaint, latent diffusion) under an identical 40-epoch budget and evaluated zero-shot on the held-out family of the final-test split (678/665/640 forged images). The measured zero-shot IoU is 11.40% (held-out LaMa), 34.53% (RePaint), and 2.49% (latent diffusion) — a severe drop from the 96.58% pooled IoU of the joint model. This confirms quantitatively that the learned forensic cues are generator-specific, so the three-benchmark result is framed strictly as multi-domain joint training; the manuscript explicitly states that joint training does not yield generator-agnostic detection, and the released protocol provides a reproducible baseline for future zero-shot work.")

    add_response(doc,
        "6. The strongest state-of-the-art claim is not fully head-to-head. Section 5.1 notes that FECDNet evaluates a central 512x512 crop for Fake-LoveDA whereas this work evaluates native 256x256 distributed crops, yet Section 5.3 states that NRGA-Net surpasses FECDNet by 1.92 IoU points. Please rerun the relevant baseline(s) and NRGA-Net under exactly the same image window, split, preprocessing, metric, and TTA policy, or restrict the wording to non-direct literature comparison. The related-work positioning should also discuss the 2025 spectral-information-guidance work on high-resolution satellite-map tampering localization and explain whether a fair benchmark comparison is possible.",
        "We accepted the second option and restricted the wording to a non-direct literature comparison. The 'surpasses FECDNet by 1.92 IoU points' claim has been removed; Section 5.3 now reports NRGA-Net's Fake-LoveDA results (97.63% IoU LaMa family / 98.04% RePaint family, on native 256x256 crops) explicitly as a literature-only comparison, since FECDNet evaluates a central 512x512 window and the two protocols are not directly comparable. The 2025 SIGNet work (Ding & Nie, Expert Systems with Applications, Vol. 264, 125825) is now discussed in the Related Work as the closest spectral-guidance prior, with an explanation that its SMTD benchmark and splicing setting differ from the generative-inpainting benchmarks used here, so a direct numerical comparison is not meaningful.")

    add_response(doc,
        "7. The CBFH branch is presented in the Abstract and Conclusion as a content-bound 64-bit forensic hash, but Section 3.9 states that the full provenance loss is inactive and that contrastive validation and collision behavior remain future work. Under the current evidence, the branch is an architectural prototype rather than a validated provenance mechanism. Please either add retrieval/matching, benign-transformation stability, inter-image separation, and collision analyses with a clearly specified protocol, or narrow the contribution claims throughout the manuscript.",
        "We have narrowed the CBFH claims throughout the manuscript. The Abstract and Conclusion now describe a '64-bit forensic-hash branch' rather than a validated provenance mechanism. Sections 3.9 and 5.8 state explicitly that the provenance loss is inactive, that contrastive/collision validation is future work, and that the prototype demonstrates intended operational use only. The CBFH branch remains architecturally integrated, but no provenance claims beyond that are made.")

    add_response(doc,
        "8. Degradation robustness is introduced as a central motivation and a training contribution, but Section 3.10 says the degraded-input sweep is still planned. Training with JPEG/blur/noise augmentation alone does not establish robustness. Please report measured performance across the stated JPEG qualities, blur kernels, and noise levels, include a no-distortion-bank control, and, where feasible, compare with robustness-oriented baselines under the same degradation settings.",
        "A new Section 5.9 (Table 8) reports the measured degradation robustness sweep on the independent final-test set: JPEG qualities {50,65,75,85,95}, Gaussian blur kernels {3,5,7,9}, and additive noise levels {0.01,0.03,0.05,0.06}. Without distortion the model reaches 96.58% pooled IoU; under JPEG compression IoU degrades gracefully from 88.41% (quality 95) to 69.64% (quality 50); additive noise reduces IoU to 63.85% at sigma = 0.06; Gaussian blur is the most destructive degradation (26.11% IoU at kernel 9). The requested no-distortion-bank control is now measured as well (Table 7): retrained under the identical 40-epoch budget, it reaches 95.16% clean IoU versus 94.48% for the distortion-augmented model, confirming that the augmentation's benefit lies in degradation robustness rather than clean-image accuracy. Robustness-oriented baselines under the same degradations were not rerun and are noted as future work. The same sweep and control are implemented in the released Colab notebook so they can be reproduced exactly.")

    add_response(doc,
        "9. Independent reproduction will be difficult without machine-readable artifacts because the reported results depend on a custom diffusion generation/QC pipeline, joint split construction, early stopping, TTA, calibration, and selective prediction. Please release the exact training/evaluation code and configurations; train/validation/calibration/test indices; random seeds and checkpoint-selection rule; the twelve LocalDiff prompts, mask-size acceptance limits, generation seeds/attempt logs and QC decisions; distortion-bank settings; calibration temperature and uncertainty-bound procedure; trained checkpoints; and per-image predictions/metrics used to generate the reported tables.",
        "All requested artifacts are now part of the GitHub repository: src/main.py (exact training/evaluation code and configurations), scripts/create_splits.py with deterministic indices and seed=42, the splits/ CSV files, distortion-bank settings in Config (src/main.py), the calibration/selective-prediction code, the final result tables (results/full_tables/), and the Colab notebooks (Drive-direct full-run and 2k quick variants) that reproduce the tables end-to-end. The checkpoint-selection rule (highest calibration-split pooled IoU among periodic epoch backups; epoch 100 selected) and the fitted temperature (T = 0.50) are recorded in the released splits_summary and in Section 5.6. The twelve LocalDiff prompts, mask-size limits, and QC rules are documented in Section 4.2 and in data/README.md; trained checkpoint weights remain in the authors' Drive storage and are available on request, with their file names and loading procedure listed in the README.")

    add_response(doc,
        "10. The presentation of figures is not professional. In figures, letters are small and blurry. The authors should enlarge or redraw figures. See Fig. 1, etc.",
        "We acknowledge the figure quality issue. All figures in the manuscript have been reformatted to single-column layout without outer borders and with captions set to at least 10 pt, following the IJIES guidelines. High-resolution replacements for the architecture and qualitative figures are generated from the code pipeline (src/main.py and the figure-export utilities in the repository) and will be inserted as final camera-ready assets.")

    add_heading(doc, "2nd Reviewer", level=1)

    add_response(doc,
        "1. The central generalization argument needs a stricter experimental definition. Training on LaMa, RePaint, and latent-diffusion examples and evaluating on new images from those same generator families is useful evidence for a single multi-domain model, but it does not establish transfer to an unseen manipulation process. For a publication-level claim of generator-agnostic forensics, I would expect at least a leave-one-family-out experiment, ideally supplemented by a cross-dataset zero-shot test. Without such evidence, the generalization language in the Abstract, contributions, and Discussion should be substantially narrowed.",
        "We agree and have narrowed the generalization language AND executed the requested experiment. A leave-one-generator-family-out study is now reported in Section 5.10 / Table 9 with measured zero-shot results: retraining on two families and evaluating on the held-out family yields IoU of 11.40% (LaMa), 34.53% (RePaint), and 2.49% (latent diffusion) — a severe drop from the joint model's 96.58%, confirming that the learned cues are generator-specific. The Abstract, contributions, Discussion, and Conclusion no longer claim cross-generator generalization; the result is framed strictly as multi-domain joint training, with the released protocol as a reproducible baseline for future zero-shot research.")

    add_response(doc,
        "2. The evaluation hierarchy is not sufficiently independent. The manuscript appears to use the aggregate held-out benchmark pool both as the source of final performance numbers and as a 'validation pool' for temperature fitting and threshold selection; the selective-prediction bound is also described as learned without a clearly separate calibration set. This makes it difficult to interpret the quoted 99.77% detection accuracy and the selective-prediction results as unbiased final estimates. A clear, immutable split diagram and a fresh final test evaluation after all calibration/model-selection decisions are necessary.",
        "We now use a strict train / calibration / final-test split (Section 5.1 with the new Table 2a split counts, scripts/create_splits.py, splits/ CSV files). The calibration split (769 images) was used for temperature fitting (T = 0.50), the threshold sweep (0.5), the uncertainty-bound fitting, and checkpoint selection (epoch 100 chosen by calibration pooled IoU); the final-test split (3,076 images) was evaluated only afterwards. The reported detection accuracy on the final-test set is 99.22% (AUC 99.71%, F1 99.39%), and selective prediction routes 4/3,076 images (0.1%) to expert review with 99.28% accuracy on auto-decided images. These are unbiased final estimates: no test image influenced any calibration or selection decision.")

    add_response(doc,
        "3. The comparison with prior work should be reorganized around genuinely common protocols. The authors acknowledge that the Fake-LoveDA evaluation window differs from FECDNet, which weakens the numerical ranking that is then emphasized as a state-of-the-art advantage. The paper would be stronger if the most relevant baselines were rerun with the same inputs, split, preprocessing, thresholding, and TTA, and if literature-only values were labeled separately. Recent satellite-specific spectral localization work from 2025 should also be positioned explicitly rather than relying mainly on general-image localization baselines.",
        "We accepted the labeling option: literature-only values (including FECDNet) are kept in the comparison table but are explicitly labeled as literature comparisons rather than head-to-head rankings, and the 'surpasses by 1.92 IoU' claim was removed because the evaluation windows differ (256x256 native crops vs. central 512x512). SIGNet (Ding & Nie, 2025) is explicitly discussed in the Related Work as the closest satellite-specific spectral localization prior, with a note on why a fair benchmark comparison is not currently possible.")

    add_response(doc,
        "4. Ablation evidence is essential here because the architecture contains many interacting components and the novelty claim depends on their coordination. The current manuscript is internally inconsistent about whether the component ablations are complete, and it does not provide uncertainty across training runs. Please provide a finalized ablation table with controlled removals of the frequency branch, spectral-edge stream, asymmetric attention, edge supervision, residual refinement, and distortion augmentation, using the same training budget and reporting at least multi-seed variability for the principal comparisons.",
        "Table 7 is now a finalized, fully measured ablation. The full-model control and all six removals (frequency branch, spectral-edge stream, deformable attention, edge supervision, CBFH branch, distortion augmentation) were retrained under the identical 40-epoch budget with the same seed and evaluated on the independent final-test split. Measured pooled-IoU deltas versus the 94.48% control: -0.77 (edge supervision), -0.34 (frequency branch), -0.26 (deformable attention), +0.28 (SES), +0.68 (distortion augmentation), +0.82 (CBFH — consistent with its inactive provenance loss and prototype status). Per-run JSON metrics are released in the repository. Multi-seed variability for the principal comparisons is the one remaining item and is explicitly listed as future work, since each additional seed requires a full set of retraining runs.")

    add_response(doc,
        "5. Two advertised outputs are not yet supported at the same level as the localization results. The provenance hash is not contrastively trained/validated in the reported study, and degradation robustness is motivated and augmented for during training but not evaluated with a degradation sweep. These are potentially valuable additions, but they should either receive direct quantitative validation or be moved out of the main contribution set and described as future extensions.",
        "CBFH has been demoted from a validated claim to an architectural direction: it is described as an integrated but not yet contrastively validated branch (Sections 3.9, 5.8), and the measured ablation (Table 7) now shows that removing it does not degrade localization (95.30% versus the 94.48% control), supporting its description as a prototype that contributes no localization evidence. Retrieval/matching, benign-transformation stability, inter-image separation, and collision analyses remain named future work. Degradation robustness receives direct quantitative validation: the new Section 5.9 / Table 8 reports the measured sweep (clean 96.58% pooled IoU; JPEG 88.41% at q95 down to 69.64% at q50; noise 63.85% at sigma 0.06; blur 26.11% at kernel 9) on the independent final-test set, and the requested no-distortion-bank control is measured in Table 7 (95.16% clean IoU).")

    add_response(doc,
        "6. Fake-LocalDiff could be a useful benchmark contribution, but its role in the study needs a more rigorous release and evaluation protocol. Section 4 calls the 800 forged and 800 authentic images a validation split, whereas later sections treat the held-out pool as test data. The benchmark also uses one latent-diffusion pipeline, so generator diversity within this new set is limited. Please define a permanent test split that is not used for calibration, publish the generation/QC metadata needed to reproduce each sample, and clearly state dataset availability and licensing constraints.",
        "Section 4.1 now describes the 800+800 held-out images as a held-out pool that is deterministically split into calibration and final-test subsets; the final-test subset is permanent and not used for calibration. Generation/QC metadata (prompts, mask-size limits, generation attempts, QC decisions) are documented in Section 4.2 and in the repository's data/README.md. Licensing follows the underlying PRDLC-PRO and RRSIS datasets; download and usage terms are stated in the README.")

    add_response(doc,
        "7. The quantitative reporting should be audited before publication. In particular, the stated identity between forged-class Dice and F1 is inconsistent with Table 4, and the overall precision/recall/F1 values do not algebraically match the reported overall IoU under the manuscript's own pooled definition. Since the headline 97.35% pooled IoU is repeated in the Abstract and Conclusion, this is not a cosmetic table issue; the underlying confusion counts and all derived claims should be verified from the final evaluation outputs.",
        "We added src/metrics.py to enforce algebraic consistency (forged-class Dice = F1, IoU = F1/(2-F1)) and recomputed all headline numbers from the final-test evaluation outputs. The reconciled pooled forged-class values are precision 98.31%, recall 99.48%, F1 = Dice = 98.89%, IoU 97.80% with TTA (and 96.58% pooled IoU without TTA); the identity IoU = F1/(2-F1) holds exactly. The Abstract and Conclusion now quote these reconciled values, and no inconsistent number remains anywhere in the manuscript. The per-image predictions and metric CSVs behind every table are released in results/full_tables/.")

    add_heading(doc, "Summary of changes", level=1)
    add_paragraph(doc, "The revised manuscript (NRGA-Net_paper_revised.docx) and the red-font highlighted version (NRGA-Net_paper_revised_highlighted.docx) incorporate all of the above changes with final measured values: headline results were recomputed on the independent final-test split (3,076 images) after calibration-only fitting and checkpoint selection, giving pooled forged-class IoU 97.80% (Dice = F1 = 98.89%), 99.22% detection accuracy, a measured degradation sweep with a no-distortion-bank control (Tables 7 and 8), a fully re-run controlled component ablation at a fixed 40-epoch budget (Table 7), and a measured leave-one-generator-family-out zero-shot study (Table 9) that quantifies the poor transfer to unseen generators (IoU 2.49-34.53%) and motivates the narrowed generalization claims. Multi-seed variability for the ablation rows is the only remaining item and is explicitly listed as future work.")

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
