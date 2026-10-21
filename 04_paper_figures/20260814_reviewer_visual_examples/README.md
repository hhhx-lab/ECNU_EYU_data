# Reviewer visual examples

This directory contains paired qualitative examples prepared from the fixed 103-case internal validation cohort.

Final figures:

- `segmentation_gt_vs_prediction.{png,pdf,svg}`: ground truth versus the selected original-E nnU-Net prediction for two illustrative cases.
- `segmentation_high_agreement_gt_vs_prediction.{png,pdf,svg}`: the two fixed-103 cases with the highest whole-volume macro label Dice, explicitly presented as selected best-case examples.
- `segmentation_option_A-D.{png,pdf,svg}`: four candidate case combinations; every figure retains the title `Qualitative segmentation results`.
- `segmentation_options_contact_sheet.png`: compact A-D overview for selection only.
- `t2w_real_vs_ensemble.{png,pdf,svg}`: real T2W versus G1 r4 Ensemble reconstruction for two illustrative cases.
- `suggested_captions.md`: manuscript-ready captions and interpretation boundaries.
- `figure_manifest.json`: exact input hashes, model binding, case roles, slice indices, display contract, and per-case metrics.
- `render_figures.py`: reproducible renderer; run with the existing `g1_t2w_bbdm` Conda environment.

The two figures use separate case selections because segmentation overlap and reconstruction fidelity answer different questions. All cases are explicitly described as illustrative rather than random or statistically representative. The complete 103-case quantitative analyses remain the cohort-level evidence.

Case allocation:

- Segmentation: `BraTS-MET-00025-000` and `BraTS-MET-01060-001`.
- High-agreement segmentation: `BraTS-MET-00532-000` and `BraTS-MET-00552-000`.
- Missing-T2W reconstruction: `BraTS-MET-00009-000` and `BraTS-MET-00807-000`.

Evidence roots:

- Original-E selection and fixed-103 segmentation results: `work_space/S2/results/s2_official_validation_inference_20260727_r1/selection_evidence/`.
- G1 r4 Ensemble fixed-103 reconstruction results: `work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/ensemble_r4/`.

These examples are internal-validation evidence; they are not the unlabeled Synapse validation cohort. The G1 r4 Ensemble result remains `experimental_unvalidated` with `operator_approved=false`, and its qualitative reconstruction fidelity does not establish a downstream nnU-Net segmentation benefit.
