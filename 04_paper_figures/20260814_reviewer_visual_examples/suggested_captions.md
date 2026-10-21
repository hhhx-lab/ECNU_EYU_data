# Suggested figure captions

## Segmentation

**Qualitative segmentation results for two illustrative cases from the fixed internal validation cohort.** Ground-truth annotations (left) and predictions from the selected nnU-Net model E (right) are overlaid on the same axial T1-weighted contrast-enhanced slice using an identical crop and intensity window. Colors denote NETC (label 1), SNFH (label 2), ET (label 3), RC (label 4). The cases show distinct lesion morphologies and strong segmentation overlap; the displayed slice was selected automatically as the axial slice with the largest ground-truth tumor area. These illustrative cases are not presented as a random or statistically representative sample; cohort-level performance is reported separately for all 103 validation cases.

## High-agreement segmentation

**Selected high-agreement segmentation examples from the fixed internal validation cohort.** The two cases with the highest whole-volume macro label Dice for model E are shown: BraTS-MET-00532-000 (0.971) and BraTS-MET-00552-000 (0.951). Ground-truth annotations (left) and nnU-Net predictions (right) are overlaid on identical axial T1-weighted contrast-enhanced images. Colors denote NETC (label 1), SNFH (label 2), ET (label 3), RC (label 4). These deliberately selected best-case examples illustrate the attainable agreement and must not be interpreted as random, cohort-representative, or statistical-significance evidence; aggregate performance is reported over all 103 validation cases.

## Missing-modality reconstruction

**Qualitative missing-T2W reconstruction for two illustrative cases from the fixed internal validation cohort.** Real T2W images (left) are compared with G1 r4 Ensemble reconstructions (right). The yellow contour denotes the ground-truth tumor region. Each pair uses the same axial slice, crop, and intensity window derived from the real T2W image; no panel-wise normalization was applied. Brain-region SSIM/PSNR values are shown for each case. These illustrative examples assess reconstruction fidelity and do not by themselves establish downstream segmentation benefit or replace the full 103-case quantitative evaluation.
