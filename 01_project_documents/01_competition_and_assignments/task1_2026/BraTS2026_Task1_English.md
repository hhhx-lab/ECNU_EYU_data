# BraTS 2026 Challenge - Task 1 Structured Notes

Source page: https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097
Accessed: 2026-05-04
Scope: Task 1 page information, organized into a readable Markdown document. This file is a comprehensive structured extraction and paraphrase, not a verbatim mirror of the full third-party webpage.

## Page Context

Challenge name: BraTS 2026 Challenge
Status: Active
Registered participants shown on the page: 52
Available challenge tabs: Overview, Instructions, Task 1, Task 2, Task 3, Task 4, Task 5, News, Community
Task 1 page contents menu: Description, Data, Data Files, Evaluation, Submission, Results

The challenge overview shown in the page header states that the Brain Tumor Segmentation Challenge, known as BraTS, began at MICCAI 2012. It has supported brain tumor image analysis by benchmarking algorithmic methods, providing annotated datasets, and asking participants to solve clinically relevant problems across the disease course. For MICCAI 2026, the 15th BraTS cluster of challenges continues this work in collaboration with AI-RANO, RSNA, ASNR, NIH, ASFNR, and CBTN.

## Description

### Clinical Issue

The Task 1 problem focuses on metastatic brain disease monitoring. The page identifies three major issues:

1. Monitoring metastatic brain disease is slow and labor-intensive, particularly when patients have multiple brain metastases and the workflow depends on manual measurements or annotations.
2. Brain metastases are often assessed under RANO-BM guidelines by measuring the largest one-dimensional lesion diameter. The task page emphasizes that volumetric estimates of lesions and surrounding edema are important for clinical decisions and for improving treatment outcome prediction.
3. Brain metastases are often small. Detecting and segmenting lesions under 10 mm is difficult and has historically produced low Dice similarity coefficients.

### Proposed Solution

The proposed approach is to use machine learning to automatically detect and segment:

- brain metastases,
- perilesional edema,
- resection cavities.

The intended benefits are higher time efficiency, better reproducibility, and stronger robustness against variability among human raters.

### Intended Impact

The challenge aims to produce algorithms that are useful in both current and post-treatment clinical settings. The stated impact is to improve, and potentially transform, the management and monitoring of patients with brain metastases.

## Task

Participants should develop a versatile autosegmentation algorithm that can detect and accurately delineate brain metastases of different sizes. The algorithm should apply to both pre-treatment and post-treatment cases.

### Four-Label Annotation System

BraTS 2026 Brain Metastases uses the following four labels:

| Label | Abbreviation | Name | Meaning |
|---:|---|---|---|
| 1 | NETC | Nonenhancing tumor core | Tumor-core tissue without contrast enhancement that is enclosed by enhancing tumor. It represents the tumor bulk generally considered for surgical excision. |
| 2 | SNFH | Surrounding non-enhancing FLAIR hyperintensity | Peritumoral edematous and infiltrated tissue defined by the abnormal hyperintense signal envelope on T2 FLAIR. It includes infiltrative non-enhancing tumor and vasogenic peritumoral edema. FLAIR abnormalities unrelated to tumor, such as prior infarcts or microvascular ischemic white-matter changes, are excluded. |
| 3 | ET | Enhancing Tumor | Tumor regions with visible contrast enhancement on post-contrast T1-weighted imaging. Adjacent vessels, hemorrhage, and intrinsic T1 hyperintensity are excluded. |
| 4 | RC | Resection Cavity | The resection region in the brain for post-treatment cases. |

For 2026, the task adds a detection leaderboard. Its goal is to encourage algorithms that are sensitive to lesion detection. The page highlights small lesions below 27 mm^3 as clinically relevant because they may need to be counted as separate entities or quantified independently.

## Task 1 Organizers

| Name | Role | Affiliation |
|---|---|---|
| Mariam Aboian, MD/PhD | Lead Co-Organizer | Department of Radiology, Children's Hospital of Philadelphia |
| Nikolay Yordanov, MD | Co-Organizer | Faculty of Medicine, Medical University - Sofia, Sofia, Bulgaria |
| Nazanin Maleki, MD | Co-Organizer | Department of Radiology, Children’s Hospital of Philadelphia (CHOP) |
| Raisa Amiruddin, MBBS | Co-Organizer | Department of Radiology, Children’s Hospital of Philadelphia (CHOP) |
| Fabian Umeh | Co-Organizer | Teesside University, UK |
| Crystal Chukwurah | Co-Organizer | Medical Student, Yale School of Medicine |
| Monika Pytlarz | Co-Organizer | PhD Student, Sano - Centre for Computational Personalised Medicine |

## Data

The BraTS 2026 Brain Metastases dataset is a retrospective collection of pre-treatment and post-treatment brain-metastasis mpMRI scans from multiple institutions. The scans were acquired under standard clinical conditions. Because they come from different institutions, equipment, and imaging protocols, the dataset reflects a broad range of image quality and real-world clinical practice.

For this challenge, the organizers are using the dataset from 2025. The page states that all images from the training and testing sets are annotated. For 2026, the organizers plan to perform quality control on all annotations in the dataset. They also plan to offer non-annotated cases to participants to support innovation in semi-supervised learning.

### MRI Series

The dataset consists of multiparametric MRI scans with the following series:

- pre-contrast T1-weighted imaging, T1W;
- post-contrast T1-weighted imaging, T1C;
- T2-weighted imaging, T2W;
- T2-weighted Fluid Attenuated Inversion Recovery, FLAIR.

In 2025, T2W became non-mandatory in BraTS-METS. Some cases contain native T2, some contain synthetic T2, and some have no T2 series. Imaging volumes were segmented using STAPLE fusion of several brain-metastasis segmentation algorithms. The fused labels were manually refined by neuroradiology experts with different levels of rank and experience, following a shared annotation protocol. Experienced board-certified attending neuroradiologists approved the refined annotations.

### Datasets Included in the Challenge

| Dataset | Training | Validation | Testing | Registered in | Contributor | Institution | Annotated |
|---|---:|---:|---:|---|---|---|---|
| Duke | 37 | 15 | 30 | SRI24 space | Devon Godfrey PhD; Scott Floyd MD/PhD | Duke University | yes |
| NCI | 35 | n/a | 1 | SRI24 space | Ayda Youssef MD | National Cancer Institute | yes |
| Missouri | 22 | 25 | 35 | SRI24 space | Nourel hoda Tahon MD, Msc; Ayman Nada MD/PhD | University of Missouri | yes |
| WashU | 39 | 2 | 12 | SRI24 space | Satrajit Chakrabarty | Washington University | yes |
| Yale | 195 | n/a | 12 | SRI24 space | Mariam Aboian MD/PhD | Yale university | yes |
| UCSF | 322 | n/a | n/a | Native space | Jeffrey Rudie MD | University of California, San Francisco | yes |
| NW | n/a | 46 | n/a | SRI24 space | Yuri S. Velichko PhD | Northwestern University | yes |
| UCSD | 646 | 91 | 213 | Native space | Maria Correia de Verdier MD; Jeffrey Rudie MD/PhD | University of California, San Diego | yes |
| Ulm | 200 | 0 | 0 | Native space | Nico Sollman MD/PhD | Ulm University, Germany | no |
| In total | 1496 | 179 | 303 | n/a | n/a | n/a | yes |

### Additional Dataset Detail: UCSD Longitudinal MRI Dataset

The University of California San Diego Brain Metastases Longitudinal MRI Dataset currently contains 646 training cases. It contains progressive longitudinal data. Some cases may have received non-surgical treatment, meaning some cases may have empty masks.

Before release, the organizers state that the dataset will undergo quality control. They will run one algorithm from the BraTS 2025 winners and one algorithm that was never trained on BraTS data, cited as Rudie et al., 2021, on the training data. Cases with Dice below 1 will be identified and re-annotated.

### Image Registration

The BraTS 2025 Metastases dataset contains a mixture of:

- cases in native space,
- cases co-registered to T1C at 1 mm^3,
- cases registered in SRI24 space.

All cases from Ulm University, UCSF, and UCSD are in native space, totaling 1268 cases. The remaining cases are registered in SRI24 space, totaling 328 cases.

The page explains that registering neuroimaging data into a shared space such as SRI24 creates a consistent anatomical reference for comparison across subjects, studies, and datasets. However, native space is more natural for radiologists because interpolation can distort images and make small lesions harder to see.

### Data Access

In addition to registering for the challenge, participants must request access to the data.

1. Submit the Data Access Google form: https://forms.gle/UiCpXos2zKFPdMnK6
2. Only one data access form is required across all five challenge tasks and their training plus validation datasets.
3. After details are verified, the BraTS Service Account will email an invitation to join the BraTS 2026 Data Access Team.
4. Participants must accept the invitation to unlock the files listed on the page.

BraTS 2026 Data Access Team: https://www.synapse.org/Team:3586605

Note: test datasets and validation ground-truth labels will not be publicly released.

## Data Files

| Name | Synapse ID | Modified On | Size | MD5 |
|---|---|---|---:|---|
| MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip | syn64919665 | 4/29/2026 3:48 AM | 31.17 GB | a67d67f756c8ef14e8dbda08ca73688c |
| MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip | syn64919141 | 4/29/2026 3:48 AM | 5.06 GB | ab2e253de48f11e8e8c4da3dc96ba113 |
| MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip | syn65888166 | 4/29/2026 3:48 AM | 20.25 KB | dbad47aca8d27bff93d3f6436b6f2cfa |

The page also shows an "Add To Download List" action.

## Evaluation

### Segmentation Evaluation Metrics

The task uses subject-wise segmentation metrics:

1. Dice Similarity Coefficient, DSC, a standard metric for segmentation performance.
2. Normalized Surface Distance, NSD, a complementary metric that uses a tolerance parameter.

### Lesion Detection Evaluation Metrics

The lesion-wise detection metrics are:

1. F1 score, the harmonic mean of precision and recall. It is used to understand whether an algorithm tends to over-segment or under-segment.

Update checked on 2026-06-15: the "AUC over multiple F1 scores" text is currently inside an HTML comment in the official Evaluation wiki, and the current Results leaderboard does not expose an AUC column. Therefore AUC should not be treated as the active official headline metric.

Detection evaluation is applied to every lesion within an MRI study. For the detection arm, the page defines a lesion as a collective term covering Enhancing Tumor, Non-enhancing Tumor Core, and Resection Cavity.

Segmentation evaluation metrics are applied only to lesions larger than 275 mm^3. The page notes that the images in the dataset are co-registered to 1 mm slice thickness.

### Ranking Details

The ranking will follow DELPHI-based recommendations for image analysis validation. The process incorporates:

1. algorithmic ranking;
2. statistical significance testing.

For multidimensional outcomes or metrics, each team receives ranks across the average of the described metrics. These ranks are summed to form a univariate overall summary measure, which determines the overall team ranking.

All teams will be ordered by rank. Their average rankings will then be randomly permuted pairwise, using 500,000 permutations. Pairwise p-values will be calculated to report pairwise statistical significance and actual differences between ordered approaches.

The p-values will be reported in an upper triangular matrix. This matrix will show statistically insignificant teams grouped into tiers and statistically superior teams indicated separately. The page describes this as an evolved version of the systematic ranking used in prior BraTS and other challenges. The ranking method will be packaged and distributed as an independent tool for reproducibility and reuse in other challenges.

If an algorithm fails to produce a result metric for a specific test case, the page states that there will be no penalty. In that case, the metric will not be set to its worst possible value, such as 0 for DSC or NSD.

### Evaluation References

1. Reinke et al. Understanding metric-related pitfalls in image analysis validation. Nature Methods. 2024 Feb;21(2):182-194.
2. Maier-Hein et al. Metrics reloaded: recommendations for image analysis validation. Nature Methods. 2024 Feb;21(2):195-212.

## Submission

The page contains a Submission Dashboard and a Your Submission Directory section.

Project SynID: syn74773222

The submission file table has the following columns:

- File Name
- Updated On
- ID

At the time of access, the table contained no rows and displayed 0-0 of 0.

Available submission actions shown on the page:

- Upload File
- Submit Selection

## Results

Update checked on 2026-06-15: the Results section now shows a Validation Leaderboard. The active displayed fields include:

- `lesionwise_dsc_mean_et`, `lesionwise_nsd_mean_et`
- `lesionwise_dsc_mean_rc`, `lesionwise_nsd_mean_rc`
- `lesionwise_dsc_mean_tc`, `lesionwise_nsd_mean_tc`
- `lesionwise_dsc_mean_wt`, `lesionwise_nsd_mean_wt`
- `small_instance_tp/fn/fp/f1_et`
- `small_instance_tp/fn/fp/f1_tc`
- `small_instance_tp/fn/fp/f1_wt`
- `small_instance_tp/fn/fp/f1_rc`

## Footer and Related Links

- Terms of Service: https://www.synapse.org/TrustCenter:TermsOfService
- About: https://sagebionetworks.org/
- Help: https://help.synapse.org/docs/Getting-Started.2055471150.html
- Version Number / source repository link: https://github.com/Sage-Bionetworks/synapse-web-monorepo
