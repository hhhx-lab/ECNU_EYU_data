# BraTS Evaluation
<img width="983" height="219" alt="BraTS Banner" src="https://github.com/user-attachments/assets/51c06fbb-fa48-46a9-acae-05c0a8079899" />

The Brain TumorS aka Brain Tumor Segmentation (BraTS) challenge is a globally recognized community benchmark for the evaluation of automated segmentation algorithms in neuro-oncology. Over the years, BraTS has expanded to encompass a variety of specialized tasks, including:

*   **Glioma Segmentation**: The flagship task, focusing on the delineation of distinct sub-regions (e.g., enhancing tumor, tumor core, and whole tumor) in adult gliomas.
*   **Pediatric Tumor Segmentation**: Targeting brain tumors in pediatric patients, addressing the distinct anatomical and pathological characteristics seen in this population.
*   **Brain Metastasis Segmentation**: Focusing on the detection and segmentation of metastatic brain lesions, which are often small, numerous, and anatomically diverse.
*   **Meningioma Segmentation**: Evaluating the accurate boundary delineation of meningiomas, the most common primary central nervous system tumor.

Robust, and rigorous evaluation of segmentation algorithms across these diverse tasks is essential to accurately gauge clinical applicability and algorithmic performance.

> Note: This package provides the official implementation of the evaluation metrics for the above [BraTS segmentation challenges](https://challenges.synapse.org/brats2026).
> For the BraTS inpainting challenge, a [separate evaluation package](https://github.com/BraTS-inpainting/inpainting) is available.

---

## Panoptica: Instance-Wise Evaluation

[Panoptica](https://github.com/BrainLesion/panoptica) is a comprehensive Python library designed to bridge the gap between global semantic evaluation and clinical necessity by enabling rigorous instance-wise and lesion-wise quantification.
While traditional metrics like the whole-volume Dice score often mask critical individual detection errors, Panoptica isolates and evaluates discrete structures such as tumor subregions through a robust pipeline of instance approximation, matching, and evaluation. 

It computes a comprehensive suite of vital detection and segmentation metrics like:

* **Detection metics** True Positive, False Positive, and False Negative detection rates, 
* **Instance-specific overlap metrics** including Intersection over Union (IoU), instance-level Dice scores, and Average Precision (AP). 
* **Instance-specific distance metrics** such as Normalized surface distance (NSD), and Hausdorff distance (HD95).

* This makes Panoptica a reliable tool for benchmarking deep learning models in medical image segmentation tasks,
standardizing clinical research pipelines, and ensuring that medical image segmentation models are evaluated on their true clinical utility rather than just gross volumetric overlap.

---

## Installation

```bash
pip install BraTS-evaluation
```

This installs the `brats_evaluation` Python package and exposes two console scripts: `brats-evaluate` and `brats-parse-metrics`.

---

## Usage

The evaluation pipeline runs in two steps: produce a JSON summary with `brats-evaluate`, then turn that JSON into a CSV report with `brats-parse-metrics`. Either step can also be driven from Python.

### 1. Run the evaluation (`brats-evaluate`)

This command evaluates prediction NIfTI files against reference (ground truth) NIfTI files using the Panoptica framework.

**Command:**
```bash
brats-evaluate \
    --config mets \
    --ref_path /path/to/reference/niftis/ \
    --pred_path /path/to/prediction/niftis/ \
    --summary_json ./panoptica_evaluation_summary.json
```

Use `--config` with a bundled config name (`mets`, `gli`, `ped`, `MenRT`, `MenPre`, `GoAT`), or `--config_path` to point at a custom YAML file.

**Arguments:**
*   `--config`: Name of a bundled Panoptica config (`mets`, `gli`, `ped`, `MenRT`, `MenPre`, `GoAT`).
*   `--config_path`: Path to a custom Panoptica configuration YAML file (mutually exclusive with `--config`).
*   `--ref_path`: Path to the directory containing reference (ground truth) NIfTI files.
*   `--pred_path`: Path to the directory containing prediction NIfTI files.
*   `--summary_json`: (Optional) Output path for the JSON file summarizing all evaluation metrics. Default: `./panoptica_evaluation_summary.json`.
*   `--num_subjects`: (Optional) Number of subjects to process (e.g. `--num_subjects 5`). Useful for quick testing. If omitted, all subjects are processed.

### 2. Parse the results (`brats-parse-metrics`)

Once the evaluation is complete, a `JSON` file will be created which includes all the quantified metrics.
In order to extract only the metrics which are used for the BraTS Leaderboard and ranking,
use the parser command to extract these metrics into a clean CSV format.

The parser supports two commands: `seg` (for all segmentation tasks except for the Metastasis) and `mets`
(for only the Metastasis task which needs both segmentation and detection metrics).

**Command (Basic Segmentation Metrics):**
```bash
brats-parse-metrics seg \
    --json_path ./panoptica_evaluation_summary.json \
    --output_csv_path ./parsed_panoptica_seg_stats.csv
```

**Command (Metastasis/Detailed Instance Metrics):**
```bash
brats-parse-metrics mets \
    --json_path ./panoptica_evaluation_summary.json \
    --vol_threshold 20.0 \
    --overlap_threshold 0.1 \
    --output_csv_path ./parsed_panoptica_mets_stats.csv
```

**Arguments for `mets` command:**
*   `--vol_threshold`: Volume threshold to differentiate between large and small lesions (e.g., 20.0 voxels/mm3 depending on your config).
*   `--overlap_threshold`: Dice score threshold to classify small lesions as True Positive (TP) or False Negative (FN).

### Python library

Call the evaluator directly from your own Python code:

```python
from panoptica import Panoptica_Evaluator
from brats_evaluation import config_path, evaluate_single_exam

# Bundled configs: "mets", "gli", "ped", "MenRT", "MenPre", "GoAT"
evaluator = Panoptica_Evaluator.load_from_config(str(config_path("mets")))
results = evaluate_single_exam(
    prediction_filepath="path/to/pred.nii.gz",
    reference_filepath="path/to/ref.nii.gz",
    subject_identifier="case-001",
    evaluator=evaluator,
)
print(results)
```

For a runnable, end-to-end example using the bundled sample data see [`./example/programmatic_example.py`](./example/programmatic_example.py).

### Example notebook
For a complete, step-by-step walkthrough of the evaluation and parsing process, see the Jupyter notebook at **[`./example/brats_mets.ipynb`](./example/brats_mets.ipynb)**.

---

## Modifying the pipeline

If you want to tweak the evaluation logic or the Panoptica configs, clone the repo and install with Poetry.

Create and activate a Python environment using **either** conda **or** the built-in `venv`:

```bash
# Option 1 — conda
conda create -n brats_eval python=3.10
conda activate brats_eval
```

```bash
# Option 2 — venv (no conda required)
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then clone the repo and install with Poetry:

```bash
git clone https://github.com/BraTS/BraTS_evaluation.git
cd BraTS_evaluation
poetry install
```

If Poetry is not yet available, install it via **either** route:
*   `conda install -c conda-forge poetry` (for conda users — keeps Poetry inside the env)
*   `curl -sSL https://install.python-poetry.org | python3 -` (official standalone installer)

---

## References

1.  **BraTS Challenge**: [Brain TumorS (BraTS) Challenge](https://www.synapse.org/Synapse:syn74274097/wiki/639571)
2.  **Panoptica Library**: [Panoptica evaluation framework](https://github.com/BrainLesion/panoptica)
