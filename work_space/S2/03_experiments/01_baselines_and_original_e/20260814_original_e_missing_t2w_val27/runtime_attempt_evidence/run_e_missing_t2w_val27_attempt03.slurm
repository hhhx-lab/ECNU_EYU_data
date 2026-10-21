#!/usr/bin/env bash
#SBATCH --job-name=s2_e_val27
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/runtime/experimental_unvalidated_e_missing_t2w_val27_20260814_r1/logs/s2_e_val27_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/runtime/experimental_unvalidated_e_missing_t2w_val27_20260814_r1/logs/s2_e_val27_%j.err

set -euo pipefail
umask 027

PROJECT_ROOT=/public/home/zqchen/projects/ECNU_EYU_data
RUNTIME_ROOT="${PROJECT_ROOT}/work_space/S2/runtime/experimental_unvalidated_e_missing_t2w_val27_20260814_r1"
RESULT_ROOT="${PROJECT_ROOT}/work_space/S2/results/experimental_unvalidated/e_missing_t2w_val27_20260814_r1"
S2_REPOSITORY="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
SOURCE_ROOT="${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3/data/output/run_3104668"
EVAL_MANIFEST="${RUNTIME_ROOT}/synthetic_accepted_evaluation_manifest_run_3104668.csv"
MASTER_SPLIT="${RUNTIME_ROOT}/splits_master_train_val_test.json"
INFERENCE_SCRIPT="${RUNTIME_ROOT}/inference_frozen.py"
MODEL_ROOT="${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres"
CHECKPOINT="${MODEL_ROOT}/fold_0/checkpoint_final.pth"
TRAIN_PY=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
EVAL_PY="${EVAL_ENV}/bin/python"

EXPECTED_CASES=27
EXPECTED_CHECKPOINT_SHA256=4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267
EXPECTED_EVAL_MANIFEST_SHA256=24c84e1b90cb4ebfd1f034e311130dfd2ea714c288f09728c8da46c7aeadbc37
EXPECTED_INFERENCE_SCRIPT_SHA256=fa367326d18d6e03ef3b313daca7213af43a96ecfa818fe29e64c52d75df9b38
EXPECTED_INFERENCE_TRAINER_SHA256=c7c464c9ec283d75ab7d2f353d50025dc52e8b59dcc0d1f0256abdff7b78df13
INFERENCE_TRAINER="${S2_REPOSITORY}/custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

write_failure() {
    local rc="${1:-1}"
    local reason="${2:-command_error}"
    trap - ERR INT TERM
    for pid in $(jobs -pr 2>/dev/null || true); do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -d "${RESULT_ROOT}" ]]; then
        RC="${rc}" REASON="${reason}" RESULT_ROOT="${RESULT_ROOT}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            "${TRAIN_PY}" - <<'PY' || true
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["RESULT_ROOT"]) / "E_MISSING_T2W_VAL27_FAILED.json"
payload = {
    "status": "experimental_unvalidated_failed",
    "operator_approved": False,
    "exit_code": int(os.environ["RC"]),
    "reason": os.environ["REASON"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
}
try:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
except FileExistsError:
    pass
PY
    fi
    exit "${rc}"
}

trap 'write_failure $? command_error' ERR
trap 'write_failure 130 SIGINT' INT
trap 'write_failure 143 SIGTERM' TERM

for path in "${RUNTIME_ROOT}" "${SOURCE_ROOT}" "${S2_REPOSITORY}" "${MODEL_ROOT}"; do
    [[ -d "${path}" ]] || { echo "Missing required directory: ${path}" >&2; exit 1; }
done
for path in "${EVAL_MANIFEST}" "${MASTER_SPLIT}" "${INFERENCE_SCRIPT}" \
    "${INFERENCE_TRAINER}" "${CHECKPOINT}" "${MODEL_ROOT}/plans.json" "${MODEL_ROOT}/dataset.json" \
    "${TRAIN_PY}" "${EVAL_PY}" "${EVAL_ENV}/bin/brats-evaluate" "${EVAL_ENV}/bin/brats-parse-metrics"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

[[ "$(sha256_file "${CHECKPOINT}")" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || { echo "E checkpoint SHA256 drift" >&2; exit 1; }
[[ "$(sha256_file "${EVAL_MANIFEST}")" == "${EXPECTED_EVAL_MANIFEST_SHA256}" ]] || { echo "Evaluation manifest SHA256 drift" >&2; exit 1; }
[[ "$(sha256_file "${INFERENCE_SCRIPT}")" == "${EXPECTED_INFERENCE_SCRIPT_SHA256}" ]] || { echo "Inference script SHA256 drift" >&2; exit 1; }
[[ "$(sha256_file "${INFERENCE_TRAINER}")" == "${EXPECTED_INFERENCE_TRAINER_SHA256}" ]] || { echo "Inference trainer SHA256 drift" >&2; exit 1; }
[[ ! -e "${RESULT_ROOT}" ]] || { echo "Exclusive result root already exists: ${RESULT_ROOT}" >&2; exit 1; }

mkdir -p "$(dirname "${RESULT_ROOT}")"
mkdir "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}/input/shard_0" "${RESULT_ROOT}/input/shard_1" \
    "${RESULT_ROOT}/prediction/shard_0" "${RESULT_ROOT}/prediction/shard_1" \
    "${RESULT_ROOT}/prediction/all" "${RESULT_ROOT}/reference" "${RESULT_ROOT}/logs" \
    "${RESULT_ROOT}/evidence"
printf '%s\n' experimental_unvalidated > "${RESULT_ROOT}/EXPERIMENTAL_UNVALIDATED"

export RESULT_ROOT SOURCE_ROOT EVAL_MANIFEST MASTER_SPLIT MODEL_ROOT CHECKPOINT
export EXPECTED_CASES EXPECTED_CHECKPOINT_SHA256 EXPECTED_EVAL_MANIFEST_SHA256
export EXPECTED_INFERENCE_SCRIPT_SHA256 EXPECTED_INFERENCE_TRAINER_SHA256
export INFERENCE_SCRIPT INFERENCE_TRAINER SLURM_JOB_ID

"${TRAIN_PY}" - <<'PY'
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


result_root = Path(os.environ["RESULT_ROOT"])
source_root = Path(os.environ["SOURCE_ROOT"])
manifest_path = Path(os.environ["EVAL_MANIFEST"])
split_path = Path(os.environ["MASTER_SPLIT"])
expected = int(os.environ["EXPECTED_CASES"])

with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
    all_rows = list(csv.DictReader(handle))
rows = [row for row in all_rows if row["source_split"].strip() == "val"]
test_rows = [row for row in all_rows if row["source_split"].strip() == "test"]
assert len(rows) == expected, (len(rows), expected)
assert len(test_rows) == 26, len(test_rows)
assert all(truthy(row["accepted_for_evaluation"]) for row in rows)
assert all(not truthy(row["accepted_for_training"]) for row in rows)
assert all(truthy(row["source_is_fake_t2w_case"]) for row in rows)
assert all(truthy(row["source_completion_mode"]) for row in rows)
assert all(truthy(row["has_t2w"]) for row in rows)
assert all(
    row["generation_mode"].strip() == "t1n_t1c_t2f_to_t2w_completion"
    for row in rows
)

source_ids = [row["source_case_id"].strip() for row in rows]
nnunet_ids = [row["nnunet_case_id"].strip() for row in rows]
assert len(source_ids) == len(set(source_ids)) == expected
assert len(nnunet_ids) == len(set(nnunet_ids)) == expected
assert not set(source_ids) & {row["source_case_id"].strip() for row in test_rows}

split_payload = json.loads(split_path.read_text(encoding="utf-8"))
split = split_payload[0] if isinstance(split_payload, list) else split_payload
assert set(nnunet_ids).issubset(set(map(str, split["val"])))
assert not set(nnunet_ids) & set(map(str, split["test"]))

channels = (("t1n", "0000"), ("t1c", "0001"), ("t2w", "0002"), ("t2f", "0003"))
audit_rows = []
for index, (source_id, nnunet_id) in enumerate(sorted(zip(source_ids, nnunet_ids))):
    case_root = source_root / source_id
    reference_path = case_root / f"{source_id}-seg.nii.gz"
    paths = {name: case_root / f"{source_id}-{name}.nii.gz" for name, _ in channels}
    for path in [reference_path, *paths.values()]:
        assert path.is_file(), path

    reference = nib.load(str(reference_path))
    reference_data = np.asanyarray(reference.dataobj)
    reference_labels = sorted(int(value) for value in np.unique(reference_data))
    assert set(reference_labels).issubset({0, 1, 2, 3, 4}), (source_id, reference_labels)
    assert np.isfinite(reference_data).all(), source_id

    affine_digest = hashlib.sha256(np.asarray(reference.affine, dtype=np.float64).tobytes()).hexdigest()
    for modality, _ in channels:
        image = nib.load(str(paths[modality]))
        assert image.shape == reference.shape, (source_id, modality, image.shape, reference.shape)
        assert np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5), (source_id, modality)
        assert np.isfinite(np.asanyarray(image.dataobj)).all(), (source_id, modality)

    shard = index % 2
    for modality, channel in channels:
        target = result_root / "input" / f"shard_{shard}" / f"{source_id}_{channel}.nii.gz"
        target.symlink_to(paths[modality])
    (result_root / "reference" / f"{source_id}.nii.gz").symlink_to(reference_path)
    audit_rows.append({
        "source_case_id": source_id,
        "nnunet_case_id": nnunet_id,
        "source_split": "val",
        "input_t2w_role": "synthesized_completion",
        "shard": shard,
        "shape": "x".join(map(str, reference.shape)),
        "spacing": "x".join(f"{value:.8g}" for value in reference.header.get_zooms()[:3]),
        "affine_sha256": affine_digest,
        "reference_labels": ";".join(map(str, reference_labels)),
    })

with (result_root / "evidence" / "VAL27_CASE_MANIFEST.csv").open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
    writer.writeheader()
    writer.writerows(audit_rows)

shard_counts = {str(shard): sum(row["shard"] == shard for row in audit_rows) for shard in (0, 1)}
assert shard_counts == {"0": 14, "1": 13}, shard_counts

plan = {
    "schema_version": 1,
    "status": "experimental_unvalidated",
    "operator_approved": False,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "S2-E-missing-T2W-complete-pipeline-val27",
    "scope": "27-case internal validation with synthesized T2W only",
    "case_count": expected,
    "source_split": "val",
    "locked_test_case_count_excluded": 26,
    "training_required": False,
    "model": "E_completion_plus_focal_loss",
    "checkpoint": str(Path(os.environ["CHECKPOINT"])),
    "checkpoint_sha256": os.environ["EXPECTED_CHECKPOINT_SHA256"],
    "inference": {
        "gpu_count": 1,
        "shard_counts": shard_counts,
        "execution": "two_shards_sequential_on_one_gpu",
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "save_probabilities": False,
    },
    "evaluation": {
        "package": "BraTS-evaluation==0.0.8",
        "panoptica": "2.1.0",
        "numpy": "1.26.4",
        "config": "mets",
        "volume_threshold": 27,
        "overlap_threshold": 0.2,
    },
    "artifact_sha256": {
        "evaluation_manifest": os.environ["EXPECTED_EVAL_MANIFEST_SHA256"],
        "master_split": sha256_file(split_path),
        "inference_script": os.environ["EXPECTED_INFERENCE_SCRIPT_SHA256"],
        "inference_trainer": os.environ["EXPECTED_INFERENCE_TRAINER_SHA256"],
        "model_plans": sha256_file(Path(os.environ["MODEL_ROOT"]) / "plans.json"),
        "model_dataset": sha256_file(Path(os.environ["MODEL_ROOT"]) / "dataset.json"),
    },
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
(result_root / "E_MISSING_T2W_VAL27_EXECUTION_PLAN.json").write_text(
    json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
preflight = {
    "status": "experimental_unvalidated_preflight_pass",
    "operator_approved": False,
    "case_count": expected,
    "shard_counts": shard_counts,
    "all_modalities_finite": True,
    "geometry_mismatch_count": 0,
    "invalid_label_count": 0,
    "locked_test_excluded": True,
}
(result_root / "evidence" / "VAL27_PREFLIGHT.json").write_text(
    json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(preflight, sort_keys=True))
PY

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1

export PYTHONPATH="${S2_REPOSITORY}:${PYTHONPATH:-}"
export nnUNet_raw="${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results"
export nnUNet_extTrainer="${S2_REPOSITORY}/custom_nnunet"
export nnUNet_compile=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

"${TRAIN_PY}" - <<'PY'
import json
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly 1 allocated CUDA device, got {torch.cuda.device_count()}")
print(json.dumps({
    "cuda_device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
}, sort_keys=True))
PY

echo "S2_E_MISSING_T2W_VAL27_INFERENCE_START utc=$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${INFERENCE_SCRIPT}" \
    --input "${RESULT_ROOT}/input/shard_0" \
    --output "${RESULT_ROOT}/prediction/shard_0" \
    --model-root "${MODEL_ROOT}" \
    --fold 0 --preprocess-workers 4 --export-workers 4 \
    > "${RESULT_ROOT}/logs/inference_shard_0.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${INFERENCE_SCRIPT}" \
    --input "${RESULT_ROOT}/input/shard_1" \
    --output "${RESULT_ROOT}/prediction/shard_1" \
    --model-root "${MODEL_ROOT}" \
    --fold 0 --preprocess-workers 4 --export-workers 4 \
    > "${RESULT_ROOT}/logs/inference_shard_1.log" 2>&1
echo "S2_E_MISSING_T2W_VAL27_INFERENCE_FINISHED utc=$(date -u +%FT%TZ)"

for shard in 0 1; do
    while IFS= read -r prediction; do
        ln -s "${prediction}" "${RESULT_ROOT}/prediction/all/$(basename "${prediction}")"
    done < <(find "${RESULT_ROOT}/prediction/shard_${shard}" -maxdepth 1 -type f -name '*.nii.gz' | sort)
done

"${TRAIN_PY}" - <<'PY'
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

root = Path(os.environ["RESULT_ROOT"])
expected = int(os.environ["EXPECTED_CASES"])
with (root / "evidence" / "VAL27_CASE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
    case_rows = list(csv.DictReader(handle))
expected_ids = {row["source_case_id"] for row in case_rows}
prediction_paths = sorted((root / "prediction" / "all").glob("*.nii.gz"))
actual_ids = {path.name.removesuffix(".nii.gz") for path in prediction_paths}
assert len(prediction_paths) == expected
assert actual_ids == expected_ids, {
    "missing": sorted(expected_ids - actual_ids),
    "unexpected": sorted(actual_ids - expected_ids),
}

rows = []
for prediction_path in prediction_paths:
    case_id = prediction_path.name.removesuffix(".nii.gz")
    reference_path = root / "reference" / f"{case_id}.nii.gz"
    prediction = nib.load(str(prediction_path))
    reference = nib.load(str(reference_path))
    labels = sorted(int(value) for value in np.unique(np.asanyarray(prediction.dataobj)))
    shape_match = prediction.shape == reference.shape
    affine_match = bool(np.allclose(prediction.affine, reference.affine, rtol=0, atol=1e-5))
    labels_valid = set(labels).issubset({0, 1, 2, 3, 4})
    assert shape_match and affine_match and labels_valid, (case_id, shape_match, affine_match, labels)
    rows.append({
        "source_case_id": case_id,
        "shape_match": shape_match,
        "affine_match": affine_match,
        "labels_valid": labels_valid,
        "prediction_labels": ";".join(map(str, labels)),
    })

with (root / "evidence" / "VAL27_PREDICTION_AUDIT.csv").open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "evidence" / "VAL27_PREDICTION_AUDIT.json").write_text(
    json.dumps({
        "status": "experimental_unvalidated_prediction_contract_pass",
        "operator_approved": False,
        "prediction_count": len(rows),
        "coverage_missing": 0,
        "coverage_unexpected": 0,
        "geometry_mismatch_count": 0,
        "invalid_label_count": 0,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

[[ "$(sha256_file "${CHECKPOINT}")" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || { echo "E checkpoint changed during inference" >&2; exit 1; }

export PATH="${EVAL_ENV}/bin:${PATH}"
[[ "$("${EVAL_PY}" -c 'import importlib.metadata as m; print(m.version("BraTS-evaluation"))')" == "0.0.8" ]] || exit 1
[[ "$("${EVAL_PY}" -c 'import importlib.metadata as m; print(m.version("panoptica"))')" == "2.1.0" ]] || exit 1
[[ "$("${EVAL_PY}" -c 'import importlib.metadata as m; print(m.version("numpy"))')" == "1.26.4" ]] || exit 1

SUMMARY_JSON="${RESULT_ROOT}/panoptica_evaluation_summary.json"
METRICS_CSV="${RESULT_ROOT}/leaderboard_metrics.csv"
brats-evaluate \
    --config mets \
    --ref_path "${RESULT_ROOT}/reference" \
    --pred_path "${RESULT_ROOT}/prediction/all" \
    --summary_json "${SUMMARY_JSON}"

"${EVAL_PY}" - "${SUMMARY_JSON}" "${EXPECTED_CASES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
metrics = data.get("metrics", [])
missings = data.get("missings", [])
if len(metrics) != expected or missings:
    raise SystemExit(
        f"BraTS evaluation coverage mismatch: expected={expected}, metrics={len(metrics)}, missings={len(missings)}"
    )
PY

brats-parse-metrics mets \
    --json_path "${SUMMARY_JSON}" \
    --vol_threshold 27 \
    --overlap_threshold 0.2 \
    --output_csv_path "${METRICS_CSV}"
[[ -s "${METRICS_CSV}" ]] || { echo "Missing parsed MET metrics" >&2; exit 1; }

export SUMMARY_JSON METRICS_CSV
"${EVAL_PY}" - <<'PY'
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(os.environ["RESULT_ROOT"])
metrics_path = Path(os.environ["METRICS_CSV"])
with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != int(os.environ["EXPECTED_CASES"]) + 3:
    raise SystemExit(f"Unexpected leaderboard row count: {len(rows)}")
by_subject = {row["subject_id"]: row for row in rows}
if set(("mean", "std", "median")) - set(by_subject):
    raise SystemExit("leaderboard_metrics.csv lacks mean/std/median rows")
metric_names = [name for name in rows[0] if name != "subject_id"]
mean_values = {name: by_subject["mean"][name] for name in metric_names}
(root / "mean_metrics.txt").write_text(
    "".join(f"{name}: {mean_values[name]}\n" for name in metric_names),
    encoding="utf-8",
)
(root / "mean_metrics.json").write_text(
    json.dumps(mean_values, indent=2, sort_keys=False) + "\n", encoding="utf-8"
)
(root / "aggregate_metrics.json").write_text(
    json.dumps(
        {
            statistic: {name: by_subject[statistic][name] for name in metric_names}
            for statistic in ("mean", "std", "median")
        },
        indent=2,
        sort_keys=False,
    ) + "\n",
    encoding="utf-8",
)
payload = {
    "schema_version": 1,
    "status": "experimental_unvalidated_pass",
    "operator_approved": False,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "S2-E-missing-T2W-complete-pipeline-val27",
    "case_count": int(os.environ["EXPECTED_CASES"]),
    "input_t2w": "synthesized_completion",
    "source_split": "val",
    "locked_test_used": False,
    "training_performed": False,
    "model": "E_completion_plus_focal_loss",
    "checkpoint_sha256": os.environ["EXPECTED_CHECKPOINT_SHA256"],
    "prediction_contract_pass": True,
    "official_met_evaluation_complete": True,
    "claim_boundary": "Internal 27-case synthesized-T2W validation only; not an independent holdout or Synapse result.",
    "artifacts": {
        "metrics_csv": str(metrics_path),
        "metrics_csv_sha256": sha256_file(metrics_path),
        "mean_metrics_txt": str(root / "mean_metrics.txt"),
        "mean_metrics_json": str(root / "mean_metrics.json"),
        "aggregate_metrics_json": str(root / "aggregate_metrics.json"),
        "summary_json": str(Path(os.environ["SUMMARY_JSON"])),
        "summary_json_sha256": sha256_file(Path(os.environ["SUMMARY_JSON"])),
    },
}
(root / "E_MISSING_T2W_VAL27_VALIDATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "E_MISSING_T2W_VAL27_COMPLETE.ok").write_text(
    "experimental_unvalidated\n", encoding="ascii"
)
PY

find "${RESULT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > "${RESULT_ROOT}/SHA256SUMS.txt"
trap - ERR INT TERM
chmod -R a-w "${RESULT_ROOT}"
echo "S2_E_MISSING_T2W_VAL27_COMPLETE result=${RESULT_ROOT} utc=$(date -u +%FT%TZ)"
