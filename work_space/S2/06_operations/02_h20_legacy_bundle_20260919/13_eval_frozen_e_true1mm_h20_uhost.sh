#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

EVAL_ROOT="${EVAL_ROOT:-/root/brats2026/runs/s2_e_baseline_true1mm_eval_20260726_r1}"
FALLBACK_ROOT="${FALLBACK_ROOT:-/root/brats2026/runs/s2_e_continue_fallback_20260726_r1}"
OVERLAY_ROOT="${FALLBACK_ROOT}/preprocessed_overlay"
OVERLAY_AUDIT="${OVERLAY_ROOT}/PREPROCESSED_OVERLAY_AUDIT.json"
SPLIT_DIR="${SPLIT_DIR:-${S2_REPOSITORY}/data/splits/completion_warmstart}"
E_CHECKPOINT="${E_CHECKPOINT:-${PROJECT_ROOT}/work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth}"
EXPECTED_E_SHA256="4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAIN_SHA256="1cfa31a71c1c5014fb6ed457277f634ef0db4a95607270f66a7eafcbf9020b52"
EXPECTED_VAL_SHA256="7027d91362adf799901544070204f0821b5ce0608f4d5c85c4d878ee5cc7219a"
EXPECTED_OVERLAY_AUDIT_SHA256="95c9b8f04f6343b44a19ce039cee69ecb89c8227976d8b03a479152df2fc9ef2"
EXPECTED_CUDA_VISIBLE_DEVICES="${S2_FROZEN_E_EVAL_GPU:-1}"
TRAINER="nnUNetTrainerBraTS2026RCFocalCompletionFineTune"
RESULTS_ROOT="${EVAL_ROOT}/nnUNet_results"
RESULT_FOLD="${RESULTS_ROOT}/Dataset264_BraTS2026_MET_Completion/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
VALIDATION_DIR="${RESULT_FOLD}/validation"
CONTRACT_PATH="${EVAL_ROOT}/EVAL_CONTRACT.json"
LAUNCH_MARKER="${EVAL_ROOT}/EVAL_LAUNCHED.ok"
COMPLETE_MARKER="${EVAL_ROOT}/EVAL_COMPLETE.ok"

export EVAL_ROOT FALLBACK_ROOT OVERLAY_ROOT OVERLAY_AUDIT SPLIT_DIR E_CHECKPOINT
export EXPECTED_E_SHA256 EXPECTED_TRAIN_SHA256 EXPECTED_VAL_SHA256
export EXPECTED_OVERLAY_AUDIT_SHA256 EXPECTED_CUDA_VISIBLE_DEVICES
export TRAINER RESULTS_ROOT CONTRACT_PATH

s2_uhost_activate_runtime
[[ "${CUDA_VISIBLE_DEVICES:-}" == "${EXPECTED_CUDA_VISIBLE_DEVICES}" ]] || {
    echo "Frozen-E evaluation must be bound to physical GPU ${EXPECTED_CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}
s2_uhost_require_single_visible_gpu
s2_uhost_require_free_space
s2_uhost_require_dir "${S2_REPOSITORY}" "S2 repository"
s2_uhost_require_dir "${OVERLAY_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres" "true-1mm overlay cache"
s2_uhost_require_file "${OVERLAY_AUDIT}" "true-1mm overlay audit"
s2_uhost_require_file "${SPLIT_DIR}/train_fixed.txt" "fixed training split"
s2_uhost_require_file "${SPLIT_DIR}/val_fixed.txt" "fixed validation split"
s2_uhost_require_file "${E_CHECKPOINT}" "frozen E checkpoint"

[[ "$(s2_uhost_sha256 "${E_CHECKPOINT}")" == "${EXPECTED_E_SHA256}" ]] || { echo "Frozen E SHA256 drifted" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${SPLIT_DIR}/train_fixed.txt")" == "${EXPECTED_TRAIN_SHA256}" ]] || { echo "Training split SHA256 drifted" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${SPLIT_DIR}/val_fixed.txt")" == "${EXPECTED_VAL_SHA256}" ]] || { echo "Validation split SHA256 drifted" >&2; exit 1; }
[[ "$(awk 'NF {count++} END {print count+0}' "${SPLIT_DIR}/val_fixed.txt")" == "103" ]] || { echo "Validation split count drifted" >&2; exit 1; }

if pgrep -af '[n]nUNetv2_train.*nnUNetTrainerBraTS2026RCFocalCompletionFineTune.*--val' >/dev/null; then
    echo "A frozen-E validation process is already active" >&2
    exit 1
fi
if [[ -e "${EVAL_ROOT}" ]]; then
    echo "Frozen-E evaluation root is immutable and already exists: ${EVAL_ROOT}" >&2
    exit 1
fi

mkdir -p "${RESULT_FOLD}"
ln -s "${E_CHECKPOINT}" "${RESULT_FOLD}/checkpoint_final.pth"

"${PYTHON_BIN}" - <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

overlay_audit = json.loads(Path(os.environ["OVERLAY_AUDIT"]).read_text(encoding="utf-8"))
if overlay_audit.get("status") != "pass" or overlay_audit.get("audit_sha256") != os.environ["EXPECTED_OVERLAY_AUDIT_SHA256"]:
    raise SystemExit("true-1mm overlay audit identity drifted")

repo = Path(os.environ["S2_REPOSITORY"])
spec = importlib.util.find_spec("nnunetv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("nnunetv2 is unavailable")
trainer_dir = Path(list(spec.submodule_search_locations)[0]) / "training" / "nnUNetTrainer"
runtime_files = [
    "nnUNetTrainerBraTS2026RC.py",
    "nnUNetTrainerBraTS2026RCCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCFocalCompletionFineTune.py",
    "small_lesion_trainer_mixins.py",
    "small_lesion_variants.py",
]
runtime_sha = {}
for name in runtime_files:
    source = repo / "custom_nnunet" / name
    deployed = trainer_dir / name
    if not source.is_file() or not deployed.is_file() or source.read_bytes() != deployed.read_bytes():
        raise SystemExit(f"deployed frozen-E trainer runtime drifted: {name}")
    runtime_sha[name] = sha256(source)

payload = {
    "schema_version": 1,
    "experiment": "S2-frozen-E-true1mm-fixed103-evaluation",
    "evaluation_root": os.environ["EVAL_ROOT"],
    "dataset": "Dataset264_BraTS2026_MET_Completion",
    "configuration": "3d_fullres",
    "fold": 0,
    "trainer": os.environ["TRAINER"],
    "checkpoint": os.environ["E_CHECKPOINT"],
    "checkpoint_sha256": os.environ["EXPECTED_E_SHA256"],
    "preprocessed_overlay": os.environ["OVERLAY_ROOT"],
    "preprocessed_overlay_audit": os.environ["OVERLAY_AUDIT"],
    "preprocessed_overlay_audit_sha256": os.environ["EXPECTED_OVERLAY_AUDIT_SHA256"],
    "train_split_sha256": os.environ["EXPECTED_TRAIN_SHA256"],
    "val_split_sha256": os.environ["EXPECTED_VAL_SHA256"],
    "val_count": 103,
    "physical_gpu": os.environ["EXPECTED_CUDA_VISIBLE_DEVICES"],
    "validation_only": True,
    "save_probabilities": True,
    "uses_met_aug": False,
    "uses_g1_g2_diffusion": False,
    "runtime_sha256": runtime_sha,
}
payload["contract_sha256"] = hashlib.sha256(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(os.environ["CONTRACT_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf 'launched_at=%s\npid=%s\ngpu=%s\ncontract=%s\n' \
    "$(date -u +%FT%TZ)" "$$" "${CUDA_VISIBLE_DEVICES:-unset}" "${CONTRACT_PATH}" > "${LAUNCH_MARKER}"

export nnUNet_raw="${NNUNET_RAW_ROOT}"
export nnUNet_preprocessed="${OVERLAY_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"
export BRATS_S2_REPO_DIR="${S2_REPOSITORY}"
export BRATS_SPLIT_DIR="${SPLIT_DIR}"
export nnUNet_extTrainer="${S2_REPOSITORY}/custom_nnunet"
export PYTHONPATH="${S2_REPOSITORY}:${PYTHONPATH:-}"
export S2_COMPLETION_EPOCHS=200
export S2_COMPLETION_INITIAL_LR=0.001
export S2_COMPLETION_SAVE_EVERY=25
export S2_FOCAL_GAMMA=2.0
export S2_MET_AUG_ENABLE=0
export nnUNet_compile=0
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${S2_REPOSITORY}"
echo "S2_FROZEN_E_EVAL_START gpu=${CUDA_VISIBLE_DEVICES:-unset} validation_count=103"
nnUNetv2_train 264 3d_fullres 0 \
    -tr "${TRAINER}" \
    -num_gpus 1 \
    --val \
    --npz

s2_uhost_require_file "${VALIDATION_DIR}/summary.json" "frozen-E validation summary"
NIFTI_COUNT="$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')"
NPZ_COUNT="$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
[[ "${NIFTI_COUNT}" == "103" ]] || { echo "Incomplete frozen-E NIfTI predictions: ${NIFTI_COUNT}/103" >&2; exit 1; }
[[ "${NPZ_COUNT}" == "103" ]] || { echo "Incomplete frozen-E NPZ predictions: ${NPZ_COUNT}/103" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${RESULT_FOLD}/checkpoint_final.pth")" == "${EXPECTED_E_SHA256}" ]] || { echo "Frozen E link SHA256 drifted" >&2; exit 1; }

SUMMARY_SHA="$(s2_uhost_sha256 "${VALIDATION_DIR}/summary.json")"
printf 'completed_at=%s\ncheckpoint_sha256=%s\nsummary_sha256=%s\nnifti_predictions=%s\nnpz_predictions=%s\n' \
    "$(date -u +%FT%TZ)" "${EXPECTED_E_SHA256}" "${SUMMARY_SHA}" "${NIFTI_COUNT}" "${NPZ_COUNT}" > "${COMPLETE_MARKER}"
s2_uhost_require_free_space
echo "S2_FROZEN_E_EVAL_PASS summary_sha256=${SUMMARY_SHA}"
