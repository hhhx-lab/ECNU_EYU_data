#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

FALLBACK_ROOT="${FALLBACK_ROOT:-/root/brats2026/runs/s2_e_continue_fallback_20260726_r1}"
EVAL_ROOT="${EVAL_ROOT:-/root/brats2026/runs/s2_e_continue_epoch100_true1mm_eval_20260727_r1}"
OVERLAY_ROOT="${FALLBACK_ROOT}/preprocessed_overlay"
OVERLAY_AUDIT="${OVERLAY_ROOT}/PREPROCESSED_OVERLAY_AUDIT.json"
PLANS_FILE="${OVERLAY_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans.json"
FALLBACK_CONTRACT="${FALLBACK_ROOT}/FALLBACK_CONTRACT.json"
SPLIT_DIR="${SPLIT_DIR:-${S2_REPOSITORY}/data/splits/completion_warmstart}"
TRAINER="nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune"
SOURCE_FOLD="${FALLBACK_ROOT}/training/nnUNet_results/Dataset264_BraTS2026_MET_Completion/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
SOURCE_CHECKPOINT="${SOURCE_FOLD}/checkpoint_latest.pth"
SOURCE_PROVENANCE="${SOURCE_FOLD}/met_aug_control_provenance.json"
SNAPSHOT_EPOCH="${S2_E_CONTINUE_SNAPSHOT_EPOCH:-100}"
EXPECTED_E_SHA256="4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAIN_SHA256="1cfa31a71c1c5014fb6ed457277f634ef0db4a95607270f66a7eafcbf9020b52"
EXPECTED_VAL_SHA256="7027d91362adf799901544070204f0821b5ce0608f4d5c85c4d878ee5cc7219a"
EXPECTED_PLANS_SHA256="c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5"
EXPECTED_OVERLAY_AUDIT_SHA256="95c9b8f04f6343b44a19ce039cee69ecb89c8227976d8b03a479152df2fc9ef2"
EXPECTED_CUDA_VISIBLE_DEVICES="${S2_E_CONTINUE_INTERIM_EVAL_GPU:-1}"
RESULTS_ROOT="${EVAL_ROOT}/nnUNet_results"
RESULT_FOLD="${RESULTS_ROOT}/Dataset264_BraTS2026_MET_Completion/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
VALIDATION_DIR="${RESULT_FOLD}/validation"
CONTRACT_PATH="${EVAL_ROOT}/EVAL_CONTRACT.json"
LAUNCH_MARKER="${EVAL_ROOT}/EVAL_LAUNCHED.ok"
COMPLETE_MARKER="${EVAL_ROOT}/EVAL_COMPLETE.ok"

export FALLBACK_ROOT EVAL_ROOT OVERLAY_ROOT OVERLAY_AUDIT PLANS_FILE FALLBACK_CONTRACT SPLIT_DIR
export TRAINER SOURCE_FOLD SOURCE_CHECKPOINT SOURCE_PROVENANCE SNAPSHOT_EPOCH
export EXPECTED_E_SHA256 EXPECTED_TRAIN_SHA256 EXPECTED_VAL_SHA256 EXPECTED_PLANS_SHA256
export EXPECTED_OVERLAY_AUDIT_SHA256 EXPECTED_CUDA_VISIBLE_DEVICES RESULTS_ROOT RESULT_FOLD CONTRACT_PATH

s2_uhost_activate_runtime
[[ "${SNAPSHOT_EPOCH}" == "100" ]] || { echo "Only the pre-registered epoch-100 snapshot is allowed" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "${EXPECTED_CUDA_VISIBLE_DEVICES}" ]] || {
    echo "E-continue interim evaluation must be bound to physical GPU ${EXPECTED_CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}
s2_uhost_require_single_visible_gpu
s2_uhost_require_free_space
s2_uhost_require_dir "${S2_REPOSITORY}" "S2 repository"
s2_uhost_require_dir "${OVERLAY_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres" "true-1mm overlay cache"
s2_uhost_require_file "${OVERLAY_AUDIT}" "true-1mm overlay audit"
s2_uhost_require_file "${PLANS_FILE}" "true-1mm plans"
s2_uhost_require_file "${FALLBACK_CONTRACT}" "E-continue fallback contract"
s2_uhost_require_file "${SPLIT_DIR}/train_fixed.txt" "fixed training split"
s2_uhost_require_file "${SPLIT_DIR}/val_fixed.txt" "fixed validation split"
s2_uhost_require_file "${SOURCE_CHECKPOINT}" "epoch-100 source checkpoint"
s2_uhost_require_file "${SOURCE_PROVENANCE}" "E-continue matched-control provenance"

[[ "$(s2_uhost_sha256 "${SPLIT_DIR}/train_fixed.txt")" == "${EXPECTED_TRAIN_SHA256}" ]] || { echo "Training split SHA256 drifted" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${SPLIT_DIR}/val_fixed.txt")" == "${EXPECTED_VAL_SHA256}" ]] || { echo "Validation split SHA256 drifted" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${PLANS_FILE}")" == "${EXPECTED_PLANS_SHA256}" ]] || { echo "true-1mm plans SHA256 drifted" >&2; exit 1; }
[[ "$(awk 'NF {count++} END {print count+0}' "${SPLIT_DIR}/val_fixed.txt")" == "103" ]] || { echo "Validation split count drifted" >&2; exit 1; }

if pgrep -af '[n]nUNetv2_train.*nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.*--val' >/dev/null; then
    echo "An E-continue interim validation process is already active" >&2
    exit 1
fi
if [[ -e "${EVAL_ROOT}" ]]; then
    echo "E-continue epoch-100 evaluation root is immutable and already exists: ${EVAL_ROOT}" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


overlay_audit = json.loads(Path(os.environ["OVERLAY_AUDIT"]).read_text(encoding="utf-8"))
if overlay_audit.get("status") != "pass" or overlay_audit.get("audit_sha256") != os.environ["EXPECTED_OVERLAY_AUDIT_SHA256"]:
    raise SystemExit("true-1mm overlay audit identity drifted")

fallback_contract = json.loads(Path(os.environ["FALLBACK_CONTRACT"]).read_text(encoding="utf-8"))
expected_contract = {
    "experiment": "S2-E-continue-fallback",
    "pretrained_weights_sha256": os.environ["EXPECTED_E_SHA256"],
    "augmentation_probability": 0.0,
    "epochs": 200,
    "initial_lr": 0.001,
    "save_every": 25,
    "focal_gamma": 2.0,
    "training_seed": 20260724,
    "augmentation_workers": 0,
    "torch_compile": False,
    "ddp": False,
}
if any(fallback_contract.get(key) != value for key, value in expected_contract.items()):
    raise SystemExit("E-continue fallback contract drifted")

provenance = json.loads(Path(os.environ["SOURCE_PROVENANCE"]).read_text(encoding="utf-8"))
if provenance.get("augmentation_probability") != 0.0 or provenance.get("base_trainer") != "nnUNetTrainerBraTS2026RCFocalCompletionFineTune":
    raise SystemExit("E-continue matched-control provenance drifted")

plans = json.loads(Path(os.environ["PLANS_FILE"]).read_text(encoding="utf-8"))
checkpoint = torch.load(os.environ["SOURCE_CHECKPOINT"], map_location="cpu", weights_only=False)
if checkpoint.get("trainer_name") != os.environ["TRAINER"]:
    raise SystemExit("E-continue source checkpoint trainer identity drifted")
if checkpoint.get("current_epoch") != int(os.environ["SNAPSHOT_EPOCH"]):
    raise SystemExit("E-continue source checkpoint is not the requested epoch-100 savepoint")
init_args = checkpoint.get("init_args", {})
if init_args.get("configuration") != "3d_fullres" or init_args.get("fold") != 0:
    raise SystemExit("E-continue source checkpoint configuration/fold drifted")
if init_args.get("plans") != plans:
    raise SystemExit("E-continue source checkpoint embedded plans differ from true-1mm plans")

repo = Path(os.environ["S2_REPOSITORY"])
spec = importlib.util.find_spec("nnunetv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("nnunetv2 is unavailable")
trainer_dir = Path(list(spec.submodule_search_locations)[0]) / "training" / "nnUNetTrainer"
runtime_files = [
    "met_aug_core.py",
    "met_aug_gate.py",
    "met_aug_paired_training.py",
    "nnUNetTrainerBraTS2026RC.py",
    "nnUNetTrainerBraTS2026RCCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.py",
    "small_lesion_trainer_mixins.py",
]
runtime_sha = {}
for name in runtime_files:
    source = repo / "custom_nnunet" / name
    deployed = trainer_dir / name
    if not source.is_file() or not deployed.is_file() or source.read_bytes() != deployed.read_bytes():
        raise SystemExit(f"deployed E-continue trainer runtime drifted: {name}")
    runtime_sha[name] = sha256(source)

Path(os.environ["EVAL_ROOT"]).mkdir(parents=True)
Path(os.environ["EVAL_ROOT"]).joinpath("PRECHECK.json").write_text(
    json.dumps(
        {
            "status": "pass",
            "snapshot_epoch": int(os.environ["SNAPSHOT_EPOCH"]),
            "source_checkpoint_sha256": sha256(Path(os.environ["SOURCE_CHECKPOINT"])),
            "fallback_contract_sha256": sha256(Path(os.environ["FALLBACK_CONTRACT"])),
            "source_provenance_sha256": sha256(Path(os.environ["SOURCE_PROVENANCE"])),
            "runtime_sha256": runtime_sha,
        },
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

SOURCE_SHA_BEFORE="$(s2_uhost_sha256 "${SOURCE_CHECKPOINT}")"
mkdir -p "${RESULT_FOLD}"
cp --reflink=auto --preserve=mode,timestamps "${SOURCE_CHECKPOINT}" "${RESULT_FOLD}/checkpoint_final.pth"
SOURCE_SHA_AFTER="$(s2_uhost_sha256 "${SOURCE_CHECKPOINT}")"
SNAPSHOT_SHA="$(s2_uhost_sha256 "${RESULT_FOLD}/checkpoint_final.pth")"
[[ "${SOURCE_SHA_BEFORE}" == "${SOURCE_SHA_AFTER}" && "${SOURCE_SHA_BEFORE}" == "${SNAPSHOT_SHA}" ]] || {
    echo "E-continue checkpoint changed during immutable snapshot copy" >&2
    exit 1
}
export SOURCE_SHA_BEFORE SNAPSHOT_SHA

"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


precheck_path = Path(os.environ["EVAL_ROOT"]) / "PRECHECK.json"
precheck = json.loads(precheck_path.read_text(encoding="utf-8"))
if precheck.get("status") != "pass" or precheck.get("source_checkpoint_sha256") != os.environ["SOURCE_SHA_BEFORE"]:
    raise SystemExit("E-continue interim precheck identity drifted")
payload = {
    "schema_version": 1,
    "experiment": "S2-E-continue-epoch100-true1mm-fixed103-evaluation",
    "evaluation_root": os.environ["EVAL_ROOT"],
    "dataset": "Dataset264_BraTS2026_MET_Completion",
    "configuration": "3d_fullres",
    "fold": 0,
    "trainer": os.environ["TRAINER"],
    "snapshot_epoch": int(os.environ["SNAPSHOT_EPOCH"]),
    "source_checkpoint": os.environ["SOURCE_CHECKPOINT"],
    "source_checkpoint_sha256": os.environ["SOURCE_SHA_BEFORE"],
    "snapshot_checkpoint": str(Path(os.environ["RESULT_FOLD"]) / "checkpoint_final.pth"),
    "snapshot_checkpoint_sha256": os.environ["SNAPSHOT_SHA"],
    "fallback_contract": os.environ["FALLBACK_CONTRACT"],
    "fallback_contract_sha256": sha256(Path(os.environ["FALLBACK_CONTRACT"])),
    "source_control_provenance": os.environ["SOURCE_PROVENANCE"],
    "source_control_provenance_sha256": sha256(Path(os.environ["SOURCE_PROVENANCE"])),
    "preprocessed_overlay": os.environ["OVERLAY_ROOT"],
    "preprocessed_overlay_audit": os.environ["OVERLAY_AUDIT"],
    "preprocessed_overlay_audit_sha256": os.environ["EXPECTED_OVERLAY_AUDIT_SHA256"],
    "plans_sha256": os.environ["EXPECTED_PLANS_SHA256"],
    "train_split_sha256": os.environ["EXPECTED_TRAIN_SHA256"],
    "val_split_sha256": os.environ["EXPECTED_VAL_SHA256"],
    "val_count": 103,
    "physical_gpu": os.environ["EXPECTED_CUDA_VISIBLE_DEVICES"],
    "validation_only": True,
    "save_probabilities": True,
    "uses_met_aug": False,
    "uses_g1_g2_diffusion": False,
    "runtime_sha256": precheck["runtime_sha256"],
}
payload["contract_sha256"] = hashlib.sha256(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(os.environ["CONTRACT_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf 'launched_at=%s\npid=%s\ngpu=%s\ncontract=%s\nsnapshot_epoch=%s\nsnapshot_sha256=%s\n' \
    "$(date -u +%FT%TZ)" "$$" "${CUDA_VISIBLE_DEVICES:-unset}" "${CONTRACT_PATH}" "${SNAPSHOT_EPOCH}" "${SNAPSHOT_SHA}" > "${LAUNCH_MARKER}"

export nnUNet_raw="${NNUNET_RAW_ROOT}"
export nnUNet_preprocessed="${OVERLAY_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"
export BRATS_S2_REPO_DIR="${S2_REPOSITORY}"
export BRATS_SPLIT_DIR="${SPLIT_DIR}"
export nnUNet_extTrainer="${S2_REPOSITORY}/custom_nnunet"
export PYTHONPATH="${S2_REPOSITORY}:${PYTHONPATH:-}"
export S2_EXPERIMENT_MODE=met_aug_route_a_control
export S2_MET_AUG_ENABLE=0
export S2_COMPLETION_EPOCHS=200
export S2_COMPLETION_INITIAL_LR=0.001
export S2_COMPLETION_SAVE_EVERY=25
export S2_FOCAL_GAMMA=2.0
export S2_PAIRED_TRAINING_SEED=20260724
export nnUNet_compile=0
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${S2_REPOSITORY}"
echo "S2_E_CONTINUE_EPOCH100_EVAL_START gpu=${CUDA_VISIBLE_DEVICES:-unset} validation_count=103 snapshot_sha256=${SNAPSHOT_SHA}"
nnUNetv2_train 264 3d_fullres 0 \
    -tr "${TRAINER}" \
    -num_gpus 1 \
    --val \
    --npz

s2_uhost_require_file "${VALIDATION_DIR}/summary.json" "E-continue epoch-100 validation summary"
NIFTI_COUNT="$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')"
NPZ_COUNT="$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
[[ "${NIFTI_COUNT}" == "103" ]] || { echo "Incomplete E-continue epoch-100 NIfTI predictions: ${NIFTI_COUNT}/103" >&2; exit 1; }
[[ "${NPZ_COUNT}" == "103" ]] || { echo "Incomplete E-continue epoch-100 NPZ predictions: ${NPZ_COUNT}/103" >&2; exit 1; }
[[ "$(s2_uhost_sha256 "${RESULT_FOLD}/checkpoint_final.pth")" == "${SNAPSHOT_SHA}" ]] || { echo "E-continue immutable snapshot SHA256 drifted" >&2; exit 1; }

SUMMARY_SHA="$(s2_uhost_sha256 "${VALIDATION_DIR}/summary.json")"
printf 'completed_at=%s\nsnapshot_epoch=%s\nsnapshot_checkpoint_sha256=%s\nsummary_sha256=%s\nnifti_predictions=%s\nnpz_predictions=%s\n' \
    "$(date -u +%FT%TZ)" "${SNAPSHOT_EPOCH}" "${SNAPSHOT_SHA}" "${SUMMARY_SHA}" "${NIFTI_COUNT}" "${NPZ_COUNT}" > "${COMPLETE_MARKER}"
s2_uhost_require_free_space
echo "S2_E_CONTINUE_EPOCH100_EVAL_PASS summary_sha256=${SUMMARY_SHA}"
