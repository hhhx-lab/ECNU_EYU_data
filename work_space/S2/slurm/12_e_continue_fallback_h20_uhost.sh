#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

STAGE="${S2_E_CONTINUE_STAGE:-${1:-}}"
case "${STAGE}" in
    preflight|smoke|train|status) ;;
    *) echo "Usage: S2_E_CONTINUE_STAGE={preflight|smoke|train|status} $0" >&2; exit 2 ;;
esac

FALLBACK_ROOT="${FALLBACK_ROOT:-/root/brats2026/runs/s2_e_continue_fallback_20260726_r1}"
SPLIT_DIR="${SPLIT_DIR:-${S2_REPOSITORY}/data/splits/completion_warmstart}"
CACHE_AUDIT="${CACHE_AUDIT:-/root/brats2026/data/s2_dataset264_true1mm_20260726_r1/audit/TRUE1MM_CACHE_AUDIT.json}"
FINGERPRINT_SOURCE="${FINGERPRINT_SOURCE:-/root/brats2026/data/s2_dataset264/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/dataset_fingerprint.json}"
E_CHECKPOINT="${E_CHECKPOINT:-${PROJECT_ROOT}/work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth}"
EXPECTED_E_SHA256="4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAIN_SHA256="1cfa31a71c1c5014fb6ed457277f634ef0db4a95607270f66a7eafcbf9020b52"
EXPECTED_VAL_SHA256="7027d91362adf799901544070204f0821b5ce0608f4d5c85c4d878ee5cc7219a"
EXPECTED_PLANS_SHA256="c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5"
TRAINER="nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune"
OVERLAY_ROOT="${FALLBACK_ROOT}/preprocessed_overlay"
OVERLAY_AUDIT="${OVERLAY_ROOT}/PREPROCESSED_OVERLAY_AUDIT.json"
SMOKE_DIR="${FALLBACK_ROOT}/training_smoke_attempt_03"
SMOKE_REPORT="${SMOKE_DIR}/e_continue_training_smoke_report.json"
SMOKE_APPROVAL="${FALLBACK_ROOT}/TRAINING_SMOKE_APPROVED.ok"
TRAIN_RESULTS="${FALLBACK_ROOT}/training/nnUNet_results"
RESULT_FOLD="${TRAIN_RESULTS}/Dataset264_BraTS2026_MET_Completion/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
LAUNCH_MARKER="${FALLBACK_ROOT}/TRAIN_LAUNCHED.ok"
COMPLETE_MARKER="${FALLBACK_ROOT}/E_CONTINUE_COMPLETE.ok"
CONTRACT_PATH="${FALLBACK_ROOT}/FALLBACK_CONTRACT.json"
DEADLINE_UTC="${S2_FINAL_DEADLINE_UTC:-2026-07-28T13:32:13Z}"
DOWNSTREAM_RESERVE_HOURS="${S2_DOWNSTREAM_RESERVE_HOURS:-12}"
ETA_SAFETY_FACTOR="${S2_ETA_SAFETY_FACTOR:-1.25}"

export FALLBACK_ROOT SPLIT_DIR CACHE_AUDIT FINGERPRINT_SOURCE E_CHECKPOINT
export NNUNET_RAW_ROOT NNUNET_PREPROCESSED_ROOT
export EXPECTED_E_SHA256 EXPECTED_TRAIN_SHA256 EXPECTED_VAL_SHA256 EXPECTED_PLANS_SHA256
export TRAINER SMOKE_REPORT CONTRACT_PATH DEADLINE_UTC DOWNSTREAM_RESERVE_HOURS ETA_SAFETY_FACTOR

require_no_related_processes() {
    local matches
    matches="$(pgrep -af '([2]5_run_e_continue_training_smoke.py|[n]nUNetv2_train.*nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune)' || true)"
    if [[ -n "${matches}" ]]; then
        echo "An E-continue process is already active:" >&2
        echo "${matches}" >&2
        return 1
    fi
}

acquire_execution_lock() {
    mkdir -p "${FALLBACK_ROOT}"
    exec 9>"${FALLBACK_ROOT}/.execution.lock"
    if ! flock -n 9; then
        echo "Another E-continue wrapper holds the execution lock" >&2
        return 1
    fi
}

write_or_validate_contract() {
    mkdir -p "${FALLBACK_ROOT}"
    "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["CONTRACT_PATH"])
payload = {
    "schema_version": 1,
    "experiment": "S2-E-continue-fallback",
    "fallback_root": os.environ["FALLBACK_ROOT"],
    "dataset": "Dataset264_BraTS2026_MET_Completion",
    "nnunet_raw": os.environ["NNUNET_RAW_ROOT"],
    "nnunet_preprocessed": os.environ["NNUNET_PREPROCESSED_ROOT"],
    "split_dir": os.environ["SPLIT_DIR"],
    "train_split_sha256": os.environ["EXPECTED_TRAIN_SHA256"],
    "val_split_sha256": os.environ["EXPECTED_VAL_SHA256"],
    "pretrained_weights": os.environ["E_CHECKPOINT"],
    "pretrained_weights_sha256": os.environ["EXPECTED_E_SHA256"],
    "plans_file_sha256": os.environ["EXPECTED_PLANS_SHA256"],
    "trainer": os.environ["TRAINER"],
    "augmentation_probability": 0.0,
    "epochs": 200,
    "initial_lr": 0.001,
    "save_every": 25,
    "focal_gamma": 2.0,
    "training_seed": 20260724,
    "augmentation_workers": 0,
    "torch_compile": False,
    "ddp": False,
    "uses_route_approval": False,
    "uses_g1_g2_diffusion": False,
    "final_deadline_utc": os.environ["DEADLINE_UTC"],
    "downstream_reserve_hours": float(os.environ["DOWNSTREAM_RESERVE_HOURS"]),
}
identity = hashlib.sha256(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
payload["contract_sha256"] = identity
encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
if path.exists():
    if path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"immutable fallback contract drifted: {path}")
else:
    path.write_text(encoded, encoding="utf-8")
PY
}

preflight() {
    s2_uhost_activate_runtime
    s2_uhost_require_free_space
    s2_uhost_require_dir "${S2_REPOSITORY}" "S2 repository"
    s2_uhost_require_dir "${NNUNET_RAW_ROOT}/Dataset264_BraTS2026_MET_Completion" "Dataset264 raw dataset"
    s2_uhost_require_dir "${NNUNET_PREPROCESSED_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres" "true-1mm Dataset264 cache"
    s2_uhost_require_file "${NNUNET_PREPROCESSED_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans.json" "true-1mm plans"
    s2_uhost_require_file "${SPLIT_DIR}/train_fixed.txt" "fixed training split"
    s2_uhost_require_file "${SPLIT_DIR}/val_fixed.txt" "fixed validation split"
    s2_uhost_require_file "${CACHE_AUDIT}" "true-1mm cache audit"
    s2_uhost_require_file "${FINGERPRINT_SOURCE}" "Dataset264 provenance fingerprint"
    s2_uhost_require_file "${E_CHECKPOINT}" "frozen E checkpoint"
    [[ "$(s2_uhost_sha256 "${E_CHECKPOINT}")" == "${EXPECTED_E_SHA256}" ]] || { echo "Frozen E SHA256 drifted" >&2; return 1; }
    [[ "$(s2_uhost_sha256 "${SPLIT_DIR}/train_fixed.txt")" == "${EXPECTED_TRAIN_SHA256}" ]] || { echo "Training split SHA256 drifted" >&2; return 1; }
    [[ "$(s2_uhost_sha256 "${SPLIT_DIR}/val_fixed.txt")" == "${EXPECTED_VAL_SHA256}" ]] || { echo "Validation split SHA256 drifted" >&2; return 1; }
    [[ "$(s2_uhost_sha256 "${NNUNET_PREPROCESSED_ROOT}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans.json")" == "${EXPECTED_PLANS_SHA256}" ]] || { echo "true-1mm plans SHA256 drifted" >&2; return 1; }
    [[ "$(awk 'NF {count++} END {print count+0}' "${SPLIT_DIR}/train_fixed.txt")" == "1035" ]] || { echo "Training split count drifted" >&2; return 1; }
    [[ "$(awk 'NF {count++} END {print count+0}' "${SPLIT_DIR}/val_fixed.txt")" == "103" ]] || { echo "Validation split count drifted" >&2; return 1; }
    write_or_validate_contract
    "${PYTHON_BIN}" "${S2_REPOSITORY}/scripts/26_prepare_e_continue_preprocessed_overlay.py" \
        --source-preprocessed-root "${NNUNET_PREPROCESSED_ROOT}" \
        --fingerprint-source "${FINGERPRINT_SOURCE}" \
        --cache-audit "${CACHE_AUDIT}" \
        --output-root "${OVERLAY_ROOT}"
    echo "S2_E_CONTINUE_PREFLIGHT_PASS contract=${CONTRACT_PATH}"
}

case "${STAGE}" in
    preflight)
        require_no_related_processes
        preflight
        ;;
    smoke)
        acquire_execution_lock
        require_no_related_processes
        preflight
        s2_uhost_require_single_visible_gpu
        [[ ! -e "${SMOKE_DIR}" ]] || { echo "Immutable smoke output already exists: ${SMOKE_DIR}" >&2; exit 1; }
        if env | grep -Eq '^(S2_MET_AUG_(COMPONENT_MANIFEST|ROUTE_CONFIG|VALID_MASK_MANIFEST|ROUTE_GATE|G1_CODE_DIR|G1_CHECKPOINT_ROOT|G1_CHECKPOINT_SELECTION|G2_QC_GATE)|G1_CODE_DIR|G1_CHECKPOINT_ROOT|G1_SELECTION|G2_PARENT_GATE)='; then
            echo "Refusing inherited Route/G1/G2/Diffusion asset variables" >&2
            exit 1
        fi
        export nnUNet_raw="${NNUNET_RAW_ROOT}"
        export nnUNet_preprocessed="${OVERLAY_ROOT}"
        export S2_MET_AUG_ENABLE=0
        "${PYTHON_BIN}" "${S2_REPOSITORY}/scripts/25_run_e_continue_training_smoke.py" \
            --nnunet-raw "${NNUNET_RAW_ROOT}" \
            --nnunet-preprocessed "${OVERLAY_ROOT}" \
            --split-dir "${SPLIT_DIR}" \
            --pretrained-weights "${E_CHECKPOINT}" \
            --cache-audit "${CACHE_AUDIT}" \
            --output-dir "${SMOKE_DIR}" \
            --steps 8 \
            --eta-safety-factor "${ETA_SAFETY_FACTOR}" \
            --max-estimated-training-hours 45
        "${PYTHON_BIN}" - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path

report_path = Path(os.environ["SMOKE_REPORT"])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "pass":
    raise SystemExit("E-continue smoke did not pass")
if report.get("validation_executed") or report.get("checkpoint_saved"):
    raise SystemExit("E-continue smoke violated no-validation/no-checkpoint contract")
if report.get("generative_assets_loaded"):
    raise SystemExit("E-continue smoke loaded a generative asset")
deadline = datetime.fromisoformat(os.environ["DEADLINE_UTC"].replace("Z", "+00:00"))
remaining_hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
reserve = float(os.environ["DOWNSTREAM_RESERVE_HOURS"])
training = float(report["timing"]["estimated_200_epochs_hours_conservative"])
available_for_training = remaining_hours - reserve
if remaining_hours <= 0 or training > available_for_training:
    hold = report_path.parent.parent / "TRAINING_ETA_EXCEEDS_BUDGET.hold"
    hold.write_text(
        f"remaining_hours={remaining_hours:.6f}\n"
        f"downstream_reserve_hours={reserve:.6f}\n"
        f"available_training_hours={available_for_training:.6f}\n"
        f"estimated_training_hours={training:.6f}\n",
        encoding="utf-8",
    )
    raise SystemExit("E-continue plus downstream reserve exceeds remaining deadline")
approval = report_path.parent.parent / "TRAINING_SMOKE_APPROVED.ok"
approval.write_text(
    f"smoke_report={report_path}\n"
    f"smoke_report_sha256={report['report_sha256']}\n"
    f"steps={report['timing']['steps']}\n"
    f"estimated_200_epochs_hours={report['timing']['estimated_200_epochs_hours']:.6f}\n"
    f"estimated_200_epochs_hours_conservative={training:.6f}\n"
    f"remaining_hours_at_approval={remaining_hours:.6f}\n"
    f"downstream_reserve_hours={reserve:.6f}\n",
    encoding="utf-8",
)
print(
    "S2_E_CONTINUE_SMOKE_PASS "
    f"estimated_200_epochs_hours_conservative={training:.3f} "
    f"remaining_hours={remaining_hours:.3f} reserve_hours={reserve:.3f}"
)
PY
        ;;
    train)
        acquire_execution_lock
        require_no_related_processes
        preflight
        s2_uhost_require_single_visible_gpu
        s2_uhost_require_file "${SMOKE_APPROVAL}" "E-continue smoke approval"
        [[ ! -e "${LAUNCH_MARKER}" ]] || { echo "E-continue has already been launched: ${LAUNCH_MARKER}" >&2; exit 1; }
        [[ ! -e "${TRAIN_RESULTS}" ]] || { echo "E-continue result root already exists: ${TRAIN_RESULTS}" >&2; exit 1; }
        mkdir -p "${FALLBACK_ROOT}/training" "${FALLBACK_ROOT}/logs"
        printf 'launched_at=%s\npid=%s\ngpu=%s\ncontract=%s\npreprocessed_overlay_audit=%s\n' \
            "$(date -u +%FT%TZ)" "$$" "${CUDA_VISIBLE_DEVICES:-unset}" "${CONTRACT_PATH}" "${OVERLAY_AUDIT}" > "${LAUNCH_MARKER}"
        export nnUNet_raw="${NNUNET_RAW_ROOT}"
        export nnUNet_preprocessed="${OVERLAY_ROOT}"
        export nnUNet_results="${TRAIN_RESULTS}"
        export BRATS_SPLIT_DIR="${SPLIT_DIR}"
        export S2_PRETRAINED_WEIGHTS="${E_CHECKPOINT}"
        export S2_EXPERIMENT_MODE=met_aug_route_a_control
        export S2_MET_AUG_ENABLE=0
        export S2_COMPLETION_EPOCHS=200
        export S2_COMPLETION_INITIAL_LR=0.001
        export S2_COMPLETION_SAVE_EVERY=25
        export S2_FOCAL_GAMMA=2.0
        export S2_PAIRED_TRAINING_SEED=20260724
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export S2_CONTINUE=0
        export S2_SKIP_COMPLETED=0
        export nnUNet_compile=0
        export nnUNet_n_proc_DA=0
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
        cd "${S2_REPOSITORY}"
        echo "S2_E_CONTINUE_TRAIN_START gpu=${CUDA_VISIBLE_DEVICES:-unset} results=${TRAIN_RESULTS}"
        bash train.sh
        s2_uhost_require_file "${RESULT_FOLD}/checkpoint_final.pth" "E-continue final checkpoint"
        s2_uhost_require_file "${RESULT_FOLD}/validation/summary.json" "E-continue validation summary"
        PREDICTION_COUNT="$(find "${RESULT_FOLD}/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')"
        [[ "${PREDICTION_COUNT}" == "103" ]] || { echo "Incomplete E-continue predictions: ${PREDICTION_COUNT}/103" >&2; exit 1; }
        CHECKPOINT_SHA="$(s2_uhost_sha256 "${RESULT_FOLD}/checkpoint_final.pth")"
        printf 'completed_at=%s\ntrainer=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nvalidation_predictions=%s\n' \
            "$(date -u +%FT%TZ)" "${TRAINER}" "${RESULT_FOLD}/checkpoint_final.pth" "${CHECKPOINT_SHA}" "${PREDICTION_COUNT}" \
            > "${COMPLETE_MARKER}"
        s2_uhost_require_free_space
        echo "S2_E_CONTINUE_TRAIN_PASS checkpoint_sha256=${CHECKPOINT_SHA}"
        ;;
    status)
        printf 'fallback_root=%s\n' "${FALLBACK_ROOT}"
        pgrep -af '([2]5_run_e_continue_training_smoke.py|[n]nUNetv2_train.*nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune)' || true
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
        for marker in "${CONTRACT_PATH}" "${SMOKE_REPORT}" "${SMOKE_APPROVAL}" "${LAUNCH_MARKER}" "${COMPLETE_MARKER}"; do
            [[ -e "${marker}" ]] && printf 'present=%s\n' "${marker}" || printf 'absent=%s\n' "${marker}"
        done
        if [[ -d "${RESULT_FOLD}" ]]; then
            find "${RESULT_FOLD}" -maxdepth 2 -type f -printf '%T@ %p %s\n' | sort -n | tail -n 12
        fi
        df -BG --output=avail "${FALLBACK_ROOT%/*}" | tail -n 1
        ;;
esac
