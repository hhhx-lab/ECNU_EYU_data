#!/usr/bin/env bash
#SBATCH --job-name=s2_met_aug_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/root/brats2026/logs/%x_%j.out
#SBATCH --error=/root/brats2026/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

STAGE="${S2_MET_AUG_STAGE:-${1:-}}"
case "${STAGE}" in
    prepare|prepare_compact_reuse|resume_prepare|gate2|finalize|training_smoke|status) ;;
    *)
        echo "Usage: S2_MET_AUG_STAGE={prepare|prepare_compact_reuse|resume_prepare|gate2|finalize|training_smoke|status} $0" >&2
        exit 2
        ;;
esac

s2_uhost_activate_runtime
s2_uhost_bind_contract
mkdir -p "${ROUTE_ROOT}" "${ROUTE_ROOT}/logs"

REUSE_SOURCE_ROUTE_ROOT="${S2_MET_AUG_REUSE_SOURCE_ROUTE_ROOT:-}"
if [[ -n "${REUSE_SOURCE_ROUTE_ROOT}" ]]; then
    COMPONENT_MANIFEST="${REUSE_SOURCE_ROUTE_ROOT}/component_pool/component_manifest.json"
    COORDINATE_AUDIT="${REUSE_SOURCE_ROUTE_ROOT}/preprocessed_coordinate_contract.json"
    VALID_MASK_MANIFEST="${REUSE_SOURCE_ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
else
    COMPONENT_MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
    COORDINATE_AUDIT="${ROUTE_ROOT}/preprocessed_coordinate_contract.json"
    VALID_MASK_MANIFEST="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
fi
ROUTE_CONFIG="${ROUTE_ROOT}/route_a_config.json"
REUSE_AUDIT="${ROUTE_ROOT}/compact_support_reuse_audit.json"
GATE1_EQUIVALENCE_DIR="${ROUTE_ROOT}/gate1_parallel_equivalence"
GATE1_REPORT="${ROUTE_ROOT}/gate1/gate1_report.json"
GATE2_SMOKE_MANIFEST="${ROUTE_ROOT}/gate2_smoke_manifest.json"
GATE2_RUN="${ROUTE_ROOT}/gate2_run"
GATE2_AUTOMATIC_REPORT="${GATE2_RUN}/gate2_automatic_report.json"
GATE2_DECISIONS="${GATE2_RUN}/manual_review_decisions.csv"
GATE2_FINAL_REPORT="${ROUTE_ROOT}/gate2_final_report.json"
ROUTE_APPROVAL="${ROUTE_ROOT}/route_a_approval.json"
TRAINING_SMOKE_DIR="${ROUTE_ROOT}/training_smoke"
TRAINING_SMOKE_REPORT="${TRAINING_SMOKE_DIR}/met_aug_training_smoke_report.json"

if [[ "${STAGE}" == "status" ]]; then
    for artifact in \
        "${COORDINATE_AUDIT}" "${COMPONENT_MANIFEST}" "${ROUTE_CONFIG}" "${VALID_MASK_MANIFEST}" \
        "${REUSE_AUDIT}" "${GATE1_EQUIVALENCE_DIR}/parallel_equivalence_report.json" \
        "${GATE1_REPORT}" "${GATE2_SMOKE_MANIFEST}" "${GATE2_AUTOMATIC_REPORT}" \
        "${GATE2_DECISIONS}" "${GATE2_FINAL_REPORT}" "${ROUTE_APPROVAL}" \
        "${TRAINING_SMOKE_REPORT}"; do
        if [[ -s "${artifact}" ]]; then
            echo "present ${artifact}"
        else
            echo "missing ${artifact}"
        fi
    done
    exit 0
fi

s2_uhost_require_contract_inputs
s2_uhost_require_free_space
cd "${S2_REPOSITORY}"

case "${STAGE}" in
    prepare)
        for output in \
            "${COORDINATE_AUDIT}" "${ROUTE_ROOT}/component_pool" "${ROUTE_CONFIG}" "${ROUTE_ROOT}/valid_masks" \
            "${ROUTE_ROOT}/gate1" "${GATE2_SMOKE_MANIFEST}"; do
            if [[ -e "${output}" ]]; then
                echo "Prepare output already exists; refusing to overwrite immutable evidence: ${output}" >&2
                exit 1
            fi
        done

        "${PYTHON_BIN}" scripts/21_validate_preprocessed_coordinate_contract.py \
            --dataset-dir "${DATASET_DIR}" \
            --preprocessed-dir "${PREPROCESSED_DIR}" \
            --output "${COORDINATE_AUDIT}"

        "${PYTHON_BIN}" scripts/12_build_met_aug_component_pool.py \
            --dataset-dir "${DATASET_DIR}" \
            --preprocessed-dir "${PREPROCESSED_DIR}" \
            --train-file "${TRAIN_FILE}" \
            --mapping-csv "${MAPPING_CSV}" \
            --preprocessed-coordinate-audit "${COORDINATE_AUDIT}" \
            --output-dir "${ROUTE_ROOT}/component_pool"

        "${PYTHON_BIN}" scripts/13_make_met_aug_route_a_config.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --output "${ROUTE_CONFIG}"

        "${PYTHON_BIN}" scripts/14_prepare_met_aug_valid_masks.py \
            --dataset-dir "${DATASET_DIR}" \
            --preprocessed-dir "${PREPROCESSED_DIR}" \
            --train-file "${TRAIN_FILE}" \
            --allow-nearest-resample \
            --output-dir "${ROUTE_ROOT}/valid_masks"

        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
            "${PYTHON_BIN}" scripts/15_run_met_aug_gate1.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output-dir "${ROUTE_ROOT}/gate1" \
            --events 100000 \
            --workers "${S2_MET_AUG_GATE1_WORKERS:-16}"

        "${PYTHON_BIN}" scripts/17_prepare_met_aug_gate2_smoke.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output "${GATE2_SMOKE_MANIFEST}"

        printf 'completed_at=%s\n' "$(date -u +%FT%TZ)" > "${ROUTE_ROOT}/PREPARE_COMPLETE.ok"
        echo "S2_MET_AUG_PREPARE_PASS route_root=${ROUTE_ROOT}"
        ;;

    prepare_compact_reuse)
        EXPECTED_SOURCE_ROUTE_ROOT="/root/brats2026/runs/s2_met_aug_route_a_20260726_r4"
        if [[ "${REUSE_SOURCE_ROUTE_ROOT}" != "${EXPECTED_SOURCE_ROUTE_ROOT}" ]]; then
            echo "R5 compact-support preparation must reference the locked R4 root." >&2
            exit 1
        fi
        for output in \
            "${ROUTE_CONFIG}" "${REUSE_AUDIT}" "${GATE1_EQUIVALENCE_DIR}" \
            "${ROUTE_ROOT}/gate1" "${GATE2_SMOKE_MANIFEST}" "${ROUTE_ROOT}/PREPARE_COMPLETE.ok"; do
            if [[ -e "${output}" ]]; then
                echo "Compact-support prepare output already exists; refusing to overwrite: ${output}" >&2
                exit 1
            fi
        done

        "${PYTHON_BIN}" scripts/24_prepare_met_aug_compact_support_reuse.py \
            --source-route-root "${REUSE_SOURCE_ROUTE_ROOT}" \
            --target-route-root "${ROUTE_ROOT}" \
            --route-config-output "${ROUTE_CONFIG}" \
            --reuse-audit-output "${REUSE_AUDIT}" \
            --seed 20260725 \
            --max-total-support-voxels 4096 \
            --max-total-to-core-ratio 20 \
            --expected-eligible-component-count 3554 \
            --expected-eligible-bin-count 27_49=950 \
            --expected-eligible-bin-count 50_275=1764 \
            --expected-eligible-bin-count gt_275=840 \
            --expected-eligible-group-count 27_49=263 \
            --expected-eligible-group-count 50_275=384 \
            --expected-eligible-group-count gt_275=289 \
            --expected-prepare-marker-sha256 3cd1481f0ffb0af4396a3481b06faea6a00cbb6c85ba7ed39ffbd97c0105cf50 \
            --expected-coordinate-audit-sha256 4b6f4038c179878e93dbc61aff6f81a4cbc81c05c2dbc83f787c4f7e99c6776a \
            --expected-component-manifest-file-sha256 db05d8ef62ef3a10ae37f3467217de331a8d1e39c089d57d5d97c30cdb731998 \
            --expected-component-manifest-identity-sha256 c8a413bc180b3a120ee3415f548eab98c305cbe3a63867db89d09f19243da26e \
            --expected-valid-mask-manifest-file-sha256 a381b24ab508b7eba758b971e4ffe05b4a10e319c6299c1aa4a67f955bc6a169 \
            --expected-valid-mask-manifest-identity-sha256 d27b71ec61c34942d33a5c3e0b67611a43daa166b8bbb6a25cb129c5099197ce

        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
            "${PYTHON_BIN}" scripts/22_validate_gate1_parallel_equivalence.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output-dir "${GATE1_EQUIVALENCE_DIR}" \
            --events 256 \
            --workers "${S2_MET_AUG_GATE1_WORKERS:-16}"
        "${PYTHON_BIN}" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["serial_parallel_jsonl_byte_identical"] is True and p["serial_parallel_report_byte_identical"] is True, p' \
            "${GATE1_EQUIVALENCE_DIR}/parallel_equivalence_report.json"

        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
            "${PYTHON_BIN}" scripts/15_run_met_aug_gate1.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output-dir "${ROUTE_ROOT}/gate1" \
            --events 100000 \
            --workers "${S2_MET_AUG_GATE1_WORKERS:-16}"

        "${PYTHON_BIN}" scripts/17_prepare_met_aug_gate2_smoke.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output "${GATE2_SMOKE_MANIFEST}"

        printf 'completed_at=%s\nmode=compact_support_reuse\nsource_route_root=%s\ngate1=deterministic_parallel\n' \
            "$(date -u +%FT%TZ)" "${REUSE_SOURCE_ROUTE_ROOT}" \
            > "${ROUTE_ROOT}/PREPARE_COMPLETE.ok"
        echo "S2_MET_AUG_PREPARE_PASS route_root=${ROUTE_ROOT} mode=compact_support_reuse"
        ;;

    resume_prepare)
        s2_uhost_require_file "${ROUTE_ROOT}/gate1_parallel_migration/PROMOTION_AUDIT.json" \
            "parallel Gate 1 promotion audit"
        s2_uhost_require_file "${GATE1_REPORT}" "promoted Gate 1 report"
        if [[ -e "${GATE2_SMOKE_MANIFEST}" || -e "${ROUTE_ROOT}/PREPARE_COMPLETE.ok" ]]; then
            echo "Resumed prepare evidence already exists; refusing to overwrite." >&2
            exit 1
        fi
        "${PYTHON_BIN}" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "pass" and p["event_count"] == 100000 and p["violations"] == [], p' \
            "${GATE1_REPORT}"
        "${PYTHON_BIN}" scripts/17_prepare_met_aug_gate2_smoke.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --output "${GATE2_SMOKE_MANIFEST}"
        printf 'completed_at=%s\nresume=parallel_gate1_promotion\n' "$(date -u +%FT%TZ)" \
            > "${ROUTE_ROOT}/PREPARE_COMPLETE.ok"
        echo "S2_MET_AUG_PREPARE_PASS route_root=${ROUTE_ROOT} resume=parallel_gate1_promotion"
        ;;

    gate2)
        s2_uhost_require_single_visible_gpu
        s2_uhost_require_file "${ROUTE_ROOT}/PREPARE_COMPLETE.ok" "prepare completion marker"
        if [[ -e "${GATE2_RUN}" ]]; then
            echo "Gate 2 output already exists; refusing to overwrite: ${GATE2_RUN}" >&2
            exit 1
        fi
        "${PYTHON_BIN}" scripts/18_run_met_aug_gate2_smoke.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --smoke-manifest "${GATE2_SMOKE_MANIFEST}" \
            --preprocessed-dir "${PREPROCESSED_DIR}" \
            --g1-code-dir "${G1_CODE_DIR}" \
            --g1-checkpoint-root "${G1_CHECKPOINT_ROOT}" \
            --g1-checkpoint-selection "${G1_SELECTION}" \
            --g2-parent-gate "${G2_PARENT_GATE}" \
            --output-dir "${GATE2_RUN}" \
            --device cuda
        "${PYTHON_BIN}" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "hold_for_manual_review", p' \
            "${GATE2_AUTOMATIC_REPORT}"
        printf 'completed_at=%s\n' "$(date -u +%FT%TZ)" > "${ROUTE_ROOT}/GATE2_HOLD_FOR_REVIEW.ok"
        echo "S2_MET_AUG_GATE2_HOLD review_template=${GATE2_RUN}/manual_review_template.csv"
        ;;

    finalize)
        s2_uhost_require_file "${GATE2_DECISIONS}" "24-case manual review decisions"
        if [[ -e "${GATE2_FINAL_REPORT}" || -e "${ROUTE_APPROVAL}" ]]; then
            echo "Final Gate evidence already exists; refusing to overwrite." >&2
            exit 1
        fi
        "${PYTHON_BIN}" scripts/19_finalize_met_aug_gate2_review.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --smoke-manifest "${GATE2_SMOKE_MANIFEST}" \
            --automatic-report "${GATE2_AUTOMATIC_REPORT}" \
            --review-decisions "${GATE2_DECISIONS}" \
            --output "${GATE2_FINAL_REPORT}"

        "${PYTHON_BIN}" scripts/16_finalize_met_aug_route_a_gate.py \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --gate1-report "${GATE1_REPORT}" \
            --gate2-report "${GATE2_FINAL_REPORT}" \
            --g1-checkpoint-selection "${G1_SELECTION}" \
            --g2-parent-gate "${G2_PARENT_GATE}" \
            --g1-code-dir "${G1_CODE_DIR}" \
            --output "${ROUTE_APPROVAL}"
        "${PYTHON_BIN}" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema_version"] == 3 and p["decision"] == "approve", p' \
            "${ROUTE_APPROVAL}"
        echo "S2_MET_AUG_ROUTE_APPROVED approval=${ROUTE_APPROVAL}"
        ;;

    training_smoke)
        s2_uhost_require_single_visible_gpu
        s2_uhost_require_file "${ROUTE_APPROVAL}" "Route A approval"
        if [[ -e "${TRAINING_SMOKE_DIR}" ]]; then
            echo "Training smoke output already exists; refusing to overwrite: ${TRAINING_SMOKE_DIR}" >&2
            exit 1
        fi
        "${PYTHON_BIN}" scripts/20_run_met_aug_training_smoke.py \
            --nnunet-raw "${nnUNet_raw}" \
            --nnunet-preprocessed "${nnUNet_preprocessed}" \
            --split-dir "${S2_REPOSITORY}/data/splits/completion_warmstart" \
            --pretrained-weights "${E_CHECKPOINT}" \
            --component-manifest "${COMPONENT_MANIFEST}" \
            --route-config "${ROUTE_CONFIG}" \
            --valid-mask-manifest "${VALID_MASK_MANIFEST}" \
            --route-approval "${ROUTE_APPROVAL}" \
            --g1-code-dir "${G1_CODE_DIR}" \
            --g1-checkpoint-root "${G1_CHECKPOINT_ROOT}" \
            --g1-checkpoint-selection "${G1_SELECTION}" \
            --g2-parent-gate "${G2_PARENT_GATE}" \
            --output-dir "${TRAINING_SMOKE_DIR}" \
            --min-steps 4 \
            --max-steps 12 \
            --min-committed-events 1 \
            --min-gpu-memory-gib 90

        SMOKE_MAX_ESTIMATED_HOURS="${SMOKE_MAX_ESTIMATED_HOURS:-45}"
        read -r SMOKE_STATUS ESTIMATED_HOURS < <(
            "${PYTHON_BIN}" -c \
                'import json,sys; p=json.load(open(sys.argv[1])); print(p["status"], p.get("timing", {}).get("estimated_200_epochs_hours", float("inf")))' \
                "${TRAINING_SMOKE_REPORT}"
        )
        if [[ "${SMOKE_STATUS}" != "pass" ]]; then
            echo "Training smoke failed." >&2
            exit 1
        fi
        "${PYTHON_BIN}" -c \
            'import sys; observed=float(sys.argv[1]); limit=float(sys.argv[2]); raise SystemExit(0 if observed <= limit else 3)' \
            "${ESTIMATED_HOURS}" "${SMOKE_MAX_ESTIMATED_HOURS}" || {
                printf 'estimated_200_epochs_hours=%s\nlimit_hours=%s\n' \
                    "${ESTIMATED_HOURS}" "${SMOKE_MAX_ESTIMATED_HOURS}" \
                    > "${ROUTE_ROOT}/TRAINING_SMOKE_TOO_SLOW.hold"
                echo "Smoke passed technically but exceeded the time budget; keep Heun-18 training on hold." >&2
                exit 3
            }
        printf 'estimated_200_epochs_hours=%s\nlimit_hours=%s\n' \
            "${ESTIMATED_HOURS}" "${SMOKE_MAX_ESTIMATED_HOURS}" \
            > "${ROUTE_ROOT}/TRAINING_SMOKE_APPROVED.ok"
        echo "S2_MET_AUG_TRAINING_SMOKE_PASS estimated_200_epochs_hours=${ESTIMATED_HOURS}"
        ;;
esac
