#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
S2_UHOST="${S2_UHOST:-s2-h20}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/brats2026/ECNU_EYU_data}"

SYNC_PATHS=(
    "time_pipeline/BraTS2026_Task1_最终全流程执行计划.md"
    "work_space/S2/BraTS2026_S2_RC_v1.0/repository"
    "work_space/S2/slurm"
    "work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md"
    "work_space/G1/扩散模型加速方法.md"
    "work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
    "work_space/G1/code/BraTS_2023_2024_solutions-main 3/model.py"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t1c/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t1n/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t2w/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t2f/weights/diffusion_150000.pt"
    "work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
    "work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721"
    "work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth"
)

CRITICAL_FILES=(
    "work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN/src/infer/diffusion_inference_utils.py"
    "work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN/src/networks/DiffusionNetwork.py"
    "work_space/G1/code/BraTS_2023_2024_solutions-main 3/model.py"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t1c/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t1n/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t2w/weights/diffusion_150000.pt"
    "work_space/G1/results/g1_diffusion_v3_final_20260720/canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/t2f/weights/diffusion_150000.pt"
    "work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json"
    "work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/g2_diffusion_qc_gate.json"
    "work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
    "work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth"
)

for path in "${SYNC_PATHS[@]}"; do
    if [[ ! -e "${LOCAL_ROOT}/${path}" ]]; then
        echo "Missing local runtime asset: ${LOCAL_ROOT}/${path}" >&2
        exit 1
    fi
done

ssh "${S2_UHOST}" "mkdir -p '${REMOTE_ROOT}' /root/brats2026/logs /root/brats2026/data /root/brats2026/runs /root/brats2026/envs"
tar -C "${LOCAL_ROOT}" \
    --exclude='.DS_Store' --exclude='__pycache__' --exclude='*.pyc' \
    -cf - "${SYNC_PATHS[@]}" \
    | ssh "${S2_UHOST}" "tar -C '${REMOTE_ROOT}' -xf -"

ssh "${S2_UHOST}" "\
    env_file='${REMOTE_ROOT}/work_space/S2/slurm/.env.uhost'; \
    env_example='${REMOTE_ROOT}/work_space/S2/slurm/.env.uhost.example'; \
    if [[ ! -e \"\${env_file}\" ]]; then cp \"\${env_example}\" \"\${env_file}\"; fi; \
    chmod 600 \"\${env_file}\"; \
    chmod +x '${REMOTE_ROOT}'/work_space/S2/slurm/*.sh '${REMOTE_ROOT}'/work_space/S2/slurm/*.slurm; \
    chmod 644 '${REMOTE_ROOT}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/model.py'"

MANIFEST=$(mktemp)
trap 'rm -f "${MANIFEST}"' EXIT
(
    cd "${LOCAL_ROOT}"
    for path in "${CRITICAL_FILES[@]}"; do
        shasum -a 256 "${path}"
    done
) > "${MANIFEST}"
scp "${MANIFEST}" "${S2_UHOST}:/root/brats2026/runtime_asset_manifest.sha256"
ssh "${S2_UHOST}" "cd '${REMOTE_ROOT}' && sha256sum -c /root/brats2026/runtime_asset_manifest.sha256"

echo "S2_UHOST_RUNTIME_SYNC_PASS host=${S2_UHOST} root=${REMOTE_ROOT}"
