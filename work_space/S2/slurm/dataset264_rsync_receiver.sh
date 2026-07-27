#!/usr/bin/env bash

set -euo pipefail

ECNU_USER="${1:?ECNU user is required}"
REMOTE_FORWARD_PORT="${2:?reverse-forward port is required}"
ECNU_STORAGE_ROOT="${3:?ECNU storage root is required}"
UHOST_DATA_ROOT="${4:?UHost data root is required}"
DATASET_NAME="${5:?dataset name is required}"
SSH_TRANSPORT="ssh -p ${REMOTE_FORWARD_PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/.ssh/known_hosts_ecnu_tunnel -o ServerAliveInterval=30"

command -v rsync >/dev/null
mkdir -p "${UHOST_DATA_ROOT}/nnUNet_raw" "${UHOST_DATA_ROOT}/nnUNet_preprocessed"

rsync -aL --partial --append-verify --info=progress2 \
    -e "${SSH_TRANSPORT}" \
    "${ECNU_USER}@127.0.0.1:${ECNU_STORAGE_ROOT}/nnUNet_raw/${DATASET_NAME}" \
    "${UHOST_DATA_ROOT}/nnUNet_raw/"

rsync -aL --partial --append-verify --info=progress2 \
    -e "${SSH_TRANSPORT}" \
    "${ECNU_USER}@127.0.0.1:${ECNU_STORAGE_ROOT}/nnUNet_preprocessed/${DATASET_NAME}" \
    "${UHOST_DATA_ROOT}/nnUNet_preprocessed/"

test -s "${UHOST_DATA_ROOT}/nnUNet_raw/${DATASET_NAME}/dataset.json"
test -s "${UHOST_DATA_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json"
test -d "${UHOST_DATA_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_3d_fullres"
du -sh \
    "${UHOST_DATA_ROOT}/nnUNet_raw/${DATASET_NAME}" \
    "${UHOST_DATA_ROOT}/nnUNet_preprocessed/${DATASET_NAME}"
df -h "${UHOST_DATA_ROOT}"
echo "DATASET264_RELAY_PASS"
