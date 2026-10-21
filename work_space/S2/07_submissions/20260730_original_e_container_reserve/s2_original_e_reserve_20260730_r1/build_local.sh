#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
fi

IMAGE="${IMAGE:-s2-original-e-reserve:experimental_unvalidated}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/../../results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_CHECKPOINT_SHA256="4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"

if [[ "${CHECKPOINT_DIR}" != /* ]]; then
    CHECKPOINT_DIR="${ROOT}/${CHECKPOINT_DIR}"
fi
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd -P)"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_final.pth"

command -v docker >/dev/null 2>&1 || {
    echo "Docker is required; no installation is performed by this script." >&2
    exit 2
}
docker buildx version >/dev/null
test -f "${CHECKPOINT}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 2
}

"${PYTHON_BIN}" "${ROOT}/validate_context.py" --checkpoint "${CHECKPOINT}"

CHECKPOINT_CONTEXT="$(mktemp -d "${ROOT}/.checkpoint-context.XXXXXX")"
cleanup() {
    rm -rf -- "${CHECKPOINT_CONTEXT}"
}
trap cleanup EXIT

if ! ln "${CHECKPOINT}" "${CHECKPOINT_CONTEXT}/checkpoint_final.pth" 2>/dev/null; then
    cp -p "${CHECKPOINT}" "${CHECKPOINT_CONTEXT}/checkpoint_final.pth"
fi

docker buildx build \
    --platform linux/amd64 \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256}" \
    --build-context "checkpoint=${CHECKPOINT_CONTEXT}" \
    --tag "${IMAGE}" \
    --load \
    "${ROOT}"

echo "Built ${IMAGE}; status=experimental_unvalidated; no registry push performed."
