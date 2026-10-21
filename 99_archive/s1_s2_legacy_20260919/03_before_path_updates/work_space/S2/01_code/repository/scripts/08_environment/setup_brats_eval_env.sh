#!/usr/bin/env bash

# Create or validate the CPU-only environment used by official BraTS evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ="${PROJ:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-brats_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
EXPECTED_BRATS_EVAL_VERSION="${EXPECTED_BRATS_EVAL_VERSION:-0.0.8}"
EXPECTED_PANOPTICA_VERSION="${EXPECTED_PANOPTICA_VERSION:-2.1.0}"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
REQUIREMENTS_FILE="${S2_REPO}/requirements_eval.txt"
MODE="setup"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_brats_eval_env.sh [--check]

Without arguments, create/update the brats_eval Conda environment and verify it.
With --check, only validate an existing environment; do not install anything.

Optional environment variables:
  CONDA_SH                    path to conda.sh
  CONDA_ENV                   environment name (default: brats_eval)
  PYTHON_VERSION              Python version for a new env (default: 3.10)
  EXPECTED_BRATS_EVAL_VERSION required BraTS-evaluation version (default: 0.0.8)
  EXPECTED_PANOPTICA_VERSION  required Panoptica version (default: 2.1.0)
EOF
}

case "${1:-}" in
    "") ;;
    --check) MODE="check" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Cannot initialize Conda. Set CONDA_SH to the server's conda.sh path." >&2
    exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Missing evaluation requirements: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

ENV_EXISTS=0
while read -r env_name _; do
    if [[ "${env_name}" == "${CONDA_ENV}" ]]; then
        ENV_EXISTS=1
        break
    fi
done < <(conda env list)

if [[ "${ENV_EXISTS}" == "0" ]]; then
    if [[ "${MODE}" == "check" ]]; then
        echo "Missing Conda environment ${CONDA_ENV}. Run this script without --check first." >&2
        exit 1
    fi
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip -y
elif ! conda run -n "${CONDA_ENV}" python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.'))) else 1)" >/dev/null 2>&1; then
    echo "Existing ${CONDA_ENV} does not use Python ${PYTHON_VERSION}; refusing to overwrite it." >&2
    echo "After checking that it is disposable, remove it manually or set CONDA_ENV to a new name." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"

if [[ "${MODE}" == "setup" ]]; then
    python -m pip install --disable-pip-version-check -r "${REQUIREMENTS_FILE}"
fi

ACTUAL_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("BraTS-evaluation"))')"
ACTUAL_PANOPTICA_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("panoptica"))')"
if [[ "${ACTUAL_VERSION}" != "${EXPECTED_BRATS_EVAL_VERSION}" ]]; then
    echo "BraTS-evaluation version mismatch: expected ${EXPECTED_BRATS_EVAL_VERSION}, got ${ACTUAL_VERSION}" >&2
    exit 1
fi
if [[ "${ACTUAL_PANOPTICA_VERSION}" != "${EXPECTED_PANOPTICA_VERSION}" ]]; then
    echo "Panoptica version mismatch: expected ${EXPECTED_PANOPTICA_VERSION}, got ${ACTUAL_PANOPTICA_VERSION}" >&2
    exit 1
fi

python -c "import brats_evaluation, nibabel, pandas, panoptica, SimpleITK; print('BraTS evaluation imports OK')"
command -v brats-evaluate
command -v brats-parse-metrics

echo "Environment ready: ${CONDA_ENV}"
echo "BraTS-evaluation: ${ACTUAL_VERSION}"
echo "Panoptica: ${ACTUAL_PANOPTICA_VERSION}"
echo "Official source snapshot: ${PROJ}/data_space/task1_2026/reference_code/BraTS_evaluation"
