#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

BASE_ENV_PATH="${BASE_ENV_PATH:-/usr/local/miniconda3/envs/py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/brats2026/envs/g1_diffusion_v3}"
CONDA_BIN="${CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
REQUIREMENTS="${SCRIPT_DIR}/requirements-cloud.txt"

test -x "${CONDA_BIN}"
test -x "${BASE_ENV_PATH}/bin/python"
test -s "${REQUIREMENTS}"
mkdir -p "$(dirname "${CONDA_ENV_PATH}")"

if [[ ! -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --prefix "${CONDA_ENV_PATH}" --clone "${BASE_ENV_PATH}"
fi

"${CONDA_ENV_PATH}/bin/python" -m pip install --no-cache-dir --requirement "${REQUIREMENTS}"
"${CONDA_ENV_PATH}/bin/python" - <<'PY'
import matplotlib
import monai
import nibabel
import nilearn
import numpy
import pandas
import scipy
import skimage
import sklearn
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert numpy.__version__.startswith("1.26."), numpy.__version__
print("ENVIRONMENT_PASS")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("monai", monai.__version__, "numpy", numpy.__version__)
PY
