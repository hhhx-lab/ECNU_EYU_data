#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

BASE_ENV_PATH="${BASE_ENV_PATH:-/usr/local/miniconda3/envs/py312}"
CONDA_BIN="${CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
REQUIREMENTS="${SCRIPT_DIR}/requirements-h20.txt"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
S2_TORCH_VERSION="${S2_TORCH_VERSION:-2.7.1}"
S2_TORCHVISION_VERSION="${S2_TORCHVISION_VERSION:-0.22.1}"
S2_TORCHAUDIO_VERSION="${S2_TORCHAUDIO_VERSION:-2.7.1}"
S2_TORCH_CUDA="${S2_TORCH_CUDA:-12.8}"
export S2_TORCH_VERSION S2_TORCHVISION_VERSION S2_TORCHAUDIO_VERSION S2_TORCH_CUDA

test -x "${CONDA_BIN}"
test -x "${BASE_ENV_PATH}/bin/python"
test -s "${REQUIREMENTS}"
mkdir -p "$(dirname "${CONDA_ENV_PATH}")" "${ROUTE_ROOT}/runtime"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    "${CONDA_BIN}" create --yes --prefix "${CONDA_ENV_PATH}" --clone "${BASE_ENV_PATH}"
fi

# nnU-Net 2.8 explicitly excludes torch 2.9.*. Replace the cloned image build
# only inside this project environment with the locally validated 2.7.1/cu128
# stack before resolving the remaining dependencies.
"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
    --index-url "${PYTORCH_INDEX_URL}" \
    "torch==${S2_TORCH_VERSION}" \
    "torchvision==${S2_TORCHVISION_VERSION}" \
    "torchaudio==${S2_TORCHAUDIO_VERSION}"
"${PYTHON_BIN}" -m pip install --no-cache-dir --requirement "${REQUIREMENTS}"
"${PYTHON_BIN}" -m pip check

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import os
import torch

required = {
    "nnunetv2": "2.8.0",
    "batchgeneratorsv2": "0.3.3",
    "numpy": "1.26.4",
    "monai": "1.5.1",
}
for package, expected in required.items():
    actual = md.version(package)
    if actual != expected:
        raise SystemExit(f"{package} expected {expected}, got {actual}")
torch_release = torch.__version__.split("+", 1)[0]
if torch_release != os.environ["S2_TORCH_VERSION"]:
    raise SystemExit(f"torch expected {os.environ['S2_TORCH_VERSION']}, got {torch.__version__}")
if torch.version.cuda != os.environ["S2_TORCH_CUDA"]:
    raise SystemExit(f"torch CUDA expected {os.environ['S2_TORCH_CUDA']}, got {torch.version.cuda}")
for package, env_name in (
    ("torchvision", "S2_TORCHVISION_VERSION"),
    ("torchaudio", "S2_TORCHAUDIO_VERSION"),
):
    actual = md.version(package).split("+", 1)[0]
    expected = os.environ[env_name]
    if actual != expected:
        raise SystemExit(f"{package} expected {expected}, got {actual}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "index": index,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_gib": props.total_memory / 1024**3,
    })
    if props.major != 9 or props.total_memory / 1024**3 < 90:
        raise SystemExit(f"unexpected GPU: {devices[-1]}")
payload = {
    "status": "pass",
    "python": os.sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "devices": devices,
    "packages": {
        **{name: md.version(name) for name in required},
        "torch": md.version("torch"),
        "torchvision": md.version("torchvision"),
        "torchaudio": md.version("torchaudio"),
    },
}
output = Path(os.environ["ROUTE_ROOT"]) / "runtime" / "h20_environment_audit.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("S2_H20_ENVIRONMENT_PASS", output)
PY

(
    cd "${S2_REPOSITORY}"
    PYTHONPATH="${S2_REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}" \
        PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m unittest discover \
        -s tests -p 'test_*.py'
)

echo "S2_H20_RUNTIME_READY env=${CONDA_ENV_PATH}"
