#!/usr/bin/env bash

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
ENV_DIR="${ENV_DIR:-/root/brats2026/envs/s2_brats_eval_numpy126_r2}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/brats2026/runs/s2_e_continue_selection_20260727_r1}"
AUDIT_PATH="${AUDIT_ROOT}/EVAL_ENVIRONMENT_AUDIT.json"
READY_MARKER="${AUDIT_ROOT}/EVAL_ENVIRONMENT_READY.ok"

test -x "${CONDA_BIN}"
test -d "${AUDIT_ROOT}"

if [[ -e "${ENV_DIR}" ]]; then
    echo "Refusing to overwrite evaluation environment: ${ENV_DIR}" >&2
    exit 2
fi
if [[ -e "${AUDIT_PATH}" || -e "${READY_MARKER}" ]]; then
    echo "Refusing to overwrite evaluation environment audit" >&2
    exit 3
fi
if pgrep -af '[c]onda create' >/dev/null; then
    echo "Another conda create process is still active" >&2
    pgrep -af '[c]onda create' >&2
    exit 4
fi

mkdir -p "$(dirname "${ENV_DIR}")"

# Serial downloads avoid the parallel-fetch stall preserved in attempt 1.
export CONDA_FETCH_THREADS=1
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30
export CONDA_REMOTE_READ_TIMEOUT_SECS=300
export CONDA_REMOTE_MAX_RETRIES=10

"${CONDA_BIN}" create --yes --prefix "${ENV_DIR}" python=3.10 pip

PYTHON="${ENV_DIR}/bin/python"
EVALUATE="${ENV_DIR}/bin/brats-evaluate"
PARSE="${ENV_DIR}/bin/brats-parse-metrics"

"${PYTHON}" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --retries 10 \
    --timeout 300 \
    "numpy==1.26.4" \
    "BraTS-evaluation==0.0.8" \
    "panoptica==2.1.0"
"${PYTHON}" -m pip check

test -x "${EVALUATE}"
test -x "${PARSE}"
"${EVALUATE}" --help >/dev/null
"${PARSE}" --help >/dev/null

export ENV_DIR AUDIT_PATH READY_MARKER EVALUATE PARSE
"${PYTHON}" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


expected = {
    "numpy": "1.26.4",
    "BraTS-evaluation": "0.0.8",
    "panoptica": "2.1.0",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Evaluation package versions drifted: {actual}")
if platform.python_version_tuple()[:2] != ("3", "10"):
    raise SystemExit(f"Expected Python 3.10, got {platform.python_version()}")

executables = {
    "python": Path(os.environ["ENV_DIR"]) / "bin" / "python3.10",
    "brats-evaluate": Path(os.environ["EVALUATE"]),
    "brats-parse-metrics": Path(os.environ["PARSE"]),
}
payload = {
    "schema_version": 1,
    "status": "pass",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": str(Path(os.environ["ENV_DIR"]).resolve()),
    "python": platform.python_version(),
    "packages": actual,
    "executables_sha256": {name: sha256(path.resolve()) for name, path in executables.items()},
}
identity = hashlib.sha256(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
payload["audit_sha256"] = identity

audit_path = Path(os.environ["AUDIT_PATH"])
audit_path.write_text(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
Path(os.environ["READY_MARKER"]).write_text(
    f"status=pass\naudit_sha256={identity}\nenvironment={payload['environment']}\n",
    encoding="utf-8",
)
print(f"S2_EVAL_ENVIRONMENT_PASS audit_sha256={identity}")
PY
