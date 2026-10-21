#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 RUN_ROOT SHARD EXPECTED_COUNT CONTRACT_ID" >&2
    exit 2
fi

RUN_ROOT=$1
SHARD=$2
EXPECTED_COUNT=$3
EXPECTED_CONTRACT_ID=$4
PROJECT_ROOT=/root/brats2026/ECNU_EYU_data
REPOSITORY="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
PYTHON_BIN=/root/brats2026/envs/s2_met_aug_h20/bin/python
EXPECTED_CHECKPOINT_SHA256=4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267
INPUT_ROOT="${RUN_ROOT}/input_${SHARD}"
OUTPUT_ROOT="${RUN_ROOT}/output_${SHARD}"
MODEL_ROOT="${RUN_ROOT}/model"
CONTRACT="${RUN_ROOT}/OFFICIAL_INFERENCE_CONTRACT.json"
PREPARE_MARKER="${RUN_ROOT}/PREPARE_COMPLETE.ok"
LAUNCH_MARKER="${RUN_ROOT}/${SHARD}_LAUNCHED.ok"
COMPLETE_MARKER="${RUN_ROOT}/${SHARD}_COMPLETE.ok"
FAILURE_MARKER="${RUN_ROOT}/${SHARD}_FAILED.ok"
OUTPUT_MANIFEST="${RUN_ROOT}/${SHARD}_output_manifest.tsv"

case "${SHARD}" in
    gpu0) EXPECTED_GPU=0 ;;
    gpu1) EXPECTED_GPU=1 ;;
    *) echo "SHARD must be gpu0 or gpu1" >&2; exit 2 ;;
esac

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "${EXPECTED_GPU}" ]]; then
    echo "${SHARD} must use physical GPU ${EXPECTED_GPU}" >&2
    exit 3
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing runtime Python: ${PYTHON_BIN}" >&2
    exit 3
fi
for path in "${CONTRACT}" "${PREPARE_MARKER}" "${MODEL_ROOT}/fold_0/checkpoint_final.pth"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing inference prerequisite: ${path}" >&2
        exit 3
    fi
done
if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "Missing shard input: ${INPUT_ROOT}" >&2
    exit 3
fi
for path in "${OUTPUT_ROOT}" "${LAUNCH_MARKER}" "${COMPLETE_MARKER}" "${FAILURE_MARKER}" "${OUTPUT_MANIFEST}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite shard artifact: ${path}" >&2
        exit 4
    fi
done

export RUN_ROOT SHARD EXPECTED_COUNT EXPECTED_CONTRACT_ID EXPECTED_CHECKPOINT_SHA256
export INPUT_ROOT OUTPUT_ROOT MODEL_ROOT CONTRACT
export BRATS_S2_REPO_DIR="${REPOSITORY}"
export PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export nnUNet_compile=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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

contract = json.loads(Path(os.environ["CONTRACT"]).read_text(encoding="utf-8"))
shard = os.environ["SHARD"]
expected_count = int(os.environ["EXPECTED_COUNT"])
if contract.get("status") != "prepared":
    raise SystemExit("Inference contract is not prepared")
if contract.get("contract_sha256") != os.environ["EXPECTED_CONTRACT_ID"]:
    raise SystemExit("Inference contract identity drifted")
if contract.get("model", {}).get("checkpoint_sha256") != os.environ["EXPECTED_CHECKPOINT_SHA256"]:
    raise SystemExit("Contract checkpoint binding drifted")
if sha256(Path(os.environ["MODEL_ROOT"]) / "fold_0/checkpoint_final.pth") != os.environ["EXPECTED_CHECKPOINT_SHA256"]:
    raise SystemExit("Deployment checkpoint SHA256 drifted")
shard_contract = contract.get("shards", {}).get(shard, {})
if shard_contract.get("case_count") != expected_count:
    raise SystemExit("Shard case count contract drifted")
input_root = Path(os.environ["INPUT_ROOT"])
names = [path.name for path in input_root.glob("*.nii.gz")]
if len(names) != expected_count * 4:
    raise SystemExit(f"Shard input count mismatch: {len(names)}/{expected_count * 4}")
case_ids = {name[:-12] for name in names}
if len(case_ids) != expected_count:
    raise SystemExit(f"Shard unique case count mismatch: {len(case_ids)}/{expected_count}")
for case_id in case_ids:
    expected = {f"{case_id}_{channel}.nii.gz" for channel in ("0000", "0001", "0002", "0003")}
    if not expected.issubset(names):
        raise SystemExit(f"Incomplete shard channels: {case_id}")
PY

trap 'printf "status=fail\nfailed_at_utc=%s\nshard=%s\n" "$(date -u +%FT%TZ)" "${SHARD}" > "${FAILURE_MARKER}"' ERR
printf 'status=launched\nlaunched_at_utc=%s\nshard=%s\nphysical_gpu=%s\npid=%s\nexpected_cases=%s\ncontract_identity=%s\n' \
    "$(date -u +%FT%TZ)" "${SHARD}" "${EXPECTED_GPU}" "$$" "${EXPECTED_COUNT}" "${EXPECTED_CONTRACT_ID}" \
    > "${LAUNCH_MARKER}"

cd "${REPOSITORY}"
"${PYTHON_BIN}" inference_frozen.py \
    --input "${INPUT_ROOT}" \
    --output "${OUTPUT_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --fold 0 \
    --preprocess-workers 4 \
    --export-workers 4

"${PYTHON_BIN}" - <<'PY'
import hashlib
import os
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

input_root = Path(os.environ["INPUT_ROOT"])
output_root = Path(os.environ["OUTPUT_ROOT"])
expected_count = int(os.environ["EXPECTED_COUNT"])
input_ids = {
    path.name[:-12]
    for path in input_root.glob("*_0000.nii.gz")
}
outputs = sorted(output_root.glob("*.nii.gz"))
output_ids = {path.name.removesuffix(".nii.gz") for path in outputs}
if len(outputs) != expected_count or output_ids != input_ids:
    raise SystemExit(
        f"Shard output coverage mismatch: outputs={len(outputs)} expected={expected_count} "
        f"missing={sorted(input_ids-output_ids)[:10]} extra={sorted(output_ids-input_ids)[:10]}"
    )
manifest = Path(os.environ["RUN_ROOT"]) / f"{os.environ['SHARD']}_output_manifest.tsv"
with manifest.open("w", encoding="utf-8") as handle:
    handle.write("filename\tbytes\tsha256\n")
    for path in outputs:
        handle.write(f"{path.name}\t{path.stat().st_size}\t{sha256(path)}\n")
PY

MANIFEST_SHA="$(${PYTHON_BIN} - "${OUTPUT_MANIFEST}" <<'PY'
import hashlib
import sys
from pathlib import Path
p = Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)"
printf 'status=pass\ncompleted_at_utc=%s\nshard=%s\nphysical_gpu=%s\npredictions=%s\ncheckpoint_sha256=%s\noutput_manifest_sha256=%s\n' \
    "$(date -u +%FT%TZ)" "${SHARD}" "${EXPECTED_GPU}" "${EXPECTED_COUNT}" \
    "${EXPECTED_CHECKPOINT_SHA256}" "${MANIFEST_SHA}" > "${COMPLETE_MARKER}"
echo "S2_OFFICIAL_INFERENCE_SHARD_PASS shard=${SHARD} cases=${EXPECTED_COUNT} manifest_sha256=${MANIFEST_SHA}"
