#!/usr/bin/env bash
#SBATCH --job-name=s2t7mf1v2
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/logs/table7_disjoint_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/logs/table7_disjoint_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100"
DATA="${PROJECT}/work_space/S2/data/ecnu_completion_emergency"
INPUT="${DATA}/nnUNet_raw/Dataset264_BraTS2026_MET_Completion/imagesTs"
REFERENCE="${DATA}/nnUNet_raw/Dataset264_BraTS2026_MET_Completion/labelsTs"
MODEL="${DATA}/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres"
CHECKPOINT="${MODEL}/fold_0/checkpoint_final.pth"
CHECKPOINT_SHA256=4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267
S2_REPOSITORY="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
TRAIN_PY=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
EVAL_PY="${EVAL_ENV}/bin/python"
FAILED="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_FAILED.json"
STARTED="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_STARTED.json"
INFERENCE_AUDIT="${ROOT}/evidence/INFERENCE_VALIDATION.json"
VALIDATION="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_VALIDATION.json"
COMPLETE="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_COMPLETE.ok"

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" SLURM_JOB_ID="${SLURM_JOB_ID:-}" EVAL_PY="$EVAL_PY" "$EVAL_PY" - <<'PY' || true
import json, os
from datetime import datetime, timezone
from pathlib import Path
payload = {
    "schema_version": 1,
    "status": "failed",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "exit_code": int(os.environ["RC"]),
    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
}
try:
    with Path(os.environ["FAILED"]).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
except FileExistsError:
    pass
PY
  exit "$rc"
}
trap fail_stage ERR INT TERM

cd "$ROOT"
test -f evidence/REMOTE_STATIC_PREFLIGHT.json
test ! -e "$FAILED"
test ! -e "$STARTED"
test ! -e "$INFERENCE_AUDIT"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e prediction
test ! -e official_reproduction
test ! -e disjoint_microf1
sha256sum -c STATIC_SHA256SUMS.txt
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/runtime:${S2_REPOSITORY}:${PYTHONPATH:-}"
export nnUNet_raw="${DATA}/nnUNet_raw"
export nnUNet_preprocessed="${DATA}/nnUNet_preprocessed"
export nnUNet_results="${DATA}/nnUNet_results"
export nnUNet_extTrainer="${ROOT}/runtime/custom_nnunet"
export nnUNet_compile=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

STARTED="$STARTED" CHECKPOINT="$CHECKPOINT" CHECKPOINT_SHA256="$CHECKPOINT_SHA256" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$TRAIN_PY" - <<'PY'
import hashlib, json, os, socket
from datetime import datetime, timezone
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one visible CUDA device is required")
device = torch.cuda.get_device_name(0)
if "V100" not in device:
    raise RuntimeError(f"V100 required, got {device}")
p = Path(os.environ["CHECKPOINT"])
digest = hashlib.sha256()
with p.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
h = digest.hexdigest()
if h != os.environ["CHECKPOINT_SHA256"]:
    raise RuntimeError("checkpoint SHA drift")
payload = {
    "schema_version": 1,
    "status": "pass",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": socket.gethostname(),
    "device_name": device,
    "gpu_count": torch.cuda.device_count(),
    "case_count": 104,
    "checkpoint_sha256": h,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
with Path(os.environ["STARTED"]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

"$TRAIN_PY" "${ROOT}/runtime/frozen_s2_inference.py" \
  --input "$INPUT" --output "${ROOT}/prediction" \
  --model-root "$MODEL" --audit-json "$INFERENCE_AUDIT" \
  --expected-count 104 --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --model-name paper_table7_completion_plus_focal \
  > "${ROOT}/logs/inference_${SLURM_JOB_ID}.log" 2>&1

mkdir "${ROOT}/official_reproduction"
{
  "$EVAL_PY" --version
  "$EVAL_PY" -c 'import importlib.metadata as m; print("BraTS-evaluation="+m.version("BraTS-evaluation")); print("panoptica="+m.version("panoptica")); print("numpy="+m.version("numpy")); print("scipy="+m.version("scipy")); print("nibabel="+m.version("nibabel"))'
  sha256sum "$EVAL_ENV/bin/brats-evaluate" "$EVAL_ENV/bin/brats-parse-metrics"
} > "${ROOT}/official_reproduction/evaluation_environment.txt"

"$EVAL_ENV/bin/brats-evaluate" \
  --config mets --ref_path "$REFERENCE" --pred_path "${ROOT}/prediction" \
  --summary_json "${ROOT}/official_reproduction/panoptica_evaluation_summary.json" \
  > "${ROOT}/logs/official_evaluate_${SLURM_JOB_ID}.log" 2>&1

for threshold in 27 50; do
  "$EVAL_ENV/bin/brats-parse-metrics" mets \
    --json_path "${ROOT}/official_reproduction/panoptica_evaluation_summary.json" \
    --vol_threshold "$threshold" --overlap_threshold 0.2 \
    --output_csv_path "${ROOT}/official_reproduction/leaderboard_metrics_threshold${threshold}.csv" \
    > "${ROOT}/logs/official_parse_t${threshold}_${SLURM_JOB_ID}.log" 2>&1
done

"$EVAL_PY" "${ROOT}/runtime/disjoint_microf1.py" \
  --prediction-root "${ROOT}/prediction" --reference-root "$REFERENCE" \
  --output-root "${ROOT}/disjoint_microf1" --expected-count 104 \
  --overlap-threshold 0.2 --thresholds 27,50 \
  > "${ROOT}/logs/disjoint_microf1_${SLURM_JOB_ID}.log" 2>&1

"$EVAL_PY" "${ROOT}/runtime/validate_run.py" \
  --root "$ROOT" --mode final --job-id "${SLURM_JOB_ID}" \
  > "${ROOT}/logs/final_validation_${SLURM_JOB_ID}.log" 2>&1

COMPLETE="$COMPLETE" VALIDATION="$VALIDATION" SLURM_JOB_ID="${SLURM_JOB_ID}" "$EVAL_PY" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
validation = Path(os.environ["VALIDATION"])
payload = {
    "status": "pass",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "job_id": os.environ["SLURM_JOB_ID"],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
}
with Path(os.environ["COMPLETE"]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

trap - ERR INT TERM
echo "TABLE7_DISJOINT_MICROF1_PASS job_id=${SLURM_JOB_ID}"
