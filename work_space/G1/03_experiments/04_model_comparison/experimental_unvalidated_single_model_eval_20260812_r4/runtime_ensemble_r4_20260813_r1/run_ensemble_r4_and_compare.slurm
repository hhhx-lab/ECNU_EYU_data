#!/bin/bash
#SBATCH -J g1v3-ensemble-r4
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/logs/g1v3_ensemble_r4_%j.out
#SBATCH -e /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/logs/g1v3_ensemble_r4_%j.err

set -euo pipefail

ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4
CODE=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3
RUNTIME="${ROOT}/runtime_ensemble_r4_20260813_r1"
TARGET="${ROOT}/ensemble_r4"
COMPARE="${ROOT}/paired_comparison_r4"
CASE_LIST="${ROOT}/val_fixed_103_case_list.csv"
LOCKED_MANIFEST="${ROOT}/encdec_only/evaluation_run.json"
PLAN="${ROOT}/ENSEMBLE_R4_EXECUTION_PLAN.json"
FAILED="${ROOT}/ENSEMBLE_R4_FAILED.json"

fail_report() {
  rc=$?
  export rc FAILED SLURM_JOB_ID
  /public/home/zqchen/.conda/envs/segmamba/bin/python - <<'PY' || true
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["FAILED"])
payload = {
    "status": "experimental_unvalidated_failed",
    "operator_approved": False,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "exit_code": int(os.environ.get("rc", "1")),
    "failed_at": datetime.now(timezone.utc).isoformat(),
}
try:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
except FileExistsError:
    pass
PY
  exit "$rc"
}
trap fail_report ERR

test ! -e "${TARGET}" || { echo "ERROR: ensemble target already exists: ${TARGET}"; exit 1; }
test ! -e "${COMPARE}" || { echo "ERROR: comparison target already exists: ${COMPARE}"; exit 1; }
test ! -e "${FAILED}" || { echo "ERROR: prior failure evidence exists: ${FAILED}"; exit 1; }
test -f "${PLAN}"
test -f "${RUNTIME}/compare_three_way_paired.py"
test -f "${CASE_LIST}"
test -f "${LOCKED_MANIFEST}"

export PLAN ROOT CODE RUNTIME TARGET COMPARE CASE_LIST LOCKED_MANIFEST
/public/home/zqchen/.conda/envs/segmamba/bin/python - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


plan_path = Path(os.environ["PLAN"]).resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
assert plan["status"] == "experimental_unvalidated"
assert plan["operator_approved"] is False
assert plan["training_required"] is False
assert plan["case_count"] == 103
assert plan["synthesis_type"] == "ensamble"
assert plan["spatial_preprocessing"] == "foreground_centered_isotropic_resample_v1"
assert plan["bootstrap"] == {"replicates": 20000, "seed": 20260813}
assert plan["comparison_scope"] == "missing_t2w_image_quality_only"
assert plan["downstream_nnunet_segmentation_evaluated"] is False

expected_paths = {
    "root": os.environ["ROOT"],
    "ensemble_output": os.environ["TARGET"],
    "comparison_output": os.environ["COMPARE"],
    "case_list": os.environ["CASE_LIST"],
    "locked_manifest": os.environ["LOCKED_MANIFEST"],
}
for key, expected in expected_paths.items():
    assert Path(plan["paths"][key]).resolve() == Path(expected).resolve(), key

artifacts = {
    "case_list": Path(os.environ["CASE_LIST"]),
    "locked_manifest": Path(os.environ["LOCKED_MANIFEST"]),
    "evaluate_slurm": Path(os.environ["CODE"]) / "slurm/05_evaluate_val.slurm",
    "evaluate_py": Path(os.environ["CODE"]) / "evaluate.py",
    "vae": Path(plan["checkpoints"]["vae"]["path"]),
    "encdec": Path(plan["checkpoints"]["encdec"]["path"]),
    "bbdm": Path(plan["checkpoints"]["bbdm"]["path"]),
    "wrapper": Path(os.environ["RUNTIME"]) / "run_ensemble_r4_and_compare.slurm",
    "comparison": Path(os.environ["RUNTIME"]) / "compare_three_way_paired.py",
}
for name, path in artifacts.items():
    assert path.is_file(), f"missing artifact: {name}: {path}"
    observed = sha256_file(path)
    expected = plan["artifact_sha256"][name]
    assert observed == expected, f"SHA mismatch: {name}: {observed} != {expected}"

with Path(os.environ["CASE_LIST"]).open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 103
column = next(iter(rows[0]))
case_ids = [str(row[column]).strip() for row in rows]
assert len(case_ids) == len(set(case_ids)) == 103
print(json.dumps({
    "status": "experimental_unvalidated_preflight_pass",
    "plan_sha256": sha256_file(plan_path),
    "case_count": len(case_ids),
}, sort_keys=True))
PY

export G1_V3_CODE_DIR="${CODE}"
export G1_V3_CONDA_ENV=segmamba
export SYNTHESIS_TYPE=ensamble
export EVAL_SPLIT=val
export EVAL_EXPECTED_CASES=103
export EVAL_RUN_ID=ensemble_r4
export EVAL_RUN_DIR="${TARGET}"
export EVAL_CASE_LIST="${CASE_LIST}"
export UPDATE_LATEST_RUN=0
export PUBLISH_EVAL_RESULT=0
export G1_LOCKED_EVALUATION_MANIFEST="${LOCKED_MANIFEST}"

bash "${CODE}/slurm/05_evaluate_val.slurm"

printf '%s\n' experimental_unvalidated > "${TARGET}/EXPERIMENTAL_UNVALIDATED"
cp -n "${PLAN}" "${TARGET}/ENSEMBLE_R4_EXECUTION_PLAN.json"
cp -n "${RUNTIME}/compare_three_way_paired.py" "${TARGET}/compare_three_way_paired.py"

/public/home/zqchen/.conda/envs/segmamba/bin/python "${RUNTIME}/compare_three_way_paired.py" \
  --root "${ROOT}" \
  --case-list "${CASE_LIST}" \
  --output-dir "${COMPARE}" \
  --bootstrap 20000 \
  --seed 20260813 \
  --vae-sha256 004dbaa00669685b1f24fbaf6044d5f592f4a7ae6156a482be34235724175294 \
  --encdec-sha256 aed9bc96d6cb285b60d092fd48f572c60f76dba2d654dba4e60754ad55c0c558 \
  --bbdm-sha256 969eef82241a9500d843bb38b4fcf3230eb0b55ecfb1876aa83c65b59ec05541

printf '%s\n' experimental_unvalidated > "${TARGET}/ENSEMBLE_R4_COMPLETE.ok"
chmod -R a-w "${TARGET}" "${COMPARE}"
trap - ERR
echo "G1_ENSEMBLE_R4_AND_PAIRED_COMPARISON_COMPLETE"
