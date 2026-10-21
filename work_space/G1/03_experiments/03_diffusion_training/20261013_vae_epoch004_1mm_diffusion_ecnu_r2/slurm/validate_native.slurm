#!/bin/bash
#SBATCH --job-name=g1d1mmp0r2
#SBATCH --partition=cpu_96G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261013_r2/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261013_r2/logs/%x_%j.err

set -euo pipefail
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261013_r2
source "$ROOT/config/p0.env"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "$G1_EXPERIMENT_ROOT" = "$ROOT"
test -f "$G1_RECONSTRUCTION_ROOT/dataset/COMPLETE.ok"
test -f "$G1_SOURCE_CODE/data/data_csv.csv"
test -f "$ROOT/runtime/validate_native.py"
test ! -e "$G1_P0_OUTPUT"
"$G1_PYTHON" -B - "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
package=json.loads((root/'plan/P0_PACKAGE_MANIFEST.json').read_text())
for relative,expected in package['files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected, 'runtime contract drift: '+relative
print('P0_RUNTIME_SHA_PASS')
PY
if [[ "${G1_STATIC_DRY_RUN:-0}" == 1 ]]; then
  printf 'P0_BASH_U_DRY_RUN_PASS\n'
  exit 0
fi
test -f "$ROOT/evidence/P0_STATIC_PREFLIGHT.json"
test -f "$ROOT/evidence/P0_LAUNCH_AUDIT.json"
"$G1_PYTHON" -B "$ROOT/runtime/validate_native.py" \
  --experiment-root "$ROOT" --reconstruction-root "$G1_RECONSTRUCTION_ROOT" \
  --source-code "$G1_SOURCE_CODE" --output "$G1_P0_OUTPUT"
test -f "$G1_P0_OUTPUT/COMPLETE.ok"
printf 'P0_NATIVE_COMPLETE\n'
