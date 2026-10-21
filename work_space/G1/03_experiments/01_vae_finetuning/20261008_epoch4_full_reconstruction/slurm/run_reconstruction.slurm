#!/bin/bash
#SBATCH -J g1vae4recon
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r1/logs/%x_%j.out
#SBATCH -e /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r1/logs/%x_%j.err

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r1
CODE=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3
CHECKPOINT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_extended_finetune_20260901_r3_attempt03/training/checkpoints/checkpoint_epoch_004.pt
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python

if [[ "${RECON_PHASE:-}" != smoke && "${RECON_PHASE:-}" != full ]]; then
  echo 'RECON_PHASE must be smoke or full' >&2
  exit 2
fi
if [[ "${RECON_PHASE}" == smoke ]]; then
  OUTPUT="$ROOT/smoke"
  CASES=(--case-id BraTS-MET-00001-000 --case-id BraTS-MET-01117-003)
else
  test -f "$ROOT/smoke/COMPLETE.ok"
  OUTPUT="$ROOT/dataset"
  CASES=()
fi

test -f "$ROOT/evidence/REMOTE_STATIC_PREFLIGHT.json"
"$PYTHON" -B "$ROOT/runtime/reconstruct_epoch004.py" \
  --code-root "$CODE" --checkpoint "$CHECKPOINT" \
  --source-csv "$CODE/data/data_csv.csv" --data-root "$CODE/data/input" \
  --output-root "$OUTPUT" "${CASES[@]}"

echo "RECONSTRUCTION_${RECON_PHASE^^}_COMPLETE"
