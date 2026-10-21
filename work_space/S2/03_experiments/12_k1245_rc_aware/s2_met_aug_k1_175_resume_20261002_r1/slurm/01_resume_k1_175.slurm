#!/bin/bash
# Restore the full latest checkpoint only after source TIMEOUT acceptance.
#SBATCH --job-name=s2k1res175r1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1/logs/%x_%j.err

set -euo pipefail
umask 077
module purge
module load apps/envs/miniconda3/25.5.1
RESUME_ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1
source "$RESUME_ROOT/runtime/research_environment.sh"
test -f "$RESUME_ROOT/evidence/REMOTE_FINAL_PREFLIGHT.json"
test -f "$RESUME_ROOT/evidence/LAUNCH_AUDIT.json"
test -f "$RESUME_ROOT/evidence/FINAL_SOURCE_ACCEPTANCE.json"
test ! -e "$RESUME_ROOT/evidence/RESUME_FAILED.json"
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B "$RESUME_ROOT/runtime/run_resume.py"
