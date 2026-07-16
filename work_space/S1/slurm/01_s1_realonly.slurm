#!/usr/bin/env bash
#SBATCH --job-name=s1_realonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=96:00:00
#SBATCH --output=logs/s1_realonly_%j.out
#SBATCH --error=logs/s1_realonly_%j.err
#SBATCH --mail-type=END,FAIL
#
# Optional site-specific lines (uncomment / override at submit time as needed):
#SBATCH --account=torch_pr_522_general
#SBATCH --mail-user=bf2260@nyu.edu
#
# =============================================================================
# S1 real-only baseline (canonical Slurm entrypoint)
#
# ALWAYS submit from the project root after creating logs/:
#
#   export PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data   # or your clone path
#   cd "${PROJECT_ROOT}"
#   mkdir -p logs
#
#   # Recommended: force a >=40GB GPU when the cluster supports it
#   sbatch --gres=gpu:a100:1 \
#     --export=ALL,PROJ="${PROJECT_ROOT}" \
#     work_space/S1/slurm/01_s1_realonly.slurm
#
# ECNU A100 example:
#   sbatch -p a100 --gres=gpu:NVIDIAA100-PCIE-40GB:1 \
#     --export=ALL,PROJ="${PROJECT_ROOT}",CONDA_SH=/path/to/conda.sh,CONDA_ENV=brats2026_s1 \
#     work_space/S1/slurm/01_s1_realonly.slurm
#
# Full manual:
#   work_space/S1/docs/S1_服务器运行手册.md
#   work_space/S1/brats2026_multitask_S1_v2/repository/docs/S1_服务器运行手册.md
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If PROJ was not exported, infer from .../work_space/S1/slurm
if [[ -z "${PROJ:-}" ]]; then
  PROJ="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PROJ

# Ensure log directory exists for this job's relative SBATCH paths when possible.
mkdir -p "${PROJ}/logs" "${PROJ}/work_space/S1/logs"

echo "Submitting body with PROJ=${PROJ}"
echo "Script dir: ${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_s1_realonly.sh"
