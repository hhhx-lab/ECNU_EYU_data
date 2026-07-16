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
#SBATCH --account=torch_pr_522_general
#SBATCH --mail-user=bf2260@nyu.edu
#
# =============================================================================
# Compatibility wrapper (historical path).
# Canonical implementation:
#   work_space/S1/slurm/01_s1_realonly.slurm
#   work_space/S1/slurm/run_s1_realonly.sh
#
# Prefer submitting the S1 path going forward. This wrapper remains so older
# docs / commands keep working.
# =============================================================================

set -euo pipefail

# Infer PROJ if not provided: .../work_space/G1/code/<repo>/slurm -> project root
if [[ -z "${PROJ:-}" ]]; then
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # slurm -> repo -> code -> G1 -> work_space -> PROJ  (5 levels up)
  PROJ="$(cd "${_HERE}/../../../../.." && pwd)"
fi
export PROJ

S1_ENTRY="${PROJ}/work_space/S1/slurm/run_s1_realonly.sh"
if [[ ! -f "${S1_ENTRY}" ]]; then
  echo "ERROR: canonical S1 runner missing: ${S1_ENTRY}" >&2
  echo "PROJ resolved to: ${PROJ}" >&2
  echo "Set PROJ explicitly: sbatch --export=ALL,PROJ=/path/to/ECNU_EYU_data ..." >&2
  exit 1
fi

mkdir -p "${PROJ}/logs" "${PROJ}/work_space/S1/logs"

echo "Compatibility entry: 05_s1_realonly_nyu.slurm"
echo "Delegating to: ${S1_ENTRY}"
echo "PROJ=${PROJ}"

# shellcheck source=/dev/null
source "${S1_ENTRY}"
