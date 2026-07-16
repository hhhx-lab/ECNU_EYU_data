#!/usr/bin/env bash
# =============================================================================
# S1 real-only baseline — executable body (sourced by Slurm entrypoints)
#
# Do NOT sbatch this file directly. Use:
#   work_space/S1/slurm/01_s1_realonly.slurm
# or the compatibility wrapper:
#   work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
#
# Guarantees (fail-fast):
#   1) required paths / conda / python packages exist
#   2) GPU present and VRAM >= S1_MIN_GPU_GB (default 40)
#   3) G2 real-only mapping + fixed split exist (refresh if needed)
#   4) S1 view rebuilt (or validated) with complete symlinks + labels
#   5) audit PASS before training starts
#   6) training uses multitask_v1_full.yaml memory-safe defaults
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
# Prefer explicit PROJ; else walk up from this script: .../work_space/S1/slurm
if [[ -z "${PROJ:-}" ]]; then
  _S1_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJ="$(cd "${_S1_SLURM_DIR}/../../.." && pwd)"
fi
export PROJ

S1_ROOT="${PROJ}/work_space/S1"
S1_REPO="${S1_REPO:-${S1_ROOT}/brats2026_multitask_S1_v2/repository}"
G2_RESULTS="${G2_RESULTS:-${PROJ}/work_space/G2/results}"
G2_RAW_INTAKE_SCRIPT="${G2_RAW_INTAKE_SCRIPT:-${PROJ}/work_space/G2/code/g2_build_realonly_from_raw.py}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJ}/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training}"

LOG_DIR="${LOG_DIR:-${PROJ}/logs}"
S1_LOG_DIR="${S1_LOG_DIR:-${S1_ROOT}/logs}"

CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-1}"
S1_REBUILD_VIEW="${S1_REBUILD_VIEW:-1}"
S1_MIN_GPU_GB="${S1_MIN_GPU_GB:-40}"
S1_ALLOW_SMALL_GPU="${S1_ALLOW_SMALL_GPU:-0}"
S1_CONFIG="${S1_CONFIG:-configs/multitask_v1_full.yaml}"

export S1_REALONLY_VIEW="${S1_REALONLY_VIEW:-${S1_ROOT}/data/real_only_cases}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S1_REPO}/data/splits}"
export BRATS_NNUNET_MAPPING_CSV="${BRATS_NNUNET_MAPPING_CSV:-${G2_RESULTS}/manifests/nnunet_case_mapping_realonly.csv}"
export BRATS_SPLIT_JSON="${BRATS_SPLIT_JSON:-${G2_RESULTS}/splits/splits_final_train_val_test.json}"
export BRATS_TRAIN_ROOT="${BRATS_TRAIN_ROOT:-${S1_REALONLY_VIEW}}"
export S1_CHECKPOINT_DIR="${S1_CHECKPOINT_DIR:-${S1_ROOT}/results/realonly/checkpoints}"
export S1_TENSORBOARD_DIR="${S1_TENSORBOARD_DIR:-${S1_ROOT}/results/realonly/tensorboard}"
export S1_OUTPUT_DIR="${S1_OUTPUT_DIR:-${S1_ROOT}/results/realonly/predictions}"

# Optional resume path (empty = train from scratch)
export S1_RESUME="${S1_RESUME:-}"

mkdir -p \
  "${LOG_DIR}" \
  "${S1_LOG_DIR}" \
  "${BRATS_SPLIT_DIR}" \
  "${S1_CHECKPOINT_DIR}" \
  "${S1_TENSORBOARD_DIR}" \
  "${S1_OUTPUT_DIR}"

echo "============================================================"
echo " S1 real-only baseline"
echo "============================================================"
echo "Date            : $(date)"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : ${SLURMD_NODENAME:-$(hostname)}"
echo "User            : ${USER:-unknown}"
echo "PWD             : $(pwd)"
echo "PROJ            : ${PROJ}"
echo "S1_REPO         : ${S1_REPO}"
echo "RAW_DATA_DIR    : ${RAW_DATA_DIR}"
echo "S1_REALONLY_VIEW: ${S1_REALONLY_VIEW}"
echo "BRATS_TRAIN_ROOT: ${BRATS_TRAIN_ROOT}"
echo "BRATS_SPLIT_DIR : ${BRATS_SPLIT_DIR}"
echo "Mapping CSV     : ${BRATS_NNUNET_MAPPING_CSV}"
echo "Split JSON      : ${BRATS_SPLIT_JSON}"
echo "Checkpoints     : ${S1_CHECKPOINT_DIR}"
echo "TensorBoard     : ${S1_TENSORBOARD_DIR}"
echo "Config          : ${S1_CONFIG}"
echo "G2_REFRESH      : ${G2_REFRESH_REALONLY}"
echo "REBUILD_VIEW    : ${S1_REBUILD_VIEW}"
echo "MIN_GPU_GB      : ${S1_MIN_GPU_GB}"
echo "ALLOW_SMALL_GPU : ${S1_ALLOW_SMALL_GPU}"
echo "S1_RESUME       : ${S1_RESUME:-<none>}"
echo "CONDA_ENV       : ${CONDA_ENV}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0: static path checks (before conda)
# ---------------------------------------------------------------------------
echo "[phase 0] static path checks"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${PROJ}" ]] || fail "PROJ does not exist: ${PROJ}"
[[ -d "${S1_REPO}" ]] || fail "S1 repository missing: ${S1_REPO}"
[[ -f "${S1_REPO}/trainers/trainer_v1_final.py" ]] || fail "trainer missing: ${S1_REPO}/trainers/trainer_v1_final.py"
if [[ ! -f "${S1_REPO}/${S1_CONFIG}" && ! -f "${S1_CONFIG}" ]]; then
  fail "config not found: ${S1_CONFIG} (looked in ${S1_REPO} and cwd)"
fi
[[ -f "${S1_REPO}/scripts/08_build_full_multitask_labels.py" ]] || fail "missing label builder script"
[[ -f "${S1_REPO}/scripts/09_dataset_audit.py" ]] || fail "missing audit script"
[[ -d "${RAW_DATA_DIR}" ]] || fail "raw data directory missing: ${RAW_DATA_DIR}
  Put MICCAI-LH-BraTS2025-MET-Challenge-Training under work_space/G1/data/raw/
  or override RAW_DATA_DIR=/absolute/path"
[[ -f "${G2_RAW_INTAKE_SCRIPT}" ]] || fail "G2 raw intake missing: ${G2_RAW_INTAKE_SCRIPT}"
[[ -f "${CONDA_SH}" ]] || fail "conda activation script missing: ${CONDA_SH}
  Override with: sbatch --export=ALL,CONDA_SH=/path/to/conda.sh,..."

# Raw data smoke: at least one case folder
raw_case_count="$(find "${RAW_DATA_DIR}" -maxdepth 2 -type d -name 'BraTS-MET-*' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${raw_case_count}" -lt 1 ]]; then
  fail "no BraTS-MET-* case folders found under RAW_DATA_DIR=${RAW_DATA_DIR}"
fi
echo "raw BraTS-MET case folders (depth<=2): ${raw_case_count}"

# ---------------------------------------------------------------------------
# Phase 1: conda + dependency preflight
# ---------------------------------------------------------------------------
echo "[phase 1] conda activate + dependency preflight"
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "python not found after conda activate (${PYTHON_BIN})"
echo "python: $(command -v "${PYTHON_BIN}")"
echo "python version: $("${PYTHON_BIN}" -c 'import sys; print(sys.version)')"

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "monai",
    "nibabel",
    "yaml",
    "numpy",
    "scipy",
    "tqdm",
    "tensorboard",
    "skimage",
]
missing = []
for name in required:
    try:
        importlib.import_module(name if name != "yaml" else "yaml")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")

if missing:
    print("ERROR: missing python packages in the active env:", file=sys.stderr)
    for line in missing:
        print("  -", line, file=sys.stderr)
    print(
        "Fix: follow work_space/S1/brats2026_multitask_S1_v2/repository/docs/S1_experiment.txt",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
import monai

print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("monai:", monai.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA is required for S1 training", file=sys.stderr)
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / (1024 ** 3)
print("gpu:", torch.cuda.get_device_name(0))
print(f"gpu_memory_gb: {total_gb:.1f}")
print("bf16_supported:", torch.cuda.is_bf16_supported())

import os
min_gb = float(os.environ.get("S1_MIN_GPU_GB", "40"))
allow_small = os.environ.get("S1_ALLOW_SMALL_GPU", "0") == "1"
if total_gb + 1e-6 < min_gb:
    msg = (
        f"S1 requires >= {min_gb:.0f}GB GPU for default 96^3 training + full-volume SWI val; "
        f"got {total_gb:.1f}GB."
    )
    if allow_small:
        print("WARNING:", msg)
        print("WARNING: S1_ALLOW_SMALL_GPU=1 set; continuing. Strongly use patch 80^3.")
    else:
        print("ERROR:", msg, file=sys.stderr)
        print(
            "Request A100/H100, or set S1_ALLOW_SMALL_GPU=1 only after switching config to 80^3.",
            file=sys.stderr,
        )
        sys.exit(1)

print("dependency preflight: PASS")
PY

# Optional resume file check
if [[ -n "${S1_RESUME}" ]]; then
  [[ -f "${S1_RESUME}" ]] || fail "S1_RESUME file not found: ${S1_RESUME}"
  echo "resume checkpoint OK: ${S1_RESUME}"
fi

cd "${S1_REPO}"
echo "cwd for training: $(pwd)"

# ---------------------------------------------------------------------------
# Phase 2: G2 real-only mapping + fixed split
# ---------------------------------------------------------------------------
echo "[phase 2] G2 real-only mapping / split"

if [[ "${G2_REFRESH_REALONLY}" == "1" || ! -f "${BRATS_NNUNET_MAPPING_CSV}" || ! -f "${BRATS_SPLIT_JSON}" ]]; then
  echo "Running G2 raw intake (g2_build_realonly_from_raw.py) ..."
  "${PYTHON_BIN}" "${G2_RAW_INTAKE_SCRIPT}" \
    --project-root "${PROJ}" \
    --data-root "${RAW_DATA_DIR}" \
    --results-root "${G2_RESULTS}" \
    --fail-if-no-valid-cases
else
  echo "Skip G2 refresh (G2_REFRESH_REALONLY=0 and files exist)"
fi

[[ -f "${BRATS_NNUNET_MAPPING_CSV}" ]] || fail "missing mapping after intake: ${BRATS_NNUNET_MAPPING_CSV}"
[[ -f "${BRATS_SPLIT_JSON}" ]] || fail "missing split after intake: ${BRATS_SPLIT_JSON}"
echo "mapping CSV: ${BRATS_NNUNET_MAPPING_CSV}"
echo "split JSON : ${BRATS_SPLIT_JSON}"

# ---------------------------------------------------------------------------
# Phase 3: build or validate S1 training view
# ---------------------------------------------------------------------------
echo "[phase 3] S1 training view"

view_ready=0
if [[ "${S1_REBUILD_VIEW}" != "1" ]]; then
  if [[ -d "${S1_REALONLY_VIEW}" \
     && -f "${BRATS_SPLIT_DIR}/train_full.txt" \
     && -f "${BRATS_SPLIT_DIR}/val_full.txt" ]]; then
    echo "S1_REBUILD_VIEW=0 — validating existing view instead of rebuild"
    if "${PYTHON_BIN}" scripts/09_dataset_audit.py \
        --train-root "${BRATS_TRAIN_ROOT}" \
        --split-dir "${BRATS_SPLIT_DIR}"; then
      view_ready=1
      echo "existing view audit: PASS"
    else
      echo "existing view audit FAILED — will rebuild"
      view_ready=0
      S1_REBUILD_VIEW=1
    fi
  else
    echo "existing view incomplete — will rebuild"
    S1_REBUILD_VIEW=1
  fi
fi

if [[ "${S1_REBUILD_VIEW}" == "1" || "${view_ready}" != "1" ]]; then
  echo "Building S1 real-only view from G2 mapping + fixed split ..."
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import shutil
import sys
from pathlib import Path

proj = Path(os.environ["PROJ"])
view_root = Path(os.environ["S1_REALONLY_VIEW"])
mapping_csv = Path(os.environ["BRATS_NNUNET_MAPPING_CSV"])
split_json = Path(os.environ["BRATS_SPLIT_JSON"])
split_dir = Path(os.environ["BRATS_SPLIT_DIR"])

def resolve(path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (proj / path).resolve()

with mapping_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"mapping CSV is empty: {mapping_csv}")

required_cols = {
    "nnunet_case_id",
    "source_case_id",
    "t1n_source_path",
    "t1c_source_path",
    "t2w_source_path",
    "t2f_source_path",
    "seg_source_path",
    "label_source",
}
missing_cols = required_cols - set(rows[0].keys())
if missing_cols:
    raise SystemExit(f"mapping CSV missing columns: {sorted(missing_cols)}")

by_nnunet = {row["nnunet_case_id"]: row for row in rows}

with split_json.open() as f:
    data = json.load(f)
split = data[0] if isinstance(data, list) else data
for key in ("train", "val"):
    if key not in split:
        raise SystemExit(f"split JSON missing key '{key}': {split_json}")

train_ids = list(split["train"])
val_ids = list(split["val"])
test_ids = list(split.get("test", []))
if not train_ids:
    raise SystemExit("split train is empty")
if not val_ids:
    raise SystemExit("split val is empty")

selected_ids = train_ids + val_ids
missing_ids = [case_id for case_id in selected_ids if case_id not in by_nnunet]
if missing_ids:
    raise SystemExit(
        f"missing mapping for {len(missing_ids)} split ids; examples: {missing_ids[:10]}"
    )

# Fresh view — remove only the view directory, never raw data.
if view_root.exists():
    if view_root.resolve() == proj.resolve():
        raise SystemExit(f"refusing to delete PROJ as view_root: {view_root}")
    # Safety: view_root must live under work_space/S1
    try:
        view_root.resolve().relative_to((proj / "work_space" / "S1").resolve())
    except ValueError as exc:
        raise SystemExit(
            f"refusing to rebuild view outside work_space/S1: {view_root}"
        ) from exc
    shutil.rmtree(view_root)

view_root.mkdir(parents=True, exist_ok=True)
split_dir.mkdir(parents=True, exist_ok=True)

modalities = {
    "t1n": "t1n_source_path",
    "t1c": "t1c_source_path",
    "t2w": "t2w_source_path",
    "t2f": "t2f_source_path",
}

def force_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)

def link_case(row: dict) -> str:
    source_case = row["source_case_id"]
    case_dir = view_root / source_case
    case_dir.mkdir(parents=True, exist_ok=True)
    for mod, column in modalities.items():
        src = resolve(row[column])
        dst = case_dir / f"{source_case}-{mod}.nii.gz"
        force_symlink(src, dst)
    seg_src = resolve(row["seg_source_path"])
    seg_dst = case_dir / f"{source_case}-seg.nii.gz"
    force_symlink(seg_src, seg_dst)
    return source_case

train_cases = []
for case_id in train_ids:
    train_cases.append(link_case(by_nnunet[case_id]))
val_cases = []
for case_id in val_ids:
    val_cases.append(link_case(by_nnunet[case_id]))

# Detect accidental train/val leakage by case id
overlap = set(train_cases) & set(val_cases)
if overlap:
    raise SystemExit(f"train/val case id overlap: {sorted(list(overlap))[:10]}")

(split_dir / "train_full.txt").write_text("\n".join(train_cases) + "\n")
(split_dir / "val_full.txt").write_text("\n".join(val_cases) + "\n")
(split_dir / "test_internal_locked.txt").write_text(
    "\n".join(test_ids) + ("\n" if test_ids else "")
)

manifest_path = view_root / "s1_realonly_view_manifest.csv"
with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "nnunet_case_id",
            "source_case_id",
            "split",
            "case_dir",
            "label_source",
        ],
    )
    writer.writeheader()
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        for nnunet_id in ids:
            row = by_nnunet[nnunet_id]
            writer.writerow(
                {
                    "nnunet_case_id": nnunet_id,
                    "source_case_id": row["source_case_id"],
                    "split": split_name,
                    "case_dir": str(view_root / row["source_case_id"]),
                    "label_source": row.get("label_source", ""),
                }
            )

print("train:", len(train_cases))
print("val:", len(val_cases))
print("test locked (ids only):", len(test_ids))
print("view:", view_root)
print("manifest:", manifest_path)

# Soft expectation for current BraTS-MET real-only fixed split.
# Do not hard-fail if G2 counts change slightly after QC updates.
if len(train_cases) < 100 or len(val_cases) < 10:
    raise SystemExit(
        f"suspiciously small split: train={len(train_cases)} val={len(val_cases)}"
    )
print("view build: PASS")
PY

  echo "[phase 3b] build tumor/RC labels"
  "${PYTHON_BIN}" scripts/08_build_full_multitask_labels.py --train-root "${BRATS_TRAIN_ROOT}"
fi

echo "[phase 3c] audit S1 view (must PASS)"
"${PYTHON_BIN}" scripts/09_dataset_audit.py \
  --train-root "${BRATS_TRAIN_ROOT}" \
  --split-dir "${BRATS_SPLIT_DIR}"

# Extra split non-empty guards
train_n="$(grep -cve '^\s*$' "${BRATS_SPLIT_DIR}/train_full.txt" || true)"
val_n="$(grep -cve '^\s*$' "${BRATS_SPLIT_DIR}/val_full.txt" || true)"
echo "split counts: train=${train_n} val=${val_n}"
[[ "${train_n}" -ge 1 ]] || fail "train_full.txt is empty"
[[ "${val_n}" -ge 1 ]] || fail "val_full.txt is empty"

# ---------------------------------------------------------------------------
# Phase 4: train
# ---------------------------------------------------------------------------
echo "[phase 4] train S1 multitask model"
echo "config: ${S1_CONFIG}"
echo "env BRATS_TRAIN_ROOT=${BRATS_TRAIN_ROOT}"
echo "env BRATS_SPLIT_DIR=${BRATS_SPLIT_DIR}"
echo "env S1_CHECKPOINT_DIR=${S1_CHECKPOINT_DIR}"
echo "env S1_TENSORBOARD_DIR=${S1_TENSORBOARD_DIR}"
if [[ -n "${S1_RESUME}" ]]; then
  echo "env S1_RESUME=${S1_RESUME}"
fi

# shellcheck disable=SC2086
"${PYTHON_BIN}" trainers/trainer_v1_final.py --config "${S1_CONFIG}"

# ---------------------------------------------------------------------------
# Phase 5: post-train sanity
# ---------------------------------------------------------------------------
echo "[phase 5] post-train checks"
[[ -f "${S1_CHECKPOINT_DIR}/latest.pth" ]] || fail "missing latest.pth under ${S1_CHECKPOINT_DIR}"
if [[ ! -f "${S1_CHECKPOINT_DIR}/best.pth" ]]; then
  echo "WARNING: best.pth not found yet (possible if training stopped before first val improve)"
else
  echo "best.pth: ${S1_CHECKPOINT_DIR}/best.pth"
fi
ls -lh "${S1_CHECKPOINT_DIR}" || true

echo "============================================================"
echo " S1 real-only DONE: $(date)"
echo " Checkpoints : ${S1_CHECKPOINT_DIR}"
echo " TensorBoard : ${S1_TENSORBOARD_DIR}"
echo " View        : ${S1_REALONLY_VIEW}"
echo " Splits      : ${BRATS_SPLIT_DIR}"
echo "============================================================"
