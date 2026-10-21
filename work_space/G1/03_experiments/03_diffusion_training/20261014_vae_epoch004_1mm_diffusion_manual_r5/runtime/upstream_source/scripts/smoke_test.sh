#!/usr/bin/env bash
# End-to-end smoke test on a few cases (a few minutes on one GPU):
#   preprocess -> CSV -> train t1c/t2w (+resume) -> checks -> evaluate (sweep, whole-brain) -> generate
# Run from TumourDiffusion_1mm/. Exits non-zero at the first failure; prints SMOKE TEST PASSED at the end.
#
#   RAW_DIRS="/data/medical/1 /data/medical/5" bash scripts/smoke_test.sh
#   RAW_DIRS="..." COMPILE=--use_compile bash scripts/smoke_test.sh    # also test torch.compile (Linux)
# All outputs are retained in unique directories; this script never deletes directories.
set -euo pipefail

RAW_DIRS=${RAW_DIRS:?set RAW_DIRS to the raw folder(s) containing the test cases}
CASES=${CASES:-"BraTS-MET-00001-000 BraTS-MET-00002-000 BraTS-MET-00004-000 BraTS-MET-00572-000"}
VAL=${VAL:-BraTS-MET-00002-000}
STEPS=${STEPS:-120}
BATCH=${BATCH:-4}
COMPILE=${COMPILE:-}
PY=${PY:-python}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
if [[ -n "${WORK:-}" ]]; then
  if [[ -e "$WORK" || -L "$WORK" ]]; then
    echo "Refusing existing WORK path: $WORK" >&2; exit 1
  fi
  mkdir -- "$WORK"
else
  WORK=$(mktemp -d "$ROOT/smoke.XXXXXXXX")
fi
WORK=$(cd -- "$WORK" && pwd -P)
mkdir -p -- Checkpoint
CKPT=$(mktemp -d "$ROOT/Checkpoint/zz_smoke.XXXXXXXX")
LOGDIR=$(basename -- "$CKPT")
trap 'echo "Smoke outputs retained: $WORK ; $CKPT"' EXIT
read -r -a RAW <<< "$RAW_DIRS"

step() { echo; echo "================ [$1] $2"; }
tr ' ' '\n' <<< "$CASES" > "$WORK/cases.txt"
echo "$VAL" > "$WORK/val.txt"

step 1/8 "preprocess ($CASES)"
$PY scripts/preprocess_dataset.py --raw_dirs "${RAW[@]}" --out_dir "$WORK/DataSet" \
    --case_list "$WORK/cases.txt" --num_workers 4
grep -q ",ok," "$WORK/DataSet/preprocess_report.csv"

step 2/8 "CSV (val: $VAL)"
$PY src/train/csv_creator.py --dataset BRATS_2024 --datadir "$WORK/DataSet" --logdir $LOGDIR \
    --crop_size 64 --merge_dist 16 --val_list "$WORK/val.txt" --debug False | tail -n 3

TRAIN=("$PY" src/train/tumour_main_diffusion.py --dataset BRATS_2024 --logdir "$LOGDIR" --batch_size "$BATCH"
       --generator_type Unet_NnU --crop_size 64 --small_lesion_weight 3.0 --noise_schedule edm
       --normalization nnunet --sigma_data 1.0 --planner_gpu_gb 48 --num_workers 2 --loader_workers 0)
COMPILE_ARGS=()
if [[ "$COMPILE" == "--use_compile" ]]; then COMPILE_ARGS=(--use_compile);
elif [[ -n "$COMPILE" ]]; then echo "COMPILE must be empty or --use_compile" >&2; exit 1; fi

step 3/8 "train t1c and t2w for $STEPS steps"
for mod in t1c t2w; do
  "${TRAIN[@]}" "${COMPILE_ARGS[@]}" --modality "$mod" --num_steps "$STEPS" > "$WORK/train_$mod.log" 2>&1 \
    || { tail -n 30 "$WORK/train_$mod.log"; exit 1; }
  grep -E "used for training|LAST SAVE" "$WORK/train_$mod.log"
done

step 4/8 "resume t1c from its first checkpoint"
first=$(ls "$CKPT/t1c/weights" | sed -E 's/diffusion_([0-9]+)\.pt/\1/' | sort -n | sed -n '1p')
# Resume uncompiled, also exercising compiled -> eager when COMPILE was enabled.
"${TRAIN[@]}" --modality t1c --num_steps $((STEPS + 20)) --resume_iter "$first" > "$WORK/resume.log" 2>&1 \
  || { tail -n 30 "$WORK/resume.log"; exit 1; }
grep -E "AMP scaler restored|Resuming from" "$WORK/resume.log"

step 5/8 "checks: crops contain their lesions, checkpoint contents, no temp files"
$PY - "$CKPT" <<'PYEOF'
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped
from src.utils.gaussian_noise_tumour import GaussianNoiseTumour, CropLesionROId
from src.utils.convert_to_multi_channel_based_on_brats_classes import \
    ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d as LT
from src.infer.diffusion_inference_utils import find_checkpoint

ckpt_dir = sys.argv[1]
df = pd.read_csv(os.path.join(ckpt_dir, os.path.basename(ckpt_dir) + ".csv"))
ints = ["center_x", "center_y", "center_z", "x_extreme_min", "x_extreme_max", "y_extreme_min", "y_extreme_max",
        "z_extreme_min", "z_extreme_max", "x_size", "y_size", "z_size", "n_voxels", "patient_n_crops"]
tf = Compose([LoadImaged(keys=["scan_t1c", "label"], image_only=False),
              EnsureChannelFirstd(keys=["scan_t1c", "label"]), EnsureTyped(keys=["scan_t1c", "label"]),
              CropLesionROId(keys=["scan_t1c", "label"], target_size=64),
              GaussianNoiseTumour(keys="scan_t1c", normalization="nnunet", target_size=64,
                                  label_transform=LT(keys="label").converter)])
worst = 1.0
for _, r in df.iterrows():
    d = {"scan_t1c": r["scan_t1c"], "label": r["label"], **{k: int(r[k]) for k in ints}}
    for _ in range(3):
        out = tf(d)
        n_in = int((np.asarray(out["label_crop_pad"]).max(axis=0) > 0).sum())
        worst = min(worst, n_in / d["n_voxels"])
        assert np.isfinite(np.asarray(out["scan_t1c_crop_pad"])).all()
print(f"{len(df)} CSV rows x 3 random offsets: minimum share of the lesion inside its crop = {worst:.0%}")
assert worst >= 0.999, "a crop does not contain its lesion"

for mod in ("t1c", "t2w"):
    wdir = os.path.join(ckpt_dir, mod, "weights")
    assert not glob.glob(os.path.join(wdir, "*.tmp")), "temporary checkpoint file left behind"
    c = torch.load(find_checkpoint(wdir), map_location="cpu")
    assert "scaler" in c and c["data_identity"]["csv_sha256"], "checkpoint lacks scaler / data_identity"
    assert c["planner_gpu_gb"] == 48 and c["normalization"] == "nnunet"
    print(f"{mod}: latest {os.path.basename(find_checkpoint(wdir))}, {c['data_identity']['n_train_rows']} train rows, ok")
PYEOF

SAMPLE="--dataset BRATS_2024 --generator_type Unet_NnU --crop_size 64 --sigma_data 1.0
        --noise_schedule edm --sampling_method edm_heun --sampling_steps 18"
EVAL="$SAMPLE --normalization nnunet"   # generation starts from a label only, so it takes no --normalization

step 6/8 "evaluate: checkpoint sweep (patch mode, val split)"
"$PY" scripts/eval_checkpoints.py --modality t1c --logdir "$LOGDIR" --every 50 --max_cases 3 \
    --out_dir "$WORK/eval_ckpts" 2>&1 | grep -E "evaluating [0-9]|step|best|^ +[0-9]+ " | grep -v Warning
test -f "$WORK/eval_ckpts/t1c/summary.csv"

step 7/8 "evaluate: whole-brain mode"
"$PY" src/infer/evaluate_generation.py --diffusion_ckpt_dir "$CKPT" --csv_path "$CKPT/$LOGDIR.csv" --modality t1c \
    --evaluation_mode whole_brain --max_cases 2 --output_dir "$WORK/eval_wb" $EVAL > "$WORK/eval_wb.log" 2>&1
grep -E "loaded:|ssim_lesion=" "$WORK/eval_wb.log" | sed -n '1,3p'
test -f "$WORK/eval_wb/metrics.json"

step 8/8 "generate from the val case's label (all trained modalities)"
$PY src/infer/generate_from_label.py --label_path "$WORK/DataSet/$VAL/$VAL-seg.nii.gz" \
    --diffusion_ckpt_dir "$CKPT" --output_dir "$WORK/generated" --modality all --merge_dist 16 $SAMPLE 2>&1 \
    | grep -E "loading:|SKIP|saved:"
$PY - "$WORK/DataSet/$VAL/$VAL-seg.nii.gz" "$WORK/generated" <<'PYEOF'
import glob, sys, nibabel as nib, numpy as np
lab = nib.load(sys.argv[1])
outs = sorted(glob.glob(sys.argv[2] + "/*.nii.gz"))
assert outs, "nothing generated"
for p in outs:
    g = nib.load(p)
    a = g.get_fdata()
    assert g.shape == lab.shape and np.allclose(g.affine, lab.affine) and np.isfinite(a).all()
print(f"{len(outs)} generated volume(s): same shape and geometry as the label, finite values")
PYEOF

echo
echo "SMOKE TEST PASSED"
echo "outputs kept in $WORK/ and $CKPT/"
