import argparse
import csv
import os
from pathlib import Path
import shutil
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("PROJ", REPO_ROOT.parents[3])).expanduser()
DEFAULT_TRAIN_ROOT = "data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"
DEFAULT_DATASET_NAME = "Dataset263_BraTS2026_MET_RealOnly_Current"


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_dataset_dir():
    if os.environ.get("NNUNET_DATASET_DIR"):
        return os.environ["NNUNET_DATASET_DIR"]
    if os.environ.get("nnUNet_raw"):
        return str(Path(os.environ["nnUNet_raw"]) / DEFAULT_DATASET_NAME)
    return f"data/nnunet_raw/{DEFAULT_DATASET_NAME}"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--src",
    default=os.environ.get("BRATS_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
)
parser.add_argument(
    "--dst",
    default=default_dataset_dir()
)
parser.add_argument(
    "--mapping-csv",
    default=os.environ.get("BRATS_NNUNET_MAPPING_CSV", "")
)
parser.add_argument(
    "--mode",
    choices=["copy", "symlink"],
    default=os.environ.get("BRATS_MATERIALIZE_MODE", "symlink")
)
parser.add_argument(
    "--clean",
    action="store_true"
)
args = parser.parse_args()

SRC = resolve_path(args.src)
DST = resolve_path(args.dst)

mods = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003"
}

if args.clean:
    shutil.rmtree(DST / "imagesTr", ignore_errors=True)
    shutil.rmtree(DST / "labelsTr", ignore_errors=True)

(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)


def materialize_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if args.mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def load_mapping_rows(mapping_csv):
    mapping_csv = resolve_project_path(mapping_csv)
    with mapping_csv.open(newline="") as f:
        return list(csv.DictReader(f))


if args.mapping_csv:
    rows = load_mapping_rows(args.mapping_csv)
    for row in rows:
        nnunet_id = row["nnunet_case_id"]
        source_paths = {
            "t1n": row["t1n_source_path"],
            "t1c": row["t1c_source_path"],
            "t2w": row["t2w_source_path"],
            "t2f": row["t2f_source_path"],
        }
        for mod, idx in mods.items():
            materialize_file(
                resolve_project_path(source_paths[mod]),
                DST / "imagesTr" / f"{nnunet_id}_{idx}.nii.gz"
            )
        materialize_file(
            resolve_project_path(row["seg_source_path"]),
            DST / "labelsTr" / f"{nnunet_id}.nii.gz"
        )
    num_training = len(rows)
else:
    cases = sorted([p for p in SRC.rglob("BraTS-MET-*") if p.is_dir()])
    for case_dir in cases:
        cid = case_dir.name
        for mod, idx in mods.items():
            materialize_file(
                case_dir / f"{cid}-{mod}.nii.gz",
                DST / "imagesTr" / f"{cid}_{idx}.nii.gz"
            )
        materialize_file(
            case_dir / f"{cid}-seg.nii.gz",
            DST / "labelsTr" / f"{cid}.nii.gz"
        )
    num_training = len(cases)

dataset_json = {

    "channel_names": {

        "0": "T1N",
        "1": "T1C",
        "2": "T2W",
        "3": "T2F"
    },

    "labels": {

        "background": 0,
        "NETC": 1,
        "SNFH": 2,
        "ET": 3,
        "RC": 4
    },

    "numTraining": num_training,

    "file_ending": ".nii.gz"
}

json.dump(
    dataset_json,
    open(DST / "dataset.json", "w"),
    indent=4
)

print("conversion finished")
print(dataset_json)
