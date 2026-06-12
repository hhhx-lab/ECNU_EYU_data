from pathlib import Path
import shutil
import json

SRC = Path(
    "/root/autodl-tmp/brats2026/data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"
)

DST = Path(
    "/root/autodl-tmp/nnunet_raw/Dataset501_BraTS2026Tumor"
)

mods = {
    "t1c": "0000",
    "t1n": "0001",
    "t2f": "0002",
    "t2w": "0003"
}

cases = sorted([p for p in SRC.rglob("BraTS-MET-*") if p.is_dir()])

for case_dir in cases:

    cid = case_dir.name

    for mod, idx in mods.items():

        shutil.copy2(
            case_dir / f"{cid}-{mod}.nii.gz",
            DST / "imagesTr" / f"{cid}_{idx}.nii.gz"
        )

    shutil.copy2(
        case_dir / f"{cid}-seg.nii.gz",
        DST / "labelsTr" / f"{cid}.nii.gz"
    )

dataset_json = {

    "channel_names": {

        "0": "T1C",
        "1": "T1N",
        "2": "T2F",
        "3": "T2W"
    },

    "labels": {

        "background": 0,
        "NETC": 1,
        "SNFH": 2,
        "ET": 3,
        "RC": 4
    },

    "numTraining": len(cases),

    "file_ending": ".nii.gz"
}

json.dump(
    dataset_json,
    open(DST / "dataset.json", "w"),
    indent=4
)

print("conversion finished")
print(dataset_json)
