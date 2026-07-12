import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
SCRIPT_PATH = CODE_DIR / "train_sam2unet_post.py"


def load_module():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("train_sam2unet_post", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_case(root: Path, split_name: str, case_id: str) -> None:
    case_dir = root / split_name / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    image = np.ones((6, 6, 6), dtype=np.float32)
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        nib.save(nib.Nifti1Image(image, affine), str(case_dir / f"{case_id}-{modality}.nii.gz"))
    seg = np.zeros((6, 6, 6), dtype=np.int16)
    seg[2:4, 2:4, 2:4] = 3
    nib.save(nib.Nifti1Image(seg, affine), str(case_dir / f"{case_id}-seg.nii.gz"))


class S4G2FixedSplitTest(unittest.TestCase):
    def test_loads_g2_split_and_keeps_test_locked(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed_root = root / "case_folders"
            make_case(fixed_root, "train", "BraTS-MET-00001-000")
            make_case(fixed_root, "val", "BraTS-MET-00002-000")
            make_case(fixed_root, "test", "BraTS-MET-00003-000")
            output = root / "output"
            output.mkdir()

            train_records, val_records = mod.load_g2_fixed_split_records(fixed_root, output)

            self.assertEqual([row["case"] for row in train_records], ["BraTS-MET-00001-000"])
            self.assertEqual([row["case"] for row in val_records], ["BraTS-MET-00002-000"])
            split = json.loads((output / "g2_fixed_split.json").read_text())
            self.assertEqual(split["test_cases"], ["BraTS-MET-00003-000"])

    def test_rejects_case_id_overlap(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed_root = root / "case_folders"
            make_case(fixed_root, "train", "BraTS-MET-00001-000")
            make_case(fixed_root, "val", "BraTS-MET-00001-000")
            make_case(fixed_root, "test", "BraTS-MET-00003-000")
            output = root / "output"
            output.mkdir()

            with self.assertRaisesRegex(ValueError, "overlapping case IDs"):
                mod.load_g2_fixed_split_records(fixed_root, output)


if __name__ == "__main__":
    unittest.main()
