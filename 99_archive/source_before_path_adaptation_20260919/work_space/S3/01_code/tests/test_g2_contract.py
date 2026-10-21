import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


S3_CODE = Path(__file__).resolve().parents[1] / "code"


def load_module(name: str, relative_path: str):
    sys.path.insert(0, str(S3_CODE))
    path = S3_CODE / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class S3G2ContractTest(unittest.TestCase):
    def test_fixed_split_uses_mapping_paths_and_locks_test(self):
        mod = load_module("s3_make_split", "make_split.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "mapping.csv"
            split = root / "split.json"
            rows = []
            for index, case_id in enumerate(("train", "val", "test"), start=1):
                row = {
                    "nnunet_case_id": f"BraTSMET_{index:06d}",
                    "source_case_id": f"BraTS-MET-0000{index}-000",
                    "seg_source_path": f"nested/{case_id}/seg.nii.gz",
                }
                for modality in ("t1n", "t1c", "t2w", "t2f"):
                    row[f"{modality}_source_path"] = f"nested/{case_id}/{modality}.nii.gz"
                rows.append(row)
            with mapping.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            split.write_text(json.dumps([{
                "train": ["BraTSMET_000001"],
                "val": ["BraTSMET_000002"],
                "test": ["BraTSMET_000003"],
            }]))
            train, val, test = mod.load_g2_split(split, mapping, root)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)
            self.assertEqual(test, ["BraTS-MET-00003-000"])
            self.assertEqual(train[0]["image"][0], str(root / "nested/train/t1n.nii.gz"))

    def test_loader_rejects_illegal_labels_instead_of_remapping(self):
        mod = load_module("s3_data_utils", "utils/data_utils.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "illegal-seg.nii.gz"
            label = np.zeros((4, 4, 4), dtype=np.int16)
            label[1, 1, 1] = 8
            nib.save(nib.Nifti1Image(label, np.eye(4)), str(path))
            with self.assertRaises(ValueError):
                mod.NibabelLoader(keys=["label"])({"label": str(path)})


if __name__ == "__main__":
    unittest.main()
