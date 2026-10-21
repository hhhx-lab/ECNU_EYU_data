import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "05_validate_fixed_split_cache.py"
SPEC = importlib.util.spec_from_file_location("validate_fixed_split_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FixedSplitCacheTests(unittest.TestCase):
    def create_fixture(self, root, cache_ids=("case_1", "case_2")):
        train_file = root / "train.txt"
        val_file = root / "val.txt"
        train_file.write_text("case_1\n", encoding="utf-8")
        val_file.write_text("case_2\n", encoding="utf-8")

        dataset_dir = root / "Dataset260"
        images_dir = dataset_dir / "imagesTr"
        labels_dir = dataset_dir / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir()
        for case_id in ("case_1", "case_2"):
            for channel in range(4):
                (images_dir / f"{case_id}_{channel:04d}.nii.gz").touch()
            (labels_dir / f"{case_id}.nii.gz").touch()
        (dataset_dir / "dataset.json").write_text(
            json.dumps({"numTraining": 2}), encoding="utf-8"
        )

        preprocessed_dir = root / "preprocessed" / "Dataset260" / "nnUNetPlans_3d_fullres"
        preprocessed_dir.mkdir(parents=True)
        for case_id in cache_ids:
            (preprocessed_dir / f"{case_id}.b2nd").touch()
            (preprocessed_dir / f"{case_id}_seg.b2nd").touch()
        (preprocessed_dir.parent / "dataset.json").write_text(
            json.dumps({"numTraining": 2}), encoding="utf-8"
        )
        return train_file, val_file, dataset_dir, preprocessed_dir

    def test_accepts_exactly_matching_b2nd_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.create_fixture(Path(temporary))
            result = MODULE.validate_fixed_split_cache(*args)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["train_count"], 1)
            self.assertEqual(result["validation_count"], 1)
            self.assertEqual(result["cache_format"], "b2nd")

    def test_rejects_missing_preprocessed_case_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.create_fixture(Path(temporary), cache_ids=("case_1",))
            with self.assertRaisesRegex(ValueError, "cache ID space differs"):
                MODULE.validate_fixed_split_cache(*args)

    def test_accepts_npz_cache_with_unpacked_npy_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_file, val_file, dataset_dir, preprocessed_dir = self.create_fixture(root)
            for path in preprocessed_dir.glob("*.b2nd"):
                path.unlink()
            for case_id in ("case_1", "case_2"):
                (preprocessed_dir / f"{case_id}.npz").touch()
                (preprocessed_dir / f"{case_id}.npy").touch()
                (preprocessed_dir / f"{case_id}.pkl").touch()
            result = MODULE.validate_fixed_split_cache(
                train_file, val_file, dataset_dir, preprocessed_dir
            )
            self.assertEqual(result["cache_format"], "npz")


if __name__ == "__main__":
    unittest.main()
