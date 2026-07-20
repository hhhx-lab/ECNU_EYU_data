import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_materialize_nnunet_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("g2_materialize_nnunet_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_case(root: Path, case_id: str, value: float) -> dict[str, str]:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    affine = np.eye(4)
    row = {"source_case_id": case_id}
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        path = case_dir / f"{case_id}-{modality}.nii.gz"
        array = np.indices((8, 8, 8)).sum(axis=0).astype(np.float32) + value
        nib.save(nib.Nifti1Image(array, affine), str(path))
        row[f"{modality}_source_path"] = str(path)
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    seg = np.zeros((8, 8, 8), dtype=np.int16)
    seg[2:6, 2:6, 2:6] = 3
    nib.save(nib.Nifti1Image(seg, affine), str(seg_path))
    row["seg_source_path"] = str(seg_path)
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class G2MaterializerTest(unittest.TestCase):
    def test_real_data_root_keeps_corrected_seg_authoritative(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ordinary = make_case(root, "BraTS-MET-00001-000", 1)
            corrected = make_case(root, "BraTS-MET-01184-002", 2)
            corrected_seg = root / "corrected-seg.nii.gz"
            corrected_seg.write_bytes(b"corrected-placeholder")
            rows = [
                {
                    "source_case_id": ordinary["source_case_id"],
                    "label_source": "raw",
                },
                {
                    "source_case_id": corrected["source_case_id"],
                    "label_source": "corrected",
                    "seg_source_path": str(corrected_seg),
                },
            ]

            updated = mod.apply_real_data_root(rows, root)

            self.assertEqual(updated, 2)
            self.assertEqual(
                Path(rows[0]["seg_source_path"]).resolve(),
                Path(ordinary["seg_source_path"]).resolve(),
            )
            self.assertEqual(rows[1]["seg_source_path"], str(corrected_seg))
            self.assertEqual(
                Path(rows[1]["t1n_source_path"]).resolve(),
                Path(corrected["t1n_source_path"]).resolve(),
            )

    def test_completion_root_overrides_machine_specific_manifest_path(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "BraTS-MET-00002-000"
            completion_root = root / "run_1"
            case_dir = completion_root / case_id
            case_dir.mkdir(parents=True)
            expected = case_dir / f"{case_id}-t2w.nii.gz"
            expected.write_bytes(b"nifti-placeholder")
            rows = [{
                "source_case_id": case_id,
                "source_completion_mode": "True",
                "label_kind": "completion",
                "raw_t2w_path": "/another/machine/completion-t2w.nii.gz",
            }]

            updated = mod.apply_completion_root(rows, completion_root)

            self.assertEqual(updated, 1)
            self.assertEqual(rows[0]["t2w_source_path"], str(expected.resolve()))

    def test_completion_replaces_only_t2w_and_synthetic_enters_train(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_a = make_case(root / "real", "BraTS-MET-00001-000", 1)
            real_b = make_case(root / "real", "BraTS-MET-00554-000", 2)
            real_a["nnunet_case_id"] = "BraTSMET_000001"
            real_b["nnunet_case_id"] = "BraTSMET_000002"
            real_b["t2w_status"] = "fake_or_broken"

            completion_t2w = root / "completion-t2w.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32) * 99, np.eye(4)),
                str(completion_t2w),
            )
            completion = {
                "synthetic_raw_id": "BraTS-MET-00554-000",
                "source_case_id": "BraTS-MET-00554-000",
                "source_completion_mode": "True",
                "label_kind": "completion",
                "accepted_for_training": "True",
                "accepted_for_evaluation": "False",
                "raw_t2w_path": str(completion_t2w),
            }
            synthetic_files = make_case(root / "synthetic", "SYN-MET-ABC", 3)
            augmentation = {
                "synthetic_raw_id": "raw-augmentation",
                "synthetic_final_id": "SYN-MET-ABC",
                "nnunet_case_id": "SYNMET_ABC",
                "source_case_id": "BraTS-MET-00001-000",
                "source_split": "train",
                "label_kind": "v2aug",
                "accepted_for_training": "True",
                "accepted_for_evaluation": "False",
                **{f"raw_{mod}_path": synthetic_files[f"{mod}_source_path"] for mod in ("t1n", "t1c", "t2w", "t2f")},
                "raw_seg_path": synthetic_files["seg_source_path"],
            }

            specs, stats = mod.build_case_specs(
                [real_a, real_b],
                [completion, augmentation],
                {"BraTS-MET-00554-000"},
                profile="real-synth",
                allow_incomplete_completion=False,
            )
            self.assertEqual(stats["completion_replacements"], 1)
            self.assertEqual(stats["synthetic_augmentation_cases"], 1)
            completion_spec = next(spec for spec in specs if spec["nnunet_case_id"] == "BraTSMET_000002")
            self.assertEqual(completion_spec["paths"]["t2w"], str(completion_t2w))
            self.assertEqual(completion_spec["paths"]["t1n"], real_b["t1n_source_path"])

            split = mod.build_output_split(
                {
                    "name": "master",
                    "train": ["BraTSMET_000001", "BraTSMET_000002"],
                    "val": [],
                    "test": [],
                },
                specs,
            )
            self.assertIn("SYNMET_ABC", split["train"])
            self.assertNotIn("SYNMET_ABC", split["val"])
            self.assertNotIn("SYNMET_ABC", split["test"])

    def test_materializes_both_views_and_runs_integrity(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_case(root / "source", "BraTS-MET-00001-000", 1)
            row["nnunet_case_id"] = "BraTSMET_000001"
            specs, _ = mod.build_case_specs(
                [row], [], set(), profile="real-only", allow_incomplete_completion=False
            )
            mod.assign_spec_splits(
                specs,
                {"train": ["BraTSMET_000001"], "val": [], "test": []},
            )
            dataset_dir = root / "nnunet"
            case_root = root / "cases"
            (dataset_dir / "imagesTr").mkdir(parents=True)
            (dataset_dir / "labelsTr").mkdir()
            (dataset_dir / "imagesTs").mkdir()
            (dataset_dir / "labelsTs").mkdir()
            case_root.mkdir()
            records = mod.materialize_specs(specs, dataset_dir, case_root, "symlink")
            self.assertEqual(len(records), 5)
            self.assertTrue((dataset_dir / "imagesTr" / "BraTSMET_000001_0000.nii.gz").is_file())
            self.assertTrue((case_root / "train" / "BraTS-MET-00001-000" / "BraTS-MET-00001-000-t1n.nii.gz").is_file())
            report = mod.verify_materialized_dataset(dataset_dir, specs, "symlink")
            self.assertTrue(report["passed"], report["errors"])

    def test_integrity_treats_signed_zero_affines_as_equal(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_case(root / "source", "BraTS-MET-00001-000", 1)
            row["nnunet_case_id"] = "BraTSMET_000001"

            negative_zero_affine = np.eye(4)
            negative_zero_affine[0, 3] = -0.0
            for modality in ("t1n", "t1c", "t2w", "t2f"):
                path = Path(row[f"{modality}_source_path"])
                image = nib.load(str(path))
                nib.save(
                    nib.Nifti1Image(np.asanyarray(image.dataobj), negative_zero_affine),
                    str(path),
                )

            specs, _ = mod.build_case_specs(
                [row], [], set(), profile="real-only", allow_incomplete_completion=False
            )
            mod.assign_spec_splits(
                specs,
                {"train": ["BraTSMET_000001"], "val": [], "test": []},
            )
            dataset_dir = root / "nnunet"
            case_root = root / "cases"
            (dataset_dir / "imagesTr").mkdir(parents=True)
            (dataset_dir / "labelsTr").mkdir()
            (dataset_dir / "imagesTs").mkdir()
            (dataset_dir / "labelsTs").mkdir()
            case_root.mkdir()
            mod.materialize_specs(specs, dataset_dir, case_root, "symlink")

            report = mod.verify_materialized_dataset(dataset_dir, specs, "symlink")

            self.assertTrue(report["passed"], report["errors"])

            t2f_path = Path(row["t2f_source_path"])
            t2f_image = nib.load(str(t2f_path))
            mismatched_affine = negative_zero_affine.copy()
            mismatched_affine[0, 3] = 0.01
            nib.save(
                nib.Nifti1Image(np.asanyarray(t2f_image.dataobj), mismatched_affine),
                str(t2f_path),
            )
            mismatched_report = mod.verify_materialized_dataset(
                dataset_dir, specs, "symlink"
            )
            self.assertIn(
                "BraTSMET_000001:geometry_mismatch",
                mismatched_report["errors"],
            )

    def test_multiple_runs_may_reuse_raw_id_but_not_run_raw_pair(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            repeated = root / "repeated.csv"
            raw_id = "BraTS-MET-00001-000_v2aug_label_0"
            write_rows(first, [{"generation_run_id": "run-a", "synthetic_raw_id": raw_id}])
            write_rows(second, [{"generation_run_id": "run-b", "synthetic_raw_id": raw_id}])
            write_rows(repeated, [{"generation_run_id": "run-a", "synthetic_raw_id": raw_id}])
            self.assertEqual(len(mod.read_synthetic_manifests([first, second])), 2)
            with self.assertRaises(ValueError):
                mod.read_synthetic_manifests([first, repeated])

    def test_completion_approval_is_rechecked_against_master_split(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = make_case(root / "real", "BraTS-MET-00554-000", 2)
            real["nnunet_case_id"] = "BraTSMET_000002"
            real["t2w_status"] = "fake_or_broken"
            completion_t2w = root / "completion-t2w.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)),
                str(completion_t2w),
            )
            completion = {
                "synthetic_raw_id": "BraTS-MET-00554-000",
                "source_case_id": "BraTS-MET-00554-000",
                "source_completion_mode": "True",
                "label_kind": "completion",
                "accepted_for_training": "True",
                "accepted_for_evaluation": "False",
                "raw_t2w_path": str(completion_t2w),
            }
            specs, _ = mod.build_case_specs(
                [real],
                [completion],
                {"BraTS-MET-00554-000"},
                profile="completion",
                allow_incomplete_completion=False,
            )
            with self.assertRaises(ValueError):
                mod.assign_spec_splits(
                    specs,
                    {"train": [], "val": ["BraTSMET_000002"], "test": []},
                )

    def test_augmentation_source_split_is_rechecked_against_master(self):
        mod = load_module()
        specs = [
            {
                "nnunet_case_id": "BraTSMET_000001",
                "source_case_id": "BraTS-MET-00001-000",
                "row_type": "real",
            },
            {
                "nnunet_case_id": "SYNMET_ABC",
                "source_case_id": "BraTS-MET-00001-000",
                "row_type": "synthetic_augmentation",
            },
        ]
        with self.assertRaises(ValueError):
            mod.assign_spec_splits(
                specs,
                {"train": [], "val": ["BraTSMET_000001"], "test": []},
            )

    def test_locked_test_is_physically_separated(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_case(root / "source", "BraTS-MET-00002-000", 1)
            row["nnunet_case_id"] = "BraTSMET_000002"
            specs, _ = mod.build_case_specs(
                [row], [], set(), profile="real-only", allow_incomplete_completion=False
            )
            mod.assign_spec_splits(
                specs,
                {"train": [], "val": [], "test": ["BraTSMET_000002"]},
            )
            dataset_dir = root / "nnunet"
            case_root = root / "cases"
            for name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
                (dataset_dir / name).mkdir(parents=True, exist_ok=True)
            case_root.mkdir()
            mod.materialize_specs(specs, dataset_dir, case_root, "symlink")
            self.assertFalse((dataset_dir / "imagesTr" / "BraTSMET_000002_0000.nii.gz").exists())
            self.assertTrue((dataset_dir / "imagesTs" / "BraTSMET_000002_0000.nii.gz").is_file())
            self.assertTrue((dataset_dir / "labelsTs" / "BraTSMET_000002.nii.gz").is_file())
            self.assertTrue((case_root / "test" / "BraTS-MET-00002-000").is_dir())
            report = mod.verify_materialized_dataset(dataset_dir, specs, "symlink")
            self.assertTrue(report["passed"], report["errors"])

    def test_nonempty_output_requires_explicit_clean(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            (output / "old.txt").write_text("old")
            with self.assertRaises(FileExistsError):
                mod.prepare_output(output, clean=False)
            mod.prepare_output(output, clean=True)
            self.assertEqual(list(output.iterdir()), [])

    def test_manifest_only_preflights_source_files(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.nii.gz"
            with self.assertRaises(FileNotFoundError):
                mod.materialize_file(missing, root / "planned.nii.gz", "manifest-only")


if __name__ == "__main__":
    unittest.main()
