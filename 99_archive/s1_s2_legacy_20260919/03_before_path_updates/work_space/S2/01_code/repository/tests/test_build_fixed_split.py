import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "04_build_fixed_split.py"
SPEC = importlib.util.spec_from_file_location("build_fixed_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FixedSplitTests(unittest.TestCase):
    def write_inputs(self, root, missing_train=False, missing_val=False):
        split_path = root / "split.json"
        split_path.write_text(
            json.dumps(
                [{"train": ["case_1", "case_2"], "val": ["case_3"], "test": ["case_4"]}]
            ),
            encoding="utf-8",
        )
        mapping_path = root / "mapping.csv"
        ids = ["case_1", "case_2", "case_3", "case_4"]
        if missing_train:
            ids.remove("case_2")
        if missing_val:
            ids.remove("case_3")
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("nnunet_case_id", "source_case_id")
            )
            writer.writeheader()
            for case_id in ids:
                writer.writerow({"nnunet_case_id": case_id, "source_case_id": case_id})
        return split_path, mapping_path

    def create_existing_artifacts(self, root, bindings, validation_ids):
        dataset_dir = root / "Dataset260"
        images_dir = dataset_dir / "imagesTr"
        images_dir.mkdir(parents=True)
        for nnunet_id, source_id in bindings:
            source_dir = root / "raw" / source_id
            source_dir.mkdir(parents=True)
            source_image = source_dir / f"{source_id}-t1n.nii.gz"
            source_image.touch()
            (images_dir / f"{nnunet_id}_0000.nii.gz").symlink_to(source_image)

        validation_dir = root / "fold_0" / "validation"
        validation_dir.mkdir(parents=True)
        for nnunet_id in validation_ids:
            (validation_dir / f"{nnunet_id}.nii.gz").touch()
        return dataset_dir, validation_dir

    def test_writes_one_fixed_split_and_locked_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root, missing_train=True)
            output_dir = root / "splits"
            mapping_output = output_dir / "trainval.csv"
            summary = MODULE.build_fixed_split(
                split_path, mapping_path, output_dir, mapping_output
            )
            self.assertEqual((output_dir / "train_fixed.txt").read_text(), "case_1\n")
            self.assertEqual((output_dir / "val_fixed.txt").read_text(), "case_3\n")
            self.assertEqual(
                (output_dir / "test_internal_locked.txt").read_text(), "case_4\n"
            )
            self.assertEqual(
                (output_dir / "test_internal_locked_source_ids.txt").read_text(),
                "case_4\n",
            )
            self.assertEqual(summary["effective_counts"]["train"], 1)
            self.assertEqual(
                summary["missing_train_ids_excluded_by_mapping"], ["case_2"]
            )
            with (output_dir / "fixed_split_membership.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                membership = list(csv.DictReader(handle))
            self.assertEqual(
                {row["id_space"] for row in membership}, {"current_g2_mapping"}
            )
            self.assertFalse((output_dir / "train_fold1.txt").exists())

    def test_missing_fixed_validation_case_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root, missing_val=True)
            with self.assertRaisesRegex(ValueError, "validation or locked-test"):
                MODULE.build_fixed_split(
                    split_path,
                    mapping_path,
                    root / "splits",
                    root / "splits" / "trainval.csv",
                )

    def test_rejects_unexpected_fixed_split_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root)
            with self.assertRaisesRegex(ValueError, "count contract failed"):
                MODULE.build_fixed_split(
                    split_path,
                    mapping_path,
                    root / "splits",
                    root / "splits" / "trainval.csv",
                    expected_train_count=99,
                    expected_val_count=1,
                    expected_test_count=1,
                )

    def test_rejects_patient_group_leakage_when_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    [{
                        "train": ["nn_1"],
                        "val": ["nn_2"],
                        "test": ["nn_3"],
                    }]
                ),
                encoding="utf-8",
            )
            mapping_path = root / "mapping.csv"
            with mapping_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("nnunet_case_id", "source_case_id")
                )
                writer.writeheader()
                writer.writerows((
                    {"nnunet_case_id": "nn_1", "source_case_id": "BraTS-MET-00001-000"},
                    {"nnunet_case_id": "nn_2", "source_case_id": "BraTS-MET-00001-001"},
                    {"nnunet_case_id": "nn_3", "source_case_id": "BraTS-MET-00002-000"},
                ))
            with self.assertRaisesRegex(ValueError, "Patient groups cross"):
                MODULE.build_fixed_split(
                    split_path,
                    mapping_path,
                    root / "splits",
                    root / "splits" / "trainval.csv",
                    require_patient_group_disjoint=True,
                )

    def test_recovers_split_without_trusting_overwritten_split_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root)
            output_dir = root / "splits"
            output_dir.mkdir()
            (output_dir / "train_full.txt").write_text("contaminated_train\n")
            (output_dir / "val_full.txt").write_text("contaminated_val\n")
            dataset_dir, validation_dir = self.create_existing_artifacts(
                root,
                (("old_train", "case_1"), ("old_val", "case_3")),
                ("old_val",),
            )

            summary = MODULE.build_fixed_split(
                split_path,
                mapping_path,
                output_dir,
                output_dir / "trainval.csv",
                reuse_existing=True,
                existing_dataset_dir=dataset_dir,
                existing_validation_dir=validation_dir,
                baseline_excluded_source_ids={"case_2"},
            )
            self.assertEqual((output_dir / "train_fixed.txt").read_text(), "old_train\n")
            self.assertEqual((output_dir / "val_fixed.txt").read_text(), "old_val\n")
            self.assertEqual(
                summary["split_source"], "existing_dataset260_and_fold0_validation"
            )
            self.assertEqual(summary["missing_train_ids_excluded_by_mapping"], [])
            self.assertTrue(summary["validation_recovered_from_fold0_outputs"])
            self.assertTrue(summary["locked_test_recovered_as_source_complement"])
            self.assertEqual(summary["baseline_excluded_source_ids"], ["case_2"])
            self.assertTrue(summary["source_identity_disjoint"])
            with (output_dir / "fixed_split_membership.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                membership = list(csv.DictReader(handle))
            self.assertEqual(
                {row["id_space"] for row in membership if row["split"] != "test_internal_locked"},
                {"dataset260_historical"},
            )

    def test_reconstructs_shifted_ids_from_existing_dataset_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    [{"train": ["new_1"], "val": ["new_2"], "test": ["test_1"]}]
                ),
                encoding="utf-8",
            )
            mapping_path = root / "mapping.csv"
            with mapping_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("nnunet_case_id", "source_case_id")
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"nnunet_case_id": "new_1", "source_case_id": "source_case_1"},
                        {"nnunet_case_id": "new_2", "source_case_id": "source_case_2"},
                        {"nnunet_case_id": "test_1", "source_case_id": "source_test_1"},
                    )
                )

            dataset_dir, validation_dir = self.create_existing_artifacts(
                root,
                (("old_1", "source_case_1"), ("old_2", "source_case_2")),
                ("old_2",),
            )
            output_dir = root / "splits"
            mapping_output = output_dir / "trainval.csv"
            summary = MODULE.build_fixed_split(
                split_path,
                mapping_path,
                output_dir,
                mapping_output,
                reuse_existing=True,
                existing_dataset_dir=dataset_dir,
                existing_validation_dir=validation_dir,
            )
            with mapping_output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {"nnunet_case_id": "old_1", "source_case_id": "source_case_1"},
                    {"nnunet_case_id": "old_2", "source_case_id": "source_case_2"},
                ],
            )
            self.assertTrue(summary["mapping_reconstructed_from_dataset_symlinks"])
            self.assertEqual(summary["source_mapping_sha256"], MODULE.sha256(mapping_path))
            self.assertEqual(
                summary["trainval_mapping_sha256"], MODULE.sha256(mapping_output)
            )

    def test_rejects_validation_id_absent_from_dataset260(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root)
            dataset_dir, validation_dir = self.create_existing_artifacts(
                root,
                (("old_1", "case_1"), ("old_3", "case_3")),
                ("unknown_val",),
            )
            with self.assertRaisesRegex(ValueError, "absent from Dataset260"):
                MODULE.build_fixed_split(
                    split_path,
                    mapping_path,
                    root / "splits",
                    root / "splits" / "trainval.csv",
                    reuse_existing=True,
                    existing_dataset_dir=dataset_dir,
                    existing_validation_dir=validation_dir,
                    baseline_excluded_source_ids={"case_2"},
                )

    def test_rejects_wrong_locked_test_complement_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path, mapping_path = self.write_inputs(root)
            dataset_dir, validation_dir = self.create_existing_artifacts(
                root,
                (("old_1", "case_1"), ("old_3", "case_3")),
                ("old_3",),
            )
            with self.assertRaisesRegex(ValueError, "locked test as the source-case complement"):
                MODULE.build_fixed_split(
                    split_path,
                    mapping_path,
                    root / "splits",
                    root / "splits" / "trainval.csv",
                    reuse_existing=True,
                    existing_dataset_dir=dataset_dir,
                    existing_validation_dir=validation_dir,
                )


if __name__ == "__main__":
    unittest.main()
