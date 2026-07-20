import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
SCRIPT_PATH = CODE_DIR / "g2_pretraining_audit.py"


def load_module():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("g2_pretraining_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, affine), str(path))


class G2SyntheticIntakeTest(unittest.TestCase):
    def test_manual_review_clearance_only_clears_soft_flags(self):
        mod = load_module()
        reasons = [
            "tiny_ratio_high",
            "z_discontinuity",
            "block_artifact_suspected",
        ]
        cleared = mod.apply_manual_review_clearance(
            reasons,
            {"cleared_review_reasons": "tiny_ratio_high;z_discontinuity"},
        )
        self.assertEqual(cleared, ["block_artifact_suspected"])

    def test_manual_review_clearance_rejects_non_whitelisted_reason(self):
        mod = load_module()
        cleared = mod.apply_manual_review_clearance(
            ["block_artifact_suspected"],
            {"cleared_review_reasons": "block_artifact_suspected"},
        )
        self.assertEqual(
            cleared,
            [
                "block_artifact_suspected",
                "release_review_clearance_invalid",
            ],
        )

    def test_affine_signed_zero_is_consistent(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = np.zeros((4, 4, 4), dtype=np.float32)
            affine_positive = np.eye(4)
            affine_negative = np.eye(4)
            affine_negative[0, 1] = -0.0
            positive_path = root / "positive.nii.gz"
            negative_path = root / "negative.nii.gz"
            save(positive_path, array, affine_positive)
            save(negative_path, array, affine_negative)
            metas = {
                "positive": mod.nifti_meta(positive_path),
                "negative": mod.nifti_meta(negative_path),
            }
            self.assertTrue(mod.affines_consistent(metas))
            self.assertEqual(
                metas["positive"]["affine_hash"], metas["negative"]["affine_hash"]
            )

    def test_workspace_raw_path_uses_configured_data_root(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            relative = Path(
                "MICCAI-LH-BraTS2025-MET-Challenge-Training"
            ) / "BraTS-MET-00554-000" / "BraTS-MET-00554-000-t1n.nii.gz"
            expected = data_root / relative
            expected.parent.mkdir(parents=True)
            expected.touch()
            old_value = os.environ.get("G2_DATA_ROOT")
            os.environ["G2_DATA_ROOT"] = str(data_root)
            try:
                resolved = mod.parse_workspace_path(
                    Path("work_space/G1/data/raw") / relative
                )
            finally:
                if old_value is None:
                    os.environ.pop("G2_DATA_ROOT", None)
                else:
                    os.environ["G2_DATA_ROOT"] = old_value
            self.assertEqual(resolved, expected.resolve())

    def test_v2_source_manifest_indexes_source_case_id(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            (results_root / "manifests").mkdir(parents=True)
            (results_root / "splits").mkdir()
            (results_root / "qc").mkdir()
            case_id = "BraTS-MET-00001-000"
            nnunet_id = "BraTSMET_000001"
            write_csv(
                results_root / "manifests" / "real_train_manifest.csv",
                [{"case_id": case_id, "final_qc_pass": True}],
            )
            write_csv(
                results_root / "manifests" / "g1_v2_source_manifest.csv",
                [{"source_case_id": case_id, "allowed_as_v2_source": True}],
            )
            write_csv(
                results_root / "manifests" / "nnunet_case_mapping_master.csv",
                [{"source_case_id": case_id, "nnunet_case_id": nnunet_id}],
            )
            (results_root / "splits" / "splits_master_train_val_test.json").write_text(
                json.dumps([{"train": [nnunet_id], "val": [], "test": []}]),
                encoding="utf-8",
            )
            context = mod.load_reference_context(results_root)
            status = mod.build_source_status(case_id, context, "full_generation")
            self.assertTrue(status["source_allowed_for_v2"])
            self.assertTrue(status["source_is_allowed"])

    def build_fixture(
        self,
        root: Path,
        include_metadata: bool,
        include_approval: bool,
        source_split: str = "train",
    ):
        case_id = "BraTS-MET-00554-000"
        nnunet_id = "BraTSMET_000001"
        source_dir = root / "source" / case_id
        source_dir.mkdir(parents=True)
        run_root = root / "run_v3"
        case_dir = run_root / case_id
        case_dir.mkdir(parents=True)
        results_root = root / "results"
        (results_root / "manifests").mkdir(parents=True)
        (results_root / "splits").mkdir()
        (results_root / "qc").mkdir()
        affine = np.diag([1.0, 1.0, 1.2, 1.0])
        shape = (12, 12, 12)
        grid = np.indices(shape).sum(axis=0).astype(np.float32)
        lesion = np.zeros(shape, dtype=bool)
        lesion[4:8, 4:8, 4:8] = True
        paths = {}
        for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f"), start=1):
            array = grid + index * 5
            if modality in {"t1c", "t2w", "t2f"}:
                array = array.copy()
                array[lesion] += 20
            source_path = source_dir / f"{case_id}-{modality}.nii.gz"
            save(source_path, array, affine)
            paths[modality] = source_path
            output = array.copy()
            if modality == "t2w":
                output[lesion] += 0.1
            save(case_dir / f"{case_id}-{modality}.nii.gz", output, affine)
        seg = np.zeros(shape, dtype=np.int16)
        seg[lesion] = 3
        seg_path = source_dir / f"{case_id}-seg.nii.gz"
        save(seg_path, seg, affine)
        save(case_dir / f"{case_id}-seg.nii.gz", seg, affine)

        write_csv(
            results_root / "manifests" / "real_train_manifest.csv",
            [{
                "case_id": case_id,
                "final_qc_pass": True,
                "effective_seg_path": str(seg_path),
                "raw_seg_path": str(seg_path),
                "t1n_path": str(paths["t1n"]),
                "t1c_path": str(paths["t1c"]),
                "t2w_path": str(paths["t2w"]),
                "t2f_path": str(paths["t2f"]),
                "shape_seg": "12x12x12",
            }],
        )
        write_csv(
            results_root / "manifests" / "real_validation_manifest.csv",
            [{"case_id": "unused"}],
        )
        write_csv(
            results_root / "manifests" / "nnunet_case_mapping_master.csv",
            [{"nnunet_case_id": nnunet_id, "source_case_id": case_id}],
        )
        write_csv(
            results_root / "manifests" / "g1_v2_source_manifest.csv",
            [{"source_case_id": case_id, "allowed_as_v2_source": False}],
        )
        write_csv(
            results_root / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv",
            [{"case_id": case_id}],
        )
        (results_root / "splits" / "splits_master_train_val_test.json").write_text(
            json.dumps([{
                "train": [nnunet_id] if source_split == "train" else [],
                "val": [nnunet_id] if source_split == "val" else [],
                "test": [nnunet_id] if source_split == "test" else [],
            }]),
            encoding="utf-8",
        )
        config = {
            "generation_run_id": "v3_unit",
            "generator_name": "g1_missing_t2w_v3",
            "generation_mode": "completion",
            "generator_io": "t1n_t1c_t2f_to_t2w_completion",
            "source_csv": "data/g1_v3_data_placement_manifest.csv",
            "source_csv_version": "g1_v3_data_placement_manifest.csv",
            "seed": 42,
            "vae_weights": "vae.pt",
            "encdec_checkpoint": "encdec.pt",
            "bbdm_checkpoint": "bbdm.pt",
            "bbdm_s": 0.005,
            "validation_run": "validation/run-unit",
        }
        if include_metadata:
            (run_root / "generation_config.json").write_text(json.dumps(config), encoding="utf-8")
            write_csv(
                run_root / "synthetic_generation_manifest.csv",
                [{
                    "synthetic_raw_id": case_id,
                    "source_case_id": case_id,
                    "label_kind": "completion",
                    "status": "success",
                    "raw_case_dir": str(case_dir),
                }],
            )
            (run_root / "generation_log.jsonl").write_text(
                json.dumps({
                    "synthetic_raw_id": case_id,
                    "source_case_id": case_id,
                    "generation_run_id": "v3_unit",
                    "seed": 42,
                    "status": "success",
                }) + "\n",
                encoding="utf-8",
            )
        if include_approval:
            write_csv(
                run_root / "g2_approval_manifest.csv",
                [{
                    "synthetic_raw_id": case_id,
                    "approved_for_training": source_split == "train",
                    "approved_for_evaluation": source_split in {"val", "test"},
                    "reviewer": "unit-test",
                    "reason": "fixture",
                }],
            )
        return case_id, nnunet_id, run_root, results_root

    def run_fixture(
        self,
        include_metadata: bool,
        include_approval: bool,
        source_split: str = "train",
    ):
        mod = load_module()
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        case_id, nnunet_id, run_root, results_root = self.build_fixture(
            root, include_metadata, include_approval, source_split
        )
        dirs = mod.ensure_dirs(results_root)
        args = argparse.Namespace(
            synthetic_run_id="",
            generation_mode="completion",
        )
        mod.ingest_synthetic_run(run_root, results_root, args, dirs)
        qc_path = results_root / "qc" / f"qc_metrics_{run_root.name if not include_metadata else 'v3_unit'}.csv"
        with qc_path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        manifest_path = results_root / "manifests" / f"synthetic_generation_manifest_{run_root.name if not include_metadata else 'v3_unit'}.csv"
        with manifest_path.open(newline="") as handle:
            manifest_row = next(csv.DictReader(handle))
        return temporary, row, manifest_row, case_id, nnunet_id

    def test_missing_metadata_is_rejected(self):
        temporary, row, _, _, _ = self.run_fixture(False, False)
        try:
            self.assertEqual(row["qc_status"], "rejected")
            self.assertIn("metadata_incomplete", row["qc_reject_reason"])
            self.assertEqual(row["accepted_for_training"], "False")
        finally:
            temporary.cleanup()

    def test_v3_completion_keeps_real_identity_and_requires_approval(self):
        temporary, row, _, case_id, nnunet_id = self.run_fixture(True, False)
        try:
            self.assertEqual(row["synthetic_final_id"], case_id)
            self.assertEqual(row["nnunet_case_id"], nnunet_id)
            self.assertEqual(row["qc_status"], "pending_review")
            self.assertIn("release_approval_missing", row["manual_review_reason"])
        finally:
            temporary.cleanup()

    def test_v3_train_completion_can_be_approved(self):
        temporary, row, _, case_id, nnunet_id = self.run_fixture(True, True)
        try:
            self.assertEqual(row["synthetic_final_id"], case_id)
            self.assertEqual(row["nnunet_case_id"], nnunet_id)
            self.assertEqual(row["qc_status"], "accepted_for_training")
            self.assertEqual(row["accepted_for_training"], "True")
            self.assertEqual(row["generator_checkpoint_t1n"], "")
            self.assertEqual(row["generator_checkpoint_t2w"], "bbdm.pt")
            self.assertEqual(row["sampling_method"], "")
            self.assertEqual(row["sampling_steps"], "")
            self.assertEqual(row["eta"], "")
        finally:
            temporary.cleanup()

    def test_generation_manifest_covers_delivery_template_fields(self):
        temporary, _, manifest_row, _, _ = self.run_fixture(True, True)
        try:
            template_path = CODE_DIR.parent / "results" / "manifests" / "synthetic_generation_manifest_template_g1.csv"
            with template_path.open(newline="") as handle:
                template_fields = next(csv.reader(handle))
            self.assertTrue(set(template_fields).issubset(manifest_row))
            self.assertEqual(manifest_row["source_shape_x"], "12")
            self.assertEqual(manifest_row["source_shape_y"], "12")
            self.assertEqual(manifest_row["source_shape_z"], "12")
        finally:
            temporary.cleanup()

    def test_v3_val_completion_is_evaluation_only(self):
        temporary, row, manifest_row, case_id, nnunet_id = self.run_fixture(True, True, "val")
        try:
            self.assertEqual(row["synthetic_final_id"], case_id)
            self.assertEqual(row["nnunet_case_id"], nnunet_id)
            self.assertEqual(row["source_split"], "val")
            self.assertEqual(row["qc_status"], "accepted_for_evaluation")
            self.assertEqual(row["accepted_for_training"], "False")
            self.assertEqual(row["accepted_for_evaluation"], "True")
            self.assertIn("/imagesTr/", manifest_row["nnunet_t1n_target_path"])
        finally:
            temporary.cleanup()

    def test_v3_locked_test_completion_targets_images_ts(self):
        temporary, row, manifest_row, _, _ = self.run_fixture(True, True, "test")
        try:
            self.assertEqual(row["source_split"], "test")
            self.assertEqual(row["qc_status"], "accepted_for_evaluation")
            self.assertIn("/imagesTs/", manifest_row["nnunet_t1n_target_path"])
            self.assertIn("/labelsTs/", manifest_row["nnunet_seg_target_path"])
        finally:
            temporary.cleanup()

    def test_case_missing_from_generation_log_is_rejected(self):
        mod = load_module()
        temporary = tempfile.TemporaryDirectory()
        try:
            root = Path(temporary.name)
            _, _, run_root, results_root = self.build_fixture(
                root,
                include_metadata=True,
                include_approval=True,
                source_split="train",
            )
            (run_root / "generation_log.jsonl").write_text(
                json.dumps({
                    "synthetic_raw_id": "BraTS-MET-99999-000",
                    "source_case_id": "BraTS-MET-99999-000",
                    "generation_run_id": "v3_unit",
                    "seed": 42,
                    "status": "success",
                }) + "\n",
                encoding="utf-8",
            )
            dirs = mod.ensure_dirs(results_root)
            mod.ingest_synthetic_run(
                run_root,
                results_root,
                argparse.Namespace(synthetic_run_id="", generation_mode="completion"),
                dirs,
            )
            with (results_root / "qc" / "qc_metrics_v3_unit.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["qc_status"], "rejected")
            self.assertIn("case_log_record", row["metadata_missing_fields"])
        finally:
            temporary.cleanup()

    def test_missing_source_modality_comparison_is_rejected(self):
        mod = load_module()
        temporary = tempfile.TemporaryDirectory()
        try:
            root = Path(temporary.name)
            case_id, _, run_root, results_root = self.build_fixture(
                root,
                include_metadata=True,
                include_approval=True,
                source_split="train",
            )
            (root / "source" / case_id / f"{case_id}-t1n.nii.gz").unlink()
            dirs = mod.ensure_dirs(results_root)
            mod.ingest_synthetic_run(
                run_root,
                results_root,
                argparse.Namespace(synthetic_run_id="", generation_mode="completion"),
                dirs,
            )
            with (results_root / "qc" / "qc_metrics_v3_unit.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["qc_status"], "rejected")
            self.assertEqual(row["source_modality_comparison_complete"], "False")
            self.assertIn("source_modality_comparison_incomplete", row["qc_reject_reason"])
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
