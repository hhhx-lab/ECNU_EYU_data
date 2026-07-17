import importlib.util
import csv
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_v3_paired_quality.py"


def load_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"paired-quality implementation is missing: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("g2_v3_paired_quality", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class G2V3PairedQualityTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_validate_geometry_rejects_shape_and_affine_mismatch(self):
        affine = np.eye(4)
        reference = nib.Nifti1Image(np.zeros((12, 12, 12)), affine)
        generated_bad_shape = nib.Nifti1Image(np.zeros((12, 12, 11)), affine)
        generated_bad_affine = nib.Nifti1Image(
            np.zeros((12, 12, 12)), np.diag([1.0, 1.0, 2.0, 1.0])
        )
        segmentation = nib.Nifti1Image(np.zeros((12, 12, 12)), affine)

        with self.assertRaisesRegex(ValueError, "shape"):
            self.mod.validate_geometry(reference, generated_bad_shape, segmentation)
        with self.assertRaisesRegex(ValueError, "affine"):
            self.mod.validate_geometry(reference, generated_bad_affine, segmentation)

    def test_identical_masked_region_has_perfect_metrics(self):
        image = np.zeros((15, 15, 15), dtype=np.float32)
        image[3:12, 3:12, 3:12] = np.linspace(0.1, 0.9, 9)[:, None, None]
        mask = np.zeros_like(image, dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        metrics = self.mod.compute_masked_metrics(image, image.copy(), mask)

        self.assertAlmostEqual(metrics["ssim"], 1.0, places=6)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertTrue(np.isinf(metrics["psnr"]))

    def test_build_region_masks_uses_brats_2026_labels(self):
        seg = np.zeros((6, 6, 6), dtype=np.int16)
        seg[1, 1, 1] = 1
        seg[2, 2, 2] = 2
        seg[3, 3, 3] = 3
        seg[4, 4, 4] = 4

        masks = self.mod.build_region_masks(seg)

        self.assertEqual(int(masks["tumor_all"].sum()), 4)
        self.assertTrue(masks["NETC"][1, 1, 1])
        self.assertTrue(masks["SNFH"][2, 2, 2])
        self.assertTrue(masks["ET"][3, 3, 3])
        self.assertTrue(masks["RC"][4, 4, 4])

    def test_extract_lesions_uses_26_connectivity_and_mm3_size_classes(self):
        seg = np.zeros((32, 32, 32), dtype=np.int16)
        seg[1:3, 1:3, 1:3] = 1       # 8 mm3: tiny
        seg[8:11, 8:11, 8:11] = 3    # 27 mm3: small
        seg[18:25, 18:25, 18:25] = 4  # 343 mm3: large

        lesions = self.mod.extract_lesions(seg, spacing=(1.0, 1.0, 1.0))

        self.assertEqual([row["size_class"] for row in lesions], ["tiny", "small", "large"])
        self.assertEqual([row["voxel_count"] for row in lesions], [8, 27, 343])
        self.assertEqual(lesions[2]["labels_present"], "RC")

    def test_choose_review_focus_prioritizes_rc_then_small_lesion(self):
        seg = np.zeros((24, 24, 24), dtype=np.int16)
        seg[2:4, 2:4, 2:4] = 1
        seg[14:17, 14:17, 14:17] = 4
        lesions = self.mod.extract_lesions(seg, spacing=(1.0, 1.0, 1.0))

        focus, reason = self.mod.choose_review_focus(seg, lesions)

        self.assertEqual(reason, "RC")
        self.assertTrue(all(14 <= coordinate <= 16 for coordinate in focus))

    def test_artifact_metrics_detect_void_and_brain_external_signal(self):
        reference = np.zeros((16, 16, 16), dtype=np.float32)
        reference[3:13, 3:13, 3:13] = 0.5
        generated = reference.copy()
        generated[6:9, 6:9, 6:9] = 0.0
        generated[0:2, 0:2, 0:2] = 0.5
        brain = reference > 0
        lesion = np.zeros_like(brain)
        lesion[6:9, 6:9, 6:9] = True

        metrics = self.mod.compute_artifact_metrics(reference, generated, brain, lesion)

        self.assertGreater(metrics["brain_void_excess"], 0.0)
        self.assertGreater(metrics["external_signal_fraction"], 0.0)
        self.assertGreater(metrics["lesion_void_fraction"], 0.9)

    def test_render_montage_writes_nonempty_png(self):
        shape = (20, 20, 20)
        reference = np.zeros(shape, dtype=np.float32)
        reference[3:17, 3:17, 3:17] = 0.4
        generated = reference.copy()
        generated[8:12, 8:12, 8:12] = 0.7
        seg = np.zeros(shape, dtype=np.int16)
        seg[8:12, 8:12, 8:12] = 3

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "montage.png"
            self.mod.render_montage(
                reference,
                generated,
                seg,
                focus=(10, 10, 10),
                case_id="BraTS-MET-UNIT-000",
                annotations={"whole_ssim": 0.8, "focus_reason": "small"},
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def _write_spatial_audit(self, path, rows):
        fieldnames = [
            "subject",
            "native_shape",
            "target_shape",
            "target_spacing_mm",
            "foreground_voxel_count",
            "lesion_voxel_count",
            "foreground_outside_voxel_count",
            "lesion_outside_voxel_count",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _spatial_row(self, case_id, **overrides):
        row = {
            "subject": case_id,
            "native_shape": "240x240x155",
            "target_shape": "256x256x160",
            "target_spacing_mm": "1.0",
            "foreground_voxel_count": "1000",
            "lesion_voxel_count": "25",
            "foreground_outside_voxel_count": "0",
            "lesion_outside_voxel_count": "0",
        }
        row.update(overrides)
        return row

    def test_spatial_audit_accepts_exact_case_set_with_zero_escape(self):
        case_ids = ["BraTS-MET-00001-000", "BraTS-MET-00002-000"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spatial_audit.csv"
            self._write_spatial_audit(
                path,
                [self._spatial_row(case_id) for case_id in reversed(case_ids)],
            )

            rows = self.mod.read_and_validate_spatial_audit(path, case_ids)

        self.assertEqual({row["subject"] for row in rows}, set(case_ids))

    def test_spatial_audit_rejects_case_id_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spatial_audit.csv"
            self._write_spatial_audit(
                path,
                [
                    self._spatial_row("BraTS-MET-00001-000"),
                    self._spatial_row("BraTS-MET-99999-000"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "case IDs do not match"):
                self.mod.read_and_validate_spatial_audit(
                    path,
                    ["BraTS-MET-00001-000", "BraTS-MET-00002-000"],
                )

    def test_spatial_audit_rejects_duplicate_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spatial_audit.csv"
            self._write_spatial_audit(
                path,
                [
                    self._spatial_row("BraTS-MET-00001-000"),
                    self._spatial_row("BraTS-MET-00001-000"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate subjects"):
                self.mod.read_and_validate_spatial_audit(
                    path,
                    ["BraTS-MET-00001-000"],
                )

    def test_spatial_audit_rejects_any_foreground_or_lesion_escape(self):
        for column in (
            "foreground_outside_voxel_count",
            "lesion_outside_voxel_count",
        ):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "spatial_audit.csv"
                self._write_spatial_audit(
                    path,
                    [self._spatial_row("BraTS-MET-00001-000", **{column: "1"})],
                )

                with self.assertRaisesRegex(ValueError, "escaped model FOV"):
                    self.mod.read_and_validate_spatial_audit(
                        path,
                        ["BraTS-MET-00001-000"],
                    )

    def test_run_persists_passing_spatial_gate_and_outputs(self):
        case_id = "BraTS-MET-00001-000"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_root = root / "real"
            synthetic_root = root / "synthetic"
            output_root = root / "qc"
            case_dir = real_root / case_id
            case_dir.mkdir(parents=True)
            synthetic_root.mkdir()

            shape = (16, 16, 16)
            affine = np.eye(4)
            image = np.zeros(shape, dtype=np.float32)
            image[2:14, 2:14, 2:14] = np.linspace(0.1, 0.9, 12)[:, None, None]
            segmentation = np.zeros(shape, dtype=np.int16)
            segmentation[6:10, 6:10, 6:10] = 3
            nib.save(
                nib.Nifti1Image(image, affine),
                case_dir / f"{case_id}-t2w.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(segmentation, affine),
                case_dir / f"{case_id}-seg.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(image.copy(), affine),
                synthetic_root / f"{case_id}-t2w.nii.gz",
            )

            metrics_path = root / "metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "subject",
                        "whole_SSIM",
                        "whole_PSNR",
                        "brain_SSIM",
                        "brain_PSNR",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "subject": case_id,
                        "whole_SSIM": "1.0",
                        "whole_PSNR": "99.0",
                        "brain_SSIM": "1.0",
                        "brain_PSNR": "99.0",
                    }
                )
            spatial_path = root / "spatial_audit.csv"
            self._write_spatial_audit(spatial_path, [self._spatial_row(case_id)])

            summary = self.mod.run(
                Namespace(
                    real_root=str(real_root),
                    synthetic_root=str(synthetic_root),
                    stage5_metrics=str(metrics_path),
                    spatial_audit=str(spatial_path),
                    output_root=str(output_root),
                    expected_cases=1,
                    seed=42,
                    overwrite=False,
                )
            )

            self.assertEqual(summary["spatial_gate"]["status"], "pass")
            self.assertEqual(summary["spatial_gate"]["case_count"], 1)
            self.assertTrue((output_root / "spatial_audit.csv").is_file())
            self.assertTrue((output_root / "case_metrics.csv").is_file())
            self.assertTrue((output_root / "montages" / f"{case_id}.png").is_file())


if __name__ == "__main__":
    unittest.main()
