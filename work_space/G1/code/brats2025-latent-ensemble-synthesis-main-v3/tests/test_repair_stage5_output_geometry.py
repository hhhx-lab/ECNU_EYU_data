from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np

from repair_stage5_output_geometry import run


class RepairStage5OutputGeometryTests(unittest.TestCase):
    def test_repairs_geometry_without_changing_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "BraTS-MET-00001-000"
            real_case = root / "real" / case_id
            synthetic = root / "synthetic"
            real_case.mkdir(parents=True)
            synthetic.mkdir()
            shape = (8, 9, 10)
            reference_affine = np.array(
                [[1.0, 0.0, 0.0, 12.0], [0.0, 1.2, 0.0, -8.0], [0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 0.0, 1.0]]
            )
            wrong_affine = reference_affine.copy()
            wrong_affine[:3, 3] += [5.0, -3.0, 8.0]
            reference_data = np.ones(shape, dtype=np.float32)
            generated_data = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
            reference = nib.Nifti1Image(reference_data, reference_affine)
            reference.set_qform(reference_affine, 1)
            reference.set_sform(reference_affine, 1)
            nib.save(reference, real_case / f"{case_id}-t2w.nii.gz")
            nib.save(
                nib.Nifti1Image(generated_data, wrong_affine),
                synthetic / f"{case_id}-t2w.nii.gz",
            )
            metrics = root / "metrics.csv"
            with metrics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject"])
                writer.writeheader()
                writer.writerow({"subject": case_id})

            repair_root = root / "fixed"
            summary = run(
                Namespace(
                    real_root=str(root / "real"),
                    synthetic_root=str(synthetic),
                    metrics_csv=str(metrics),
                    audit_root=str(root / "audit"),
                    repair_output_root=str(repair_root),
                    expected_cases=1,
                    overwrite=False,
                )
            )
            fixed = nib.load(str(repair_root / f"{case_id}-t2w.nii.gz"))
            self.assertEqual(summary["geometry_mismatch_before_count"], 1)
            self.assertEqual(summary["repaired_count"], 1)
            self.assertFalse(summary["voxel_resampling_performed"])
            np.testing.assert_allclose(fixed.affine, reference_affine, atol=1e-6)
            np.testing.assert_array_equal(fixed.get_fdata(dtype=np.float32), generated_data)


if __name__ == "__main__":
    unittest.main()
