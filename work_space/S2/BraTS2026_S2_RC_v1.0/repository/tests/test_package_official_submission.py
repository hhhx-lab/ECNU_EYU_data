import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = None
    np = None


SCRIPT = Path(__file__).parents[1] / "scripts" / "07_package_official_submission.py"
SPEC = importlib.util.spec_from_file_location("package_official_submission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(nib is not None and np is not None, "nibabel/numpy not installed")
class PackageOfficialSubmissionTests(unittest.TestCase):
    def write_nifti(self, path: Path, shape=(4, 5, 6), labels=False):
        data = np.zeros(shape, dtype=np.uint8 if labels else np.float32)
        if labels:
            data[1, 1, 1] = 3
        affine = np.array(
            [[1.0, 0.0, 0.0, 10.0],
             [0.0, 1.0, 0.0, -20.0],
             [0.0, 0.0, 1.0, 30.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        nib.save(nib.Nifti1Image(data, affine), path)

    def create_case(self, root: Path, case_id: str):
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        for modality in MODULE.MODALITIES:
            self.write_nifti(case_dir / f"{case_id}-{modality}.nii.gz")
        return case_dir

    def test_builds_flat_179_style_zip_after_full_contract_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Validation"
            predictions = root / "predictions"
            source.mkdir()
            predictions.mkdir()
            case_ids = ["BraTS-MET-00833-000", "BraTS-MET-00834-000"]
            for case_id in case_ids:
                self.create_case(source, case_id)
                self.write_nifti(predictions / f"{case_id}.nii.gz", labels=True)

            output_zip = root / "submission" / "s2.zip"
            summary = MODULE.validate_and_package(
                source,
                predictions,
                output_zip,
                root / "submission" / "manifest.csv",
                root / "submission" / "validation.json",
                expected_count=2,
            )

            self.assertEqual(summary["prediction_count"], 2)
            self.assertEqual(summary["empty_prediction_count"], 0)
            with zipfile.ZipFile(output_zip) as archive:
                self.assertEqual(archive.namelist(), [f"{case_id}.nii.gz" for case_id in case_ids])
                self.assertTrue(all("/" not in name for name in archive.namelist()))

    def test_rejects_prediction_with_wrong_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Validation"
            predictions = root / "predictions"
            source.mkdir()
            predictions.mkdir()
            case_id = "BraTS-MET-00833-000"
            self.create_case(source, case_id)
            self.write_nifti(predictions / f"{case_id}.nii.gz", shape=(3, 5, 6), labels=True)
            with self.assertRaisesRegex(ValueError, "array dimensions differ"):
                MODULE.validate_and_package(
                    source,
                    predictions,
                    root / "s2.zip",
                    root / "manifest.csv",
                    root / "validation.json",
                    expected_count=1,
                )

    def test_rejects_illegal_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Validation"
            predictions = root / "predictions"
            source.mkdir()
            predictions.mkdir()
            case_id = "BraTS-MET-00833-000"
            self.create_case(source, case_id)
            prediction = predictions / f"{case_id}.nii.gz"
            self.write_nifti(prediction, labels=True)
            image = nib.load(prediction)
            data = np.asanyarray(image.dataobj).copy()
            data[0, 0, 0] = 8
            nib.save(nib.Nifti1Image(data, image.affine), prediction)
            with self.assertRaisesRegex(ValueError, "illegal labels.*8"):
                MODULE.validate_and_package(
                    source,
                    predictions,
                    root / "s2.zip",
                    root / "manifest.csv",
                    root / "validation.json",
                    expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()
