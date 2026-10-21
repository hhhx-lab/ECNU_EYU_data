import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "06_prepare_official_validation.py"
SPEC = importlib.util.spec_from_file_location("prepare_official_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareOfficialValidationTests(unittest.TestCase):
    def create_case(self, root: Path, case_id: str, *, missing=None, extra=None):
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        for modality, _ in MODULE.CHANNELS:
            if modality == missing:
                continue
            (case_dir / f"{case_id}-{modality}.nii.gz").write_bytes(b"nifti")
        if extra:
            (case_dir / extra).write_bytes(b"nifti")
        return case_dir

    def test_materializes_exact_four_channel_input_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Validation"
            source.mkdir()
            self.create_case(source, "BraTS-MET-00833-000")
            self.create_case(source, "BraTS-MET-00834-000")
            destination = root / "imagesTs"
            manifest = root / "audit" / "manifest.csv"
            summary_path = root / "audit" / "summary.json"

            summary = MODULE.prepare_official_validation(
                source,
                destination,
                manifest,
                summary_path,
                expected_count=2,
                mode="symlink",
                clean=True,
            )

            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["nifti_count"], 8)
            self.assertEqual(summary["segmentation_count"], 0)
            self.assertTrue((destination / "BraTS-MET-00833-000_0000.nii.gz").is_symlink())
            self.assertFalse((destination / "BraTS-MET-00833-000_0004.nii.gz").exists())
            with manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["case_id"] for row in rows], [
                "BraTS-MET-00833-000",
                "BraTS-MET-00834-000",
            ])

    def test_rejects_missing_modality(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Validation"
            source.mkdir()
            self.create_case(source, "BraTS-MET-00833-000", missing="t2w")
            with self.assertRaisesRegex(ValueError, "missing=.*t2w"):
                MODULE.discover_cases(source, expected_count=1)

    def test_rejects_segmentation_or_other_unexpected_nifti(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Validation"
            source.mkdir()
            self.create_case(
                source,
                "BraTS-MET-00833-000",
                extra="BraTS-MET-00833-000-seg.nii.gz",
            )
            with self.assertRaisesRegex(ValueError, "unexpected_nifti=.*seg"):
                MODULE.discover_cases(source, expected_count=1)

    def test_rejects_stale_input_without_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Validation"
            source.mkdir()
            self.create_case(source, "BraTS-MET-00833-000")
            destination = root / "imagesTs"
            destination.mkdir()
            (destination / "stale_0000.nii.gz").touch()
            with self.assertRaisesRegex(ValueError, "stale NIfTI"):
                MODULE.prepare_official_validation(
                    source,
                    destination,
                    root / "manifest.csv",
                    root / "summary.json",
                    expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()
