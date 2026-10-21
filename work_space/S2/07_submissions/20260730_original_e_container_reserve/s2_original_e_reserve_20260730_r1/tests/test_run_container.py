from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import run_container as target  # noqa: E402


class RunContainerContractTests(unittest.TestCase):
    def make_case(self, root: Path, case_id: str = "BraTS-MET-00001-000") -> Path:
        case_dir = root / case_id
        case_dir.mkdir()
        for modality, _ in target.CHANNELS:
            (case_dir / f"{case_id}-{modality}.nii.gz").write_bytes(b"test")
        return case_dir

    def test_discovers_and_materializes_four_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_case(root)
            cases = target.discover_cases(root)
            self.assertEqual([case.case_id for case in cases], ["BraTS-MET-00001-000"])

            prepared = root / "prepared"
            target.materialize_nnunet_input(cases, prepared)
            self.assertEqual(
                sorted(path.name for path in prepared.iterdir()),
                [
                    "BraTS-MET-00001-000_0000.nii.gz",
                    "BraTS-MET-00001-000_0001.nii.gz",
                    "BraTS-MET-00001-000_0002.nii.gz",
                    "BraTS-MET-00001-000_0003.nii.gz",
                ],
            )
            self.assertTrue(all(path.is_symlink() for path in prepared.iterdir()))

    def test_rejects_missing_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = self.make_case(root)
            (case_dir / f"{case_dir.name}-t2f.nii.gz").unlink()
            with self.assertRaises(target.ContractError):
                target.discover_cases(root)

    def test_rejects_unexpected_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_case(root, "not-a-brats-case")
            with self.assertRaises(target.ContractError):
                target.discover_cases(root)

    def test_requires_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "stale.nii.gz").write_bytes(b"stale")
            with self.assertRaises(target.ContractError):
                target.require_empty_output(output)

    def test_accepts_integer_prediction_labels(self) -> None:
        self.assertEqual(
            target.validate_label_values([0.0, 1.0, 2.0, 3.0, 4.0], "case"),
            {0, 1, 2, 3, 4},
        )

    def test_rejects_fractional_prediction_label(self) -> None:
        with self.assertRaisesRegex(target.ContractError, "Non-integer"):
            target.validate_label_values([0.0, 1.5, 4.0], "case")

    def test_rejects_nonfinite_prediction_label(self) -> None:
        with self.assertRaisesRegex(target.ContractError, "Non-finite"):
            target.validate_label_values([0.0, float("nan")], "case")


if __name__ == "__main__":
    unittest.main()
