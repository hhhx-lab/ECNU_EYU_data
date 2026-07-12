import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_build_realonly_from_raw.py"


def load_module():
    spec = importlib.util.spec_from_file_location("g2_build_realonly_from_raw", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def touch_case(root: Path, case_id: str, missing: set[str] | None = None) -> None:
    missing = missing or set()
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    for suffix in ["t1n", "t1c", "t2w", "t2f", "seg"]:
        if suffix not in missing:
            (case_dir / f"{case_id}-{suffix}.nii.gz").write_text("placeholder")


class G2BuildRealOnlyFromRawTest(unittest.TestCase):
    def test_skips_case_missing_t2w(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            touch_case(raw, "BraTS-MET-00001-000")
            touch_case(raw, "BraTS-MET-00002-000", missing={"t2w"})

            rows, skipped = mod.build_mapping_rows(
                [raw],
                root,
                label_value_reader=lambda _path: {0, 1, 2, 3, 4},
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_case_id"], "BraTS-MET-00001-000")
            self.assertEqual(rows[0]["nnunet_case_id"], "BraTSMET_000001")
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["source_case_id"], "BraTS-MET-00002-000")
            self.assertIn("t2w", skipped[0]["missing_files"])

    def test_writes_mapping_csv_with_expected_columns(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            touch_case(raw, "BraTS-MET-00001-000")
            rows, _ = mod.build_mapping_rows(
                [raw],
                root,
                label_value_reader=lambda _path: {0, 1, 2, 3, 4},
            )
            out = root / "mapping.csv"

            mod.write_csv(out, mod.MAPPING_FIELDNAMES, rows)

            with out.open(newline="") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[0]["t2w_source_path"], "raw/BraTS-MET-00001-000/BraTS-MET-00001-000-t2w.nii.gz")

    def test_marks_fake_t2w_as_completion_only(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            case_id = "BraTS-MET-00554-000"
            touch_case(raw, case_id)

            rows, skipped = mod.build_mapping_rows(
                [raw],
                root,
                label_value_reader=lambda _path: {0, 1, 2, 3, 4},
                fake_t2w_case_ids={case_id},
            )

            self.assertEqual(skipped, [])
            self.assertEqual(rows[0]["patient_group"], "BraTS-MET-00554")
            self.assertEqual(rows[0]["t2w_status"], "fake_or_broken")
            self.assertEqual(rows[0]["eligible_for_realonly"], "False")
            self.assertEqual(rows[0]["completion_required"], "True")

    def test_uses_clean_corrected_label_when_raw_label_has_illegal_values(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            corrected = root / "corrected"
            case_id = "BraTS-MET-01184-002"
            touch_case(raw, case_id)
            corrected.mkdir()
            corrected_seg = corrected / f"{case_id}-seg.nii.gz"
            corrected_seg.write_text("corrected-placeholder")

            def label_values(path):
                return {0, 1, 2, 3, 4} if path == corrected_seg else {0, 1, 2, 3, 4, 8}

            rows, skipped = mod.build_mapping_rows(
                [raw],
                root,
                corrected_label_roots=[corrected],
                label_value_reader=label_values,
            )

            self.assertEqual(len(skipped), 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_case_id"], case_id)
            self.assertEqual(rows[0]["label_source"], "corrected")
            self.assertEqual(rows[0]["seg_source_path"], f"corrected/{case_id}-seg.nii.gz")

    def test_skips_case_with_illegal_raw_label_when_no_corrected_label_exists(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            case_id = "BraTS-MET-01094-002"
            touch_case(raw, case_id)

            def label_values(_path):
                return {0, 3, 6}

            rows, skipped = mod.build_mapping_rows(
                [raw],
                root,
                corrected_label_roots=[],
                label_value_reader=label_values,
            )

            self.assertEqual(rows, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["source_case_id"], case_id)
            self.assertEqual(skipped[0]["reason"], "illegal_label_values")
            self.assertIn("6", skipped[0]["missing_files"])


if __name__ == "__main__":
    unittest.main()
