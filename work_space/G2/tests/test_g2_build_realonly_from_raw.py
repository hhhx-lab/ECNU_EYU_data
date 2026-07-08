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

            rows, skipped = mod.build_mapping_rows([raw], root)

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
            rows, _ = mod.build_mapping_rows([raw], root)
            out = root / "mapping.csv"

            mod.write_csv(out, mod.MAPPING_FIELDNAMES, rows)

            with out.open(newline="") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[0]["t2w_source_path"], "raw/BraTS-MET-00001-000/BraTS-MET-00001-000-t2w.nii.gz")


if __name__ == "__main__":
    unittest.main()
