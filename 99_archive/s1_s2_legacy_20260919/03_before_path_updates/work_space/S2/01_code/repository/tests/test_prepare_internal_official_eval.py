import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "10_prepare_internal_official_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_internal_official_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareInternalOfficialEvalTests(unittest.TestCase):
    def test_materializes_exact_fixed_val_source_id_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            for case_id in ("BraTSMET_000006", "BraTSMET_000009"):
                (predictions / f"{case_id}.nii.gz").write_bytes(b"prediction")
                (references / f"{case_id}.nii.gz").write_bytes(b"reference")

            mapping = root / "mapping.tsv"
            mapping.write_text(
                "BraTSMET_000006\tBraTS-MET-00006-000\n"
                "BraTSMET_000009\tBraTS-MET-00009-000\n"
                "BraTSMET_000010\tBraTS-MET-00010-000\n",
                encoding="utf-8",
            )
            val_list = root / "val_fixed.txt"
            val_list.write_text(
                "BraTSMET_000006\nBraTSMET_000009\n", encoding="utf-8"
            )
            checkpoint = root / "checkpoint_final.pth"
            checkpoint.write_bytes(b"checkpoint")

            summary = MODULE.prepare_internal_eval(
                predictions,
                references,
                mapping,
                val_list,
                root / "output",
                checkpoint,
                expected_count=2,
                mode="hardlink",
            )

            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["status"], "pass")
            self.assertTrue(
                (root / "output" / "prediction" / "BraTS-MET-00006-000.nii.gz").is_file()
            )
            self.assertTrue(
                (root / "output" / "reference" / "BraTS-MET-00009-000.nii.gz").is_file()
            )
            written = json.loads((root / "output" / "preparation_summary.json").read_text())
            self.assertEqual(written["checkpoint_sha256"], summary["checkpoint_sha256"])

    def test_rejects_prediction_set_that_differs_from_fixed_val(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            (predictions / "BraTSMET_000006.nii.gz").write_bytes(b"prediction")
            (references / "BraTSMET_000006.nii.gz").write_bytes(b"reference")
            mapping = root / "mapping.tsv"
            mapping.write_text(
                "BraTSMET_000006\tBraTS-MET-00006-000\n", encoding="utf-8"
            )
            val_list = root / "val_fixed.txt"
            val_list.write_text(
                "BraTSMET_000006\nBraTSMET_000009\n", encoding="utf-8"
            )
            checkpoint = root / "checkpoint_final.pth"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(ValueError, "prediction/fixed-val ID mismatch"):
                MODULE.prepare_internal_eval(
                    predictions,
                    references,
                    mapping,
                    val_list,
                    root / "output",
                    checkpoint,
                    expected_count=2,
                )


if __name__ == "__main__":
    unittest.main()
