from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_prepare_completion_online_split.py"


class CompletionOnlineSplitTests(unittest.TestCase):
    def test_small_contract_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Dataset264_fixture"
            dataset.mkdir()
            split = {"train": ["A", "B"], "val": ["C"], "test": ["D"]}
            (dataset / "g2_fixed_split.json").write_text(
                json.dumps([split]), encoding="utf-8")
            rows = [
                {"nnunet_case_id": "A", "modality": "seg", "row_type": "real"},
                {"nnunet_case_id": "B", "modality": "seg", "row_type": "real_with_completion_t2w"},
                {"nnunet_case_id": "C", "modality": "seg", "row_type": "real"},
                {"nnunet_case_id": "D", "modality": "seg", "row_type": "real"},
            ]
            with (dataset / "g2_materialization_manifest.csv").open(
                "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["nnunet_case_id", "modality", "row_type"])
                writer.writeheader()
                writer.writerows(rows)
            output = root / "splits"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-dir", str(dataset),
                    "--output-split-dir", str(output),
                    "--expected-train", "2",
                    "--expected-val", "1",
                    "--expected-test", "1",
                    "--expected-completions", "1",
                ],
                check=True,
            )
            self.assertEqual(
                (output / "train_fixed.txt").read_text().splitlines(), ["A", "B"])
            self.assertEqual(
                (output / "val_fixed.txt").read_text().splitlines(), ["C"])


if __name__ == "__main__":
    unittest.main()
