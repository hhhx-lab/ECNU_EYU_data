import csv
import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_create_train_val_test_split.py"


def load_split_module():
    spec = importlib.util.spec_from_file_location("g2_create_train_val_test_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainValTestSplitTest(unittest.TestCase):
    def test_patient_group_never_crosses_splits(self):
        mod = load_split_module()
        mapping_rows = []
        for idx, source_case_id in enumerate(
            [
                "BraTS-MET-00001-000",
                "BraTS-MET-00001-001",
                "BraTS-MET-00002-000",
                "BraTS-MET-00003-000",
                "BraTS-MET-00004-000",
                "BraTS-MET-00005-000",
                "BraTS-MET-00006-000",
                "BraTS-MET-00007-000",
                "BraTS-MET-00008-000",
                "BraTS-MET-00009-000",
            ],
            start=1,
        ):
            mapping_rows.append(
                {"nnunet_case_id": f"BraTSMET_{idx:06d}", "source_case_id": source_case_id}
            )

        result = mod.create_train_val_test_split(
            mapping_rows,
            val_fraction_of_train_pool=0.2,
            test_fraction=0.2,
            seed="42",
        )
        split_by_id = {
            case_id: split_name
            for split_name in ("train", "val", "test")
            for case_id in result[split_name]
        }
        self.assertEqual(split_by_id["BraTSMET_000001"], split_by_id["BraTSMET_000002"])

    def test_unanchored_completion_group_does_not_move_anchor(self):
        mod = load_split_module()
        mapping_rows = [
            {
                "nnunet_case_id": f"BraTSMET_{idx:06d}",
                "source_case_id": source_case_id,
            }
            for idx, source_case_id in enumerate(
                [
                    "BraTS-MET-00001-000",
                    "BraTS-MET-00002-000",
                    "BraTS-MET-00003-000",
                    "BraTS-MET-00004-000",
                    "BraTS-MET-00005-000",
                    "BraTS-MET-00006-000",
                    "BraTS-MET-00007-000",
                    "BraTS-MET-00008-000",
                    "BraTS-MET-00009-000",
                    "BraTS-MET-00010-000",
                    "BraTS-MET-99999-000",
                ],
                start=1,
            )
        ]
        anchor_ids = {row["source_case_id"] for row in mapping_rows[:-1]}
        anchored_only = mod.create_train_val_test_split(
            mapping_rows[:-1],
            val_fraction_of_train_pool=0.2,
            test_fraction=0.2,
            seed="42",
            anchor_case_ids=anchor_ids,
        )
        with_completion = mod.create_train_val_test_split(
            mapping_rows,
            val_fraction_of_train_pool=0.2,
            test_fraction=0.2,
            seed="42",
            anchor_case_ids=anchor_ids,
        )
        for split_name in ("train", "val", "test"):
            self.assertEqual(
                set(anchored_only[split_name]),
                set(with_completion[split_name]) - {"BraTSMET_000011"},
            )

    def test_legacy_val_is_locked_as_test_and_train_pool_is_split(self):
        mod = load_split_module()
        mapping_rows = [
            {"nnunet_case_id": f"BraTSMET_{idx:06d}", "source_case_id": f"BraTS-MET-{idx:05d}-000"}
            for idx in range(1, 11)
        ]
        base_split = [
            {
                "train": [f"BraTSMET_{idx:06d}" for idx in range(1, 9)],
                "val": [f"BraTSMET_{idx:06d}" for idx in range(9, 11)],
            }
        ]

        result = mod.create_train_val_test_split(
            mapping_rows,
            base_split=base_split,
            val_fraction_of_train_pool=0.25,
            seed="unit-test",
        )

        self.assertEqual(set(result["test"]), {"BraTSMET_000009", "BraTSMET_000010"})
        self.assertEqual(len(result["val"]), 2)
        self.assertEqual(len(result["train"]), 6)
        self.assertEqual(
            set(result["train"]) | set(result["val"]) | set(result["test"]),
            {row["nnunet_case_id"] for row in mapping_rows},
        )
        self.assertFalse(set(result["train"]) & set(result["val"]))
        self.assertFalse(set(result["train"]) & set(result["test"]))
        self.assertFalse(set(result["val"]) & set(result["test"]))

    def test_write_outputs_contains_membership_for_each_case(self):
        mod = load_split_module()
        mapping_rows = [
            {"nnunet_case_id": f"BraTSMET_{idx:06d}", "source_case_id": f"BraTS-MET-{idx:05d}-000"}
            for idx in range(1, 6)
        ]
        split = {
            "name": "fold0_train_val_test",
            "policy": "unit",
            "seed": "unit-test",
            "val_fraction_of_train_pool": 0.25,
            "test_fraction": "",
            "source_split_json": "",
            "mapping_csv": "",
            "counts": {"train": 2, "val": 1, "test": 2},
            "train": ["BraTSMET_000001", "BraTSMET_000002"],
            "val": ["BraTSMET_000003"],
            "test": ["BraTSMET_000004", "BraTSMET_000005"],
        }
        output_json = Path(self.tmpdir) / "split.json"
        output_csv = Path(self.tmpdir) / "membership.csv"

        mod.write_split_outputs(split, mapping_rows, output_json, output_csv)

        saved = json.loads(output_json.read_text(encoding="utf-8"))[0]
        self.assertEqual(saved["counts"], {"train": 2, "val": 1, "test": 2})
        with output_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            {row["split"] for row in rows},
            {"train", "val", "test"},
        )
        self.assertTrue(all(row["patient_group"] for row in rows))

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
