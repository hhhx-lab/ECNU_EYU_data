from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "46_validate_met_aug_fix_v3_training.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fix_v3_training_validation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_row(epoch: int, patch: int, *, committed: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "epoch": epoch,
        "patch_index": patch,
        "event_id": f"{epoch * 100 + patch:024x}",
        "event_seed": epoch * 100 + patch,
        "rank": 0,
        "worker": 0,
        "route_id": "MET-AUG-A",
        "target_case_id": "train-a",
        "target_patient_group": "target-group",
    }
    if committed:
        row.update(
            {
                "state": "COMMITTED",
                "reason": None,
                "donor_patient_group": "donor-group",
                "fix_v3": {"status": "pass", "score": 1.0},
            }
        )
    else:
        row.update({"state": "NO_OP", "reason": "NOT_SELECTED"})
    return row


class FixV3TrainingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script()

    def test_validates_complete_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_log.txt"
            path.write_text(
                "2026-01-01: Epoch 0\n"
                "2026-01-01: train_loss 0.2\n"
                "2026-01-01: val_loss 0.1\n"
                "2026-01-01: Pseudo dice [0.1, 0.2, 0.3, 0.4]\n"
                "2026-01-01: Epoch time: 2.0 s\n"
                "2026-01-01: Epoch 1\n"
                "2026-01-01: train_loss 0.1\n"
                "2026-01-01: val_loss 0.05\n"
                "2026-01-01: Pseudo dice [0.2, 0.3, 0.4, 0.5]\n"
                "2026-01-01: Epoch time: 3.0 s\n"
                "2026-01-01: Training done.\n"
                "2026-01-01: Validation complete\n",
                encoding="utf-8",
            )
            result = self.module.validate_training_log(path, expected_epochs=2)
            self.assertEqual(result["epoch_count"], 2)
            self.assertEqual(result["last_pseudo_dice"], [0.2, 0.3, 0.4, 0.5])

    def test_rejects_incomplete_or_nonfinite_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_log.txt"
            path.write_text(
                "Epoch 0\ntrain_loss nan\nval_loss 0.1\n"
                "Pseudo dice [0.1]\nEpoch time: 1.0 s\n"
                "Training done.\nValidation complete\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(self.module.TrainingValidationError, "red flags"):
                self.module.validate_training_log(path, expected_epochs=1)

    def test_validates_full_audit_cartesian_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            rows = [
                _audit_row(epoch, patch, committed=(patch == 0))
                for epoch in range(2)
                for patch in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = self.module.validate_audit(
                path,
                train_ids={"train-a"},
                expected_epochs=2,
                expected_rows=4,
            )
            self.assertEqual(result["state_counts"], {"COMMITTED": 2, "NO_OP": 2})
            self.assertTrue(result["event_ids_unique"])
            self.assertTrue(result["epoch_patch_cartesian_complete"])

    def test_rejects_audit_duplicate_event_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            first = _audit_row(0, 0, committed=False)
            second = _audit_row(0, 1, committed=False)
            second["event_id"] = first["event_id"]
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(self.module.TrainingValidationError, "duplicate audit"):
                self.module.validate_audit(
                    path,
                    train_ids={"train-a"},
                    expected_epochs=1,
                    expected_rows=2,
                )

    def test_rejects_nonfinite_audit_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            row = _audit_row(0, 0, committed=True)
            row["fix_v3"] = {"status": "pass", "score": math.nan}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(self.module.TrainingValidationError, "non-finite"):
                self.module.validate_audit(
                    path,
                    train_ids={"train-a"},
                    expected_epochs=1,
                    expected_rows=1,
                )

    def test_validates_checkpoint_metadata_with_injected_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint_final.pth"
            checkpoint.write_bytes(b"checkpoint")
            logging = {
                "train_losses": [0.2, 0.1],
                "val_losses": [0.1, 0.05],
                "dice_per_class_or_region": [[0.1], [0.2]],
                "mean_fg_dice": [0.1, 0.2],
                "epoch_start_timestamps": [1.0, 3.0],
                "epoch_end_timestamps": [2.0, 4.0],
            }

            result = self.module.validate_checkpoint(
                checkpoint,
                expected_epochs=2,
                reader=lambda _: {
                    "current_epoch": 2,
                    "trainer_name": self.module.EXPECTED_TRAINER,
                    "logging": logging,
                    "__network_tensor_count": 10,
                    "__network_element_count": 100,
                },
            )
            self.assertEqual(result["current_epoch"], 2)
            self.assertEqual(result["network_element_count"], 100)

    def test_validates_prediction_coverage_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation"
            reference = root / "reference"
            validation.mkdir()
            reference.mkdir()
            for case_id in ("case-a", "case-b"):
                (validation / f"{case_id}.nii.gz").write_bytes(case_id.encode())
                (reference / f"{case_id}.nii.gz").write_bytes(case_id.encode())
            summary = {
                "mean": {str(label): {"Dice": 0.5} for label in range(1, 5)},
                "foreground_mean": {"Dice": 0.5},
                "metric_per_case": [
                    {
                        "prediction_file": str(validation / f"{case_id}.nii.gz"),
                        "metrics": {"4": {"Dice": math.nan}},
                    }
                    for case_id in ("case-a", "case-b")
                ],
            }
            (validation / "summary.json").write_text(
                json.dumps(summary, allow_nan=True),
                encoding="utf-8",
            )
            result = self.module.validate_predictions(
                validation,
                reference,
                val_ids=["case-a", "case-b"],
                inspector=lambda _prediction, _reference: {"shape": [1, 1, 1]},
            )
            self.assertEqual(result["prediction_count"], 2)
            self.assertEqual(result["aggregate_metrics"]["foreground_mean"]["Dice"], 0.5)

    def test_rejects_nonfinite_aggregate_summary(self):
        with self.assertRaisesRegex(self.module.TrainingValidationError, "non-finite"):
            self.module._finite_aggregate_metrics(
                {
                    "mean": {
                        **{str(label): {"Dice": 0.5} for label in range(1, 4)},
                        "4": {"Dice": math.nan},
                    },
                    "foreground_mean": {"Dice": 0.5},
                }
            )


if __name__ == "__main__":
    unittest.main()
