from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPOSITORY_ROOT / "scripts" / "20_run_met_aug_training_smoke.py"
    spec = importlib.util.spec_from_file_location("test_met_aug_training_smoke_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load_script()


class MetAugTrainingSmokeTests(unittest.TestCase):
    def test_authorization_environment_is_trainer_specific(self):
        path = Path("/tmp/authorization.json")

        self.assertEqual(
            SMOKE.authorization_environment(SMOKE.EXPECTED_TRAINER, path),
            {"S2_MET_AUG_ROUTE_GATE": str(path)},
        )
        self.assertEqual(
            SMOKE.authorization_environment(SMOKE.EMERGENCY_FIX_V3_TRAINER, path),
            {"S2_MET_AUG_EMERGENCY_DECISION": str(path)},
        )
        with self.assertRaises(ValueError):
            SMOKE.authorization_environment("unknown", path)

    def test_audit_summary_counts_commits_and_no_op_reasons(self):
        summary = SMOKE.summarize_audit_events([
            {"state": "NO_OP", "reason": "NOT_SELECTED"},
            {"state": "COMMITTED", "reason": None},
            {"state": "NO_OP", "reason": "NOT_SELECTED"},
        ])

        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["committed_events"], 1)
        self.assertEqual(summary["state_counts"], {"COMMITTED": 1, "NO_OP": 2})
        self.assertEqual(summary["reason_counts"], {"NOT_SELECTED": 2})

    def test_timing_summary_excludes_first_step_from_steady_estimate(self):
        summary = SMOKE.summarize_steps([
            {"batch_load_seconds": 10.0, "train_step_seconds": 20.0, "total_seconds": 30.0},
            {"batch_load_seconds": 2.0, "train_step_seconds": 3.0, "total_seconds": 5.0},
            {"batch_load_seconds": 4.0, "train_step_seconds": 5.0, "total_seconds": 9.0},
        ], iterations_per_epoch=250)

        self.assertEqual(summary["first_step_seconds"], 30.0)
        self.assertEqual(summary["steady_total_median_seconds"], 7.0)
        self.assertEqual(summary["estimated_epoch_seconds"], 1750.0)
        self.assertAlmostEqual(summary["estimated_200_epochs_hours"], 1750.0 * 200 / 3600)

    def test_split_validation_requires_exact_fixed_1035_103_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train_fixed.txt").write_text(
                "".join(f"train-{index}\n" for index in range(1035)),
                encoding="utf-8",
            )
            (root / "val_fixed.txt").write_text(
                "".join(f"val-{index}\n" for index in range(103)),
                encoding="utf-8",
            )

            contract = SMOKE.validate_split_dir(root)

        self.assertEqual(contract["train_count"], 1035)
        self.assertEqual(contract["val_count"], 103)
        self.assertEqual(len(contract["train_sha256"]), 64)
        self.assertEqual(len(contract["val_sha256"]), 64)

    def test_read_audit_rejects_rows_without_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            path.write_text(json.dumps({"reason": "NOT_SELECTED"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "invalid MET-AUG audit row"):
                SMOKE.read_audit_events(path)

    def test_locked_environment_rejects_preexisting_drift(self):
        with patch.dict(os.environ, {"S2_PAIRED_TRAINING_SEED": "9"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "environment drift"):
                SMOKE._lock_environment("S2_PAIRED_TRAINING_SEED", "20260724")


if __name__ == "__main__":
    unittest.main()
