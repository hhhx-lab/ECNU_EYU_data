from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPOSITORY_ROOT / "scripts" / "25_run_e_continue_training_smoke.py"
    spec = importlib.util.spec_from_file_location("test_e_continue_training_smoke_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load_script()


class EContinueTrainingSmokeTests(unittest.TestCase):
    def test_timing_uses_real_steps_and_conservative_factor(self):
        summary = SMOKE.summarize_steps(
            [
                {"batch_load_seconds": 10.0, "train_step_seconds": 20.0, "total_seconds": 30.0},
                {"batch_load_seconds": 4.0, "train_step_seconds": 8.0, "total_seconds": 12.0},
                {"batch_load_seconds": 1.0, "train_step_seconds": 4.0, "total_seconds": 5.0},
                {"batch_load_seconds": 3.0, "train_step_seconds": 6.0, "total_seconds": 9.0},
            ],
            iterations_per_epoch=250,
            eta_safety_factor=1.25,
        )

        self.assertEqual(summary["warmup_steps_excluded"], 2)
        self.assertEqual(summary["steady_total_median_seconds"], 7.0)
        self.assertAlmostEqual(summary["estimated_200_epochs_hours"], 7.0 * 250 * 200 / 3600)
        self.assertAlmostEqual(
            summary["estimated_200_epochs_hours_conservative"],
            7.0 * 250 * 200 / 3600 * 1.25,
        )

    def test_split_validation_requires_locked_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train_fixed.txt"
            val = root / "val_fixed.txt"
            train.write_text("".join(f"train-{index}\n" for index in range(1035)), encoding="utf-8")
            val.write_text("".join(f"val-{index}\n" for index in range(103)), encoding="utf-8")
            with patch.object(SMOKE, "EXPECTED_TRAIN_SHA256", SMOKE.sha256_file(train)), patch.object(
                SMOKE, "EXPECTED_VAL_SHA256", SMOKE.sha256_file(val)
            ):
                contract = SMOKE.validate_split_dir(root)

        self.assertEqual(contract["train_count"], 1035)
        self.assertEqual(contract["val_count"], 103)

    def test_split_validation_rejects_sha_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train_fixed.txt").write_text(
                "".join(f"train-{index}\n" for index in range(1035)), encoding="utf-8"
            )
            (root / "val_fixed.txt").write_text(
                "".join(f"val-{index}\n" for index in range(103)), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256 drifted"):
                SMOKE.validate_split_dir(root)

    def test_rejects_route_or_generative_asset_environment(self):
        with self.assertRaisesRegex(RuntimeError, "refuses Route/G1/G2/Diffusion"):
            SMOKE.reject_generative_environment({"G1_SELECTION": "/forbidden"})
        with self.assertRaisesRegex(RuntimeError, "S2_MET_AUG_ENABLE=0"):
            SMOKE.reject_generative_environment({"S2_MET_AUG_ENABLE": "1"})
        SMOKE.reject_generative_environment({"S2_MET_AUG_ENABLE": "0"})

    def test_locked_environment_rejects_drift(self):
        with patch.dict(os.environ, {"S2_COMPLETION_EPOCHS": "199"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "environment drift"):
                SMOKE._lock_environment("S2_COMPLETION_EPOCHS", "200")


if __name__ == "__main__":
    unittest.main()
