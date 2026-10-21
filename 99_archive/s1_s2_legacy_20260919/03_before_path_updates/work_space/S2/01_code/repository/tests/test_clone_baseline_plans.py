from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_clone_baseline_plans.py"


class CloneBaselinePlansTests(unittest.TestCase):
    @staticmethod
    def dataset_payload() -> dict:
        return {
            "channel_names": {
                "0": "t1n",
                "1": "t1c",
                "2": "t2w",
                "3": "t2f",
            },
            "labels": {
                "background": 0,
                "NETC": 1,
                "SNFH": 2,
                "ET": 3,
                "RC": 4,
            },
            "numTraining": 1138,
            "file_ending": ".nii.gz",
        }

    def test_clones_locked_plan_and_dataset_contract_without_deleting_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            payload = {
                "dataset_name": "Dataset263_RealOnly",
                "plans_name": "nnUNetPlans",
                "configurations": {
                    "3d_fullres": {
                        "batch_size": 2,
                        "patch_size": [112, 160, 128],
                        "normalization_schemes": ["ZScoreNormalization"] * 4,
                        "architecture": {"network_class_name": "PlainConvUNet"},
                    }
                },
            }
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            source_dataset_json = root / "dataset.json"
            dataset_payload = self.dataset_payload()
            source_dataset_json.write_text(
                json.dumps(dataset_payload), encoding="utf-8"
            )
            target = root / "Dataset264_Completion"
            target.mkdir()
            fingerprint = target / "dataset_fingerprint.json"
            fingerprint.write_text('{"keep": true}', encoding="utf-8")
            gt_dir = target / "gt_segmentations"
            gt_dir.mkdir()
            gt_sentinel = gt_dir / "BraTSMET_000001.nii.gz"
            gt_sentinel.write_bytes(b"keep-gt")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--baseline-plans", str(baseline),
                    "--source-dataset-json", str(source_dataset_json),
                    "--target-preprocessed-dir", str(target),
                    "--target-dataset-name", "Dataset264_Completion",
                ],
                check=True,
            )
            cloned = json.loads((target / "nnUNetPlans.json").read_text())
            self.assertEqual(cloned["dataset_name"], "Dataset264_Completion")
            self.assertEqual(
                cloned["configurations"], payload["configurations"]
            )
            self.assertEqual(
                json.loads((target / "dataset.json").read_text()), dataset_payload
            )
            self.assertEqual(fingerprint.read_text(encoding="utf-8"), '{"keep": true}')
            self.assertEqual(gt_sentinel.read_bytes(), b"keep-gt")

    def test_clones_plans_embedded_in_the_actual_warmstart_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "dataset_name": "Dataset263_Current",
                "plans_name": "nnUNetPlans",
                "configurations": {
                    "3d_fullres": {
                        "batch_size": 2,
                        "patch_size": [128, 128, 128],
                        "normalization_schemes": ["ZScoreNormalization"] * 4,
                        "architecture": {"network_class_name": "PlainConvUNet"},
                    }
                },
            }
            checkpoint = root / "checkpoint_final.pth"
            torch.save({"init_args": {"plans": payload}}, checkpoint)
            source_dataset_json = root / "dataset.json"
            source_dataset_json.write_text(
                json.dumps(self.dataset_payload()), encoding="utf-8"
            )
            target = root / "Dataset264_Completion"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--baseline-checkpoint", str(checkpoint),
                    "--source-dataset-json", str(source_dataset_json),
                    "--target-preprocessed-dir", str(target),
                    "--target-dataset-name", "Dataset264_Completion",
                ],
                check=True,
            )
            cloned = json.loads((target / "nnUNetPlans.json").read_text())
            self.assertEqual(cloned["dataset_name"], "Dataset264_Completion")
            self.assertEqual(cloned["configurations"], payload["configurations"])
            audit = json.loads((target / "completion_plans_audit.json").read_text())
            self.assertEqual(audit["plans_source_kind"], "checkpoint_init_args")
            self.assertEqual(audit["baseline_checkpoint"], str(checkpoint.resolve()))


if __name__ == "__main__":
    unittest.main()
