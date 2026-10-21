from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from custom_nnunet.met_aug_core import MetAugContractError, sha256_file
from custom_nnunet.met_aug_fix_v3_emergency import (
    FIX_V3_EMERGENCY_SKIPPED_STAGES,
    make_fix_v3_emergency_decision,
    validate_fix_v3_emergency_decision,
)
from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune,
)
from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune,
)


class FixV3EmergencyDecisionTests(unittest.TestCase):
    def test_training_launcher_has_isolated_emergency_mode_and_import_closure(self):
        train_script = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("met_aug_route_a_fix_v3_emergency)", train_script)
        self.assertIn(
            "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune.py",
            train_script,
        )
        for filename in (
            "met_aug_fix_v2.py",
            "met_aug_fix_v3.py",
            "met_aug_fix_v3_emergency.py",
        ):
            self.assertIn(f'"{filename}"', train_script)
        for variable in (
            "S2_MET_AUG_EMERGENCY_DECISION",
            "S2_MET_AUG_FIX_V3_CALIBRATION",
            "S2_MET_AUG_ORIGINAL_E_CHECKPOINT",
            "S2_MET_AUG_FIX_V2_FAILURE_AUDIT",
        ):
            self.assertIn(variable, train_script)

    def test_emergency_trainer_is_separate_and_uses_separate_authorization_env(self):
        trainer = nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune

        self.assertTrue(issubclass(trainer, nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune))
        self.assertEqual(
            trainer.authorization_env_name,
            "S2_MET_AUG_EMERGENCY_DECISION",
        )
        self.assertEqual(trainer.authorization_kind, "fix_v3_emergency_unvalidated")

    def test_decision_is_explicitly_unvalidated_and_sha_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in (
                "component_manifest",
                "route_config",
                "valid_mask_manifest",
                "calibration",
                "original_e",
                "fix_v2_failure",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                paths[name] = path
            with patch(
                "custom_nnunet.met_aug_fix_v3_emergency.ORIGINAL_E_SHA256",
                sha256_file(paths["original_e"]),
            ), patch(
                "custom_nnunet.met_aug_fix_v3_emergency.FIX_V2_FAILURE_SHA256",
                sha256_file(paths["fix_v2_failure"]),
            ):
                decision = make_fix_v3_emergency_decision(**paths)
                decision_path = root / "decision.json"
                decision_path.write_text(decision, encoding="utf-8")
                validated = validate_fix_v3_emergency_decision(
                    decision_path,
                    **paths,
                )

            self.assertEqual(validated["formal_validation_status"], "skipped")
            self.assertEqual(
                tuple(validated["skipped_stages"]),
                FIX_V3_EMERGENCY_SKIPPED_STAGES,
            )
            self.assertEqual(
                validated["decision"],
                "allow_experimental_smoke_training_and_fixed_103_only",
            )
            self.assertTrue(
                validated["deployment_rule"]["original_e_fallback_required"]
            )

    def test_any_bound_input_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in (
                "component_manifest",
                "route_config",
                "valid_mask_manifest",
                "calibration",
                "original_e",
                "fix_v2_failure",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                paths[name] = path
            with patch(
                "custom_nnunet.met_aug_fix_v3_emergency.ORIGINAL_E_SHA256",
                sha256_file(paths["original_e"]),
            ), patch(
                "custom_nnunet.met_aug_fix_v3_emergency.FIX_V2_FAILURE_SHA256",
                sha256_file(paths["fix_v2_failure"]),
            ):
                decision_path = root / "decision.json"
                decision_path.write_text(
                    make_fix_v3_emergency_decision(**paths),
                    encoding="utf-8",
                )
                paths["route_config"].write_text("drift", encoding="utf-8")

                with self.assertRaises(MetAugContractError):
                    validate_fix_v3_emergency_decision(decision_path, **paths)


if __name__ == "__main__":
    unittest.main()
