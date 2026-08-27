"""Experimental Fix-v3 trainer with an explicit non-Gate authorization contract."""

from __future__ import annotations

try:
    from .met_aug_core import MetAugContractError
    from .met_aug_fix_v3_emergency import validate_fix_v3_emergency_decision
    from .nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune import (
        _required_env,
        nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune,
    )
except ImportError:
    from met_aug_core import MetAugContractError  # type: ignore
    from met_aug_fix_v3_emergency import validate_fix_v3_emergency_decision  # type: ignore
    from nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune import (  # type: ignore
        _required_env,
        nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune,
    )


class nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune(
    nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune
):
    authorization_env_name = "S2_MET_AUG_EMERGENCY_DECISION"
    authorization_kind = "fix_v3_emergency_unvalidated"

    def _validate_met_aug_authorization(
        self,
        authorization_path,
        *,
        manifest_path,
        manifest,
        config,
        config_path,
        valid_mask_manifest,
        calibration_path,
        selection_path,
        gate_path,
        g1_code_dir,
    ):
        del manifest, selection_path, gate_path, g1_code_dir
        if config.fix_v3 is None or config.fix_v2 is not None:
            raise MetAugContractError(
                "Fix-v3 emergency trainer requires exactly one schema-5 Fix-v3 policy"
            )
        if calibration_path is None:
            raise MetAugContractError("Fix-v3 emergency trainer lacks calibration")
        original_e = _required_env("S2_MET_AUG_ORIGINAL_E_CHECKPOINT")
        fix_v2_failure = _required_env("S2_MET_AUG_FIX_V2_FAILURE_AUDIT")
        return validate_fix_v3_emergency_decision(
            authorization_path,
            component_manifest=manifest_path,
            route_config=config_path,
            valid_mask_manifest=valid_mask_manifest,
            calibration=calibration_path,
            original_e=original_e,
            fix_v2_failure=fix_v2_failure,
        )
