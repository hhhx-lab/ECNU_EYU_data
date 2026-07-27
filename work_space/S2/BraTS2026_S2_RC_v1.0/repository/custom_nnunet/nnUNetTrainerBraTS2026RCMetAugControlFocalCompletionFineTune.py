"""Matched p=0 control for the E warm-start Route A second stage."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from .met_aug_core import MetAugContractError, write_or_validate_immutable_json
    from .met_aug_gate import ROUTE_A_TRAINING_CONTRACT
    from .met_aug_paired_training import MetAugPairedSecondStageMixin
    from .nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
except ImportError:
    from met_aug_core import MetAugContractError, write_or_validate_immutable_json
    from met_aug_gate import ROUTE_A_TRAINING_CONTRACT
    from met_aug_paired_training import MetAugPairedSecondStageMixin
    from nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )


class nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune(
    MetAugPairedSecondStageMixin,
    nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
):
    """Continue E with the same budget as Route A while keeping p_select=0."""

    def _prepare_paired_second_stage(self):
        if os.environ.get("S2_MET_AUG_ENABLE", "0") != "0":
            raise MetAugContractError("MET-AUG matched control must keep S2_MET_AUG_ENABLE=0")
        provenance = {
            "experiment": "MET-AUG-A-matched-control",
            "augmentation_probability": 0.0,
            "training_contract": ROUTE_A_TRAINING_CONTRACT,
            "base_trainer": "nnUNetTrainerBraTS2026RCFocalCompletionFineTune",
        }
        provenance_path = Path(self.output_folder) / "met_aug_control_provenance.json"
        state = write_or_validate_immutable_json(
            provenance_path,
            provenance,
            label="MET-AUG matched-control provenance",
        )
        self.print_to_log_file(f"MET_AUG_MATCHED_CONTROL_READY provenance={state}")
