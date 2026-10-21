"""Single-GPU 125-epoch exploratory screening trainer for MET-AUG B-F."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

try:
    from .met_aug_core import MetAugContractError, sha256_file, write_or_validate_immutable_json
    from .met_aug_paired_training import configure_paired_training_runtime, seed_paired_training_epoch
    from .met_aug_screening import ScreenMetAugEngine, load_screening_policy
    from .met_aug_transform import MetAugRouteATransform
    from .nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
    from .nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune,
    )
except ImportError:
    from met_aug_core import MetAugContractError, sha256_file, write_or_validate_immutable_json  # type: ignore
    from met_aug_paired_training import configure_paired_training_runtime, seed_paired_training_epoch  # type: ignore
    from met_aug_screening import ScreenMetAugEngine, load_screening_policy  # type: ignore
    from met_aug_transform import MetAugRouteATransform  # type: ignore
    from nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (  # type: ignore
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
    from nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune import (  # type: ignore
        nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune,
    )


class nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune(
    nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune
):
    """Use the validated Fix-v3 processor with one frozen B-F screening policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("S2_SCREEN_EPOCHS", "125"))
        self.initial_lr = float(os.environ.get("S2_SCREEN_INITIAL_LR", "0.001"))
        self.save_every = int(os.environ.get("S2_SCREEN_SAVE_EVERY", "25"))
        if self.num_epochs != 125 or self.initial_lr != 0.001 or self.save_every != 25:
            raise MetAugContractError("screening trainer requires 125 epochs, LR 0.001, save_every 25")
        self._screening_policy = None

    def _initialize_met_aug(self) -> None:
        if self._screening_policy is not None:
            return
        super()._initialize_met_aug()
        assert self._met_aug_transform is not None
        base_engine = self._met_aug_transform.engine
        screen_config_value = os.environ.get("S2_MET_AUG_SCREEN_CONFIG", "").strip()
        if not screen_config_value:
            raise MetAugContractError("missing S2_MET_AUG_SCREEN_CONFIG")
        screen_config = Path(screen_config_value).expanduser().resolve()
        counts_value = os.environ.get("S2_MET_AUG_TARGET_LESION_COUNTS", "").strip()
        policy = load_screening_policy(
            screen_config,
            base_config=base_engine.config,
            manifest=base_engine.manifest,
            target_counts_path=counts_value or None,
        )
        screen_engine = ScreenMetAugEngine(
            screening_policy=policy,
            manifest=base_engine.manifest,
            config=base_engine.config,
            backend=base_engine.backend,
            audit_sink=base_engine.audit_sink,
            candidate_processor=base_engine.candidate_processor,
        )
        self._met_aug_transform = MetAugRouteATransform(screen_engine)
        self._screening_policy = policy
        plan_path_value = os.environ.get("S2_MET_AUG_SCREEN_PLAN", "").strip()
        if not plan_path_value:
            raise MetAugContractError("missing S2_MET_AUG_SCREEN_PLAN")
        plan_path = Path(plan_path_value).expanduser().resolve()
        if sha256_file(plan_path) != policy.plan_sha256:
            raise MetAugContractError("screening plan SHA256 drifted")
        provenance = {
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "status": "started",
            "route_id": policy.route_id,
            "screen_config": str(screen_config),
            "screen_config_sha256": sha256_file(screen_config),
            "plan": str(plan_path),
            "plan_sha256": policy.plan_sha256,
            "base_route_config": str(base_engine.config.path),
            "base_route_config_sha256": policy.base_route_config_sha256,
            "epochs": 125,
            "save_every": 25,
            "checkpoint_epochs": [25, 50, 75, 100, 125],
            "initial_lr": 0.001,
            "training_seed": 20260724,
            "resume": False,
            "checkpoint_selection_scope": "development_validation_only",
        }
        state = write_or_validate_immutable_json(
            Path(self.output_folder) / "met_aug_screening_provenance.json",
            provenance,
            label="MET-AUG screening provenance",
        )
        self.print_to_log_file(
            f"MET_AUG_SCREENING_INITIALIZED route={policy.route_id} provenance={state}"
        )

    def on_train_start(self):
        self._paired_training_seed = configure_paired_training_runtime()
        if self.is_ddp:
            raise MetAugContractError("screening routes require one A100 and non-DDP execution")
        if self._do_i_compile():
            raise MetAugContractError("screening routes require nnUNet_compile=0")
        if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            raise MetAugContractError("screening deterministic cuDNN contract drifted")
        if not self.was_initialized:
            self.initialize()
        self._prepare_paired_second_stage()
        self.print_to_log_file(
            "MET_AUG_SCREENING_RUNTIME_PASS epochs=125 save_every=25 seed=20260724 "
            "workers=0 cudnn_deterministic=1 cudnn_benchmark=0 torch_compile=0"
        )
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune.on_train_start(self)

    def on_train_epoch_start(self):
        self._paired_epoch_seed = seed_paired_training_epoch(
            self._paired_training_seed,
            self.current_epoch,
        )
        self._before_paired_epoch()
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune.on_train_epoch_start(self)

    def on_epoch_end(self):
        super().on_epoch_end()
        completed_epoch = int(self.current_epoch)
        if completed_epoch % 25 == 0:
            # ``save_checkpoint`` serializes ``current_epoch + 1`` because the
            # base trainer normally calls it before incrementing the counter.
            self.current_epoch -= 1
            try:
                self.save_checkpoint(
                    str(Path(self.output_folder) / f"checkpoint_epoch_{completed_epoch:04d}.pth")
                )
            finally:
                self.current_epoch += 1
