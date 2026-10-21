"""Independent 125-to-175 convergence extension for selected MET-AUG routes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

try:
    from .met_aug_core import MetAugContractError, sha256_file, write_or_validate_immutable_json
    from .met_aug_paired_training import configure_paired_training_runtime
    from .met_aug_screening import ScreenMetAugEngine, load_screening_policy
    from .met_aug_transform import MetAugRouteATransform
    from .nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
    from .nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune,
    )
    from .nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune,
    )
except ImportError:
    from met_aug_core import MetAugContractError, sha256_file, write_or_validate_immutable_json  # type: ignore
    from met_aug_paired_training import configure_paired_training_runtime  # type: ignore
    from met_aug_screening import ScreenMetAugEngine, load_screening_policy  # type: ignore
    from met_aug_transform import MetAugRouteATransform  # type: ignore
    from nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (  # type: ignore
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
    from nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune import (  # type: ignore
        nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune,
    )
    from nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune import (  # type: ignore
        nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune,
    )


SOURCE_EPOCH = 125
TARGET_EPOCH = 175
EXTENSION_EPOCHS = TARGET_EPOCH - SOURCE_EPOCH
EXTENSION_START_LR = 1.2965252773542109e-05
LR_POWER = 0.9


class ContinuousExtensionPolyLR:
    """Poly decay over epochs 125-174 without reviving the completed LR curve."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, current_epoch: int) -> None:
        offset = int(current_epoch) - SOURCE_EPOCH
        if offset < 0 or offset >= EXTENSION_EPOCHS:
            raise MetAugContractError(f"extension scheduler received epoch {current_epoch}")
        factor = math.pow(1.0 - offset / EXTENSION_EPOCHS, LR_POWER)
        lr = EXTENSION_START_LR * factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune(
    nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune
):
    """Resume a frozen Screen125 route and extend it to total epoch 175."""

    def __init__(self, *args, **kwargs):
        previous = {
            name: os.environ.get(name)
            for name in ("S2_SCREEN_EPOCHS", "S2_SCREEN_INITIAL_LR", "S2_SCREEN_SAVE_EVERY")
        }
        os.environ["S2_SCREEN_EPOCHS"] = "125"
        os.environ["S2_SCREEN_INITIAL_LR"] = "0.001"
        os.environ["S2_SCREEN_SAVE_EVERY"] = "25"
        try:
            super().__init__(*args, **kwargs)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.num_epochs = TARGET_EPOCH
        self.initial_lr = EXTENSION_START_LR
        self.save_every = 25

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            EXTENSION_START_LR,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        return optimizer, ContinuousExtensionPolyLR(optimizer)

    def _initialize_met_aug(self) -> None:
        if self._screening_policy is not None:
            return
        nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune._initialize_met_aug(self)
        assert self._met_aug_transform is not None
        base_engine = self._met_aug_transform.engine
        screen_config = Path(os.environ["S2_MET_AUG_SCREEN_CONFIG"]).expanduser().resolve()
        counts_value = os.environ.get("S2_MET_AUG_TARGET_LESION_COUNTS", "").strip()
        policy = load_screening_policy(
            screen_config,
            base_config=base_engine.config,
            manifest=base_engine.manifest,
            target_counts_path=counts_value or None,
        )
        source_plan = Path(os.environ["S2_MET_AUG_SCREEN_PLAN"]).expanduser().resolve()
        if sha256_file(source_plan) != policy.plan_sha256:
            raise MetAugContractError("source Screen125 plan SHA256 drifted")
        extension_plan = Path(os.environ["S2_MET_AUG_EXTENSION_PLAN"]).expanduser().resolve()
        source_checkpoint = Path(os.environ["S2_MET_AUG_EXTENSION_SOURCE_CHECKPOINT"]).expanduser().resolve()
        expected_source_sha = os.environ["S2_MET_AUG_EXTENSION_SOURCE_SHA256"]
        if sha256_file(source_checkpoint) != expected_source_sha:
            raise MetAugContractError("extension source checkpoint SHA256 drifted")
        self._met_aug_transform = MetAugRouteATransform(
            ScreenMetAugEngine(
                screening_policy=policy,
                manifest=base_engine.manifest,
                config=base_engine.config,
                backend=base_engine.backend,
                audit_sink=base_engine.audit_sink,
                candidate_processor=base_engine.candidate_processor,
            )
        )
        self._screening_policy = policy
        provenance = {
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "status": "started",
            "route_id": policy.route_id,
            "source_screen_config": str(screen_config),
            "source_screen_config_sha256": sha256_file(screen_config),
            "source_screen_plan": str(source_plan),
            "source_screen_plan_sha256": sha256_file(source_plan),
            "extension_plan": str(extension_plan),
            "extension_plan_sha256": sha256_file(extension_plan),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": expected_source_sha,
            "source_total_epoch": SOURCE_EPOCH,
            "target_total_epoch": TARGET_EPOCH,
            "extension_epochs": EXTENSION_EPOCHS,
            "candidate_total_epochs": [125, 150, 175],
            "extension_lr_policy": "continuous_poly_decay_from_epoch125_optimizer_lr",
            "extension_start_lr": EXTENSION_START_LR,
            "extension_lr_power": LR_POWER,
            "resume_network_state": True,
            "resume_optimizer_state": True,
            "resume_logging_state": True,
            "seed": 20260724,
        }
        write_or_validate_immutable_json(
            Path(self.output_folder) / "met_aug_extension_provenance.json",
            provenance,
            label="MET-AUG Screen175 extension provenance",
        )
        self.print_to_log_file(
            f"MET_AUG_SCREEN175_EXTENSION_INITIALIZED route={policy.route_id} "
            f"source_epoch={SOURCE_EPOCH} target_epoch={TARGET_EPOCH}"
        )

    def on_train_start(self):
        self._paired_training_seed = configure_paired_training_runtime()
        if self.is_ddp:
            raise MetAugContractError("Screen175 extension requires one GPU and non-DDP execution")
        if self._do_i_compile():
            raise MetAugContractError("Screen175 extension requires nnUNet_compile=0")
        if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            raise MetAugContractError("Screen175 extension deterministic cuDNN contract drifted")
        if not self.was_initialized:
            self.initialize()
        if int(self.current_epoch) != SOURCE_EPOCH:
            raise MetAugContractError(
                f"Screen175 extension must resume at epoch {SOURCE_EPOCH}, got {self.current_epoch}"
            )
        observed_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        if any(abs(value - EXTENSION_START_LR) > 1e-15 for value in observed_lrs):
            raise MetAugContractError(f"source optimizer LR drifted: {observed_lrs}")
        self._prepare_paired_second_stage()
        self.print_to_log_file(
            "MET_AUG_SCREEN175_EXTENSION_RUNTIME_PASS source_epoch=125 target_epoch=175 "
            f"extension_start_lr={EXTENSION_START_LR:.17g} seed=20260724 workers=0"
        )
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune.on_train_start(self)
