"""Shared reproducibility controls for the Route A matched training pair."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

try:
    from .met_aug_core import MetAugContractError
    from .met_aug_gate import ROUTE_A_TRAINING_CONTRACT, validate_route_a_training_contract
except ImportError:
    from met_aug_core import MetAugContractError  # type: ignore
    from met_aug_gate import ROUTE_A_TRAINING_CONTRACT, validate_route_a_training_contract  # type: ignore


def _seed_all(seed: int) -> None:
    bounded = int(seed)
    random.seed(bounded)
    np.random.seed(bounded % (2**32 - 1))
    torch.manual_seed(bounded)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bounded)


def configure_paired_training_runtime() -> int:
    """Apply the frozen seed and override nnU-Net's nondeterministic cuDNN defaults."""
    expected_seed = int(ROUTE_A_TRAINING_CONTRACT["training_seed"])
    configured_seed = int(os.environ.get("S2_PAIRED_TRAINING_SEED", str(expected_seed)))
    if configured_seed != expected_seed:
        raise MetAugContractError(
            "Route A/control training seed drifted: "
            f"observed={configured_seed}, expected={expected_seed}"
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _seed_all(configured_seed)
    return configured_seed


def seed_paired_training_epoch(base_seed: int, epoch: int) -> int:
    """Make resume-at-epoch reproduce the same crop and standard augmentation stream."""
    epoch_seed = int(base_seed) + int(epoch)
    _seed_all(epoch_seed)
    return epoch_seed


class MetAugPairedSecondStageMixin:
    """Keep Route A and its p=0 control on an identical deterministic budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_single_threaded_augmentation = True
        self.skip_training_dataloader_warmup = True
        self._paired_training_seed = int(ROUTE_A_TRAINING_CONTRACT["training_seed"])
        self._paired_epoch_seed = None

    def _prepare_paired_second_stage(self) -> None:
        """Hook executed after initialization and before dataloader construction."""

    def _before_paired_epoch(self) -> None:
        """Hook executed after epoch seeding and before the base epoch callback."""

    def on_train_start(self):
        self._paired_training_seed = configure_paired_training_runtime()
        focal_gamma = float(os.environ.get("S2_FOCAL_GAMMA", "2.0"))
        validate_route_a_training_contract(
            num_epochs=self.num_epochs,
            initial_lr=self.initial_lr,
            save_every=self.save_every,
            focal_gamma=focal_gamma,
            training_seed=self._paired_training_seed,
            augmentation_workers=0,
            discarded_train_warmup_batches=0,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            torch_compile=self._do_i_compile(),
        )
        if self.is_ddp:
            raise MetAugContractError("Route A matched pair requires one GPU")
        if not self.was_initialized:
            self.initialize()
        self._prepare_paired_second_stage()
        self.print_to_log_file(
            "MET_AUG_PAIRED_RUNTIME_PASS "
            f"seed={self._paired_training_seed} workers=0 train_warmup_batches=0 "
            "cudnn_deterministic=1 cudnn_benchmark=0 torch_compile=0"
        )
        super().on_train_start()

    def on_train_epoch_start(self):
        self._paired_epoch_seed = seed_paired_training_epoch(
            self._paired_training_seed,
            self.current_epoch,
        )
        self._before_paired_epoch()
        super().on_train_epoch_start()
