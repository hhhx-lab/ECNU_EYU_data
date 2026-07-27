"""Composable nnU-Net trainer behavior for the S2 ablation candidates."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

try:
    from .small_lesion_variants import (
        FocalCrossEntropyLoss,
        a1_deep_supervision_scales,
        load_matching_pretrained_weights,
        truncate_architecture_kwargs_for_a1,
    )
except ImportError:
    from small_lesion_variants import (
        FocalCrossEntropyLoss,
        a1_deep_supervision_scales,
        load_matching_pretrained_weights,
        truncate_architecture_kwargs_for_a1,
    )


class A1ArchitectureMixin:
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        architecture_kwargs = truncate_architecture_kwargs_for_a1(
            configuration_manager.network_arch_init_kwargs
        )
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            architecture_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

    def _get_deep_supervision_scales(self):
        if not self.enable_deep_supervision:
            return None
        return a1_deep_supervision_scales(
            self.configuration_manager.pool_op_kernel_sizes
        )

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        source = os.environ.get("S2_PARTIAL_PRETRAINED_WEIGHTS", "").strip()
        if self.current_epoch == 0 and not getattr(
            self, "_a1_partial_warmstart_loaded", False
        ):
            if not source:
                raise RuntimeError(
                    "A-1 training requires S2_PARTIAL_PRETRAINED_WEIGHTS on a fresh run"
                )
            audit_path = Path(self.output_folder) / "partial_warmstart_audit.json"
            audit = load_matching_pretrained_weights(
                self.network,
                source,
                audit_path=audit_path,
            )
            self._a1_partial_warmstart_loaded = True
            self.print_to_log_file(
                "A1_PARTIAL_WARMSTART_PASS "
                f"loaded={len(audit['loaded_keys'])} "
                f"fraction={audit['loaded_parameter_fraction']:.6f} "
                f"audit={audit_path}"
            )
        super().on_train_start()


class FocalLossMixin:
    def _build_loss(self):
        gamma = float(os.environ.get("S2_FOCAL_GAMMA", "2.0"))
        if gamma < 0:
            raise ValueError(f"S2_FOCAL_GAMMA must be non-negative, got {gamma}")
        ce_kwargs = {
            "weight": torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 3.0], device=self.device
            )
        }
        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        loss.ce = FocalCrossEntropyLoss(
            weight=ce_kwargs["weight"],
            ignore_index=(
                self.label_manager.ignore_label
                if self.label_manager.ignore_label is not None
                else -100
            ),
            gamma=gamma,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**index) for index in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
