"""Minimal network builder for frozen nnUNetTrainerBraTS2026RC inference."""

from torch import nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class nnUNetTrainerBraTS2026RC:
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
