from __future__ import annotations

import os
import random
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from custom_nnunet.nnUNetTrainerBraTS2026RCA1CompletionFineTune import (
    nnUNetTrainerBraTS2026RCA1CompletionFineTune,
)
from custom_nnunet.nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune,
)
from custom_nnunet.nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
)
from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune,
)
from custom_nnunet.met_aug_gate import validate_route_a_training_contract
from custom_nnunet.met_aug_paired_training import (
    configure_paired_training_runtime,
    seed_paired_training_epoch,
)
from custom_nnunet.nnUNetTrainerBraTS2026RC import nnUNetTrainerBraTS2026RC
from custom_nnunet.small_lesion_variants import FocalCrossEntropyLoss


def baseline_architecture() -> dict:
    return {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "kernel_sizes": [[3, 3, 3]] * 6,
        "strides": [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2],
        ],
        "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "conv_bias": True,
    }


class SmallLesionTrainerTests(unittest.TestCase):
    def test_a1_trainer_passes_a_five_stage_copy_to_the_network_builder(self):
        configuration = types.SimpleNamespace(
            network_arch_class_name="PlainConvUNet",
            network_arch_init_kwargs=baseline_architecture(),
            network_arch_init_kwargs_req_import=[],
        )
        sentinel = object()
        with patch(
            "custom_nnunet.small_lesion_trainer_mixins.get_network_from_plans",
            return_value=sentinel,
        ) as builder:
            actual = nnUNetTrainerBraTS2026RCA1CompletionFineTune.build_network_architecture(
                object(), configuration, 4, 5, True
            )

        self.assertIs(actual, sentinel)
        arch_kwargs = builder.call_args.args[1]
        self.assertEqual(arch_kwargs["n_stages"], 5)
        self.assertEqual(len(arch_kwargs["n_conv_per_stage_decoder"]), 4)
        self.assertEqual(configuration.network_arch_init_kwargs["n_stages"], 6)

    def test_a1_focal_trainer_uses_four_deep_supervision_outputs(self):
        trainer = nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune.__new__(
            nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune
        )
        trainer.enable_deep_supervision = True
        trainer.configuration_manager = types.SimpleNamespace(
            pool_op_kernel_sizes=baseline_architecture()["strides"]
        )

        self.assertEqual(len(trainer._get_deep_supervision_scales()), 4)

    def test_focal_trainer_keeps_rc_weight_and_replaces_only_cross_entropy(self):
        trainer = nnUNetTrainerBraTS2026RCFocalCompletionFineTune.__new__(
            nnUNetTrainerBraTS2026RCFocalCompletionFineTune
        )
        trainer.device = torch.device("cpu")
        trainer.configuration_manager = types.SimpleNamespace(batch_dice=False)
        trainer.is_ddp = False
        trainer.label_manager = types.SimpleNamespace(ignore_label=None)
        trainer.enable_deep_supervision = False
        trainer._do_i_compile = lambda: False

        with patch.dict(os.environ, {"S2_FOCAL_GAMMA": "2.0"}, clear=False):
            loss = trainer._build_loss()

        self.assertIsInstance(loss.ce, FocalCrossEntropyLoss)
        self.assertEqual(loss.ce.gamma, 2.0)
        torch.testing.assert_close(
            loss.ce.weight,
            torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0]),
        )
        self.assertEqual(loss.weight_ce, 1)
        self.assertEqual(loss.weight_dice, 1)

    def test_candidate_trainers_have_unique_result_folder_names(self):
        names = {
            nnUNetTrainerBraTS2026RCA1CompletionFineTune.__name__,
            nnUNetTrainerBraTS2026RCFocalCompletionFineTune.__name__,
            nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune.__name__,
            nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.__name__,
        }
        self.assertEqual(len(names), 4)

    def test_route_a_second_stage_training_budget_is_frozen(self):
        validate_route_a_training_contract(
            num_epochs=200,
            initial_lr=0.001,
            save_every=25,
            focal_gamma=2.0,
        )
        with self.assertRaisesRegex(RuntimeError, "training contract drifted"):
            validate_route_a_training_contract(
                num_epochs=199,
                initial_lr=0.001,
                save_every=25,
                focal_gamma=2.0,
            )
        with self.assertRaisesRegex(RuntimeError, "torch_compile"):
            validate_route_a_training_contract(
                num_epochs=200,
                initial_lr=0.001,
                save_every=25,
                focal_gamma=2.0,
                torch_compile=True,
            )

    def test_paired_trainer_contract_skips_the_phantom_train_warmup(self):
        trainer = nnUNetTrainerBraTS2026RC.__new__(nnUNetTrainerBraTS2026RC)
        trainer.skip_training_dataloader_warmup = True
        train = iter(["phantom-train-batch"])
        validation = iter(["validation-health-check"])

        trainer._prime_dataloaders(train, validation)

        self.assertEqual(next(train), "phantom-train-batch")
        with self.assertRaises(StopIteration):
            next(validation)

    def test_paired_control_mixin_locks_single_thread_and_zero_train_warmup(self):
        trainer = nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.__new__(
            nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune
        )
        with patch.object(nnUNetTrainerBraTS2026RCFocalCompletionFineTune, "__init__", return_value=None):
            trainer.__init__()
        self.assertTrue(trainer.requires_single_threaded_augmentation)
        self.assertTrue(trainer.skip_training_dataloader_warmup)

    def test_paired_epoch_seed_reproduces_python_numpy_and_torch_streams(self):
        seed_paired_training_epoch(20260724, 7)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        seed_paired_training_epoch(20260724, 7)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_paired_runtime_rejects_seed_drift_and_overrides_cudnn_defaults(self):
        with patch.dict(os.environ, {"S2_PAIRED_TRAINING_SEED": "9"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "training seed drifted"):
                configure_paired_training_runtime()
        with patch.dict(os.environ, {"S2_PAIRED_TRAINING_SEED": "20260724"}, clear=False):
            original_deterministic = torch.backends.cudnn.deterministic
            original_benchmark = torch.backends.cudnn.benchmark
            try:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                self.assertEqual(configure_paired_training_runtime(), 20260724)
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cudnn.benchmark)
            finally:
                torch.backends.cudnn.deterministic = original_deterministic
                torch.backends.cudnn.benchmark = original_benchmark


if __name__ == "__main__":
    unittest.main()
