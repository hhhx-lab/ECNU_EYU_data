from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from custom_nnunet.small_lesion_variants import (
    FocalCrossEntropyLoss,
    a1_deep_supervision_scales,
    load_matching_pretrained_weights,
    truncate_architecture_kwargs_for_a1,
)


class A1ArchitectureTests(unittest.TestCase):
    @staticmethod
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

    def test_a1_truncates_the_encoder_and_decoder_to_five_stages(self):
        original = self.baseline_architecture()
        original_copy = copy.deepcopy(original)

        actual = truncate_architecture_kwargs_for_a1(original)

        self.assertEqual(actual["n_stages"], 5)
        self.assertEqual(actual["features_per_stage"], [32, 64, 128, 256, 320])
        self.assertEqual(actual["strides"], [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ])
        self.assertEqual(actual["n_conv_per_stage_decoder"], [2, 2, 2, 2])
        self.assertEqual(original, original_copy)

    def test_a1_rejects_a_plan_that_cannot_supply_five_stages(self):
        architecture = self.baseline_architecture()
        architecture["features_per_stage"] = [32, 64, 128, 256]

        with self.assertRaisesRegex(ValueError, "features_per_stage"):
            truncate_architecture_kwargs_for_a1(architecture)

    def test_a1_deep_supervision_has_one_scale_per_decoder_output(self):
        scales = a1_deep_supervision_scales(self.baseline_architecture()["strides"])

        self.assertEqual(scales, [
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.125, 0.125, 0.125],
        ])


class FocalCrossEntropyLossTests(unittest.TestCase):
    def test_gamma_zero_matches_weighted_cross_entropy(self):
        logits = torch.tensor(
            [[[[2.0, -1.0]], [[-0.5, 1.5]], [[0.2, 0.1]]]],
            dtype=torch.float32,
        )
        target = torch.tensor([[[[0, 1]]]], dtype=torch.long)
        class_weights = torch.tensor([1.0, 2.0, 3.0])

        actual = FocalCrossEntropyLoss(weight=class_weights, gamma=0.0)(logits, target)
        expected = F.cross_entropy(logits, target[:, 0], weight=class_weights)

        torch.testing.assert_close(actual, expected)

    def test_focal_downweights_an_easy_voxel_more_than_a_hard_voxel(self):
        loss = FocalCrossEntropyLoss(gamma=2.0, reduction="none")
        easy_logits = torch.tensor([[[[5.0]], [[-5.0]]]])
        hard_logits = torch.tensor([[[[0.1]], [[0.0]]]])
        target = torch.zeros((1, 1, 1, 1), dtype=torch.long)

        easy_ratio = loss(easy_logits, target) / F.cross_entropy(
            easy_logits, target[:, 0], reduction="none"
        )
        hard_ratio = loss(hard_logits, target) / F.cross_entropy(
            hard_logits, target[:, 0], reduction="none"
        )

        self.assertLess(easy_ratio.item(), hard_ratio.item())
        self.assertLess(easy_ratio.item(), 1e-6)

    def test_ignore_label_does_not_contribute_to_the_mean(self):
        logits = torch.tensor(
            [[[[3.0, -2.0]], [[-1.0, 2.0]]]],
            dtype=torch.float32,
        )
        target = torch.tensor([[[[0, -1]]]], dtype=torch.long)

        actual = FocalCrossEntropyLoss(gamma=2.0, ignore_index=-1)(logits, target)
        expected = FocalCrossEntropyLoss(gamma=2.0)(logits[..., :1], target[..., :1])

        torch.testing.assert_close(actual, expected)


class PartialWarmStartTests(unittest.TestCase):
    class TinyNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 3)
            self.decoder = torch.nn.Linear(3, 2)
            self.seg_layers = torch.nn.Linear(2, 2)

    def test_loads_only_matching_non_segmentation_parameters_and_writes_audit(self):
        network = self.TinyNetwork()
        original_decoder = network.decoder.weight.detach().clone()
        source = {
            "encoder.weight": torch.full_like(network.encoder.weight, 7.0),
            "encoder.bias": torch.full_like(network.encoder.bias, 5.0),
            "decoder.weight": torch.zeros((4, 4)),
            "seg_layers.weight": torch.full_like(network.seg_layers.weight, 9.0),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint_final.pth"
            audit_path = root / "partial_warmstart_audit.json"
            torch.save({"network_weights": source}, checkpoint)

            audit = load_matching_pretrained_weights(
                network,
                checkpoint,
                audit_path=audit_path,
            )

            torch.testing.assert_close(
                network.encoder.weight,
                torch.full_like(network.encoder.weight, 7.0),
            )
            torch.testing.assert_close(network.decoder.weight, original_decoder)
            self.assertEqual(
                audit["loaded_keys"], ["encoder.bias", "encoder.weight"]
            )
            self.assertEqual(audit["skipped_shape_keys"], ["decoder.weight"])
            self.assertEqual(audit["skipped_segmentation_keys"], ["seg_layers.weight"])
            self.assertEqual(len(audit["source_sha256"]), 64)
            self.assertEqual(
                json.loads(audit_path.read_text(encoding="utf-8")), audit
            )

    def test_rejects_a_checkpoint_without_matching_trainable_parameters(self):
        network = self.TinyNetwork()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_final.pth"
            torch.save(
                {"network_weights": {"unrelated.weight": torch.ones(1)}},
                checkpoint,
            )

            with self.assertRaisesRegex(RuntimeError, "no compatible parameters"):
                load_matching_pretrained_weights(network, checkpoint)

if __name__ == "__main__":
    unittest.main()
