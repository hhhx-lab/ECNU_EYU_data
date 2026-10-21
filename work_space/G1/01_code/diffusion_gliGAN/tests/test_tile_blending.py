#!/usr/bin/env python3
"""Regression tests for large-lesion tile blending."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.infer import generate_from_label


class TileBlendingTests(unittest.TestCase):
    def test_constant_tiles_remain_constant_after_inner_blending(self) -> None:
        original_sampler = generate_from_label.sample_tumour_diffusion_full

        def constant_sampler(*, spatial_size, **_kwargs):
            return torch.ones((1, 1, *spatial_size), dtype=torch.float32)

        generate_from_label.sample_tumour_diffusion_full = constant_sampler
        try:
            label = np.ones((1, 12, 12, 12), dtype=np.float32)
            generated, weights = generate_from_label._tile_generate_lesion(
                label_mc=label,
                coords=(0, 12, 0, 12, 0, 12),
                crop_size=8,
                model=None,
                spatial_size=(8, 8, 8),
                device="cpu",
                sample_kwargs={},
            )
        finally:
            generate_from_label.sample_tumour_diffusion_full = original_sampler

        self.assertTrue(np.all(weights > 0))
        np.testing.assert_allclose(generated, 1.0, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
