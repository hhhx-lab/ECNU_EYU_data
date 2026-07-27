#!/usr/bin/env python3
"""Tests for training-aligned whole-brain reference support construction."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from src.infer import evaluate_generation


class ReferenceSupportTests(unittest.TestCase):
    def test_reference_helpers_exist(self) -> None:
        self.assertTrue(hasattr(evaluate_generation, "_extract_reference_content"))
        self.assertTrue(hasattr(evaluate_generation, "_tile_reference_content"))
        self.assertTrue(hasattr(evaluate_generation, "_save_array_like"))

    @unittest.skipUnless(
        hasattr(evaluate_generation, "_save_array_like"),
        "NIfTI save helper is not implemented",
    )
    def test_saved_support_volume_preserves_geometry(self) -> None:
        affine = np.array(
            [
                [-1.0, 0.0, 0.0, 31.0],
                [0.0, 1.0, 0.0, -14.0],
                [0.0, 0.0, 1.2, -12.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        reference = nib.Nifti1Image(np.zeros((8, 7, 6), dtype=np.float32), affine)
        values = np.ones((8, 7, 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "support.nii.gz"
            evaluate_generation._save_array_like(values, reference, output, np.float32)
            written = nib.load(output)

        self.assertEqual(written.shape, reference.shape)
        np.testing.assert_allclose(written.affine, affine)
        np.testing.assert_allclose(written.header.get_zooms()[:3], (1.0, 1.0, 1.2))

    @unittest.skipUnless(
        hasattr(evaluate_generation, "_extract_reference_content"),
        "reference content helper is not implemented",
    )
    def test_single_crop_reference_is_zscore_invariant(self) -> None:
        scan = np.arange(12**3, dtype=np.float32).reshape(12, 12, 12) + 1.0
        transformed = scan * 3.5 + 17.0
        kwargs = {
            "coords": (2, 10, 2, 10, 2, 10),
            "content_shape": (8, 8, 8),
            "crop_size": 12,
        }

        reference = evaluate_generation._extract_reference_content(scan, **kwargs)
        transformed_reference = evaluate_generation._extract_reference_content(
            transformed, **kwargs
        )

        self.assertEqual(reference.shape, (8, 8, 8))
        self.assertAlmostEqual(float(reference.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(reference.std()), 1.0, places=5)
        np.testing.assert_allclose(reference, transformed_reference, atol=2e-5)

    @unittest.skipUnless(
        hasattr(evaluate_generation, "_tile_reference_content"),
        "tile reference helper is not implemented",
    )
    def test_tiled_reference_restores_window_shape_and_is_zscore_invariant(self) -> None:
        scan = np.arange(14**3, dtype=np.float32).reshape(14, 14, 14) + 1.0
        transformed = scan * 2.0 + 11.0
        kwargs = {"coords": (1, 13, 1, 13, 1, 13), "crop_size": 8}

        reference, weights = evaluate_generation._tile_reference_content(scan, **kwargs)
        transformed_reference, transformed_weights = (
            evaluate_generation._tile_reference_content(transformed, **kwargs)
        )

        self.assertEqual(reference.shape, (12, 12, 12))
        self.assertTrue(np.all(weights > 0))
        np.testing.assert_allclose(weights, transformed_weights, atol=0.0)
        np.testing.assert_allclose(reference, transformed_reference, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
