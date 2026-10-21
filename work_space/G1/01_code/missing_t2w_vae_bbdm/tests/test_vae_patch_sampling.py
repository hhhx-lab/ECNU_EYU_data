import unittest

import numpy as np

from vae_patch_sampling import (
    choose_tumor_component_center,
    crop_or_pad_around_center,
    sample_synchronized_patch,
)


class PatchSamplingTests(unittest.TestCase):
    def test_components_are_selected_uniformly_not_by_volume(self):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[2:4, 2:4, 2:4] = 1
        mask[15:27, 15:27, 15:27] = 1
        rng = np.random.default_rng(7)
        counts = {1: 0, 2: 0}
        for _ in range(2000):
            _, component_id = choose_tumor_component_center(mask, rng)
            counts[component_id] += 1
        fraction = counts[1] / sum(counts.values())
        self.assertGreater(fraction, 0.45)
        self.assertLess(fraction, 0.55)

    def test_tumor_and_brain_modes_follow_probability(self):
        base = np.ones((32, 32, 32), dtype=np.float32)
        seg = np.zeros_like(base, dtype=np.int16)
        seg[12:16, 12:16, 12:16] = 3
        rng = np.random.default_rng(11)
        tumor_count = 0
        for _ in range(1000):
            result = sample_synchronized_patch(
                [base] * 4, seg, (16, 16, 16), 0.8, rng
            )
            tumor_count += result["mode"] == "tumor"
        self.assertGreater(tumor_count / 1000, 0.75)
        self.assertLess(tumor_count / 1000, 0.85)

    def test_modalities_and_masks_share_the_exact_crop(self):
        marker = np.arange(32**3, dtype=np.float32).reshape(32, 32, 32)
        seg = np.zeros((32, 32, 32), dtype=np.int16)
        seg[20, 20, 20] = 3
        result = sample_synchronized_patch(
            [marker + offset for offset in range(4)],
            seg,
            (16, 16, 16),
            1.0,
            np.random.default_rng(3),
        )
        self.assertEqual(result["images"][0].shape, (16, 16, 16))
        for offset in range(1, 4):
            np.testing.assert_array_equal(
                result["images"][offset] - offset, result["images"][0]
            )
        self.assertEqual(result["seg"][8, 8, 8], 3)
        self.assertEqual(result["tumor_mask"][8, 8, 8], 1)

    def test_boundary_crop_is_padded_to_stable_shape(self):
        array = np.ones((8, 8, 8), dtype=np.float32)
        patch = crop_or_pad_around_center(array, (0, 0, 0), (8, 8, 8))
        self.assertEqual(patch.shape, (8, 8, 8))
        self.assertEqual(patch[4, 4, 4], 1)
        self.assertEqual(patch[0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
