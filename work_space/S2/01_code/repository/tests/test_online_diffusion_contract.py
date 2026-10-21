from __future__ import annotations

import unittest

import numpy as np

from custom_nnunet.online_diffusion_contract import (
    g1_to_s2_layout,
    s2_to_g1_layout,
)


class OnlineDiffusionContractTests(unittest.TestCase):
    def test_channel_and_axis_round_trip(self) -> None:
        image = np.zeros((4, 3, 4, 5), dtype=np.float32)
        for channel in range(4):
            image[channel] = channel * 1000 + np.arange(60).reshape(3, 4, 5)
        segmentation = np.arange(60, dtype=np.int16).reshape(1, 3, 4, 5) % 5

        g1_image, g1_seg = s2_to_g1_layout(image, segmentation)
        self.assertEqual(g1_image.shape, (4, 5, 4, 3))
        self.assertTrue(np.array_equal(g1_image[0], image[1].transpose(2, 1, 0)))
        self.assertTrue(np.array_equal(g1_image[1], image[0].transpose(2, 1, 0)))
        self.assertTrue(np.array_equal(g1_seg[0], segmentation[0].transpose(2, 1, 0)))

        restored_image, restored_seg = g1_to_s2_layout(g1_image, g1_seg)
        self.assertTrue(np.array_equal(restored_image, image))
        self.assertTrue(np.array_equal(restored_seg, segmentation))

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            s2_to_g1_layout(
                np.zeros((3, 8, 8, 8), dtype=np.float32),
                np.zeros((1, 8, 8, 8), dtype=np.int16),
            )


if __name__ == "__main__":
    unittest.main()
