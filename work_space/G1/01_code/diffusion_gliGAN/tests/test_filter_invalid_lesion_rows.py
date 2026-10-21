#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "filter_invalid_lesion_rows.py"
SPEC = importlib.util.spec_from_file_location("filter_invalid_lesion_rows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row_for_bbox(lower: int = 30, upper: int = 35) -> dict[str, str]:
    return {
        "x_extreme_min": str(lower),
        "x_extreme_max": str(upper),
        "y_extreme_min": str(lower),
        "y_extreme_max": str(upper),
        "z_extreme_min": str(lower),
        "z_extreme_max": str(upper),
    }


class LesionScanContentTest(unittest.TestCase):
    def test_bbox_signal_is_present_in_every_allowed_crop(self):
        volume = np.zeros((96, 96, 96), dtype=np.int16)
        volume[32, 32, 32] = 1
        minimum, _ = MODULE.minimum_possible_crop_nonzero(
            volume, row_for_bbox(), target_size=64)
        self.assertEqual(minimum, 1)

    def test_dark_bbox_with_surrounding_brain_is_not_rejected(self):
        volume = np.ones((96, 96, 96), dtype=np.int16)
        volume[30:35, 30:35, 30:35] = 0
        minimum, _ = MODULE.minimum_possible_crop_nonzero(
            volume, row_for_bbox(), target_size=64)
        self.assertGreater(minimum, 0)

    def test_background_component_can_produce_zero_crop(self):
        volume = np.zeros((96, 96, 96), dtype=np.int16)
        minimum, _ = MODULE.minimum_possible_crop_nonzero(
            volume, row_for_bbox(), target_size=64)
        self.assertEqual(minimum, 0)


if __name__ == "__main__":
    unittest.main()
