from __future__ import annotations

import unittest

import numpy as np

from custom_nnunet.met_aug_fix_v2_selection import (
    expanded_interval,
    reference_development_interval,
)


class FixV2SelectionTests(unittest.TestCase):
    def test_expanded_interval_contains_all_observations(self) -> None:
        values = [-2.0, -1.0, 0.5, 3.0]
        lower, upper = expanded_interval(values, label="example")
        self.assertLess(lower, min(values))
        self.assertGreater(upper, max(values))

    def test_reference_development_interval_uses_reference_tails_and_accepts(self) -> None:
        reference = np.linspace(-5.0, 5.0, 1001)
        lower, upper = reference_development_interval(
            reference,
            [-6.0, 4.0],
            label="example",
        )
        self.assertLess(lower, -6.0)
        self.assertGreater(upper, 4.99)

    def test_iou_interval_is_clamped(self) -> None:
        lower, upper = reference_development_interval(
            [0.0, 0.5, 1.0],
            [0.0, 1.0],
            label="iou",
            lower_limit=0.0,
            upper_limit=1.0,
        )
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)

    def test_nonfinite_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expanded_interval([1.0, float("nan")], label="bad")


if __name__ == "__main__":
    unittest.main()
