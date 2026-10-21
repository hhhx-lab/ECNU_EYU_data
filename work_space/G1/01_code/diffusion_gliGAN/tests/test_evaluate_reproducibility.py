#!/usr/bin/env python3
"""Regression tests for deterministic Diffusion evaluation sampling."""

from __future__ import annotations

import unittest

from src.infer import evaluate_generation


class EvaluateReproducibilityTests(unittest.TestCase):
    def test_case_seed_helper_exists(self) -> None:
        self.assertTrue(hasattr(evaluate_generation, "_derive_case_seed"))

    @unittest.skipUnless(
        hasattr(evaluate_generation, "_derive_case_seed"),
        "case seed helper is not implemented",
    )
    def test_case_seed_is_stable_and_modality_specific(self) -> None:
        first = evaluate_generation._derive_case_seed(20260720, "00001-000", "t1c")
        repeated = evaluate_generation._derive_case_seed(
            20260720, "00001-000", "t1c"
        )
        other_modality = evaluate_generation._derive_case_seed(
            20260720, "00001-000", "t2w"
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_modality)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)


if __name__ == "__main__":
    unittest.main()
