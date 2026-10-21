#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "runtime" / "disjoint_microf1.py"
SPEC = importlib.util.spec_from_file_location("disjoint_microf1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DisjointMicroF1Tests(unittest.TestCase):
    def test_exact_small_and_large_matches_are_additive(self):
        reference = np.zeros((12, 12, 12), dtype=np.uint8)
        prediction = np.zeros_like(reference)
        reference[0:2, 0:2, 0:2] = 3
        prediction[0:2, 0:2, 0:2] = 3
        reference[5:9, 5:9, 5:9] = 3
        prediction[5:9, 5:9, 5:9] = 3
        evaluation = MODULE.evaluate_region(reference, prediction, (3,), 0.2)
        counts = MODULE.classify_counts(evaluation, 50)
        self.assertEqual(counts["all"], {"tp": 2, "fp": 0, "fn": 0})
        self.assertEqual(counts["small"], {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(counts["large"], {"tp": 1, "fp": 0, "fn": 0})

    def test_low_dsc_overlap_is_one_fn_plus_one_fp(self):
        reference = np.zeros((20, 4, 4), dtype=np.uint8)
        prediction = np.zeros_like(reference)
        reference[0:10, 0, 0] = 3
        prediction[9:19, 0, 0] = 3
        evaluation = MODULE.evaluate_region(reference, prediction, (3,), 0.2)
        counts = MODULE.classify_counts(evaluation, 27)
        self.assertEqual(counts["all"], {"tp": 0, "fp": 1, "fn": 1})

    def test_tp_uses_reference_size_and_fp_is_not_reused(self):
        reference = np.zeros((12, 12, 12), dtype=np.uint8)
        prediction = np.zeros_like(reference)
        reference[3:6, 3:6, 3:6] = 3
        prediction[2:7, 2:7, 2:7] = 3
        evaluation = MODULE.evaluate_region(reference, prediction, (3,), 0.2)
        counts = MODULE.classify_counts(evaluation, 50)
        self.assertEqual(counts["small"], {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(counts["large"], {"tp": 0, "fp": 0, "fn": 0})

    def test_threshold_boundary_is_large(self):
        reference = np.zeros((60, 3, 3), dtype=np.uint8)
        prediction = np.zeros_like(reference)
        reference[0:50, 0, 0] = 3
        prediction[0:50, 0, 0] = 3
        evaluation = MODULE.evaluate_region(reference, prediction, (3,), 0.2)
        counts = MODULE.classify_counts(evaluation, 50)
        self.assertEqual(counts["large"]["tp"], 1)
        self.assertEqual(counts["small"]["tp"], 0)


if __name__ == "__main__":
    unittest.main()
