#!/usr/bin/env python3
"""Tests for deterministic, patient-level Diffusion smoke selection."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_stratified_smoke_cases.py"
MODULE = None
if SCRIPT.is_file():
    spec = importlib.util.spec_from_file_location("select_stratified_smoke_cases", SCRIPT)
    MODULE = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(MODULE)


class SelectStratifiedSmokeCasesTests(unittest.TestCase):
    def test_selector_module_exists(self) -> None:
        self.assertIsNotNone(MODULE)

    @unittest.skipIf(MODULE is None, "selector module is not implemented")
    def test_selection_meets_risk_and_burden_contract(self) -> None:
        features = []
        for index in range(8):
            features.append(
                {
                    "patient_id": f"rc_{index:02d}",
                    "patient_group": f"group_{index % 4}",
                    "burden_mm3": float(50 + index * 80),
                    "has_rc": True,
                    "has_tiny_small": False,
                }
            )
        for index in range(8):
            features.append(
                {
                    "patient_id": f"small_{index:02d}",
                    "patient_group": f"group_{index % 4}",
                    "burden_mm3": float(30 + index * 100),
                    "has_rc": False,
                    "has_tiny_small": True,
                }
            )
        for index in range(8):
            features.append(
                {
                    "patient_id": f"regular_{index:02d}",
                    "patient_group": f"regular_group_{index % 4}",
                    "burden_mm3": float(360 + index * 5),
                    "has_rc": False,
                    "has_tiny_small": False,
                }
            )
        for index in range(10):
            features.append(
                {
                    "patient_id": f"filler_{index:02d}",
                    "patient_group": f"filler_group_{index % 2}",
                    "burden_mm3": float(800 + index * 100),
                    "has_rc": index % 2 == 0,
                    "has_tiny_small": index % 3 == 0,
                }
            )

        selected = MODULE.select_smoke_cases(
            features,
            case_count=20,
            min_rc=8,
            min_tiny_small=8,
            regular_count=4,
        )

        self.assertEqual(len(selected), 20)
        self.assertEqual(len({row["patient_id"] for row in selected}), 20)
        self.assertGreaterEqual(sum(bool(row["has_rc"]) for row in selected), 8)
        self.assertGreaterEqual(
            sum(bool(row["has_tiny_small"]) for row in selected), 8
        )
        self.assertGreaterEqual(
            sum(
                not bool(row["has_rc"]) and not bool(row["has_tiny_small"])
                for row in selected
            ),
            4,
        )
        self.assertEqual(
            {row["burden_stratum"] for row in selected}, {"low", "mid", "high"}
        )

    @unittest.skipIf(MODULE is None, "selector module is not implemented")
    def test_filter_keeps_every_lesion_row_for_selected_patients(self) -> None:
        lesion_rows = [
            {"patient_id": "case_a", "lesion_id": "a_0"},
            {"patient_id": "case_a", "lesion_id": "a_1"},
            {"patient_id": "case_b", "lesion_id": "b_0"},
            {"patient_id": "case_c", "lesion_id": "c_0"},
        ]

        selected_rows = MODULE.filter_lesion_rows(
            lesion_rows, {"case_a", "case_c"}
        )

        self.assertEqual(
            [row["lesion_id"] for row in selected_rows], ["a_0", "a_1", "c_0"]
        )


if __name__ == "__main__":
    unittest.main()
