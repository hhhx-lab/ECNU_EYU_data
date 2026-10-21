#!/usr/bin/env python3
"""Unit tests for final Diffusion checkpoint gate freezing."""

from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "code" / "g2_finalize_diffusion_gate.py"
SPEC = importlib.util.spec_from_file_location("g2_finalize_diffusion_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalizeDiffusionGateTests(unittest.TestCase):
    @staticmethod
    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_required_review_covers_all_mandatory_strata(self) -> None:
        rows = [
            {
                "source_case_id": "rc",
                "has_rc": "True",
                "tiny_count": "0",
                "large_count": "0",
                "artifact_flag_count": "0",
                "artifact_flags": "",
                "min_tumour_ssim": "0.9",
            },
            {
                "source_case_id": "tiny",
                "has_rc": "False",
                "tiny_count": "1",
                "large_count": "0",
                "artifact_flag_count": "0",
                "artifact_flags": "",
                "min_tumour_ssim": "0.8",
            },
            {
                "source_case_id": "large",
                "has_rc": "False",
                "tiny_count": "0",
                "large_count": "1",
                "artifact_flag_count": "0",
                "artifact_flags": "large_tiled_support",
                "min_tumour_ssim": "0.7",
            },
            {
                "source_case_id": "artifact",
                "has_rc": "False",
                "tiny_count": "0",
                "large_count": "0",
                "artifact_flag_count": "1",
                "artifact_flags": "z_continuity_shift",
                "min_tumour_ssim": "0.6",
            },
            {
                "source_case_id": "low",
                "has_rc": "False",
                "tiny_count": "0",
                "large_count": "0",
                "artifact_flag_count": "0",
                "artifact_flags": "",
                "min_tumour_ssim": "0.1",
            },
        ]
        required, reasons = MODULE.required_review_ids(rows, low_score_count=1)
        self.assertEqual(required, {"rc", "tiny", "large", "artifact", "low"})
        self.assertIn("tiled", reasons["large"])
        self.assertIn("low_score", reasons["low"])

    def test_bool_parser_is_strict_for_common_csv_values(self) -> None:
        self.assertTrue(MODULE.as_bool("True"))
        self.assertTrue(MODULE.as_bool("yes"))
        self.assertFalse(MODULE.as_bool("False"))
        self.assertFalse(MODULE.as_bool(""))

    def test_complete_94_plus_9_evidence_freezes_approved_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qc_root = root / "qc"
            qc_root.mkdir()
            case_ids = [f"BraTS-MET-{index:05d}-000" for index in range(94)]
            checkpoint_hashes = {
                modality: f"sha-{modality}" for modality in MODULE.MODALITIES
            }
            (qc_root / "summary.json").write_text(
                json.dumps(
                    {
                        "technical_gate": "pass",
                        "case_count": 94,
                        "expected_case_count": 94,
                        "modality_row_count": 376,
                        "artifact_row_count": 376,
                        "montage_count": 94,
                        "hard_failure_count": 0,
                        "checkpoint_sha256": checkpoint_hashes,
                    }
                ),
                encoding="utf-8",
            )
            review_rows = []
            for index, case_id in enumerate(case_ids):
                review_rows.append(
                    {
                        "source_case_id": case_id,
                        "has_rc": "True" if index == 0 else "False",
                        "tiny_count": 0,
                        "large_count": 0,
                        "artifact_flag_count": 0,
                        "artifact_flags": "",
                        "min_tumour_ssim": 0.1 + index / 1000,
                    }
                )
            self.write_csv(qc_root / "review_index.csv", review_rows)
            self.write_csv(
                qc_root / "artifact_metrics.csv",
                [
                    {"source_case_id": case_id, "modality": modality}
                    for case_id in case_ids
                    for modality in MODULE.MODALITIES
                ],
            )

            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps(
                    {
                        "fixed_val_count": 103,
                        "generated_positive_count": 94,
                        "strict_noop_negative_count": 9,
                        "strict_noop_pass_count": 9,
                        "selected_source_case_ids": case_ids,
                        "validation_pipeline_contract": {"training_only": True},
                    }
                ),
                encoding="utf-8",
            )
            noop_path = root / "noop.csv"
            self.write_csv(
                noop_path,
                [
                    {
                        "source_case_id": f"negative-{index}",
                        "was_modified": False,
                        "image_equal": True,
                        "seg_equal": True,
                        "image_sha256_before": f"image-{index}",
                        "image_sha256_after": f"image-{index}",
                        "seg_sha256_before": f"seg-{index}",
                        "seg_sha256_after": f"seg-{index}",
                    }
                    for index in range(9)
                ],
            )
            metrics_path = root / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "dataset": "BRATS_2024",
                            "split": "val",
                            "evaluation_mode": "whole_brain",
                            "checkpoint_step": 150000,
                            "normalization": "zscore",
                            "noise_schedule": "edm",
                            "sampling_method": "edm_heun",
                            "sampling_steps": 18,
                            "seed": 20260720,
                            "large_lesion_mode": "tile",
                            "crop_size": 64,
                            "max_cases": 0,
                            "save_support_volumes": True,
                            "generation_manifest_rows": 376,
                            "modalities": list(MODULE.MODALITIES),
                            "checkpoints": {
                                modality: {
                                    "step": 150000,
                                    "bytes": 100,
                                    "sha256": checkpoint_hashes[modality],
                                }
                                for modality in MODULE.MODALITIES
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.csv"
            self.write_csv(
                manifest_path,
                [
                    {"source_case_id": case_id, "modality": modality}
                    for case_id in case_ids
                    for modality in MODULE.MODALITIES
                ],
            )
            inventory_path = root / "inventory.csv"
            self.write_csv(
                inventory_path,
                [
                    {
                        "modality": modality,
                        "step": 150000,
                        "bytes": 100,
                        "sha256": checkpoint_hashes[modality],
                        "checksum_verified": "yes",
                        "local_canonical_relative_path": f"{modality}/diffusion_150000.pt",
                    }
                    for modality in MODULE.MODALITIES
                ],
            )
            mandatory, _ = MODULE.required_review_ids(review_rows, low_score_count=10)
            decisions_path = root / "manual.csv"
            self.write_csv(
                decisions_path,
                [
                    {
                        "source_case_id": case_id,
                        "manual_decision": "pass_technical_visual",
                        "risk_accepted": False,
                    }
                    for case_id in sorted(mandatory)
                ],
            )
            report_path = root / "HUMAN_REVIEW.md"
            report_path.write_text("# reviewed\n", encoding="utf-8")

            gate = MODULE.finalize_gate(
                qc_root,
                cohort_path,
                noop_path,
                metrics_path,
                manifest_path,
                inventory_path,
                decisions_path,
                report_path,
                root / "gates",
            )
            self.assertEqual(gate["decision"], "approve")
            self.assertEqual(gate["generated_positive_count"], 94)
            self.assertEqual(gate["strict_noop_pass_count"], 9)
            selection = json.loads(
                (root / "gates" / "checkpoint_selection.json").read_text()
            )
            self.assertEqual(set(selection["checkpoint_steps"]), set(MODULE.MODALITIES))
            self.assertFalse(gate["rollback_comparison_required"])


if __name__ == "__main__":
    unittest.main()
