#!/usr/bin/env python3
"""Tests for the Diffusion mandatory manual-review queue."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPT = CODE_ROOT / "g2_prepare_diffusion_manual_review.py"
SPEC = importlib.util.spec_from_file_location("g2_prepare_diffusion_manual_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareDiffusionManualReviewTests(unittest.TestCase):
    def test_builds_union_queue_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qc = root / "qc"
            montages = qc / "montages"
            montages.mkdir(parents=True)
            rows = []
            for index in range(94):
                case_id = f"BraTS-MET-{index:05d}-000"
                rows.append(
                    {
                        "source_case_id": case_id,
                        "has_rc": index == 20,
                        "tiny_count": 1 if index == 21 else 0,
                        "small_count": 0,
                        "large_count": 1 if index == 22 else 0,
                        "min_tumour_ssim": index / 100,
                        "mean_support_ssim": 0.8,
                        "artifact_flag_count": 1 if index == 23 else 0,
                        "artifact_flags": "z_continuity_shift" if index == 23 else "",
                    }
                )
            with (qc / "review_index.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            expected = {f"BraTS-MET-{index:05d}-000" for index in range(10)}
            expected.update(f"BraTS-MET-{index:05d}-000" for index in (20, 21, 22, 23))
            for case_id in expected:
                (montages / f"{case_id}.png").write_bytes(b"png")

            summary = MODULE.prepare_review(qc, root / "review", batch_size=5)
            self.assertEqual(summary["mandatory_review_count"], len(expected))
            self.assertEqual(summary["batch_count"], 3)
            with (root / "review" / "mandatory_review_template.csv").open(
                newline=""
            ) as handle:
                review_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["source_case_id"] for row in review_rows}, expected
            )


if __name__ == "__main__":
    unittest.main()
