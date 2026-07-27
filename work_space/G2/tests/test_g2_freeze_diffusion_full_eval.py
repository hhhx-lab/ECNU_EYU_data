#!/usr/bin/env python3
"""Tests for freezing the fixed Diffusion full-evaluation cohort."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = Path(__file__).parents[1] / "code" / "g2_freeze_diffusion_full_eval.py"
SPEC = importlib.util.spec_from_file_location("g2_freeze_diffusion_full_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FreezeDiffusionFullEvalTests(unittest.TestCase):
    def test_freezes_positive_and_strict_noop_negative_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = []
            lesions = []
            affine = np.eye(4)
            for index in range(3):
                case_id = f"BraTS-MET-{index:05d}-000"
                case_root = root / case_id
                case_root.mkdir()
                seg = np.zeros((8, 8, 8), dtype=np.uint8)
                if index < 2:
                    seg[3:5, 3:5, 3:5] = 3
                paths = {}
                for modality in ("t1n", "t1c", "t2w", "t2f"):
                    path = case_root / f"{case_id}-{modality}.nii.gz"
                    nib.save(nib.Nifti1Image(np.ones(seg.shape, np.float32), affine), path)
                    paths[f"{modality}_path"] = str(path)
                seg_path = case_root / f"{case_id}-seg.nii.gz"
                nib.save(nib.Nifti1Image(seg, affine), seg_path)
                membership.append(
                    {
                        "source_case_id": case_id,
                        "patient_group": case_id.rsplit("-", 1)[0],
                        "split": "val",
                        **paths,
                        "seg_path": str(seg_path),
                    }
                )
                if index < 2:
                    lesions.append(
                        {
                            "patient_id": f"{index:05d}-000",
                            "patient_group": case_id.rsplit("-", 1)[0],
                            "lesion_id": f"{index:05d}-000_cc0",
                            "label": str(seg_path),
                            "split": "val",
                        }
                    )
            lesions_path = root / "lesions.csv"
            membership_path = root / "membership.csv"
            write_csv(lesions_path, lesions)
            write_csv(membership_path, membership)
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "lesion_negative_source_case_ids": ["BraTS-MET-00002-000"],
                        "source_files": {
                            "lesions_csv_sha256": hashlib.sha256(
                                lesions_path.read_bytes()
                            ).hexdigest(),
                            "membership_csv_sha256": hashlib.sha256(
                                membership_path.read_bytes()
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = MODULE.freeze_cohort(
                lesions_path,
                membership_path,
                selection_path,
                root / "output",
                expected_fixed=3,
                expected_positive=2,
                expected_negative=1,
            )
            self.assertEqual(summary["generated_positive_count"], 2)
            self.assertEqual(summary["strict_noop_negative_count"], 1)
            self.assertEqual(summary["strict_noop_pass_count"], 1)
            with (root / "output" / "val_negative9_noop.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["was_modified"], "False")
            self.assertEqual(row["image_equal"], "True")
            self.assertEqual(row["seg_equal"], "True")

    def test_strict_validation_noop_returns_copies_unchanged(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
        seg = np.zeros((1, 4, 4, 4), dtype=np.int16)
        output_image, output_seg, was_modified = MODULE.strict_validation_noop(image, seg)
        self.assertFalse(was_modified)
        self.assertTrue(np.array_equal(image, output_image))
        self.assertTrue(np.array_equal(seg, output_seg))
        self.assertIsNot(image, output_image)
        self.assertIsNot(seg, output_seg)


if __name__ == "__main__":
    unittest.main()
