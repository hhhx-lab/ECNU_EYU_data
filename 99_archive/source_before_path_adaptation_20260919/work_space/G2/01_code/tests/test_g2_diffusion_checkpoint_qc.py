#!/usr/bin/env python3
"""Tests for four-modality Diffusion checkpoint support QC."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = Path(__file__).parents[1] / "code" / "g2_diffusion_checkpoint_qc.py"
MODULE = None
if SCRIPT.is_file():
    spec = importlib.util.spec_from_file_location("g2_diffusion_checkpoint_qc", SCRIPT)
    MODULE = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path, *, outside_support_signal: bool = False) -> tuple[Path, Path, Path]:
    case_id = "BraTS-MET-00001-000"
    shape = (12, 11, 10)
    affine = np.diag([-1.0, 1.0, 1.2, 1.0])
    support = np.zeros(shape, dtype=np.uint8)
    support[2:10, 2:9, 1:9] = 1
    label = np.zeros(shape, dtype=np.uint8)
    label[5:7, 5:7, 4:6] = 4
    support_path = root / f"{case_id}-support.nii.gz"
    label_path = root / f"{case_id}-seg.nii.gz"
    nib.save(nib.Nifti1Image(support, affine), support_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)

    manifest_rows = []
    for index, modality in enumerate(("t1c", "t1n", "t2w", "t2f"), start=1):
        reference = np.zeros(shape, dtype=np.float32)
        coordinates = np.argwhere(support > 0)
        reference[support > 0] = (
            coordinates[:, 0] * 0.1 + coordinates[:, 1] * 0.03 + index
        )
        generated = reference.copy()
        generated[support > 0] += 0.05 * index
        if outside_support_signal and modality == "t1c":
            generated[0, 0, 0] = 1.0
        generated_path = root / f"{case_id}-{modality}-generated.nii.gz"
        reference_path = root / f"{case_id}-{modality}-reference.nii.gz"
        nib.save(nib.Nifti1Image(generated, affine), generated_path)
        nib.save(nib.Nifti1Image(reference, affine), reference_path)
        manifest_rows.append(
            {
                "source_case_id": case_id,
                "patient_id": "00001-000",
                "modality": modality,
                "case_seed": 100 + index,
                "checkpoint_path": f"/{modality}/diffusion_150000.pt",
                "checkpoint_step": 150000,
                "generated_zscore_path": generated_path,
                "reference_zscore_path": reference_path,
                "support_path": support_path,
                "label_path": label_path,
                "support_voxels": int(support.sum()),
                "tumour_voxels": int((label > 0).sum()),
                "tumour_outside_support": 0,
                "normalization": "per_crop_or_tile_brain_zscore",
            }
        )
    manifest = root / "generation_manifest.csv"
    write_csv(manifest, manifest_rows)

    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "frozen",
                "smoke_case_count": 1,
                "selected_source_case_ids": [case_id],
            }
        ),
        encoding="utf-8",
    )
    inventory = root / "checkpoint_inventory.csv"
    checkpoint_paths = {}
    for modality in ("t1c", "t1n", "t2w", "t2f"):
        checkpoint_path = root / modality / "diffusion_150000.pt"
        checkpoint_path.parent.mkdir()
        checkpoint_path.write_bytes(f"checkpoint-{modality}".encode())
        checkpoint_paths[modality] = checkpoint_path
    for row in manifest_rows:
        row["checkpoint_path"] = str(checkpoint_paths[row["modality"]])
    write_csv(manifest, manifest_rows)
    write_csv(
        inventory,
        [
            {
                "modality": modality,
                "step": 150000,
                "bytes": checkpoint_paths[modality].stat().st_size,
                "sha256": MODULE.sha256_file(checkpoint_paths[modality]),
                "checksum_verified": "yes",
            }
            for modality in ("t1c", "t1n", "t2w", "t2f")
        ],
    )
    return manifest, selection, inventory


class G2DiffusionCheckpointQCTests(unittest.TestCase):
    def test_qc_module_exists(self) -> None:
        self.assertIsNotNone(MODULE)

    @unittest.skipIf(MODULE is None, "QC module is not implemented")
    def test_support_projection_crop_tightens_review_panel(self) -> None:
        self.assertTrue(hasattr(MODULE, "support_projection_crop"))
        support = np.zeros((20, 18, 16), dtype=bool)
        support[8:12, 7:11, 6:10] = True

        row_slice, column_slice = MODULE.support_projection_crop(
            support, axis=0, padding=2
        )
        cropped = MODULE.plane(support, axis=0, index=9)[row_slice, column_slice]

        self.assertTrue(cropped.any())
        self.assertLess(cropped.shape[0], 18)
        self.assertLess(cropped.shape[1], 16)

    @unittest.skipIf(MODULE is None, "QC module is not implemented")
    def test_artifact_metrics_detects_repeated_or_shifted_output(self) -> None:
        support = np.ones((12, 11, 10), dtype=bool)
        reference = np.indices(support.shape).sum(axis=0).astype(np.float32)
        generated = np.repeat(reference[:, :, :1], support.shape[2], axis=2)
        metrics = MODULE.artifact_metrics(reference, generated, support)
        self.assertGreater(metrics["generated_repeated_adjacent_z_slices"], 0)
        self.assertIn("repeated_adjacent_z_slices", metrics["artifact_flags"])

    @unittest.skipIf(MODULE is None, "QC module is not implemented")
    def test_valid_four_modality_support_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, selection, inventory = build_fixture(root)
            summary = MODULE.run_qc(
                manifest,
                selection,
                inventory,
                root / "qc",
                expected_cases=1,
            )

            self.assertEqual(summary["technical_gate"], "pass")
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["modality_row_count"], 4)
            self.assertEqual(summary["montage_count"], 1)
            self.assertEqual(summary["hard_failure_count"], 0)
            self.assertEqual(summary["artifact_row_count"], 4)

    @unittest.skipIf(MODULE is None, "QC module is not implemented")
    def test_signal_outside_support_fails_technical_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, selection, inventory = build_fixture(
                root, outside_support_signal=True
            )
            summary = MODULE.run_qc(
                manifest,
                selection,
                inventory,
                root / "qc",
                expected_cases=1,
            )

            self.assertEqual(summary["technical_gate"], "fail")
            self.assertGreaterEqual(summary["hard_failure_count"], 1)
            self.assertTrue(
                any("outside support" in reason for reason in summary["hard_failures"])
            )


if __name__ == "__main__":
    unittest.main()
