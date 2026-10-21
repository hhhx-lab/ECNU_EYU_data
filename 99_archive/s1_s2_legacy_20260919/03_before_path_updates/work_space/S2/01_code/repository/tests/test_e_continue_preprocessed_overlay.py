from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPOSITORY_ROOT / "scripts" / "26_prepare_e_continue_preprocessed_overlay.py"
    spec = importlib.util.spec_from_file_location("test_e_continue_overlay_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OVERLAY = _load_script()


class EContinuePreprocessedOverlayTests(unittest.TestCase):
    def _sources(self, root: Path):
        source_root = root / "source"
        dataset = source_root / OVERLAY.DATASET_NAME
        dataset.mkdir(parents=True)
        (dataset / "dataset.json").write_text("{}\n", encoding="utf-8")
        (dataset / "nnUNetPlans.json").write_text('{"plans": true}\n', encoding="utf-8")
        (dataset / "splits_final.json").write_text("[]\n", encoding="utf-8")
        (dataset / "gt_segmentations").mkdir()
        (dataset / "nnUNetPlans_3d_fullres").mkdir()
        fingerprint = root / "dataset_fingerprint.json"
        fingerprint.write_text(json.dumps({
            "foreground_intensity_properties_per_channel": {},
            "median_relative_size_after_cropping": 1,
            "shapes_after_crop": [[1, 1, 1]] * 1138,
            "spacings": [[1, 1, 1]] * 1138,
        }), encoding="utf-8")
        audit = root / "cache_audit.json"
        audit.write_text(json.dumps({
            "status": "pass",
            "audit_identity_sha256": OVERLAY.EXPECTED_CACHE_AUDIT_IDENTITY,
            "plans": {"file_sha256": OVERLAY.sha256_file(dataset / "nnUNetPlans.json")},
        }), encoding="utf-8")
        return source_root, fingerprint, audit

    def test_creates_and_revalidates_immutable_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fingerprint, audit = self._sources(root)
            output = root / "overlay"
            plans_sha = OVERLAY.sha256_file(source / OVERLAY.DATASET_NAME / "nnUNetPlans.json")
            fingerprint_sha = OVERLAY.sha256_file(fingerprint)
            with patch.object(OVERLAY, "EXPECTED_PLANS_SHA256", plans_sha), patch.object(
                OVERLAY, "EXPECTED_FINGERPRINT_SHA256", fingerprint_sha
            ):
                created = OVERLAY.create_or_validate_overlay(source, fingerprint, audit, output)
                validated = OVERLAY.create_or_validate_overlay(source, fingerprint, audit, output)

            self.assertEqual(created["state"], "created")
            self.assertEqual(validated["state"], "validated")
            dataset = output / OVERLAY.DATASET_NAME
            self.assertTrue((dataset / "nnUNetPlans_3d_fullres").is_symlink())
            self.assertFalse((dataset / "dataset_fingerprint.json").is_symlink())

    def test_rejects_overlay_entry_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fingerprint, audit = self._sources(root)
            output = root / "overlay"
            plans_sha = OVERLAY.sha256_file(source / OVERLAY.DATASET_NAME / "nnUNetPlans.json")
            fingerprint_sha = OVERLAY.sha256_file(fingerprint)
            with patch.object(OVERLAY, "EXPECTED_PLANS_SHA256", plans_sha), patch.object(
                OVERLAY, "EXPECTED_FINGERPRINT_SHA256", fingerprint_sha
            ):
                OVERLAY.create_or_validate_overlay(source, fingerprint, audit, output)
                (output / OVERLAY.DATASET_NAME / "unexpected.txt").write_text("drift", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "entries drifted"):
                    OVERLAY.create_or_validate_overlay(source, fingerprint, audit, output)


if __name__ == "__main__":
    unittest.main()
