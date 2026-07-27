from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from custom_nnunet.met_aug_core import MetAugContractError, canonical_json_sha256
from custom_nnunet.met_aug_diffusion import (
    G1_RUNTIME_FILE_KEYS,
    G1FourModalityInpaintingBackend,
    g1_runtime_code_snapshot,
    resolve_selected_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str):
    path = REPOSITORY_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD_POOL = _load_script("12_build_met_aug_component_pool.py")
PREPARE_MASKS = _load_script("14_prepare_met_aug_valid_masks.py")
RUN_GATE2 = _load_script("18_run_met_aug_gate2_smoke.py")
COORDINATE_CONTRACT = _load_script("21_validate_preprocessed_coordinate_contract.py")
PROMOTE_GATE1 = _load_script("23_promote_gate1_parallel_candidate.py")


class _FakeDataset:
    def __init__(self, *, previous_stage=None):
        self.previous_stage = previous_stage

    def load_case(self, case_id):
        del case_id
        data = np.zeros((4, 7, 8, 9), dtype=np.float32)
        segmentation = np.zeros((1, 7, 8, 9), dtype=np.int16)
        return data, segmentation, self.previous_stage, {"spacing": [1.0, 1.0, 1.0]}


class MetAugPreparationScriptTests(unittest.TestCase):
    def test_parallel_gate1_promotion_preserves_serial_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            route_root = Path(temporary) / "route"
            serial_dir = route_root / "gate1"
            candidate_dir = route_root / "gate1_parallel_candidate"
            migration_dir = route_root / "gate1_parallel_migration"
            legacy_dir = migration_dir / "legacy_source"
            component_dir = route_root / "component_pool"
            valid_dir = route_root / "valid_masks"
            for directory in (
                serial_dir,
                candidate_dir,
                legacy_dir,
                component_dir,
                valid_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            events = [json.dumps({"patch_index": index}, sort_keys=True) + "\n" for index in range(10)]
            (serial_dir / "gate1_events.jsonl").write_text("".join(events[:4]), encoding="utf-8")
            (candidate_dir / "gate1_events.jsonl").write_text("".join(events), encoding="utf-8")
            (legacy_dir / "15_run_met_aug_gate1.py").write_text("legacy runner\n", encoding="utf-8")
            (legacy_dir / "met_aug_core.py").write_text("legacy core\n", encoding="utf-8")
            component_manifest = {"manifest_sha256": "a" * 64}
            (component_dir / "component_manifest.json").write_text(
                json.dumps(component_manifest), encoding="utf-8"
            )
            route_config = route_root / "route_a_config.json"
            valid_manifest = valid_dir / "valid_mask_manifest.json"
            route_config.write_text("{}\n", encoding="utf-8")
            valid_manifest.write_text("{}\n", encoding="utf-8")
            candidate_report = {
                "status": "pass",
                "event_count": 10,
                "violations": [],
                "component_manifest_sha256": "a" * 64,
                "route_config_sha256": PROMOTE_GATE1.sha256_file(route_config),
                "valid_mask_manifest_sha256": PROMOTE_GATE1.sha256_file(valid_manifest),
            }
            (candidate_dir / "gate1_report.json").write_text(
                json.dumps(candidate_report), encoding="utf-8"
            )
            equivalence = migration_dir / "equivalence.json"
            equivalence.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "serial_parallel_jsonl_byte_identical": True,
                        "serial_parallel_report_byte_identical": True,
                        "legacy_reference_prefix_byte_identical": True,
                    }
                ),
                encoding="utf-8",
            )

            audit = PROMOTE_GATE1.promote_gate1_candidate(
                route_root=route_root,
                equivalence_report_path=equivalence,
                expected_events=10,
                stopped_pids=[],
            )

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["serial_partial"]["event_count"], 4)
            self.assertTrue((route_root / "gate1_serial_partial" / "gate1_events.jsonl").is_file())
            self.assertEqual(
                PROMOTE_GATE1.line_count(route_root / "gate1" / "gate1_events.jsonl"),
                10,
            )
            self.assertTrue((migration_dir / "PROMOTION_AUDIT.json").is_file())

    def test_nnUNet_28_four_value_case_api_is_used_everywhere(self):
        dataset = _FakeDataset()

        label = BUILD_POOL.load_preprocessed_label(dataset, "case")
        shape, properties, foreground = PREPARE_MASKS.load_preprocessed_case(dataset, "case")
        image, segmentation = RUN_GATE2.load_preprocessed_case(dataset, "case")

        self.assertEqual(label.shape, (7, 8, 9))
        self.assertEqual(shape, (7, 8, 9))
        self.assertEqual(properties["spacing"], [1.0, 1.0, 1.0])
        self.assertEqual(foreground.shape, (7, 8, 9))
        self.assertFalse(np.any(foreground))
        self.assertEqual(image.shape, (4, 7, 8, 9))
        self.assertEqual(segmentation.shape, (1, 7, 8, 9))

    def test_cascaded_preprocessed_data_is_rejected(self):
        dataset = _FakeDataset(previous_stage=np.zeros((7, 8, 9), dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "cascaded"):
            BUILD_POOL.load_preprocessed_label(dataset, "case")
        with self.assertRaisesRegex(ValueError, "cascaded"):
            PREPARE_MASKS.load_preprocessed_case(dataset, "case")
        with self.assertRaisesRegex(ValueError, "cascaded"):
            RUN_GATE2.load_preprocessed_case(dataset, "case")

    def test_component_pool_normalizes_nnunet_ignore_only_in_donor_view(self):
        source = np.array(
            [
                [[-1, 0], [1, 2]],
                [[3, -1], [0, 1]],
            ],
            dtype=np.int16,
        )

        normalized, ignore_voxels = (
            BUILD_POOL.normalize_preprocessed_label_for_component_extraction(
                source,
                case_id="case_with_ignore",
            )
        )

        self.assertEqual(ignore_voxels, 2)
        self.assertEqual(set(np.unique(normalized)), {0, 1, 2, 3})
        self.assertEqual(int(np.count_nonzero(source == -1)), 2)
        self.assertFalse(np.shares_memory(source, normalized))

    def test_component_pool_rejects_labels_other_than_nnunet_ignore_and_brats(self):
        for unsupported in (-2, 5):
            with self.subTest(unsupported=unsupported):
                label = np.array([[[0, unsupported]]], dtype=np.int16)
                with self.assertRaisesRegex(MetAugContractError, "unsupported classes"):
                    BUILD_POOL.normalize_preprocessed_label_for_component_extraction(
                        label,
                        case_id="invalid_case",
                    )

    def test_component_pool_audits_per_case_and_total_ignore_voxels(self):
        audit = BUILD_POOL.build_preprocessed_label_normalization_audit(
            {"case_b": 0, "case_a": 7, "case_c": 11}
        )

        self.assertEqual(audit["scope"], "donor_component_extraction_view_only")
        self.assertEqual(audit["source_ignore_label"], -1)
        self.assertEqual(audit["replacement_label"], 0)
        self.assertEqual(audit["allowed_source_labels"], [-1, 0, 1, 2, 3, 4])
        self.assertEqual(
            audit["case_ignore_voxel_counts"],
            {"case_a": 7, "case_b": 0, "case_c": 11},
        )
        self.assertEqual(audit["cases_with_ignore_voxels"], 2)
        self.assertEqual(audit["total_ignore_voxels"], 18)

    def test_component_pool_records_native_geometry_without_requiring_one_mm_raw_data(self):
        audit = BUILD_POOL.build_raw_source_geometry_audit(
            {
                "native_case": ((240, 240, 155), (0.8594, 0.8594, 1.5)),
                "one_mm_case": ((240, 240, 155), (1.0, 1.0, 1.0)),
            }
        )

        self.assertEqual(
            audit["role"],
            "native_nifti_provenance_and_raw_label_to_modality_alignment_only",
        )
        self.assertEqual(audit["component_coordinate_space"], "nnUNetPlans_3d_fullres_preprocessed")
        self.assertEqual(audit["component_spacing_mm"], [1.0, 1.0, 1.0])
        self.assertEqual(audit["case_count"], 2)
        self.assertEqual(audit["native_spacing_unique_count"], 2)
        self.assertEqual(audit["native_one_mm_case_count"], 1)
        self.assertTrue(audit["raw_label_to_each_modality_geometry_match_required"])

    def test_valid_mask_replays_transpose_before_preprocessing_crop(self):
        raw_mask = np.zeros((2, 3, 4), dtype=bool)
        raw_mask[1, 1, 2] = True
        transpose_forward = (2, 1, 0)
        transformed = np.transpose(raw_mask, transpose_forward)
        properties = {"bbox_used_for_cropping": [[1, 4], [0, 2], [0, 2]]}
        expected = transformed[1:4, 0:2, 0:2]

        observed = PREPARE_MASKS.align_mask(
            raw_mask,
            properties=properties,
            target_shape=expected.shape,
            transpose_forward=transpose_forward,
            allow_nearest_resample=False,
        )

        self.assertTrue(np.array_equal(observed, expected))

    def test_valid_mask_uses_the_pinned_nnunet_segmentation_resampler(self):
        source = np.ones((2, 3, 4), dtype=bool)
        calls = []

        def fake_resampling_fn_seg(data, new_shape, current_spacing, target_spacing):
            calls.append((data.copy(), new_shape, current_spacing, target_spacing))
            return np.ones((1, *new_shape), dtype=np.int16)

        observed = PREPARE_MASKS.align_mask(
            source,
            properties={"spacing": [2.0, 1.5, 1.0]},
            target_shape=(4, 6, 8),
            transpose_forward=(0, 1, 2),
            allow_nearest_resample=True,
            resampling_fn_seg=fake_resampling_fn_seg,
            target_spacing=(1.0, 1.0, 1.0),
        )

        self.assertTrue(np.all(observed))
        self.assertEqual(len(calls), 1)
        data, new_shape, current_spacing, target_spacing = calls[0]
        self.assertEqual(data.shape, (1, 2, 3, 4))
        self.assertEqual(new_shape, (4, 6, 8))
        self.assertEqual(current_spacing, (2.0, 1.5, 1.0))
        self.assertEqual(target_spacing, (1.0, 1.0, 1.0))

    def test_raw_segmentation_replay_marks_only_brain_exterior_background_as_ignore(self):
        source = np.array(
            [
                [[0, 0], [1, 0]],
                [[2, 0], [0, 3]],
            ],
            dtype=np.int16,
        )
        nonzero = np.array(
            [
                [[False, True], [True, True]],
                [[True, False], [True, True]],
            ],
            dtype=bool,
        )

        replay = PREPARE_MASKS.prepare_raw_segmentation_for_nnunet_replay(source, nonzero)

        expected = source.copy()
        expected[0, 0, 0] = -1
        expected[1, 0, 1] = -1
        self.assertTrue(np.array_equal(replay, expected))
        self.assertTrue(np.array_equal(source, np.array(
            [
                [[0, 0], [1, 0]],
                [[2, 0], [0, 3]],
            ],
            dtype=np.int16,
        )))

    def test_segmentation_replay_preserves_multiclass_labels_for_nnunet_resampler(self):
        source = np.array(
            [
                [[-1, 0], [1, 2]],
                [[3, 4], [0, 1]],
            ],
            dtype=np.int16,
        )
        calls = []

        def fake_resampling_fn_seg(data, new_shape, current_spacing, target_spacing):
            calls.append((data.copy(), new_shape, current_spacing, target_spacing))
            result = np.zeros((1, *new_shape), dtype=np.int16)
            result[0, 0, 0, 0] = 4
            return result

        observed = PREPARE_MASKS.align_segmentation(
            source,
            properties={"spacing": [2.0, 1.5, 1.0]},
            target_shape=(4, 4, 4),
            transpose_forward=(0, 1, 2),
            allow_nearest_resample=True,
            resampling_fn_seg=fake_resampling_fn_seg,
            target_spacing=(1.0, 1.0, 1.0),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(set(np.unique(calls[0][0])), {-1, 0, 1, 2, 3, 4})
        self.assertEqual(observed.shape, (4, 4, 4))
        self.assertEqual(observed[0, 0, 0], 4)

    def test_coordinate_contract_detects_a_cache_built_at_different_spacing(self):
        properties = {
            "spacing": [1.0, 1.0, 1.0],
            "shape_after_cropping_and_before_resampling": (129, 160, 138),
        }
        record = COORDINATE_CONTRACT.coordinate_case_record(
            case_id="BraTSMET_000001",
            properties=properties,
            data_shape=(4, 129, 186, 161),
            segmentation_shape=(1, 129, 186, 161),
            transpose_forward=(0, 1, 2),
            target_spacing=(1.0, 1.0, 1.0),
        )

        self.assertEqual(record["expected_spatial_shape"], [129, 160, 138])
        self.assertFalse(record["matches"])

    def test_component_pool_binds_the_exact_3d_fullres_spacing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preprocessed = root / "nnUNetPlans_3d_fullres"
            preprocessed.mkdir()
            plans = {
                "configurations": {
                    "3d_fullres": {
                        "data_identifier": "nnUNetPlans_3d_fullres",
                        "spacing": [1.0, 1.0, 1.0],
                    }
                }
            }
            plans_path = root / "nnUNetPlans.json"
            plans_path.write_text(json.dumps(plans), encoding="utf-8")

            spacing, observed_path = BUILD_POOL.load_preprocessed_contract(preprocessed)

        self.assertEqual(spacing, (1.0, 1.0, 1.0))
        self.assertEqual(observed_path, plans_path)

    def test_g1_checkpoint_uses_the_frozen_canonical_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "step": 150000,
                "canonical_relative_path": (
                    "canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/"
                    "t1c/weights/diffusion_150000.pt"
                ),
            }

            observed = resolve_selected_checkpoint(root, metadata, modality="t1c")

        self.assertEqual(
            observed,
            (root / metadata["canonical_relative_path"]).resolve(),
        )

    def test_g1_checkpoint_selection_cannot_escape_the_archive_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = {
                "step": 150000,
                "canonical_relative_path": "../diffusion_150000.pt",
            }
            with self.assertRaisesRegex(MetAugContractError, "escapes"):
                resolve_selected_checkpoint(temporary, metadata, modality="t1c")

    def test_g1_runtime_snapshot_covers_every_inference_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "g1_repository"
            g1_code_dir = repository / "Segmentation_Tasks" / "GliGAN"
            paths = (
                g1_code_dir / "src" / "infer" / "diffusion_inference_utils.py",
                g1_code_dir / "src" / "networks" / "DiffusionNetwork.py",
                repository / "model.py",
            )
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"source-{index}\n", encoding="utf-8")

            snapshot = g1_runtime_code_snapshot(g1_code_dir)

        self.assertEqual(set(snapshot["files"]), set(G1_RUNTIME_FILE_KEYS))
        self.assertEqual(snapshot["sha256"], canonical_json_sha256(snapshot["files"]))

    def test_g1_import_rejects_a_stale_same_named_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            g1_code_dir = Path(temporary) / "Segmentation_Tasks" / "GliGAN"
            (g1_code_dir / "src" / "infer").mkdir(parents=True)
            stale = types.ModuleType("diffusion_inference_utils")
            stale.__file__ = str(Path(temporary) / "stale" / "diffusion_inference_utils.py")
            backend = G1FourModalityInpaintingBackend.__new__(G1FourModalityInpaintingBackend)
            backend.g1_code_dir = g1_code_dir.resolve()
            original_sys_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {"diffusion_inference_utils": stale}):
                    with self.assertRaisesRegex(MetAugContractError, "outside the approved G1 tree"):
                        backend._prepare_imports()
            finally:
                sys.path[:] = original_sys_path


if __name__ == "__main__":
    unittest.main()
