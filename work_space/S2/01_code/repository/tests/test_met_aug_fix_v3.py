from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from custom_nnunet.met_aug_core import (
    FIX_V3_ROUTE_CONFIG_SCHEMA,
    ComponentManifest,
    MemoryAuditSink,
    MetAugContractError,
    MetAugEngine,
    RouteConfig,
    make_fix_v3_route_a_config,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Geometry, _Reject

from custom_nnunet.met_aug_fix_v3 import (
    FIX_V3_PROCESSOR_POLICY,
    FIX_V3_SMALL_BOUNDARY_AREA_MM2,
    FixV3CandidateProcessor,
    _boundary_failure_decision,
    _content_failure_decision,
    _fix_v3_boundary_threshold,
    _large_region_low_salience_failure,
    _raw_failure_decision,
)
from tests.test_met_aug_route_a import _write_manifest


class FixV3DecisionTests(unittest.TestCase):
    def test_small_boundary_forces_robust_fallback_without_ks(self):
        threshold = _fix_v3_boundary_threshold({"min_standard_area_mm2": 1.0})

        self.assertEqual(
            threshold["min_standard_area_mm2"],
            FIX_V3_SMALL_BOUNDARY_AREA_MM2,
        )

    def test_one_soft_boundary_family_does_not_reject(self):
        decision = _boundary_failure_decision(
            ["ks_signed", "ks_abs"],
            ratio=1.1,
            event_limit=1.0,
        )

        self.assertFalse(decision["reject"])
        self.assertEqual(decision["soft_families"], ["ks"])

    def test_independent_boundary_families_or_local_patch_reject(self):
        independent = _boundary_failure_decision(
            ["ks_signed", "signed_q95"],
            ratio=1.1,
            event_limit=1.0,
        )
        localized = _boundary_failure_decision(
            ["max_patch_area_mm2"],
            ratio=1.1,
            event_limit=1.0,
        )

        self.assertTrue(independent["reject"])
        self.assertTrue(localized["reject"])

    def test_one_marginal_raw_tail_does_not_reject(self):
        decision = _raw_failure_decision(
            failures=["q99"],
            quantiles={"q99": 1.1},
            intervals={"q99": [0.0, 1.0]},
        )

        self.assertFalse(decision["reject"])
        self.assertEqual(decision["soft_families"], ["upper_tail"])

    def test_severe_or_independent_raw_evidence_rejects(self):
        severe = _raw_failure_decision(
            failures=["q99"],
            quantiles={"q99": 2.1},
            intervals={"q99": [0.0, 1.0]},
        )
        independent = _raw_failure_decision(
            failures=["q01", "q99"],
            quantiles={"q01": -1.1, "q99": 1.1},
            intervals={"q01": [-1.0, 0.0], "q99": [0.0, 1.0]},
        )

        self.assertTrue(severe["reject"])
        self.assertTrue(independent["reject"])

    def test_candidate_content_needs_multimodal_or_severe_support(self):
        thresholds = {
            "t1n": {"residual_retention": [0.5, 1.5], "candidate_abs_z_q99": 10.0},
            "t1c": {"residual_retention": [0.5, 1.5], "candidate_abs_z_q99": 10.0},
        }
        one_marginal = _content_failure_decision(
            {
                "t1n": {
                    "failures": ["candidate_abs_z_q99"],
                    "candidate_abs_z_q99": 11.0,
                    "residual_retention": 1.0,
                },
                "t1c": {
                    "failures": [],
                    "candidate_abs_z_q99": 5.0,
                    "residual_retention": 1.0,
                },
            },
            thresholds,
        )
        multimodal = _content_failure_decision(
            {
                "t1n": {
                    "failures": ["candidate_abs_z_q99"],
                    "candidate_abs_z_q99": 11.0,
                    "residual_retention": 1.0,
                },
                "t1c": {
                    "failures": ["candidate_abs_z_q99"],
                    "candidate_abs_z_q99": 11.0,
                    "residual_retention": 1.0,
                },
            },
            thresholds,
        )

        self.assertFalse(one_marginal["reject"])
        self.assertTrue(multimodal["reject"])

    def test_large_low_salience_et_is_a_narrow_hard_failure(self):
        missed_bad = _large_region_low_salience_failure(
            label_value=3,
            support_voxels=2554,
            modality_metrics={
                "t1c": {
                    "median_contrast": 0.137,
                    "affected_fraction": 0.269,
                }
            },
        )
        small_region = _large_region_low_salience_failure(
            label_value=3,
            support_voxels=2047,
            modality_metrics={
                "t1c": {
                    "median_contrast": 0.137,
                    "affected_fraction": 0.269,
                }
            },
        )
        salient = _large_region_low_salience_failure(
            label_value=3,
            support_voxels=2554,
            modality_metrics={
                "t1c": {
                    "median_contrast": 0.4,
                    "affected_fraction": 0.5,
                }
            },
        )

        self.assertTrue(missed_bad)
        self.assertFalse(small_region)
        self.assertFalse(salient)


class FixV3BindingTests(unittest.TestCase):
    @staticmethod
    def _processor(payload: dict) -> FixV3CandidateProcessor:
        payload = {
            "source_audit": {
                "component_manifest_sha256": "b" * 64,
                "target_groups_sha256": "c" * 64,
            },
            **payload,
        }
        calibration = types.SimpleNamespace(
            payload=payload,
            boundary_policy="label_only_qc_v1",
            sha256="a" * 64,
            epsilon=1e-6,
        )
        return FixV3CandidateProcessor(calibration)

    @staticmethod
    def _geometry(support: np.ndarray) -> FixV2Geometry:
        return FixV2Geometry(
            label_support=support,
            image_support=support,
            harmonization_ring=np.zeros_like(support),
            reference_ring=~support,
            alpha=support.astype(np.float32),
            distance_from_label_mm=np.zeros_like(support, dtype=np.float64),
        )

    def test_schema_five_binds_fix_v3_processor_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _ = _write_manifest(root)
            manifest = ComponentManifest.load(manifest_path)
            payload = make_fix_v3_route_a_config(
                manifest,
                boundary_policy="label_only_qc_v1",
                calibration_sha256="a" * 64,
            )
            config_path = root / "fix_v3_route.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = RouteConfig.load(config_path, manifest)

            self.assertEqual(config.schema_version, FIX_V3_ROUTE_CONFIG_SCHEMA)
            self.assertIsNone(config.fix_v2)
            self.assertIsNotNone(config.fix_v3)
            self.assertEqual(config.fix_v3.processor_policy, FIX_V3_PROCESSOR_POLICY)

    def test_engine_accepts_only_matching_fix_v3_processor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _ = _write_manifest(root)
            manifest = ComponentManifest.load(manifest_path)
            payload = make_fix_v3_route_a_config(
                manifest,
                boundary_policy="label_only_qc_v1",
                calibration_sha256="a" * 64,
            )
            config_path = root / "fix_v3_route.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = RouteConfig.load(config_path, manifest)
            processor = types.SimpleNamespace(
                processor_policy=FIX_V3_PROCESSOR_POLICY,
                boundary_policy="label_only_qc_v1",
                calibration_sha256="a" * 64,
                component_manifest_sha256=manifest.identity_sha256,
                target_groups_sha256=manifest.target_groups_sha256,
            )

            engine = MetAugEngine(
                manifest=manifest,
                config=config,
                backend=None,
                audit_sink=MemoryAuditSink(),
                candidate_processor=processor,
            )

            self.assertIs(engine.candidate_processor, processor)
            processor.processor_policy = "fix_v2_qc_v1"
            with self.assertRaises(MetAugContractError):
                MetAugEngine(
                    manifest=manifest,
                    config=config,
                    backend=None,
                    audit_sink=MemoryAuditSink(),
                    candidate_processor=processor,
                )

    def test_processor_rejects_large_low_salience_et(self):
        cross_modal = {
            str(label): {
                "minimum_voxels": 1,
                "contrast_intervals": {
                    modality: [-100.0, 100.0]
                    for modality in ("t1n", "t1c", "t2w", "t2f")
                },
                "mean": [0.0, 0.0, 0.0, 0.0],
                "inverse_covariance": np.eye(4).tolist(),
                "max_mahalanobis": 100.0,
                "affected_abs_threshold": {
                    modality: 0.5
                    for modality in ("t1n", "t1c", "t2w", "t2f")
                },
                "pairwise": {},
            }
            for label in (1, 2, 3)
        }
        calibration = types.SimpleNamespace(
            payload={
                "source_audit": {
                    "component_manifest_sha256": "b" * 64,
                    "target_groups_sha256": "c" * 64,
                },
                "cross_modal_qc": {"classes": cross_modal},
            },
            boundary_policy="label_only_qc_v1",
            sha256="a" * 64,
        )
        processor = FixV3CandidateProcessor(calibration)
        label = np.zeros((64, 64, 64), dtype=np.int16)
        label[25:39, 25:39, 25:39] = 3
        support = label != 0
        geometry = FixV2Geometry(
            label_support=support,
            image_support=support,
            harmonization_ring=np.zeros_like(support),
            reference_ring=~support,
            alpha=support.astype(np.float32),
            distance_from_label_mm=np.zeros_like(label, dtype=np.float64),
        )

        with self.assertRaises(_Reject) as raised:
            processor._cross_modal_qc(
                candidate=np.zeros((4, 64, 64, 64), dtype=np.float32),
                label_cube=label,
                geometry=geometry,
                spacing_mm=(1.0, 1.0, 1.0),
                scales={modality: 1.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
                reference={modality: 0.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
            )

        self.assertEqual(raised.exception.reason, "CANDIDATE_CROSS_MODAL_QC_FAIL")
        self.assertIn("large_et_low_salience", raised.exception.detail)

    def test_processor_allows_one_marginal_raw_quantile_but_not_two_families(self):
        intervals = {
            name: [-100.0, 100.0]
            for name in ("q01", "q05", "q50", "q90", "q95", "q99")
        }
        intervals["q99"] = [0.0, 1.0]
        raw_threshold = {
            "residual_quantile_intervals": intervals,
            "extreme_abs_z": 100.0,
            "max_extreme_fraction": 1.0,
            "max_component_voxels": 64**3,
            "max_bbox_fill_ratio": 1.0,
            "max_axis_ratio": 64.0,
            "max_plane_fraction": 1.0,
        }
        processor = self._processor(
            {
                "raw_qc": {
                    "modalities": {
                        modality: dict(raw_threshold)
                        for modality in ("t1n", "t1c", "t2w", "t2f")
                    }
                }
            }
        )
        support = np.zeros((64, 64, 64), dtype=bool)
        support.reshape(-1)[:100] = True
        geometry = self._geometry(support)
        original = np.zeros((4, 64, 64, 64), dtype=np.float32)
        generated = original.copy()
        channel_values = generated[0][support]
        channel_values[:2] = 1.1
        generated[0][support] = channel_values

        report = processor._raw_qc(
            original=original,
            generated=generated,
            geometry=geometry,
            scales={modality: 1.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            report["modalities"]["t1n"]["decision"]["soft_families"],
            ["upper_tail"],
        )

        processor.calibration.payload["raw_qc"]["modalities"]["t1n"][
            "residual_quantile_intervals"
        ]["q01"] = [-1.0, 0.0]
        channel_values[:2] = -1.1
        channel_values[2:4] = 1.1
        generated[0][support] = channel_values
        with self.assertRaises(_Reject) as raised:
            processor._raw_qc(
                original=original,
                generated=generated,
                geometry=geometry,
                scales={modality: 1.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
            )
        self.assertEqual(raised.exception.reason, "RAW_GENERATION_QC_FAIL")

    def test_processor_allows_one_marginal_content_modality(self):
        thresholds = {
            modality: {
                "residual_retention": [0.5, 1.5],
                "candidate_abs_z_q99": 10.0,
            }
            for modality in ("t1n", "t1c", "t2w", "t2f")
        }
        processor = self._processor({"candidate_qc": {"modalities": thresholds}})
        support = np.zeros((64, 64, 64), dtype=bool)
        support.reshape(-1)[:100] = True
        geometry = self._geometry(support)
        original = np.zeros((4, 64, 64, 64), dtype=np.float32)
        raw = original.copy()
        candidate = original.copy()
        values = np.zeros(100, dtype=np.float32)
        values[:6] = 11.0
        raw[0][support] = values
        candidate[0][support] = values

        report = processor._candidate_content_qc(
            original=original,
            raw=raw,
            candidate=candidate,
            geometry=geometry,
            scales={modality: 1.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["decision"]["failed_modalities"], ["t1n"])

    def test_processor_uses_small_boundary_fallback_below_128_mm2(self):
        quantile_intervals = {
            **{
                f"signed_{name}": [-100.0, 100.0]
                for name in ("q01", "q05", "q50", "q90", "q95", "q99")
            },
            **{
                f"abs_{name}": [0.0, 100.0]
                for name in ("q01", "q05", "q50", "q90", "q95", "q99")
            },
        }
        threshold = {
            "label": 3,
            "modality": "t1n",
            "core_volume_mm3": [0.0, 1000.0],
            "boundary_area_mm2": [0.0, 1000.0],
            "min_standard_area_mm2": 1.0,
            "reference_signed_values": [-100.0, 100.0],
            "reference_signed_weights": [1.0, 1.0],
            "reference_abs_values": [0.0, 100.0],
            "reference_abs_weights": [1.0, 1.0],
            "ks_signed_max": 0.1,
            "ks_abs_max": 0.1,
            "quantile_intervals": quantile_intervals,
            "signed_envelope": [-100.0, 100.0],
            "abs_upper": 100.0,
            "max_abnormal_fraction": 1.0,
            "max_patch_area_mm2": 1000.0,
            "max_patch_fraction": 1.0,
            "small_q95_abs_max": 100.0,
            "small_max_abs": 100.0,
        }
        thresholds = []
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            thresholds.append({**threshold, "modality": modality})
        processor = self._processor(
            {
                "boundary_qc": {
                    "thresholds": thresholds,
                    "event_max_ratio": 1.0,
                }
            }
        )
        label = np.zeros((64, 64, 64), dtype=np.int16)
        label[31:33, 31:33, 31:33] = 3
        support = label != 0

        report = processor._boundary_qc(
            candidate=np.zeros((4, 64, 64, 64), dtype=np.float32),
            label_cube=label,
            geometry=self._geometry(support),
            spacing_mm=(1.0, 1.0, 1.0),
            core_volume_mm3=8.0,
            scales={modality: 1.0 for modality in ("t1n", "t1c", "t2w", "t2f")},
        )

        self.assertEqual(report["strata"]["3:t1n"]["branch"], "small_sample")


if __name__ == "__main__":
    unittest.main()
