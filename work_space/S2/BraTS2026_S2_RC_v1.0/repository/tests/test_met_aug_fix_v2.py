from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from custom_nnunet.met_aug_core import (
    FIX_V2_ROUTE_CONFIG_SCHEMA,
    S2_MODALITIES,
    ComponentManifest,
    ComponentRecord,
    EventContext,
    MetAugAuditError,
    MetAugEngine,
    RouteConfig,
    make_fix_v2_route_a_config,
    sha256_file,
)
from custom_nnunet.met_aug_diffusion import (
    G1_MODALITIES,
    G1FourModalityInpaintingBackend,
)
from custom_nnunet.met_aug_fix_v2 import (
    LABEL_SEMANTICS,
    FixV2Calibration,
    FixV2CandidateProcessor,
    _extract_boundary_faces,
    _inside_boundary_voxels,
)


def _modality_mapping(factory):
    return {modality: factory() for modality in S2_MODALITIES}


def _calibration_payload(
    policy: str = "label_only_qc_v1",
    *,
    component_manifest_sha256: str = "2" * 64,
    target_groups_sha256: str = "3" * 64,
) -> dict:
    quantile_intervals = {
        **{f"signed_{name}": [-100.0, 100.0] for name in ("q01", "q05", "q50", "q90", "q95", "q99")},
        **{f"abs_{name}": [0.0, 100.0] for name in ("q01", "q05", "q50", "q90", "q95", "q99")},
    }
    boundary_thresholds = []
    for label_value in (1, 2, 3):
        for modality in S2_MODALITIES:
            boundary_thresholds.append(
                {
                    "label": label_value,
                    "modality": modality,
                    "core_volume_mm3": [0.0, 1_000_000.0],
                    "boundary_area_mm2": [0.0, 1_000_000.0],
                    "min_standard_area_mm2": 1.0,
                    "reference_signed_values": [-100.0, 100.0],
                    "reference_signed_weights": [1.0, 1.0],
                    "reference_abs_values": [0.0, 100.0],
                    "reference_abs_weights": [1.0, 1.0],
                    "ks_signed_max": 1.0,
                    "ks_abs_max": 1.0,
                    "quantile_intervals": quantile_intervals,
                    "signed_envelope": [-100.0, 100.0],
                    "abs_upper": 100.0,
                    "max_abnormal_fraction": 1.0,
                    "max_patch_area_mm2": 1_000_000.0,
                    "max_patch_fraction": 1.0,
                    "small_q95_abs_max": 100.0,
                    "small_max_abs": 100.0,
                }
            )
    raw_qc = _modality_mapping(
        lambda: {
            "residual_quantile_intervals": {
                name: [-100.0, 100.0]
                for name in ("q01", "q05", "q50", "q90", "q95", "q99")
            },
            "extreme_abs_z": 100.0,
            "max_extreme_fraction": 1.0,
            "max_component_voxels": 64**3,
            "max_bbox_fill_ratio": 1.0,
            "max_axis_ratio": 64.0,
            "max_plane_fraction": 1.0,
        }
    )
    candidate_qc = _modality_mapping(
        lambda: {
            "residual_retention": [0.0, 100.0],
            "candidate_abs_z_q99": 100.0,
        }
    )
    cross_modal = {}
    for label_value in (1, 2, 3):
        cross_modal[str(label_value)] = {
            "minimum_voxels": 1,
            "contrast_intervals": {
                modality: [-100.0, 100.0] for modality in S2_MODALITIES
            },
            "mean": [0.0, 0.0, 0.0, 0.0],
            "inverse_covariance": (np.eye(4) * 1e-6).tolist(),
            "max_mahalanobis": 100.0,
            "affected_abs_threshold": {
                modality: 0.01 for modality in S2_MODALITIES
            },
            "pairwise": {
                "t1c:t2f": {
                    "iou": [0.0, 1.0],
                    "centroid_distance_mm": 1_000.0,
                }
            },
        }
    return {
        "schema_version": 1,
        "status": "frozen",
        "boundary_policy": policy,
        "modality_order": list(S2_MODALITIES),
        "label_semantics": LABEL_SEMANTICS,
        "epsilon": 1e-6,
        "geometry": {
            "halo_radius_mm": 0.0 if policy == "label_only_qc_v1" else 3.0,
            "reference_ring_inner_mm": 1.0,
            "reference_ring_outer_mm": 6.0,
            "minimum_reference_voxels": 8,
            "harmonization_ring_inner_fraction": 0.25,
            "harmonization_ring_outer_fraction": 0.75,
        },
        "raw_qc": {"modalities": raw_qc},
        "boundary_qc": {
            "minimum_mad": {modality: 0.001 for modality in S2_MODALITIES},
            "thresholds": boundary_thresholds,
            "event_max_ratio": 2.0,
        },
        "cross_modal_qc": {"classes": cross_modal},
        "candidate_qc": {"modalities": candidate_qc},
        "harmonization": {
            "minimum_ring_voxels": 1,
            "shell_edges": [0.0, 0.5, 1.0],
            "modalities": _modality_mapping(
                lambda: {
                    "gain": [0.01, 10.0],
                    "offset": [-100.0, 100.0],
                    "max_amplification_ratio": 100.0,
                    "max_halo_to_lesion_ratio": 100.0,
                    "radial_shell_upper": [100.0, 100.0],
                }
            ),
        },
        "halo_qc": {
            "modalities": _modality_mapping(
                lambda: {
                    "residual_abs_z_q95": 100.0,
                    "gradient_difference_q99": 100.0,
                    "ncc_min": -1.0,
                    "gradient_cosine_q05_min": -1.0,
                    "outer_residual_abs_z_q99": 100.0,
                    "outer_gradient_delta_abs_z_q99": 100.0,
                    "outer_max_abs_z": 100.0,
                    "outer_abnormal_abs_z": 100.0,
                    "outer_max_abnormal_fraction": 1.0,
                    "outer_max_patch_area_mm2": 1_000_000.0,
                    "structure_tensor_sigma_mm": 1.0,
                    "structure_anisotropy_min": 0.0,
                    "structure_direction_cosine_q05_min": 0.0,
                    "minimum_structure_voxels": 1,
                }
            )
        },
        "source_audit": {
            "partition_sha256": "0" * 64,
            "partition_audit_sha256": "4" * 64,
            "reference_cdf_sha256": "1" * 64,
            "reference_cdf_audit_sha256": "5" * 64,
            "component_manifest_sha256": component_manifest_sha256,
            "target_groups_sha256": target_groups_sha256,
            "patient_group_count": 10,
            "component_count": 20,
        },
    }


def _write_calibration(
    root: Path,
    policy: str = "label_only_qc_v1",
    *,
    component_manifest_sha256: str = "2" * 64,
    target_groups_sha256: str = "3" * 64,
) -> FixV2Calibration:
    path = root / f"calibration-{policy}.json"
    path.write_text(
        json.dumps(
            _calibration_payload(
                policy,
                component_manifest_sha256=component_manifest_sha256,
                target_groups_sha256=target_groups_sha256,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return FixV2Calibration.load(path, expected_policy=policy)


def _base_crop() -> np.ndarray:
    coordinates = np.indices((64, 64, 64), dtype=np.float32)
    base = 0.03 * coordinates[0] + 0.02 * coordinates[1] - 0.01 * coordinates[2]
    return np.stack([base + channel for channel in range(4)]).astype(np.float32)


def _label_cube(label_value: int = 3) -> np.ndarray:
    label = np.zeros((64, 64, 64), dtype=np.int16)
    label[29:35, 29:35, 29:35] = label_value
    return label


class _ResidualBackend:
    def __init__(self, residual: float = 1.0):
        self.residual = residual
        self.last_support = None

    def generate(self, image, label, *, seed, inpaint_support=None):
        support = label != 0 if inpaint_support is None else inpaint_support
        self.last_support = support.copy()
        result = image.copy()
        result[:, support] += self.residual
        return result


class _PatternBackend:
    def __init__(self, callback):
        self.callback = callback

    def generate(self, image, label, *, seed, inpaint_support=None):
        support = label != 0 if inpaint_support is None else inpaint_support
        result = image.copy()
        self.callback(result, image, label, support)
        result[:, ~support] = image[:, ~support]
        return result


def _process(processor, backend, label):
    return processor.process(
        original_image=_base_crop(),
        original_segmentation=np.zeros((64, 64, 64), dtype=np.int16),
        label_cube=label,
        valid_mask=np.ones((64, 64, 64), dtype=bool),
        spacing_mm=(1.0, 1.0, 1.0),
        core_volume_mm3=float(np.count_nonzero(np.isin(label, (1, 3)))),
        seed=7,
        backend=backend,
    )


class FixV2CandidateTests(unittest.TestCase):
    def test_cross_modal_pair_keys_follow_frozen_modality_order(self):
        payload = _calibration_payload()
        expected_pairs = {
            f"{left}:{right}"
            for index, left in enumerate(S2_MODALITIES)
            for right in S2_MODALITIES[index + 1 :]
        }
        limits = {"iou": [0.0, 1.0], "centroid_distance_mm": 1_000.0}
        for class_config in payload["cross_modal_qc"]["classes"].values():
            class_config["pairwise"] = {
                pair: dict(limits) for pair in expected_pairs
            }

        FixV2Calibration.validate_payload(
            payload,
            expected_policy=payload["boundary_policy"],
        )

    def test_label_only_candidate_passes_and_preserves_outside(self):
        with tempfile.TemporaryDirectory() as temporary:
            processor = FixV2CandidateProcessor(_write_calibration(Path(temporary)))
            label = _label_cube(3)
            original = _base_crop()
            result = _process(processor, _ResidualBackend(), label)

        self.assertIsNone(result.reason)
        support = label != 0
        self.assertTrue(np.array_equal(result.image[:, ~support], original[:, ~support]))
        self.assertTrue(np.array_equal(result.segmentation[support], label[support]))
        self.assertEqual(
            result.metadata["candidate_qc"]["boundary"]["strata"]["1:t1n"]["status"],
            "not_present",
        )

    def test_inserted_rc_is_rejected_and_returns_bit_identical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            processor = FixV2CandidateProcessor(_write_calibration(Path(temporary)))
            label = _label_cube(4)
            original = _base_crop()
            segmentation = np.zeros((64, 64, 64), dtype=np.int16)
            result = processor.process(
                original_image=original,
                original_segmentation=segmentation,
                label_cube=label,
                valid_mask=np.ones(label.shape, dtype=bool),
                spacing_mm=(1.0, 1.0, 1.0),
                core_volume_mm3=216.0,
                seed=7,
                backend=_ResidualBackend(),
            )

        self.assertEqual(result.reason, "LABEL_CONTRACT_FAIL")
        self.assertTrue(np.array_equal(result.image, original))
        self.assertTrue(np.array_equal(result.segmentation, segmentation))

    def test_subregion_specific_hard_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = _calibration_payload()
            for threshold in payload["boundary_qc"]["thresholds"]:
                if threshold["label"] == 2 and threshold["modality"] == "t2f":
                    threshold["signed_envelope"] = [-0.1, 0.1]
                    threshold["abs_upper"] = 0.1
                    threshold["max_abnormal_fraction"] = 0.0
                    threshold["max_patch_area_mm2"] = 0.0
                    threshold["max_patch_fraction"] = 0.0
            path = root / "strict.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _ResidualBackend(5.0),
                _label_cube(2),
            )

        self.assertEqual(result.reason, "CANDIDATE_BOUNDARY_QC_FAIL")

    def test_halo_policy_generates_only_inside_h_and_keeps_labels_inside_l(self):
        with tempfile.TemporaryDirectory() as temporary:
            processor = FixV2CandidateProcessor(
                _write_calibration(Path(temporary), "halo_cosine_v1")
            )
            backend = _ResidualBackend()
            label = _label_cube(3)
            result = _process(processor, backend, label)

        self.assertIsNone(result.reason)
        label_support = label != 0
        image_support = result.evidence["image_support"]
        self.assertTrue(np.all(image_support[label_support]))
        self.assertGreater(np.count_nonzero(image_support), np.count_nonzero(label_support))
        self.assertTrue(np.array_equal(backend.last_support, image_support))
        self.assertTrue(np.all(result.segmentation[~label_support] == 0))
        self.assertTrue(np.array_equal(result.image[:, ~image_support], _base_crop()[:, ~image_support]))

    def test_oriented_faces_use_anisotropic_physical_area(self):
        label = np.zeros((5, 5, 5), dtype=np.int16)
        label[2, 2, 2] = 3
        image = np.indices(label.shape, dtype=np.float64).sum(axis=0)
        faces = _extract_boundary_faces(
            label_cube=label,
            image=image,
            label_value=3,
            scale=1.0,
            spacing_mm=(2.0, 1.0, 1.0),
        )

        self.assertEqual(faces.signed.size, 6)
        self.assertAlmostEqual(float(np.sum(faces.weights)), 10.0)
        boundary = _inside_boundary_voxels(label, 3)
        self.assertEqual(int(np.count_nonzero(boundary)), 1)

    def test_small_outer_boundary_uses_frozen_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _calibration_payload()
            for threshold in payload["boundary_qc"]["thresholds"]:
                threshold["min_standard_area_mm2"] = 100.0
            path = root / "small.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            label = np.zeros((64, 64, 64), dtype=np.int16)
            label[32, 32, 32] = 3
            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _ResidualBackend(1.0),
                label,
            )

        self.assertIsNone(result.reason)
        self.assertEqual(
            result.metadata["candidate_qc"]["boundary"]["strata"]["3:t1c"]["branch"],
            "small_sample",
        )

    def test_enclosed_subregion_has_no_paste_boundary_and_is_not_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            processor = FixV2CandidateProcessor(_write_calibration(Path(temporary)))
            label = np.zeros((64, 64, 64), dtype=np.int16)
            label[27:37, 27:37, 27:37] = 2
            label[30:34, 30:34, 30:34] = 3
            result = _process(processor, _ResidualBackend(), label)

        self.assertIsNone(result.reason)
        self.assertEqual(
            result.metadata["candidate_qc"]["boundary"]["strata"]["3:t1c"]["status"],
            "not_present",
        )

    def test_cross_modal_non_identical_nested_ranges_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _calibration_payload()
            for class_config in payload["cross_modal_qc"]["classes"].values():
                class_config["affected_abs_threshold"] = {
                    modality: 5.0 for modality in S2_MODALITIES
                }
                class_config["pairwise"]["t1c:t2f"]["iou"] = [0.05, 1.0]
            path = root / "cross-modal.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            def nested(result, image, label, support):
                points = np.argwhere(label != 0)
                midpoint = int(np.median(points[:, 0]))
                t1c = support & (np.indices(support.shape)[0] <= midpoint)
                result[1, t1c] += 10.0
                result[3, support] += 10.0

            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _PatternBackend(nested),
                _label_cube(3),
            )

        self.assertIsNone(result.reason)
        pair = result.metadata["raw_qc"]["cross_modal"]["classes"]["3"]["pairwise"][
            "t1c:t2f"
        ]
        self.assertGreater(pair["iou"], 0.05)
        self.assertLess(pair["iou"], 1.0)

    def test_cross_modal_spatial_misalignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _calibration_payload()
            for class_config in payload["cross_modal_qc"]["classes"].values():
                class_config["affected_abs_threshold"] = {
                    modality: 5.0 for modality in S2_MODALITIES
                }
                class_config["pairwise"]["t1c:t2f"]["iou"] = [0.5, 1.0]
            path = root / "misaligned.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            def misaligned(result, image, label, support):
                x = np.indices(support.shape)[0]
                midpoint = 31
                result[1, support & (x <= midpoint)] += 10.0
                result[3, support & (x > midpoint)] += 10.0

            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _PatternBackend(misaligned),
                _label_cube(3),
            )

        self.assertEqual(result.reason, "RAW_GENERATION_QC_FAIL")

    def test_harmonization_gain_outside_frozen_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _calibration_payload("halo_cosine_harmonized_v1")
            for modality in S2_MODALITIES:
                payload["harmonization"]["modalities"][modality]["gain"] = [0.5, 2.0]
            path = root / "gain.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            def low_contrast(result, image, label, support):
                result[:, support] = 0.1 * image[:, support]

            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _PatternBackend(low_contrast),
                _label_cube(3),
            )

        self.assertEqual(result.reason, "HARMONIZATION_FAIL")

    def test_harmonization_amplification_limit_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _calibration_payload("halo_cosine_harmonized_v1")
            for modality in S2_MODALITIES:
                values = payload["harmonization"]["modalities"][modality]
                values["max_amplification_ratio"] = 0.01
            path = root / "amplification.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            def distorted(result, image, label, support):
                coordinate = np.indices(support.shape, dtype=np.float32)[0]
                for channel in range(4):
                    result[channel, support] = (
                        0.5 * image[channel, support]
                        + 0.2 * np.sin(coordinate[support] * 0.7 + channel)
                    )

            result = _process(
                FixV2CandidateProcessor(FixV2Calibration.load(path)),
                _PatternBackend(distorted),
                _label_cube(3),
            )

        self.assertEqual(result.reason, "CANDIDATE_CONTENT_QC_FAIL")


class FixV2RouteConfigTests(unittest.TestCase):
    def test_schema_four_binds_policy_and_calibration(self):
        record = ComponentRecord(
            component_id="component",
            manifest_version="test",
            source_case_id="BraTS-MET-00001-000",
            patient_group="BraTS-MET-00001",
            split="train",
            component_path="component.npz",
            label_sha256="0" * 64,
            source_label_sha256="0" * 64,
            source_modalities_sha256={modality: "0" * 64 for modality in S2_MODALITIES},
            source_affine_sha256="0" * 64,
            spacing_mm=(1.0, 1.0, 1.0),
            core_volume_mm3=27.0,
            total_volume_mm3=27.0,
            bbox_mm=(3.0, 3.0, 3.0),
            bbox_voxels=(3, 3, 3),
            class_counts={"3": 27},
            classes_present=(3,),
            core_centroid_norm=(0.5, 0.5, 0.5),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ComponentManifest(
                path=root / "manifest.json",
                root=root,
                identity_sha256="1" * 64,
                records_sha256="2" * 64,
                records=(record,),
                target_groups_path=root / "groups.json",
                target_groups_sha256="3" * 64,
                target_groups={"target": "BraTS-MET-00002"},
            )
            payload = make_fix_v2_route_a_config(
                manifest,
                boundary_policy="label_only_qc_v1",
                calibration_sha256="4" * 64,
            )
            path = root / "route.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = RouteConfig.load(path, manifest)

        self.assertEqual(loaded.schema_version, FIX_V2_ROUTE_CONFIG_SCHEMA)
        self.assertEqual(loaded.fix_v2.boundary_policy, "label_only_qc_v1")
        self.assertEqual(loaded.fix_v2.calibration_sha256, "4" * 64)

    def test_fix_v2_audit_failure_never_commits_caller_arrays(self):
        class FailingAudit:
            def append(self, event):
                raise OSError("injected audit failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            component_path = root / "component.npz"
            np.savez_compressed(
                component_path,
                label=np.full((3, 3, 3), 3, dtype=np.int16),
            )
            record = ComponentRecord(
                component_id="component",
                manifest_version="test",
                source_case_id="BraTS-MET-00001-000",
                patient_group="BraTS-MET-00001",
                split="train",
                component_path=component_path.name,
                label_sha256=sha256_file(component_path),
                source_label_sha256="0" * 64,
                source_modalities_sha256={
                    modality: "0" * 64 for modality in S2_MODALITIES
                },
                source_affine_sha256="0" * 64,
                spacing_mm=(1.0, 1.0, 1.0),
                core_volume_mm3=27.0,
                total_volume_mm3=27.0,
                bbox_mm=(3.0, 3.0, 3.0),
                bbox_voxels=(3, 3, 3),
                class_counts={"3": 27},
                classes_present=(3,),
                core_centroid_norm=(0.5, 0.5, 0.5),
            )
            manifest = ComponentManifest(
                path=root / "manifest.json",
                root=root,
                identity_sha256="1" * 64,
                records_sha256="2" * 64,
                records=(record,),
                target_groups_path=root / "groups.json",
                target_groups_sha256="3" * 64,
                target_groups={"target": "BraTS-MET-00002"},
            )
            calibration = _write_calibration(
                root,
                component_manifest_sha256=manifest.identity_sha256,
                target_groups_sha256=manifest.target_groups_sha256,
            )
            config_payload = make_fix_v2_route_a_config(
                manifest,
                boundary_policy=calibration.boundary_policy,
                calibration_sha256=calibration.sha256,
            )
            config_path = root / "route.json"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            engine = MetAugEngine(
                manifest=manifest,
                config=RouteConfig.load(config_path, manifest),
                backend=_ResidualBackend(),
                audit_sink=FailingAudit(),
                candidate_processor=FixV2CandidateProcessor(calibration),
            )
            shape = (80, 80, 80)
            coordinates = np.indices(shape, dtype=np.float32)
            base = 0.03 * coordinates[0] + 0.02 * coordinates[1] - 0.01 * coordinates[2]
            image = np.stack([base + channel for channel in range(4)]).astype(np.float32)
            segmentation = np.zeros((1,) + shape, dtype=np.int16)
            valid = np.ones(shape, dtype=bool)
            context = None
            planning_states = {}
            for patch_index in range(1_000):
                candidate_context = EventContext(
                    epoch=0,
                    rank=0,
                    worker=0,
                    case_id="target",
                    patch_index=patch_index,
                    full_shape=shape,
                )
                planned = engine.plan(
                    segmentation=segmentation,
                    valid_mask=valid,
                    context=candidate_context,
                )
                planning_key = (
                    f"{planned.state}:{planned.reason}:"
                    f"{planned.metadata.get('detail', '')}"
                )
                planning_states[planning_key] = planning_states.get(planning_key, 0) + 1
                if planned.state == "PLACEMENT_VALID":
                    context = candidate_context
                    break
            self.assertIsNotNone(context, planning_states)
            image_before = image.copy()
            segmentation_before = segmentation.copy()

            with self.assertRaises(MetAugAuditError):
                engine.apply(
                    image=image,
                    segmentation=segmentation,
                    valid_mask=valid,
                    context=context,
                )

        self.assertTrue(np.array_equal(image, image_before))
        self.assertTrue(np.array_equal(segmentation, segmentation_before))


class HaloBackendTests(unittest.TestCase):
    @staticmethod
    def _backend():
        backend = G1FourModalityInpaintingBackend.__new__(G1FourModalityInpaintingBackend)
        backend.device = torch.device("cpu")
        backend.models = {modality: object() for modality in G1_MODALITIES}
        backend.n_steps = 1000
        backend.sampling_steps = 18
        backend.schedule_cfg = types.SimpleNamespace(
            betas=None,
            alphas_bar_sqrt=None,
            one_minus_alphas_bar_sqrt=None,
            alphas_bar=None,
        )
        backend.add_gaussian_noise_tumour_zscore = lambda image, label: (image.copy(), None)
        backend.sample_tumour_diffusion_inpaint = lambda **kwargs: torch.full(
            (1, 1, 64, 64, 64), 9.0, dtype=torch.float32
        )
        backend.sample_edm = lambda **kwargs: torch.full(
            (1, 1, 64, 64, 64), 7.0, dtype=torch.float32
        )
        return backend

    def test_run_l_and_run_h_have_explicit_distinct_commit_supports(self):
        image = np.zeros((4, 64, 64, 64), dtype=np.float32)
        label = np.zeros((64, 64, 64), dtype=np.int16)
        label[31, 32, 33] = 3
        halo = np.zeros_like(label, dtype=bool)
        halo[29:35, 30:36, 31:37] = True
        backend = self._backend()

        run_l = backend.generate(image, label, seed=123)
        run_h = backend.generate(image, label, seed=123, inpaint_support=halo)

        label_support = label != 0
        self.assertTrue(np.all(run_l[:, label_support] == 9.0))
        self.assertTrue(np.all(run_l[:, ~label_support] == 0.0))
        self.assertTrue(np.all(run_h[:, halo] == 7.0))
        self.assertTrue(np.all(run_h[:, ~halo] == 0.0))

    def test_halo_must_contain_label_support(self):
        image = np.zeros((4, 64, 64, 64), dtype=np.float32)
        label = np.zeros((64, 64, 64), dtype=np.int16)
        label[31, 32, 33] = 3
        with self.assertRaisesRegex(RuntimeError, "must contain label support"):
            self._backend().generate(
                image,
                label,
                seed=1,
                inpaint_support=np.zeros(label.shape, dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
