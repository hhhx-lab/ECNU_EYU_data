from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_nnunet.met_aug_core import (
    S2_MODALITIES,
    ComponentManifest,
    ComponentRecord,
    canonical_json_sha256,
)
from custom_nnunet.met_aug_fix_v2_calibration import (
    ReferenceCase,
    build_reference_evidence,
    component_instances,
    compressed_weighted_sample,
    expected_component_id,
    measure_reference_component,
    normalize_preprocessed_segmentation,
    reference_ring,
    validate_reference_evidence,
)


class FixV2ReferenceCalibrationTests(unittest.TestCase):
    def _complete_reference_fixture(self, root: Path):
        shape = (24, 24, 24)
        segmentation = np.zeros(shape, dtype=np.int16)
        segmentation[7:17, 7:17, 7:17] = 2
        segmentation[9:12, 9:12, 9:12] = 1
        segmentation[12:15, 12:15, 12:15] = 3
        case_id = "BraTSMET_000001"
        group = "BraTS-MET-00001"
        instances = component_instances(
            segmentation,
            case_id=case_id,
            manifest_version="test-reference",
            spacing_mm=(1.0, 1.0, 1.0),
        )
        self.assertEqual(len(instances), 1)
        component_id, _label, stats = instances[0]
        core_centroid = np.argwhere(np.isin(segmentation, (1, 3, 4))).mean(axis=0)
        core_centroid /= np.asarray(shape, dtype=np.float64)
        record = ComponentRecord(
            component_id=component_id,
            manifest_version="test-reference",
            source_case_id="BraTS-MET-00001-000",
            patient_group=group,
            split="train",
            component_path="component.npz",
            label_sha256="0" * 64,
            source_label_sha256="1" * 64,
            source_modalities_sha256={modality: "2" * 64 for modality in S2_MODALITIES},
            source_affine_sha256="3" * 64,
            spacing_mm=(1.0, 1.0, 1.0),
            core_volume_mm3=float(stats["core_volume_mm3"]),
            total_volume_mm3=float(stats["total_volume_mm3"]),
            bbox_mm=tuple(float(value) for value in stats["bbox_mm"]),
            bbox_voxels=tuple(int(value) for value in stats["bbox_voxels"]),
            class_counts=dict(stats["class_counts"]),
            classes_present=tuple(stats["classes_present"]),
            core_centroid_norm=tuple(float(value) for value in core_centroid),
        )
        manifest = ComponentManifest(
            path=root / "manifest.json",
            root=root,
            identity_sha256="a" * 64,
            records_sha256="b" * 64,
            records=(record,),
            target_groups_path=root / "groups.json",
            target_groups_sha256="c" * 64,
            target_groups={case_id: group},
        )
        partition = {
            "schema_version": 1,
            "status": "pass",
            "component_manifest_sha256": manifest.identity_sha256,
            "target_groups_sha256": manifest.target_groups_sha256,
            "partitions": {"reference": [group], "development": [], "qc_holdout": []},
            "target_case_ids": {
                "reference": [case_id], "development": [], "qc_holdout": []
            },
            "component_ids": {
                "reference": [component_id], "development": [], "qc_holdout": []
            },
        }
        partition["partition_audit_sha256"] = canonical_json_sha256(
            partition, exclude=("partition_audit_sha256",)
        )
        partition_path = root / "partitions.json"
        partition_path.write_text(json.dumps(partition), encoding="utf-8")
        coordinates = np.indices(shape).astype(np.float32)
        base = coordinates[0] + 1.5 * coordinates[1] + 2.0 * coordinates[2]
        image = np.stack(
            tuple(base * (index + 1) for index in range(4)), axis=0
        ).astype(np.float32)
        case = ReferenceCase(
            case_id=case_id,
            patient_group=group,
            image=image,
            segmentation=segmentation,
            valid_mask=np.ones(shape, dtype=bool),
            spacing_mm=(1.0, 1.0, 1.0),
        )
        evidence = build_reference_evidence(
            manifest=manifest,
            partition_path=partition_path,
            valid_mask_manifest_sha256="d" * 64,
            preprocessed_contract_sha256="e" * 64,
            case_loader=lambda _case_id, _group: case,
        )
        parallel = build_reference_evidence(
            manifest=manifest,
            partition_path=partition_path,
            valid_mask_manifest_sha256="d" * 64,
            preprocessed_contract_sha256="e" * 64,
            case_loader=lambda _case_id, _group: case,
            workers=2,
        )
        self.assertEqual(evidence, parallel)
        evidence_path = root / "reference.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return manifest, partition_path, evidence, evidence_path

    def test_strict_reference_validator_checks_nested_cdf_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partition_path, evidence, evidence_path = (
                self._complete_reference_fixture(root)
            )
            summary = validate_reference_evidence(
                evidence,
                reference_path=evidence_path,
                partition_path=partition_path,
                manifest=manifest,
                expected_valid_mask_manifest_sha256="d" * 64,
                expected_preprocessed_contract_sha256="e" * 64,
            )
            self.assertEqual(summary["usable_component_count"], 1)

            drift = deepcopy(evidence)
            measured = next(
                value
                for value in drift["components"][0]["boundary"].values()
                if value["status"] == "measured"
            )
            measured["signed_weights"][0] *= 2.0
            drift["reference_cdf_audit_sha256"] = canonical_json_sha256(
                drift, exclude=("reference_cdf_audit_sha256",)
            )
            drift_path = root / "reference_drift.json"
            drift_path.write_text(json.dumps(drift), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CDF is invalid"):
                validate_reference_evidence(
                    drift,
                    reference_path=drift_path,
                    partition_path=partition_path,
                    manifest=manifest,
                )

    def test_preprocessed_ignore_is_normalized_without_touching_real_labels(self):
        source = np.asarray([[[-1, 0, 1, 2, 3, 4]]], dtype=np.int16)
        observed = normalize_preprocessed_segmentation(source[None])
        np.testing.assert_array_equal(
            observed,
            np.asarray([[[0, 0, 1, 2, 3, 4]]], dtype=np.int16),
        )

    def test_component_rebuild_uses_the_frozen_builder_identity(self):
        segmentation = np.zeros((24, 24, 24), dtype=np.int16)
        segmentation[5:9, 5:9, 5:9] = 3
        segmentation[4:10, 4:10, 4:10][segmentation[4:10, 4:10, 4:10] == 0] = 2
        instances = component_instances(
            segmentation,
            case_id="BraTSMET_000001",
            manifest_version="met_aug_component_pool_v1",
            spacing_mm=(1.0, 1.0, 1.0),
        )
        self.assertEqual(len(instances), 1)
        self.assertEqual(
            instances[0][0],
            expected_component_id(
                "met_aug_component_pool_v1", "BraTSMET_000001", 1
            ),
        )
        np.testing.assert_array_equal(instances[0][1], segmentation)

    def test_weighted_compression_is_finite_and_preserves_total_area(self):
        values, weights = compressed_weighted_sample(
            np.asarray([-2.0, 1.0, 5.0]),
            np.asarray([1.0, 2.0, 3.0]),
            points=5,
        )
        self.assertEqual(len(values), 5)
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(weights), 6.0)
        self.assertTrue(all(np.isfinite(value) for value in values))

    def test_reference_measurement_is_subregion_and_modality_conditioned(self):
        shape = (24, 24, 24)
        label = np.zeros(shape, dtype=np.int16)
        label[9:12, 9:12, 9:12] = 3
        label[8:13, 8:13, 8:13][label[8:13, 8:13, 8:13] == 0] = 2
        coordinates = np.indices(shape).sum(axis=0).astype(np.float32)
        image = np.stack(
            (coordinates, coordinates * 1.5, coordinates * 2.0, coordinates * 2.5),
            axis=0,
        )
        case = ReferenceCase(
            case_id="case-1",
            patient_group="group-1",
            image=image,
            segmentation=label.copy(),
            valid_mask=np.ones(shape, dtype=bool),
            spacing_mm=(1.0, 1.0, 1.0),
        )
        row = measure_reference_component(
            case,
            component_id="component-1",
            label=label,
            stats={
                "core_volume_mm3": 27.0,
                "total_volume_mm3": float(np.count_nonzero(label)),
                "classes_present": (2, 3),
                "class_counts": {
                    "2": int(np.count_nonzero(label == 2)),
                    "3": int(np.count_nonzero(label == 3)),
                },
            },
        )
        self.assertEqual(row["effects"]["1"]["status"], "not_present")
        self.assertEqual(row["effects"]["2"]["status"], "measured")
        self.assertEqual(row["effects"]["3"]["status"], "measured")
        # ET is enclosed by SNFH here, so only the SNFH-owned outer faces are
        # part of the paste boundary. Internal subregion contacts are excluded.
        self.assertEqual(row["boundary"]["3:t1c"]["status"], "not_present")
        self.assertEqual(row["boundary"]["2:t1c"]["status"], "measured")
        self.assertGreater(row["boundary"]["2:t1c"]["area_mm2"], 0)

    def test_reference_ring_excludes_existing_lesion_and_invalid_brain(self):
        support = np.zeros((16, 16, 16), dtype=bool)
        support[7:9, 7:9, 7:9] = True
        segmentation = np.zeros_like(support, dtype=np.int16)
        segmentation[support] = 3
        segmentation[5, 5, 5] = 3
        valid = np.ones_like(support)
        valid[6, 6, 6] = False
        ring = reference_ring(
            component_support=support,
            segmentation=segmentation,
            valid_mask=valid,
            spacing_mm=(1.0, 1.0, 1.0),
        )
        self.assertFalse(ring[5, 5, 5])
        self.assertFalse(ring[6, 6, 6])
        self.assertFalse(np.any(ring & support))


if __name__ == "__main__":
    unittest.main()
