from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from custom_nnunet.met_aug_core import (
    S2_MODALITIES,
    ComponentManifest,
    ComponentRecord,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration
from tests.test_met_aug_fix_v2 import _calibration_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str):
    path = REPOSITORY_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTITION = _load_script("29_make_met_aug_fix_v2_partitions.py")
FREEZE = _load_script("30_freeze_met_aug_fix_v2_calibration.py")
REFERENCE = _load_script("33_extract_met_aug_fix_v2_reference.py")
DEVELOPMENT = _load_script("36_run_met_aug_fix_v2_development.py")


def _manifest(root: Path, count: int = 10) -> ComponentManifest:
    records = []
    target_groups = {}
    for index in range(count):
        patient = f"BraTS-MET-{index:05d}"
        target_groups[f"target_{index:03d}"] = patient
        records.append(
            ComponentRecord(
                component_id=f"component_{index:03d}",
                manifest_version="test",
                source_case_id=f"{patient}-000",
                patient_group=patient,
                split="train",
                component_path=f"component_{index:03d}.npz",
                label_sha256="0" * 64,
                source_label_sha256="1" * 64,
                source_modalities_sha256={
                    modality: "2" * 64 for modality in S2_MODALITIES
                },
                source_affine_sha256="3" * 64,
                spacing_mm=(1.0, 1.0, 1.0),
                core_volume_mm3=27.0,
                total_volume_mm3=27.0,
                bbox_mm=(3.0, 3.0, 3.0),
                bbox_voxels=(3, 3, 3),
                class_counts={"3": 27},
                classes_present=(3,),
                core_centroid_norm=(0.5, 0.5, 0.5),
            )
        )
    return ComponentManifest(
        path=root / "component_manifest.json",
        root=root,
        identity_sha256="a" * 64,
        records_sha256="b" * 64,
        records=tuple(records),
        target_groups_path=root / "target_case_groups.json",
        target_groups_sha256="c" * 64,
        target_groups=target_groups,
    )


def _reference_evidence(partition: dict, partition_path: Path) -> dict:
    result = {
        "schema_version": 1,
        "status": "pass",
        "source_partition": "reference",
        "partition_sha256": sha256_file(partition_path),
        "partition_audit_sha256": partition["partition_audit_sha256"],
        "component_manifest_sha256": partition["component_manifest_sha256"],
        "target_groups_sha256": partition["target_groups_sha256"],
        "patient_groups": partition["partitions"]["reference"],
        "target_case_ids": partition["target_case_ids"]["reference"],
        "component_ids": partition["component_ids"]["reference"],
        "patient_group_count": len(partition["partitions"]["reference"]),
        "target_case_count": len(partition["target_case_ids"]["reference"]),
        "component_count": len(partition["component_ids"]["reference"]),
    }
    result["reference_cdf_audit_sha256"] = canonical_json_sha256(
        result,
        exclude=("reference_cdf_audit_sha256",),
    )
    return result


class FixV2PreparationScriptTests(unittest.TestCase):
    def test_reference_spacing_comes_from_matching_post_resampling_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preprocessed = root / "nnUNetPlans_3d_fullres"
            preprocessed.mkdir()
            plans = root / "nnUNetPlans.json"
            plans.write_text(
                json.dumps(
                    {
                        "configurations": {
                            "3d_fullres": {
                                "data_identifier": preprocessed.name,
                                "spacing": [1.0, 1.0, 1.0],
                            },
                            "3d_lowres": {
                                "data_identifier": "nnUNetPlans_3d_lowres",
                                "spacing": [2.0, 2.0, 2.0],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            spacing, configuration = REFERENCE._preprocessed_spacing(
                plans, preprocessed
            )

        self.assertEqual(spacing, (1.0, 1.0, 1.0))
        self.assertEqual(configuration, "3d_fullres")

    def test_development_candidate_order_is_explicit_and_pairs_b_before_c(self):
        candidates = [
            {
                "candidate_id": "C_halo_harmonized_3mm",
                "boundary_policy": "halo_cosine_harmonized_v1",
                "halo_radius_mm": 3.0,
            },
            {
                "candidate_id": "B_halo_2mm",
                "boundary_policy": "halo_cosine_v1",
                "halo_radius_mm": 2.0,
            },
            {
                "candidate_id": "A_label_only",
                "boundary_policy": "label_only_qc_v1",
                "halo_radius_mm": 0.0,
            },
            {
                "candidate_id": "C_halo_harmonized_2mm",
                "boundary_policy": "halo_cosine_harmonized_v1",
                "halo_radius_mm": 2.0,
            },
            {
                "candidate_id": "B_halo_3mm",
                "boundary_policy": "halo_cosine_v1",
                "halo_radius_mm": 3.0,
            },
        ]

        ordered = DEVELOPMENT._candidate_groups({"candidates": candidates})

        self.assertEqual(
            [row[0]["candidate_id"] for row in ordered],
            [
                "A_label_only",
                "B_halo_2mm",
                "C_halo_harmonized_2mm",
                "B_halo_3mm",
                "C_halo_harmonized_3mm",
            ],
        )
        self.assertIsNone(ordered[1][1])
        self.assertEqual(ordered[2][1]["candidate_id"], "B_halo_2mm")

    def test_freezer_writes_one_read_only_calibration_with_bound_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition = PARTITION.build_partition_payload(
                _manifest(root),
                seed=20260728,
                reference_fraction=0.70,
                development_fraction=0.15,
            )
            partition_path = root / "partitions.json"
            partition_path.write_text(
                json.dumps(partition, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            reference = _reference_evidence(partition, partition_path)
            reference_path = root / "reference.json"
            reference_path.write_text(
                json.dumps(reference, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            draft = _calibration_payload(
                component_manifest_sha256=partition["component_manifest_sha256"],
                target_groups_sha256=partition["target_groups_sha256"],
            )
            draft["status"] = "draft"
            draft_path = root / "draft.json"
            draft_path.write_text(json.dumps(draft), encoding="utf-8")
            output = root / "frozen.json"
            arguments = [
                "30_freeze_met_aug_fix_v2_calibration.py",
                "--draft",
                str(draft_path),
                "--partition-audit",
                str(partition_path),
                "--reference-cdf",
                str(reference_path),
                "--patient-group-count",
                str(partition["patient_group_count"]),
                "--component-count",
                str(partition["component_count"]),
                "--boundary-policy",
                "label_only_qc_v1",
                "--output",
                str(output),
            ]
            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                FREEZE.main()
            frozen = FixV2Calibration.load(
                output,
                expected_policy="label_only_qc_v1",
            )

            self.assertEqual(
                frozen.payload["source_audit"]["component_manifest_sha256"],
                partition["component_manifest_sha256"],
            )
            self.assertEqual(
                frozen.payload["source_audit"]["reference_cdf_audit_sha256"],
                reference["reference_cdf_audit_sha256"],
            )
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o444)

    def test_partition_is_deterministic_disjoint_and_exactly_enumerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _manifest(Path(temporary))
            first = PARTITION.build_partition_payload(
                manifest,
                seed=20260728,
                reference_fraction=0.70,
                development_fraction=0.15,
            )
            second = PARTITION.build_partition_payload(
                manifest,
                seed=20260728,
                reference_fraction=0.70,
                development_fraction=0.15,
            )

        self.assertEqual(first, second)
        groups = [set(first["partitions"][name]) for name in FREEZE.PARTITION_NAMES]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(sum(map(len, groups)), first["patient_group_count"])
        self.assertEqual(
            sum(len(first["target_case_ids"][name]) for name in FREEZE.PARTITION_NAMES),
            first["target_case_count"],
        )
        self.assertEqual(
            sum(len(first["component_ids"][name]) for name in FREEZE.PARTITION_NAMES),
            first["component_count"],
        )
        self.assertEqual(
            first["partition_audit_sha256"],
            canonical_json_sha256(first, exclude=("partition_audit_sha256",)),
        )

    def test_partition_validator_rejects_overlap_and_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            partition = PARTITION.build_partition_payload(
                _manifest(Path(temporary)),
                seed=20260728,
                reference_fraction=0.70,
                development_fraction=0.15,
            )
        overlap = deepcopy(partition)
        overlap["partitions"]["development"][0] = overlap["partitions"]["reference"][0]
        overlap["partition_audit_sha256"] = canonical_json_sha256(
            overlap,
            exclude=("partition_audit_sha256",),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            FREEZE._validate_partition_audit(
                overlap,
                expected_count=10,
                expected_component_count=10,
            )

        drift = deepcopy(partition)
        drift["component_count"] += 1
        with self.assertRaisesRegex(ValueError, "SHA256 has drifted"):
            FREEZE._validate_partition_audit(
                drift,
                expected_count=10,
                expected_component_count=10,
            )

    def test_reference_evidence_rejects_partition_membership_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition = PARTITION.build_partition_payload(
                _manifest(root),
                seed=20260728,
                reference_fraction=0.70,
                development_fraction=0.15,
            )
            partition_path = root / "partitions.json"
            partition_path.write_text(
                json.dumps(partition, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            reference = _reference_evidence(partition, partition_path)
            FREEZE._validate_reference_evidence(
                reference,
                partition=partition,
                partition_path=partition_path,
            )

            drift = deepcopy(reference)
            drift["patient_groups"] = list(reversed(drift["patient_groups"]))
            drift["reference_cdf_audit_sha256"] = canonical_json_sha256(
                drift,
                exclude=("reference_cdf_audit_sha256",),
            )
            with self.assertRaisesRegex(ValueError, "patient_groups drifted"):
                FREEZE._validate_reference_evidence(
                    drift,
                    partition=partition,
                    partition_path=partition_path,
                )


if __name__ == "__main__":
    unittest.main()
