from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_nnunet.met_aug_core import (
    COMPONENT_MANIFEST_SCHEMA,
    VALID_MASK_MANIFEST_SCHEMA,
    ComponentManifest,
    ComponentRecord,
    MetAugContractError,
    RouteConfig,
    canonical_json_sha256,
    make_route_a_config,
    sha256_file,
)
from custom_nnunet.met_aug_diffusion import g1_runtime_code_snapshot
from custom_nnunet.met_aug_gate import (
    ROUTE_A_RUNTIME_FILES,
    build_route_a_approval,
    validate_route_a_approval,
)
from custom_nnunet.met_aug_gate2 import (
    GATE2_VOLUME_BINS,
    REVIEW_TEMPLATE_FIELDS,
    gate2_runtime_code_snapshot,
    load_case_results_evidence,
    load_smoke_manifest,
    load_valid_mask_assets,
    prepare_smoke_manifest,
    validate_manual_review,
)


def _write_gate2_inputs(
    root: Path, *, compact_support: bool = False
) -> tuple[Path, Path, Path]:
    components_dir = root / "components"
    masks_dir = root / "masks"
    components_dir.mkdir()
    masks_dir.mkdir()
    component_rows = []
    target_groups = {}
    valid_rows = []

    def add_target(serial: int) -> None:
        target_case_id = f"BraTS-MET-{20000 + serial:05d}-000"
        target_groups[target_case_id] = target_case_id.rsplit("-", 1)[0]
        mask_path = masks_dir / f"{target_case_id}.npz"
        valid = np.ones((72, 72, 72), dtype=np.uint8)
        foreground = np.zeros_like(valid)
        np.savez_compressed(mask_path, valid_mask=valid, foreground_mask=foreground)
        valid_rows.append({
            "case_id": target_case_id,
            "mask_path": str(Path("masks") / mask_path.name),
            "sha256": sha256_file(mask_path),
            "shape": [72, 72, 72],
        })

    shapes = ((3, 3, 3), (4, 4, 4), (7, 7, 7))
    for bin_index, shape in enumerate(shapes):
        for index in range(8):
            serial = bin_index * 8 + index
            component_id = f"component_{serial:02d}"
            source_case_id = f"BraTS-MET-{10000 + serial:05d}-000"
            component_path = components_dir / f"{component_id}.npz"
            label = np.full(shape, 3, dtype=np.int16)
            np.savez_compressed(component_path, label=label)
            volume = float(np.prod(shape))
            record = ComponentRecord(
                component_id=component_id,
                manifest_version="gate2-test",
                source_case_id=source_case_id,
                patient_group=source_case_id.rsplit("-", 1)[0],
                split="train",
                component_path=str(Path("components") / component_path.name),
                label_sha256=sha256_file(component_path),
                source_label_sha256="0" * 64,
                source_modalities_sha256={
                    modality: "0" * 64 for modality in ("t1n", "t1c", "t2w", "t2f")
                },
                source_affine_sha256="0" * 64,
                spacing_mm=(1.0, 1.0, 1.0),
                core_volume_mm3=volume,
                total_volume_mm3=volume,
                bbox_mm=tuple(float(value) for value in shape),
                bbox_voxels=shape,
                class_counts={"3": int(volume)},
                classes_present=(3,),
                core_centroid_norm=(0.5, 0.5, 0.5),
            )
            component_rows.append(record.as_mapping())

            add_target(serial)

    # The production pool has 1,035 targets. A larger target fixture avoids a
    # coupon-collector wait caused by forcing 24 unique donors onto only 24
    # unique targets while preserving the exact Gate 2 constraints.
    for serial in range(24, 96):
        add_target(serial)

    if compact_support:
        component_id = "component_ineligible_large_support"
        source_case_id = "BraTS-MET-19999-000"
        component_path = components_dir / f"{component_id}.npz"
        label = np.full((17, 17, 17), 3, dtype=np.int16)
        np.savez_compressed(component_path, label=label)
        volume = int(label.size)
        component_rows.append(ComponentRecord(
            component_id=component_id,
            manifest_version="gate2-test",
            source_case_id=source_case_id,
            patient_group=source_case_id.rsplit("-", 1)[0],
            split="train",
            component_path=str(Path("components") / component_path.name),
            label_sha256=sha256_file(component_path),
            source_label_sha256="0" * 64,
            source_modalities_sha256={
                modality: "0" * 64 for modality in ("t1n", "t1c", "t2w", "t2f")
            },
            source_affine_sha256="0" * 64,
            spacing_mm=(1.0, 1.0, 1.0),
            core_volume_mm3=float(volume),
            total_volume_mm3=float(volume),
            bbox_mm=(17.0, 17.0, 17.0),
            bbox_voxels=(17, 17, 17),
            class_counts={"3": volume},
            classes_present=(3,),
            core_centroid_norm=(0.5, 0.5, 0.5),
        ).as_mapping())

    records_path = root / "components.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in component_rows),
        encoding="utf-8",
    )
    groups_path = root / "target_case_groups.json"
    groups_path.write_text(
        json.dumps({"schema_version": 1, "case_to_patient_group": target_groups}, sort_keys=True),
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": COMPONENT_MANIFEST_SCHEMA,
        "manifest_version": "gate2-test",
        "coordinate_space": "nnUNetPlans_3d_fullres_preprocessed",
        "builder_code_sha256": "0" * 64,
        "component_core_sha256": "0" * 64,
        "nnunet_plans_sha256": "0" * 64,
        "train_file_sha256": "0" * 64,
        "mapping_csv_sha256": "0" * 64,
        "component_count": len(component_rows),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "target_groups_file": groups_path.name,
        "target_groups_sha256": sha256_file(groups_path),
    }
    manifest_payload["manifest_sha256"] = canonical_json_sha256(
        manifest_payload, exclude=("manifest_sha256",)
    )
    manifest_path = root / "component_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")

    manifest = ComponentManifest.load(manifest_path)
    config_path = root / "route_a.json"
    config_path.write_text(
        json.dumps(
            make_route_a_config(
                manifest,
                max_total_support_voxels=4096 if compact_support else None,
                max_total_to_core_ratio=20.0 if compact_support else None,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    valid_records_path = root / "valid_mask_records.jsonl"
    valid_records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in valid_rows),
        encoding="utf-8",
    )
    valid_payload = {
        "schema_version": VALID_MASK_MANIFEST_SCHEMA,
        "builder_code_sha256": "0" * 64,
        "dataset_json_sha256": "0" * 64,
        "nnunet_plans_sha256": "0" * 64,
        "train_file_sha256": "0" * 64,
        "train_count": len(valid_rows),
        "records_file": valid_records_path.name,
        "records_sha256": sha256_file(valid_records_path),
    }
    valid_payload["manifest_sha256"] = canonical_json_sha256(
        valid_payload, exclude=("manifest_sha256",)
    )
    valid_path = root / "valid_mask_manifest.json"
    valid_path.write_text(json.dumps(valid_payload, sort_keys=True), encoding="utf-8")
    return manifest_path, config_path, valid_path


class Gate2ManifestTests(unittest.TestCase):
    def test_scoped_pre_registration_and_reload_keep_targets_and_donors_in_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, valid_path = _write_gate2_inputs(root)
            manifest = ComponentManifest.load(manifest_path)
            config = RouteConfig.load(config_path, manifest)
            target_ids = set(sorted(manifest.target_groups)[:48])
            target_groups = {manifest.target_groups[case_id] for case_id in target_ids}
            donor_groups = {record.patient_group for record in manifest.records}
            assets = load_valid_mask_assets(valid_path, expected_ids=target_ids)
            payload = prepare_smoke_manifest(
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
                assets=assets,
                search_seed=41,
                max_candidates=10000,
                allowed_target_groups=target_groups,
                allowed_donor_groups=donor_groups,
                smoke_id_prefix="development",
            )
            smoke_path = root / "scoped.json"
            smoke_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            loaded = load_smoke_manifest(
                smoke_path,
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
            )

        self.assertEqual(loaded["selection_scope"]["target_case_count"], 48)
        self.assertTrue(
            all(row["target_case_id"] in target_ids for row in loaded["smoke_cases"])
        )
        self.assertTrue(
            all(
                row["donor_patient_group"] in donor_groups
                for row in loaded["smoke_cases"]
            )
        )

    def test_pre_registration_is_fixed_stratified_and_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, valid_path = _write_gate2_inputs(root)
            manifest = ComponentManifest.load(manifest_path)
            config = RouteConfig.load(config_path, manifest)
            assets = load_valid_mask_assets(valid_path, expected_ids=set(manifest.target_groups))
            payload = prepare_smoke_manifest(
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
                assets=assets,
                search_seed=29,
                max_candidates=10000,
            )
            smoke_path = root / "gate2_smoke.json"
            smoke_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            loaded = load_smoke_manifest(
                smoke_path,
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
            )

        self.assertEqual(loaded["smoke_count"], 24)
        self.assertEqual(loaded["per_volume_bin"], {key: 8 for key in GATE2_VOLUME_BINS})
        self.assertEqual(len({row["target_case_id"] for row in loaded["smoke_cases"]}), 24)
        self.assertEqual(len({row["donor_component_id"] for row in loaded["smoke_cases"]}), 24)

    def test_tampered_smoke_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, valid_path = _write_gate2_inputs(root)
            manifest = ComponentManifest.load(manifest_path)
            config = RouteConfig.load(config_path, manifest)
            assets = load_valid_mask_assets(valid_path, expected_ids=set(manifest.target_groups))
            payload = prepare_smoke_manifest(
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
                assets=assets,
                search_seed=31,
                max_candidates=10000,
            )
            payload["smoke_cases"][0]["target_case_id"] = "tampered"
            payload["smoke_manifest_sha256"] = canonical_json_sha256(
                payload, exclude=("smoke_manifest_sha256",)
            )
            smoke_path = root / "tampered.json"
            smoke_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "entry SHA256"):
                load_smoke_manifest(
                    smoke_path,
                    manifest=manifest,
                    config=config,
                    valid_mask_manifest_path=valid_path,
                )

    def test_compact_support_gate2_excludes_ineligible_donors_and_binds_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, valid_path = _write_gate2_inputs(
                root, compact_support=True
            )
            manifest = ComponentManifest.load(manifest_path)
            config = RouteConfig.load(config_path, manifest)
            assets = load_valid_mask_assets(
                valid_path, expected_ids=set(manifest.target_groups)
            )
            payload = prepare_smoke_manifest(
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
                assets=assets,
                search_seed=37,
                max_candidates=10000,
            )
            smoke_path = root / "gate2_compact.json"
            smoke_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            loaded = load_smoke_manifest(
                smoke_path,
                manifest=manifest,
                config=config,
                valid_mask_manifest_path=valid_path,
            )

        self.assertEqual(loaded["eligible_component_count"], 24)
        self.assertEqual(loaded["donor_eligibility"]["excluded_component_count"], 1)
        self.assertNotIn(
            "component_ineligible_large_support",
            {row["donor_component_id"] for row in loaded["smoke_cases"]},
        )
        self.assertTrue(
            all(row["total_support_voxels"] <= 4096 for row in loaded["smoke_cases"])
        )
        self.assertTrue(
            all(row["total_to_core_ratio"] <= 20.0 for row in loaded["smoke_cases"])
        )


class Gate2ManualReviewTests(unittest.TestCase):
    def test_runtime_snapshot_binds_only_the_gate2_execution_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_files = (
                "scripts/18_run_met_aug_gate2_smoke.py",
                "custom_nnunet/met_aug_gate2.py",
                "custom_nnunet/met_aug_core.py",
                "custom_nnunet/met_aug_diffusion.py",
                "custom_nnunet/met_aug_fix_v2.py",
                "custom_nnunet/online_diffusion_contract.py",
            )
            for index, relative in enumerate(expected_files):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"source-{index}\n", encoding="utf-8")

            snapshot = gate2_runtime_code_snapshot(root)

        self.assertEqual(set(snapshot["files"]), set(expected_files))
        self.assertEqual(snapshot["sha256"], canonical_json_sha256(snapshot["files"]))

    def test_route_approval_binds_gate2_to_the_deployed_g1_and_s2_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, valid_path = _write_gate2_inputs(root)
            manifest = ComponentManifest.load(manifest_path)

            selection_path = root / "checkpoint_selection.json"
            selection_path.write_text(json.dumps({
                "status": "frozen",
                "normalization": "zscore",
                "sampling_method": "edm_heun",
                "sampling_steps": 18,
                "crop_size": 64,
            }, sort_keys=True), encoding="utf-8")
            parent_gate_path = root / "g2_parent_gate.json"
            parent_gate_path.write_text(json.dumps({
                "decision": "approve",
                "checkpoint_selection_sha256": sha256_file(selection_path),
            }, sort_keys=True), encoding="utf-8")

            g1_repository = root / "g1_repository"
            g1_code_dir = g1_repository / "Segmentation_Tasks" / "GliGAN"
            g1_files = (
                g1_code_dir / "src" / "infer" / "diffusion_inference_utils.py",
                g1_code_dir / "src" / "networks" / "DiffusionNetwork.py",
                g1_repository / "model.py",
            )
            for index, path in enumerate(g1_files):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"g1-source-{index}\n", encoding="utf-8")
            g1_runtime = g1_runtime_code_snapshot(g1_code_dir)

            code_dir = root / "custom_nnunet"
            code_dir.mkdir()
            for index, filename in enumerate(ROUTE_A_RUNTIME_FILES):
                (code_dir / filename).write_text(f"s2-source-{index}\n", encoding="utf-8")

            shared = {
                "route_id": "MET-AUG-A",
                "status": "pass",
                "component_manifest_sha256": manifest.identity_sha256,
                "route_config_sha256": sha256_file(config_path),
                "valid_mask_manifest_sha256": sha256_file(valid_path),
            }
            gate1_path = root / "gate1.json"
            gate1_path.write_text(json.dumps(shared, sort_keys=True), encoding="utf-8")
            gate2 = {
                **shared,
                "schema_version": 2,
                "manual_review_status": "pass",
                "smoke_count": 24,
                "per_volume_bin": {"27_49": 8, "50_275": 8, "gt_275": 8},
                "smoke_manifest_sha256": "1" * 64,
                "automatic_report_sha256": "2" * 64,
                "review_decisions_sha256": "3" * 64,
                "g1_checkpoint_selection_sha256": sha256_file(selection_path),
                "g2_parent_gate_sha256": sha256_file(parent_gate_path),
                "g1_runtime_code": g1_runtime,
            }
            gate2["gate2_report_sha256"] = canonical_json_sha256(
                gate2, exclude=("gate2_report_sha256",)
            )
            gate2_path = root / "gate2.json"
            gate2_path.write_text(json.dumps(gate2, sort_keys=True), encoding="utf-8")

            approval = build_route_a_approval(
                component_manifest_path=manifest_path,
                route_config_path=config_path,
                valid_mask_manifest_path=valid_path,
                gate1_report_path=gate1_path,
                gate2_report_path=gate2_path,
                g1_checkpoint_selection_path=selection_path,
                g2_parent_gate_path=parent_gate_path,
                g1_code_dir=g1_code_dir,
                code_dir=code_dir,
            )
            approval_path = root / "route_approval.json"
            approval_path.write_text(json.dumps(approval, sort_keys=True), encoding="utf-8")
            validated = validate_route_a_approval(
                approval_path,
                component_manifest=manifest,
                route_config_path=config_path,
                valid_mask_manifest_path=valid_path,
                g1_checkpoint_selection_path=selection_path,
                g2_parent_gate_path=parent_gate_path,
                g1_code_dir=g1_code_dir,
                code_dir=code_dir,
            )
            self.assertEqual(validated["g1_runtime_code"], g1_runtime)

            g1_files[0].write_text("drifted\n", encoding="utf-8")
            with self.assertRaisesRegex(MetAugContractError, "deployed G1 runtime code"):
                validate_route_a_approval(
                    approval_path,
                    component_manifest=manifest,
                    route_config_path=config_path,
                    valid_mask_manifest_path=valid_path,
                    g1_checkpoint_selection_path=selection_path,
                    g2_parent_gate_path=parent_gate_path,
                    g1_code_dir=g1_code_dir,
                    code_dir=code_dir,
                )

    def test_case_evidence_must_match_the_automatic_result_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts" / "route-a-smoke-001.npz"
            montage = root / "montages" / "route-a-smoke-001.png"
            artifact.parent.mkdir()
            montage.parent.mkdir()
            artifact.write_bytes(b"immutable-npz-evidence")
            montage.write_bytes(b"immutable-montage-evidence")
            entry = {
                "smoke_id": "route-a-smoke-001",
                "entry_sha256": "a" * 64,
                "event_id": "event-1",
                "event_seed": 17,
                "target_case_id": "target",
                "donor_component_id": "donor",
                "core_volume_bin": "27_49",
                "core_volume_mm3": 27.0,
            }
            row = {
                **entry,
                "transaction_state": "COMMITTED",
                "transaction_reason": None,
                "artifact_path": str(artifact.relative_to(root)),
                "artifact_sha256": sha256_file(artifact),
                "montage_path": str(montage.relative_to(root)),
                "montage_sha256": sha256_file(montage),
                "automatic_qc_status": "pass",
                "violations": [],
            }
            row["evidence_fingerprint"] = canonical_json_sha256(row)
            results_path = root / "automatic_case_results.jsonl"
            results_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            smoke_manifest = {"smoke_cases": [entry]}

            loaded = load_case_results_evidence(
                results_path,
                evidence_root=root,
                smoke_manifest=smoke_manifest,
            )
            self.assertEqual(set(loaded), {"route-a-smoke-001"})

            montage.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "evidence drifted"):
                load_case_results_evidence(
                    results_path,
                    evidence_root=root,
                    smoke_manifest=smoke_manifest,
                )

    def test_manual_review_requires_acceptance_and_preserves_evidence_fields(self):
        case_result = {
            "smoke_id": "route-a-smoke-001",
            "evidence_fingerprint": "a" * 64,
            "target_case_id": "target",
            "donor_component_id": "donor",
            "core_volume_bin": "27_49",
            "artifact_path": "artifacts/a.npz",
            "montage_path": "montages/a.png",
            "automatic_qc_status": "pass",
        }
        with tempfile.TemporaryDirectory() as temporary:
            review_path = Path(temporary) / "review.csv"
            row = {
                **case_result,
                "review_decision": "accept",
                "reviewer": "reviewer-a",
                "reviewed_at_utc": "2026-07-25T08:00:00Z",
                "notes": "",
            }
            with review_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_TEMPLATE_FIELDS)
                writer.writeheader()
                writer.writerow(row)

            result = validate_manual_review(
                review_path,
                case_results={case_result["smoke_id"]: case_result},
            )
            self.assertEqual(result["status"], "pass")

            row["target_case_id"] = "changed"
            with review_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_TEMPLATE_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(RuntimeError, "immutable field"):
                validate_manual_review(
                    review_path,
                    case_results={case_result["smoke_id"]: case_result},
                )


if __name__ == "__main__":
    unittest.main()
