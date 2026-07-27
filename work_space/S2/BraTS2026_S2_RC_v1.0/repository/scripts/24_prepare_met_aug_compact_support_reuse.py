#!/usr/bin/env python3
"""Freeze an R5 compact-support config against read-only R4 preparation assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (
    VALID_MASK_MANIFEST_SCHEMA,
    ComponentManifest,
    RouteConfig,
    canonical_json_sha256,
    make_route_a_config,
    sha256_file,
)


def parse_key_value_counts(values: list[str], *, label: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        key, separator, count = value.partition("=")
        if not separator or not key or key in parsed:
            raise ValueError(f"invalid or duplicated {label}: {value!r}")
        parsed[key] = int(count)
    return dict(sorted(parsed.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-route-root", required=True)
    parser.add_argument("--target-route-root", required=True)
    parser.add_argument("--route-config-output", required=True)
    parser.add_argument("--reuse-audit-output", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-total-support-voxels", type=int, required=True)
    parser.add_argument("--max-total-to-core-ratio", type=float, required=True)
    parser.add_argument("--expected-eligible-component-count", type=int, required=True)
    parser.add_argument(
        "--expected-eligible-bin-count", action="append", default=[], required=True
    )
    parser.add_argument(
        "--expected-eligible-group-count", action="append", default=[], required=True
    )
    parser.add_argument("--expected-prepare-marker-sha256", required=True)
    parser.add_argument("--expected-coordinate-audit-sha256", required=True)
    parser.add_argument("--expected-component-manifest-file-sha256", required=True)
    parser.add_argument("--expected-component-manifest-identity-sha256", required=True)
    parser.add_argument("--expected-valid-mask-manifest-file-sha256", required=True)
    parser.add_argument("--expected-valid-mask-manifest-identity-sha256", required=True)
    return parser.parse_args()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def require_file_sha256(path: Path, expected: str, *, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA256 drifted: observed={observed} expected={expected}"
        )
    return observed


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def require_target_path(path: Path, target_root: Path, *, label: str) -> None:
    if path == target_root or target_root not in path.parents:
        raise ValueError(f"{label} must be inside the target route root: {path}")
    if path.exists():
        raise FileExistsError(f"immutable {label} already exists: {path}")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_route_root).expanduser().resolve()
    target_root = Path(args.target_route_root).expanduser().resolve()
    route_config_path = Path(args.route_config_output).expanduser().resolve()
    reuse_audit_path = Path(args.reuse_audit_output).expanduser().resolve()
    if source_root == target_root:
        raise ValueError("source and target route roots must be independent")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source route root is missing: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    require_target_path(route_config_path, target_root, label="route config")
    require_target_path(reuse_audit_path, target_root, label="reuse audit")

    expected_bin_counts = parse_key_value_counts(
        args.expected_eligible_bin_count,
        label="eligible bin count",
    )
    expected_group_counts = parse_key_value_counts(
        args.expected_eligible_group_count,
        label="eligible group count",
    )

    prepare_marker = source_root / "PREPARE_COMPLETE.ok"
    coordinate_path = source_root / "preprocessed_coordinate_contract.json"
    component_manifest_path = source_root / "component_pool" / "component_manifest.json"
    valid_mask_manifest_path = source_root / "valid_masks" / "valid_mask_manifest.json"
    for path, expected, label in (
        (
            prepare_marker,
            args.expected_prepare_marker_sha256,
            "source prepare marker",
        ),
        (
            coordinate_path,
            args.expected_coordinate_audit_sha256,
            "source coordinate audit",
        ),
        (
            component_manifest_path,
            args.expected_component_manifest_file_sha256,
            "source component manifest file",
        ),
        (
            valid_mask_manifest_path,
            args.expected_valid_mask_manifest_file_sha256,
            "source valid-mask manifest file",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        require_file_sha256(path, expected, label=label)

    coordinate = load_json(coordinate_path, label="coordinate audit")
    if coordinate.get("status") != "pass" or int(coordinate.get("case_count", -1)) != 1138:
        raise ValueError("source coordinate audit is not the locked 1138-case pass")

    component_manifest_payload = load_json(
        component_manifest_path, label="component manifest"
    )
    manifest = ComponentManifest.load(component_manifest_path)
    if manifest.identity_sha256 != args.expected_component_manifest_identity_sha256:
        raise ValueError("source component manifest identity SHA256 drifted")
    if int(component_manifest_payload.get("component_count", -1)) != len(manifest.records):
        raise ValueError("source component count is inconsistent")
    if int(component_manifest_payload.get("train_count", -1)) != len(
        manifest.target_groups
    ):
        raise ValueError("source component manifest train count is inconsistent")

    valid_mask_manifest = load_json(
        valid_mask_manifest_path, label="valid-mask manifest"
    )
    if valid_mask_manifest.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
        raise ValueError("source valid-mask manifest schema drifted")
    if valid_mask_manifest.get("manifest_sha256") != canonical_json_sha256(
        valid_mask_manifest, exclude=("manifest_sha256",)
    ):
        raise ValueError("source valid-mask manifest identity SHA256 is invalid")
    if (
        valid_mask_manifest["manifest_sha256"]
        != args.expected_valid_mask_manifest_identity_sha256
    ):
        raise ValueError("source valid-mask manifest identity SHA256 drifted")
    if int(valid_mask_manifest.get("train_count", -1)) != len(manifest.target_groups):
        raise ValueError("source valid-mask train count drifted")
    if (
        valid_mask_manifest.get("resampling_backend")
        != "nnunet_configuration_resampling_fn_seg"
    ):
        raise ValueError("source valid-mask resampling backend drifted")
    valid_records_path = valid_mask_manifest_path.parent / str(
        valid_mask_manifest.get("records_file", "")
    )
    if not valid_records_path.is_file() or sha256_file(valid_records_path) != valid_mask_manifest.get(
        "records_sha256"
    ):
        raise ValueError("source valid-mask records SHA256 drifted")
    valid_case_ids = {
        str(json.loads(line)["case_id"])
        for line in valid_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if valid_case_ids != set(manifest.target_groups):
        raise ValueError("source valid-mask IDs do not match source train targets")

    config_payload = make_route_a_config(
        manifest,
        seed=args.seed,
        max_total_support_voxels=args.max_total_support_voxels,
        max_total_to_core_ratio=args.max_total_to_core_ratio,
    )
    eligibility = config_payload["donor_eligibility"]
    if eligibility["eligible_component_count"] != args.expected_eligible_component_count:
        raise ValueError("compact-support eligible component count is unexpected")
    if eligibility["eligible_by_core_volume_bin"] != expected_bin_counts:
        raise ValueError("compact-support eligible volume-bin counts are unexpected")
    if (
        eligibility["eligible_patient_groups_by_core_volume_bin"]
        != expected_group_counts
    ):
        raise ValueError("compact-support eligible patient-group counts are unexpected")

    write_new_json(route_config_path, config_payload)
    loaded_config = RouteConfig.load(route_config_path, manifest)
    if loaded_config.donor_eligibility is None:
        raise RuntimeError("written Route A config did not retain compact-support policy")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reuse_mode": "direct_read_only_path_reference_no_asset_copy",
        "source_route_root": str(source_root),
        "target_route_root": str(target_root),
        "source_prepare_marker": {
            "path": str(prepare_marker),
            "sha256": args.expected_prepare_marker_sha256,
        },
        "coordinate_audit": {
            "path": str(coordinate_path),
            "sha256": args.expected_coordinate_audit_sha256,
            "status": coordinate["status"],
            "case_count": coordinate["case_count"],
        },
        "component_manifest": {
            "path": str(component_manifest_path),
            "file_sha256": args.expected_component_manifest_file_sha256,
            "identity_sha256": manifest.identity_sha256,
            "records_sha256": manifest.records_sha256,
            "component_count": len(manifest.records),
            "train_count": len(manifest.target_groups),
        },
        "valid_mask_manifest": {
            "path": str(valid_mask_manifest_path),
            "file_sha256": args.expected_valid_mask_manifest_file_sha256,
            "identity_sha256": valid_mask_manifest["manifest_sha256"],
            "records_sha256": valid_mask_manifest["records_sha256"],
            "train_count": valid_mask_manifest["train_count"],
            "resampling_backend": valid_mask_manifest["resampling_backend"],
        },
        "route_config": {
            "path": str(route_config_path),
            "sha256": sha256_file(route_config_path),
            "schema_version": loaded_config.schema_version,
            "seed": loaded_config.seed,
            "p_select": loaded_config.p_select,
            "donor_eligibility": eligibility,
        },
        "source_route_files_written": False,
    }
    audit["audit_sha256"] = canonical_json_sha256(
        audit, exclude=("audit_sha256",)
    )
    write_new_json(reuse_audit_path, audit)
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
