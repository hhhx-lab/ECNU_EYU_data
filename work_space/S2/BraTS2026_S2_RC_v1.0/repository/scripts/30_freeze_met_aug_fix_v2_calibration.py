#!/usr/bin/env python3
"""Freeze an empirically prepared Fix-v2 calibration with source-evidence hashes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file  # noqa: E402
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--reference-cdf", required=True)
    parser.add_argument("--patient-group-count", required=True, type=int)
    parser.add_argument("--component-count", required=True, type=int)
    parser.add_argument("--boundary-policy", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


PARTITION_NAMES = ("reference", "development", "qc_holdout")


def _sha256_field(value: object, *, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} is not a lowercase SHA256")
    return result


def _validate_partition_audit(
    partition: dict,
    *,
    expected_count: int,
    expected_component_count: int,
) -> None:
    if partition.get("schema_version") != 1 or partition.get("status") != "pass":
        raise ValueError("train-only partition audit is not a passing schema-1 artifact")
    if partition.get("source_split") != "train_only":
        raise ValueError("calibration partition is not explicitly train-only")
    if partition.get("split_unit") != "patient_group":
        raise ValueError("calibration partition must be grouped by patient")
    partitions = partition.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITION_NAMES):
        raise ValueError("calibration partition must define Reference/Development/QC holdout")
    expected_audit = canonical_json_sha256(
        partition,
        exclude=("partition_audit_sha256",),
    )
    if partition.get("partition_audit_sha256") != expected_audit:
        raise ValueError("train-only partition audit SHA256 has drifted")
    for key in ("component_manifest_sha256", "target_groups_sha256"):
        _sha256_field(partition.get(key), label=f"partition {key}")
    observed: set[str] = set()
    for name in PARTITION_NAMES:
        groups = partitions[name]
        if (
            not isinstance(groups, list)
            or not groups
            or any(not isinstance(value, str) or not value for value in groups)
            or len(groups) != len(set(groups))
        ):
            raise ValueError(f"calibration partition {name} is empty or duplicated")
        overlap = observed & set(groups)
        if overlap:
            raise ValueError(f"calibration patient groups overlap across partitions: {sorted(overlap)}")
        observed.update(groups)
    if len(observed) != expected_count or int(partition.get("patient_group_count", -1)) != expected_count:
        raise ValueError("calibration partition patient-group count disagrees with CLI audit")

    for member_key, count_key, total_key, expected_total in (
        ("target_case_ids", "target_case_counts", "target_case_count", None),
        ("component_ids", "component_counts", "component_count", expected_component_count),
    ):
        members = partition.get(member_key)
        counts = partition.get(count_key)
        if not isinstance(members, dict) or set(members) != set(PARTITION_NAMES):
            raise ValueError(f"partition {member_key} does not cover all partitions")
        if not isinstance(counts, dict) or set(counts) != set(PARTITION_NAMES):
            raise ValueError(f"partition {count_key} does not cover all partitions")
        observed_members: set[str] = set()
        total = 0
        for name in PARTITION_NAMES:
            values = members[name]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"partition {name} has empty or duplicate {member_key}")
            overlap = observed_members & set(values)
            if overlap:
                raise ValueError(f"{member_key} overlap across partitions: {sorted(overlap)}")
            observed_members.update(values)
            total += len(values)
            if int(counts.get(name, -1)) != len(values):
                raise ValueError(f"partition {name} {member_key} count has drifted")
        if int(partition.get(total_key, -1)) != total:
            raise ValueError(f"partition {total_key} has drifted")
        if expected_total is not None and total != expected_total:
            raise ValueError("calibration component count disagrees with CLI audit")


def _validate_reference_evidence(
    reference: dict,
    *,
    partition: dict,
    partition_path: Path,
) -> None:
    required = {
        "schema_version",
        "status",
        "source_partition",
        "partition_sha256",
        "partition_audit_sha256",
        "component_manifest_sha256",
        "target_groups_sha256",
        "patient_groups",
        "target_case_ids",
        "component_ids",
        "patient_group_count",
        "target_case_count",
        "component_count",
        "reference_cdf_audit_sha256",
    }
    missing = sorted(required - set(reference))
    if missing:
        raise ValueError(f"reference CDF evidence misses fields: {missing}")
    if reference.get("schema_version") != 1 or reference.get("status") != "pass":
        raise ValueError("reference CDF evidence is not a passing schema-1 artifact")
    if reference.get("source_partition") != "reference":
        raise ValueError("reference CDF evidence is not restricted to the Reference partition")
    expected_audit = canonical_json_sha256(
        reference,
        exclude=("reference_cdf_audit_sha256",),
    )
    if reference.get("reference_cdf_audit_sha256") != expected_audit:
        raise ValueError("reference CDF evidence audit SHA256 has drifted")
    if reference.get("partition_sha256") != sha256_file(partition_path):
        raise ValueError("reference CDF evidence does not bind the partition file")
    for key in (
        "partition_audit_sha256",
        "component_manifest_sha256",
        "target_groups_sha256",
    ):
        if reference.get(key) != partition.get(key):
            raise ValueError(f"reference CDF evidence {key} disagrees with the partition audit")
    expected_members = {
        "patient_groups": partition["partitions"]["reference"],
        "target_case_ids": partition["target_case_ids"]["reference"],
        "component_ids": partition["component_ids"]["reference"],
    }
    for key, expected in expected_members.items():
        if reference.get(key) != expected:
            raise ValueError(f"reference CDF evidence {key} drifted from the frozen partition")
    for key, member_key in (
        ("patient_group_count", "patient_groups"),
        ("target_case_count", "target_case_ids"),
        ("component_count", "component_ids"),
    ):
        if int(reference.get(key, -1)) != len(expected_members[member_key]):
            raise ValueError(f"reference CDF evidence {key} has drifted")


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"frozen Fix-v2 calibration already exists: {output}")
    draft_path = Path(args.draft).expanduser().resolve()
    partition_path = Path(args.partition_audit).expanduser().resolve()
    reference_path = Path(args.reference_cdf).expanduser().resolve()
    draft = _read_json_object(draft_path, "calibration draft")
    partition = _read_json_object(partition_path, "partition audit")
    reference = _read_json_object(reference_path, "reference CDF evidence")
    if draft.get("status") != "draft":
        raise ValueError("calibration input must have status=draft")
    if args.patient_group_count <= 0 or args.component_count <= 0:
        raise ValueError("calibration source counts must be positive")
    _validate_partition_audit(
        partition,
        expected_count=args.patient_group_count,
        expected_component_count=args.component_count,
    )
    _validate_reference_evidence(
        reference,
        partition=partition,
        partition_path=partition_path,
    )

    payload = dict(draft)
    payload["status"] = "frozen"
    payload["boundary_policy"] = args.boundary_policy
    payload["source_audit"] = {
        "partition_sha256": sha256_file(partition_path),
        "partition_audit_sha256": partition["partition_audit_sha256"],
        "reference_cdf_sha256": sha256_file(reference_path),
        "reference_cdf_audit_sha256": reference["reference_cdf_audit_sha256"],
        "component_manifest_sha256": partition["component_manifest_sha256"],
        "target_groups_sha256": partition["target_groups_sha256"],
        "patient_group_count": args.patient_group_count,
        "component_count": args.component_count,
    }
    FixV2Calibration.validate_payload(
        payload,
        expected_policy=args.boundary_policy,
    )
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    frozen = FixV2Calibration.load(
        output,
        expected_policy=args.boundary_policy,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "boundary_policy": frozen.boundary_policy,
                "calibration": str(output),
                "calibration_sha256": frozen.sha256,
                "partition_file_sha256": payload["source_audit"]["partition_sha256"],
                "partition_audit_sha256": payload["source_audit"][
                    "partition_audit_sha256"
                ],
                "reference_cdf_sha256": payload["source_audit"]["reference_cdf_sha256"],
                "reference_cdf_audit_sha256": payload["source_audit"][
                    "reference_cdf_audit_sha256"
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
