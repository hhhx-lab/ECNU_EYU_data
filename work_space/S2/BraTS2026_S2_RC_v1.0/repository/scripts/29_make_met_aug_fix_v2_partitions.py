#!/usr/bin/env python3
"""Freeze deterministic train-only Reference/Development/QC patient partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    canonical_json_sha256,
    patient_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--development-fraction", type=float, default=0.15)
    return parser.parse_args()


def build_partition_payload(
    manifest: ComponentManifest,
    *,
    seed: int,
    reference_fraction: float,
    development_fraction: float,
) -> dict:
    if not 0 < reference_fraction < 1 or not 0 < development_fraction < 1:
        raise ValueError("partition fractions must be strictly between zero and one")
    if reference_fraction + development_fraction >= 1:
        raise ValueError("Reference and Development fractions leave no QC holdout")

    case_groups = {str(case_id): str(group) for case_id, group in manifest.target_groups.items()}
    malformed_groups = sorted(
        group for group in set(case_groups.values()) if patient_group(group) != group
    )
    if malformed_groups:
        raise ValueError(
            "target group map contains case/timepoint IDs instead of patient groups: "
            f"{malformed_groups[:10]}"
        )
    groups = sorted(
        set(case_groups.values()),
        key=lambda group: hashlib.sha256(
            f"{seed}|{group}".encode("utf-8")
        ).hexdigest(),
    )
    missing_record_groups = sorted(
        {record.patient_group for record in manifest.records} - set(groups)
    )
    if missing_record_groups:
        raise ValueError(
            "component donors are absent from the frozen train group map: "
            f"{missing_record_groups[:10]}"
        )
    if len(groups) < 7:
        raise ValueError("Fix-v2 calibration requires at least seven patient groups")
    reference_count = max(1, int(round(len(groups) * reference_fraction)))
    development_count = max(1, int(round(len(groups) * development_fraction)))
    if reference_count + development_count >= len(groups):
        development_count = len(groups) - reference_count - 1
    if development_count <= 0:
        raise ValueError("partition rounding left an empty Development or QC holdout")
    partitions = {
        "reference": groups[:reference_count],
        "development": groups[
            reference_count : reference_count + development_count
        ],
        "qc_holdout": groups[reference_count + development_count :],
    }
    group_partition = {
        group: name for name, values in partitions.items() for group in values
    }
    target_case_ids = {
        name: sorted(
            case_id
            for case_id, group in case_groups.items()
            if group_partition[group] == name
        )
        for name in partitions
    }
    component_ids = {
        name: sorted(
            record.component_id
            for record in manifest.records
            if group_partition.get(record.patient_group) == name
        )
        for name in partitions
    }
    case_counts = {name: len(values) for name, values in target_case_ids.items()}
    component_counts = {name: len(values) for name, values in component_ids.items()}
    payload = {
        "schema_version": 1,
        "status": "pass",
        "source_split": "train_only",
        "split_unit": "patient_group",
        "seed": int(seed),
        "hash_policy": "sha256(seed|patient_group)",
        "component_manifest_sha256": manifest.identity_sha256,
        "target_groups_sha256": manifest.target_groups_sha256,
        "reference_fraction_requested": float(reference_fraction),
        "development_fraction_requested": float(development_fraction),
        "patient_group_count": len(groups),
        "target_case_count": len(case_groups),
        "component_count": len(manifest.records),
        "partitions": partitions,
        "target_case_ids": target_case_ids,
        "component_ids": component_ids,
        "target_case_counts": case_counts,
        "component_counts": component_counts,
    }
    payload["partition_audit_sha256"] = canonical_json_sha256(
        payload,
        exclude=("partition_audit_sha256",),
    )
    return payload


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Fix-v2 train-only partition already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    payload = build_partition_payload(
        manifest,
        seed=args.seed,
        reference_fraction=args.reference_fraction,
        development_fraction=args.development_fraction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
