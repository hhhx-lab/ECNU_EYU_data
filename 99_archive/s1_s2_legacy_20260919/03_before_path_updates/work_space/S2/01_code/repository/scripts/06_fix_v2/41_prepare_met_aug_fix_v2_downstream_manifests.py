#!/usr/bin/env python3
"""Prepare one disjoint Reference pool for Gate-1B (96) and Gate-2 (120)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    RouteConfig,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2_calibration import load_partition  # noqa: E402
from custom_nnunet.met_aug_gate2 import (  # noqa: E402
    load_valid_mask_assets,
    prepare_smoke_manifest,
)


VOLUME_BINS = ("27_49", "50_275", "gt_275")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    return parser.parse_args()


def _encoded(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _child_manifest(
    master: dict[str, Any],
    *,
    selected_ids: set[str],
    per_volume_bin: int,
    stage: str,
    master_file_sha256: str,
) -> dict[str, Any]:
    child = deepcopy(master)
    child["smoke_cases"] = [
        value for value in master["smoke_cases"] if value["smoke_id"] in selected_ids
    ]
    child["smoke_count"] = len(child["smoke_cases"])
    child["per_volume_bin_quota"] = per_volume_bin
    child["per_volume_bin"] = {
        volume_bin: sum(
            value["core_volume_bin"] == volume_bin for value in child["smoke_cases"]
        )
        for volume_bin in VOLUME_BINS
    }
    child["downstream_stage"] = stage
    child["downstream_master_file_sha256"] = master_file_sha256
    child["smoke_manifest_sha256"] = canonical_json_sha256(
        child, exclude=("smoke_manifest_sha256",)
    )
    return child


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"downstream manifest output already exists: {output_dir}")
    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    if config.fix_v2 is None or config.fix_v2.boundary_policy != "label_only_qc_v1":
        raise ValueError("downstream manifests require the selected A route config")
    partition_path = Path(args.partition_audit).expanduser().resolve()
    partition = load_partition(partition_path)
    reference_groups = set(str(value) for value in partition["partitions"]["reference"])
    target_ids = set(
        str(value) for value in partition["target_case_ids"]["reference"]
    )
    if target_ids != {
        case_id
        for case_id, group in manifest.target_groups.items()
        if group in reference_groups
    }:
        raise ValueError("Reference target scope drifted from the component manifest")
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=target_ids)
    master = prepare_smoke_manifest(
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        assets=assets,
        search_seed=args.search_seed,
        per_volume_bin=72,
        max_candidates=args.max_candidates,
        allowed_target_groups=reference_groups,
        allowed_donor_groups=reference_groups,
        smoke_id_prefix="fix-v2-downstream",
    )
    master["calibration_partition"] = "reference"
    master["partition_file_sha256"] = sha256_file(partition_path)
    master["downstream_split_contract"] = {
        "gate1b_per_volume_bin": 32,
        "gate2_per_volume_bin": 40,
        "target_and_donor_unique_across_both_stages": True,
        "split_rule": "within-bin frozen search order: first 32 Gate-1B, next 40 Gate-2",
    }
    master["smoke_manifest_sha256"] = canonical_json_sha256(
        master, exclude=("smoke_manifest_sha256",)
    )
    master_bytes = _encoded(master)
    master_file_sha256 = hashlib.sha256(master_bytes).hexdigest()
    by_bin = {
        volume_bin: [
            value for value in master["smoke_cases"] if value["core_volume_bin"] == volume_bin
        ]
        for volume_bin in VOLUME_BINS
    }
    if any(len(values) != 72 for values in by_bin.values()):
        raise RuntimeError("downstream master pool does not contain exactly 72 per bin")
    gate1b_ids = {
        value["smoke_id"]
        for volume_bin in VOLUME_BINS
        for value in by_bin[volume_bin][:32]
    }
    gate2_ids = {
        value["smoke_id"]
        for volume_bin in VOLUME_BINS
        for value in by_bin[volume_bin][32:]
    }
    if gate1b_ids & gate2_ids or len(gate1b_ids) != 96 or len(gate2_ids) != 120:
        raise RuntimeError("downstream split is not disjoint 96/120")
    gate1b = _child_manifest(
        master,
        selected_ids=gate1b_ids,
        per_volume_bin=32,
        stage="gate1b_replay",
        master_file_sha256=master_file_sha256,
    )
    gate2 = _child_manifest(
        master,
        selected_ids=gate2_ids,
        per_volume_bin=40,
        stage="gate2",
        master_file_sha256=master_file_sha256,
    )
    target_overlap = {
        value["target_case_id"] for value in gate1b["smoke_cases"]
    } & {value["target_case_id"] for value in gate2["smoke_cases"]}
    donor_overlap = {
        value["donor_component_id"] for value in gate1b["smoke_cases"]
    } & {value["donor_component_id"] for value in gate2["smoke_cases"]}
    if target_overlap or donor_overlap:
        raise RuntimeError("downstream child manifests overlap targets or donors")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "master_count": 216,
        "gate1b_count": 96,
        "gate2_count": 120,
        "master_file_sha256": master_file_sha256,
        "gate1b_file_sha256": hashlib.sha256(_encoded(gate1b)).hexdigest(),
        "gate2_file_sha256": hashlib.sha256(_encoded(gate2)).hexdigest(),
        "target_overlap_count": 0,
        "donor_overlap_count": 0,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(config.path),
        "partition_file_sha256": sha256_file(partition_path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "search_seed": args.search_seed,
    }
    audit["audit_sha256"] = canonical_json_sha256(
        audit, exclude=("audit_sha256",)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "DOWNSTREAM_MASTER_216.json", master_bytes)
    _write(output_dir / "GATE1B_EVENTS_96.json", _encoded(gate1b))
    _write(output_dir / "GATE2_EVENTS_120.json", _encoded(gate2))
    _write(output_dir / "DOWNSTREAM_MANIFEST_VALIDATION.json", _encoded(audit))
    output_dir.chmod(0o555)
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
