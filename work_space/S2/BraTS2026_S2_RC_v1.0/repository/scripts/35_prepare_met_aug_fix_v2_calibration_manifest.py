#!/usr/bin/env python3
"""Freeze Development or QC-holdout Fix-v2 backend events."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    RouteConfig,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2_calibration import load_partition  # noqa: E402
from custom_nnunet.met_aug_gate2 import (  # noqa: E402
    load_valid_mask_assets,
    prepare_smoke_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--partition", choices=("development", "qc_holdout", "reference"), required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--per-volume-bin", type=int, required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument("--smoke-id-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"calibration event manifest already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    partition = load_partition(args.partition_audit)
    groups = set(str(value) for value in partition["partitions"][args.partition])
    target_ids = set(str(value) for value in partition["target_case_ids"][args.partition])
    if target_ids != {
        case_id for case_id, group in manifest.target_groups.items() if group in groups
    }:
        raise ValueError("partition target IDs drifted from component manifest")
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=target_ids)
    payload = prepare_smoke_manifest(
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        assets=assets,
        search_seed=args.search_seed,
        per_volume_bin=args.per_volume_bin,
        max_candidates=args.max_candidates,
        allowed_target_groups=groups,
        allowed_donor_groups=groups,
        smoke_id_prefix=args.smoke_id_prefix,
    )
    payload["calibration_partition"] = args.partition
    payload["partition_file_sha256"] = sha256_file(args.partition_audit)
    # Added fields are part of the immutable identity.
    from custom_nnunet.met_aug_core import canonical_json_sha256

    payload["smoke_manifest_sha256"] = canonical_json_sha256(
        payload, exclude=("smoke_manifest_sha256",)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "pass",
                "partition": args.partition,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "event_count": payload["smoke_count"],
                "per_volume_bin": payload["per_volume_bin"],
                "selection_scope": payload["selection_scope"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
