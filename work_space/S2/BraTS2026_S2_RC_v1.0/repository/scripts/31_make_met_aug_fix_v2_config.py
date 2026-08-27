#!/usr/bin/env python3
"""Freeze a schema-4 Route A config against an approved Fix-v2 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    make_fix_v2_route_a_config,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--boundary-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-total-support-voxels", type=int)
    parser.add_argument("--max-total-to-core-ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Fix-v2 Route A config already exists: {output}")
    if (args.max_total_support_voxels is None) != (
        args.max_total_to_core_ratio is None
    ):
        raise ValueError("compact-support thresholds must be supplied together")

    manifest = ComponentManifest.load(args.component_manifest)
    calibration = FixV2Calibration.load(
        args.calibration,
        expected_policy=args.boundary_policy,
    )
    source_audit = calibration.payload["source_audit"]
    if source_audit["component_manifest_sha256"] != manifest.identity_sha256:
        raise ValueError("Fix-v2 calibration was frozen from a different component manifest")
    if source_audit["target_groups_sha256"] != manifest.target_groups_sha256:
        raise ValueError("Fix-v2 calibration was frozen from a different target group map")
    config = make_fix_v2_route_a_config(
        manifest,
        boundary_policy=calibration.boundary_policy,
        calibration_sha256=calibration.sha256,
        seed=args.seed,
        max_total_support_voxels=args.max_total_support_voxels,
        max_total_to_core_ratio=args.max_total_to_core_ratio,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "route_id": config["route_id"],
                "schema_version": config["schema_version"],
                "boundary_policy": calibration.boundary_policy,
                "calibration_sha256": calibration.sha256,
                "output": str(output),
                "donor_eligibility": config.get("donor_eligibility"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
