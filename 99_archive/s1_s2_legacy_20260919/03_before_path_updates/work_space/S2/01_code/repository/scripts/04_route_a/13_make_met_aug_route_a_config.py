#!/usr/bin/env python3
"""Freeze the Route A sampling distribution after component-pool creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import ComponentManifest, make_route_a_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-total-support-voxels", type=int)
    parser.add_argument("--max-total-to-core-ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Route A config is immutable and already exists: {output}")
    if (args.max_total_support_voxels is None) != (
        args.max_total_to_core_ratio is None
    ):
        raise ValueError(
            "--max-total-support-voxels and --max-total-to-core-ratio must be provided together"
        )
    manifest = ComponentManifest.load(args.component_manifest)
    config = make_route_a_config(
        manifest,
        seed=args.seed,
        max_total_support_voxels=args.max_total_support_voxels,
        max_total_to_core_ratio=args.max_total_to_core_ratio,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "route_id": config["route_id"],
        "output": str(output),
        "component_manifest_sha256": manifest.identity_sha256,
        "strata": len(config["strata"]),
        "donor_eligibility": config.get("donor_eligibility"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
