#!/usr/bin/env python3
"""Pre-register the immutable, no-Diffusion Route A Gate 2 smoke set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import ComponentManifest, RouteConfig
from custom_nnunet.met_aug_gate2 import (
    MIN_SMOKE_PER_VOLUME_BIN,
    load_valid_mask_assets,
    prepare_smoke_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-seed", type=int, default=20260725)
    parser.add_argument("--per-volume-bin", type=int, default=MIN_SMOKE_PER_VOLUME_BIN)
    parser.add_argument("--max-candidates", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Gate 2 smoke manifest is immutable and already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=set(manifest.target_groups))
    payload = prepare_smoke_manifest(
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        assets=assets,
        search_seed=args.search_seed,
        per_volume_bin=args.per_volume_bin,
        max_candidates=args.max_candidates,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "route_id": payload["route_id"],
        "smoke_manifest": str(output),
        "smoke_manifest_sha256": payload["smoke_manifest_sha256"],
        "smoke_count": payload["smoke_count"],
        "per_volume_bin": payload["per_volume_bin"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
