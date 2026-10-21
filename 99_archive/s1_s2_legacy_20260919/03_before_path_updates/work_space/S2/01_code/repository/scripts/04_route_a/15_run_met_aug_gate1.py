#!/usr/bin/env python3
"""Run the Route A 100,000-event strategy gate without calling Diffusion."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_gate1 import MIN_GATE1_EVENTS, run_gate1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--target-seed", type=int, default=20260725)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_gate1(
        component_manifest_path=args.component_manifest,
        route_config_path=args.route_config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        output_dir=args.output_dir,
        events=args.events,
        target_seed=args.target_seed,
        workers=args.workers,
        minimum_events=MIN_GATE1_EVENTS,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
