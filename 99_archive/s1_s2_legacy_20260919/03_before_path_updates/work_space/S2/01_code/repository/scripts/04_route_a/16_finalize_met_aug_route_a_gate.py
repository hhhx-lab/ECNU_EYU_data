#!/usr/bin/env python3
"""Bind passing Route A gates into the only training approval artifact.

This command is intentionally read-only. It cannot call G1 Diffusion or start
nnU-Net; it only verifies immutable evidence and writes a new approval JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_gate import build_route_a_approval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--gate1-report", required=True)
    parser.add_argument("--gate2-report", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--code-dir", default=str(REPOSITORY_ROOT / "custom_nnunet"))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Route A approval is immutable and already exists: {output}")
    approval = build_route_a_approval(
        component_manifest_path=args.component_manifest,
        route_config_path=args.route_config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        gate1_report_path=args.gate1_report,
        gate2_report_path=args.gate2_report,
        g1_checkpoint_selection_path=args.g1_checkpoint_selection,
        g2_parent_gate_path=args.g2_parent_gate,
        g1_code_dir=args.g1_code_dir,
        code_dir=args.code_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(approval, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "route_id": approval["route_id"],
        "decision": approval["decision"],
        "approval": str(output),
        "approval_sha256": approval["approval_sha256"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
