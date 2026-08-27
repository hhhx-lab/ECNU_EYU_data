#!/usr/bin/env python3
"""Bind completed human review to an immutable passing Route A Gate 2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import ComponentManifest, RouteConfig, canonical_json_sha256, sha256_file
from custom_nnunet.met_aug_gate2 import (
    GATE2_FINAL_REPORT_SCHEMA,
    load_case_results_evidence,
    load_smoke_manifest,
    validate_automatic_report,
    validate_manual_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--automatic-report", required=True)
    parser.add_argument("--review-decisions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Gate 2 final report is immutable and already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    smoke_manifest = load_smoke_manifest(
        args.smoke_manifest,
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    automatic_report_path = Path(args.automatic_report).expanduser().resolve()
    automatic = validate_automatic_report(
        automatic_report_path,
        smoke_manifest=smoke_manifest,
        repository_root=REPOSITORY_ROOT,
    )
    case_results_path = automatic_report_path.parent / automatic["case_results_file"]
    if not case_results_path.is_file() or sha256_file(case_results_path) != automatic.get("case_results_sha256"):
        raise RuntimeError("Gate 2 automatic case-results evidence is missing or drifted")
    case_results = load_case_results_evidence(
        case_results_path,
        evidence_root=automatic_report_path.parent,
        smoke_manifest=smoke_manifest,
    )
    review = validate_manual_review(args.review_decisions, case_results=case_results)
    report = {
        "schema_version": GATE2_FINAL_REPORT_SCHEMA,
        "route_id": config.route_id,
        "status": review["status"],
        "manual_review_status": review["status"],
        "smoke_count": int(smoke_manifest["smoke_count"]),
        "per_volume_bin": smoke_manifest["per_volume_bin"],
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(config.path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "smoke_manifest_sha256": smoke_manifest["smoke_manifest_sha256"],
        "automatic_report_sha256": sha256_file(args.automatic_report),
        "review_decisions_sha256": sha256_file(args.review_decisions),
        "g1_checkpoint_selection_sha256": automatic["g1_checkpoint_selection_sha256"],
        "g2_parent_gate_sha256": automatic["g2_parent_gate_sha256"],
        "g1_runtime_code": automatic["g1_runtime_code"],
        "review": review,
    }
    if config.fix_v2 is not None:
        if automatic.get("fix_v2") != config.fix_v2.as_mapping():
            raise RuntimeError("Gate 2 automatic report Fix-v2 policy drifted")
        report["fix_v2"] = config.fix_v2.as_mapping()
    report["gate2_report_sha256"] = canonical_json_sha256(report, exclude=("gate2_report_sha256",))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "route_id": report["route_id"],
        "output": str(output),
        "gate2_report_sha256": report["gate2_report_sha256"],
        "review": report["review"],
    }, ensure_ascii=True, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
