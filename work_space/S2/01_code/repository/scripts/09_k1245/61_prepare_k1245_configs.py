#!/usr/bin/env python3
"""Freeze K1/K2/K4/K5 configs from the immutable train-only pool evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file  # noqa: E402
from custom_nnunet.met_aug_k1245 import K1245Config, K1245Manifest  # noqa: E402


def exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--legacy-route-config", required=True, type=Path)
    parser.add_argument("--legacy-component-manifest", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_candidates = sorted(root.glob("K1245_RC_AWARE_EXECUTION_PLAN_*.json"))
    if len(plan_candidates) != 1:
        raise SystemExit(f"expected one K1245 plan, found {len(plan_candidates)}")
    plan = plan_candidates[0]
    evidence_path = root / "pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") not in {"pass", "blocked_f_gate"}:
        raise SystemExit("train-only pool prerequisite did not finish")
    empirical = evidence["empirical"]
    legacy_route = json.loads(args.legacy_route_config.read_text(encoding="utf-8"))
    legacy_manifest = json.loads(args.legacy_component_manifest.read_text(encoding="utf-8"))
    calibration_identity = str(legacy_route.get("fix_v3", {}).get("calibration_sha256", ""))
    legacy_manifest_identity = str(legacy_manifest.get("manifest_sha256", ""))
    if len(calibration_identity) != 64 or len(legacy_manifest_identity) != 64:
        raise SystemExit("legacy FixV3 route/manifest identity is incomplete")
    route_contracts = {
        "K1": {"manifest": "rc27", "features": {"b": True, "d": False, "e": True, "f": True, "x2": False, "x3": False}},
        "K2": {"manifest": "rc27", "features": {"b": True, "d": True, "e": True, "f": False, "x2": True, "x3": False}},
        "K4": {"manifest": "small8", "features": {"b": False, "d": False, "e": True, "f": False, "x2": False, "x3": True}},
        "K5": {"manifest": "rc27", "features": {"b": True, "d": True, "e": True, "f": True, "x2": False, "x3": False}},
    }
    config_paths = {}
    for route, contract in route_contracts.items():
        if contract["features"]["f"] and not empirical.get("f_gate_pass"):
            continue
        manifest_path = root / "pools" / contract["manifest"] / "component_manifest.json"
        manifest = K1245Manifest.load(manifest_path)
        payload = {
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "schema_version": 1,
            "route_id": route,
            "plan_sha256": sha256_file(plan),
            "manifest_sha256": manifest.identity_sha256,
            "legacy_fix_v3_calibration_sha256": calibration_identity,
            "legacy_component_manifest_identity_sha256": legacy_manifest_identity,
            "legacy_component_manifest_file_sha256": sha256_file(args.legacy_component_manifest),
            "p_select": 0.20,
            "min_anchor_volume_mm3": 8.0 if route == "K4" else 27.0,
            "max_bbox_mm": 56.0,
            "x1_rc_probability": 0.0 if route == "K4" else 0.023,
            "second_tumour_probability": 0.20 if contract["features"]["d"] else 0.0,
            "e_probability": 0.10,
            "features": contract["features"],
            "empirical": {
                "e_shell_factor": empirical["e_shell_factor"],
                "e_volume_p10_mm3": empirical["e_volume_p10_mm3"],
                "e_volume_p90_mm3": empirical["e_volume_p90_mm3"],
                "burden_probabilities": empirical["burden_probabilities"] if contract["features"]["f"] else {},
                "target_counts": empirical["target_counts"] if contract["features"]["f"] else {},
                "rank_volume_samples": empirical["rank_volume_samples"] if contract["features"]["d"] else {},
                "nn_distance_samples_mm": empirical["nn_distance_samples_mm"] if contract["features"]["d"] else [],
            },
        }
        payload["config_sha256"] = canonical_json_sha256(payload, exclude=("config_sha256",))
        path = root / "config" / f"K1245_ROUTE_{route}.json"
        exclusive_json(path, payload)
        K1245Config.load(path, manifest=manifest, plan_path=plan)
        config_paths[route] = {"path": str(path), "sha256": sha256_file(path)}
    exclusive_json(root / "evidence/K1245_CONFIG_PREPARATION.json", {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan_sha256": sha256_file(plan),
        "pool_evidence_sha256": sha256_file(evidence_path),
        "legacy_route_config_sha256": sha256_file(args.legacy_route_config),
        "legacy_component_manifest_file_sha256": sha256_file(args.legacy_component_manifest),
        "configs": config_paths,
    })


if __name__ == "__main__":
    main()
