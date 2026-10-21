#!/usr/bin/env python3
"""Prepare only the K1 config after an independently validated fallback gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "runtime/repository"
sys.path.insert(0, str(REPOSITORY))
from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file
from custom_nnunet.met_aug_k1245 import K1245Config, K1245Manifest


def exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-root", required=True, type=Path)
    parser.add_argument("--legacy-route-config", required=True, type=Path)
    parser.add_argument("--legacy-component-manifest", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    args = parser.parse_args()
    plan = ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
    gate_path = ROOT / "evidence/F_FALLBACK_GATE_VALIDATION.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass" or gate.get("F_STRICT_PREREG_GATE") is not False:
        raise SystemExit("fallback eligibility is not pass with strict gate preserved false")
    if gate.get("gate_name") != "F_FALLBACK_ENGINEERING_ELIGIBILITY_GATE":
        raise SystemExit("wrong fallback gate identity")
    r2_evidence = json.loads((args.r2_root / "pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json").read_text())
    if r2_evidence.get("status") != "blocked_f_gate":
        raise SystemExit("r2 strict gate result drifted")
    empirical = r2_evidence["empirical"]
    legacy_route = json.loads(args.legacy_route_config.read_text(encoding="utf-8"))
    legacy_manifest = json.loads(args.legacy_component_manifest.read_text(encoding="utf-8"))
    r2_plan = json.loads((args.r2_root / "K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json").read_text())
    calibration_sha = legacy_route.get("fix_v3", {}).get("calibration_sha256", "")
    legacy_manifest_identity = legacy_manifest.get("manifest_sha256", "")
    if len(calibration_sha) != 64 or len(legacy_manifest_identity) != 64:
        raise SystemExit("legacy FixV3 identity incomplete")
    parent = r2_plan["parent_contract"]
    checks = (
        (sha256_file(args.legacy_route_config), parent["fix_v3_route_config_file_sha256"]),
        (sha256_file(args.legacy_component_manifest), parent["legacy_component_manifest_file_sha256"]),
        (legacy_manifest_identity, parent["legacy_component_manifest_identity_sha256"]),
        (sha256_file(args.calibration), parent["fix_v3_calibration_file_sha256"]),
    )
    if any(actual != expected for actual, expected in checks):
        raise SystemExit("legacy FixV3 inputs drifted from immutable r2 plan")
    manifest_path = args.r2_root / "pools/rc27/component_manifest.json"
    manifest = K1245Manifest.load(manifest_path)
    expected = json.loads(plan.read_text(encoding="utf-8"))
    if manifest.identity_sha256 != expected["fallback_gate"]["source_rc27_manifest_identity_sha256"]:
        raise SystemExit("K1 manifest is not the immutable r2 RC27 manifest")
    payload = {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "schema_version": 1,
        "route_id": "K1",
        "plan_sha256": sha256_file(plan),
        "manifest_sha256": manifest.identity_sha256,
        "legacy_fix_v3_calibration_sha256": calibration_sha,
        "legacy_component_manifest_identity_sha256": legacy_manifest_identity,
        "legacy_component_manifest_file_sha256": sha256_file(args.legacy_component_manifest),
        "fallback_gate": "F_FALLBACK_ENGINEERING_ELIGIBILITY_GATE",
        "fallback_gate_sha256": sha256_file(gate_path),
        "F_STRICT_PREREG_GATE": False,
        "p_select": 0.20,
        "min_anchor_volume_mm3": 27.0,
        "max_bbox_mm": 56.0,
        "x1_rc_probability": 0.023,
        "second_tumour_probability": 0.0,
        "e_probability": 0.10,
        "features": {"b": True, "d": False, "e": True, "f": True, "x2": False, "x3": False},
        "empirical": {
            "e_shell_factor": empirical["e_shell_factor"],
            "e_volume_p10_mm3": empirical["e_volume_p10_mm3"],
            "e_volume_p90_mm3": empirical["e_volume_p90_mm3"],
            "burden_probabilities": gate["burden_probabilities_recomputed"],
            "target_counts": empirical["target_counts"],
        },
    }
    payload["config_sha256"] = canonical_json_sha256(payload, exclude=("config_sha256",))
    config_path = ROOT / "config/K1245_ROUTE_K1_FALLBACK.json"
    exclusive_json(config_path, payload)
    K1245Config.load(config_path, manifest=manifest, plan_path=plan)
    exclusive_json(ROOT / "evidence/K1_FALLBACK_CONFIG_PREPARATION.json", {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "route_id": "K1",
        "config_sha256": sha256_file(config_path),
        "plan_sha256": sha256_file(plan),
        "fallback_gate_sha256": sha256_file(gate_path),
        "manifest_identity_sha256": manifest.identity_sha256,
        "F_STRICT_PREREG_GATE": False,
        "scientific_gate_pass": False,
    })
    print(json.dumps({"status": "pass", "config_sha256": sha256_file(config_path), "strict_gate": False}))


if __name__ == "__main__":
    main()
