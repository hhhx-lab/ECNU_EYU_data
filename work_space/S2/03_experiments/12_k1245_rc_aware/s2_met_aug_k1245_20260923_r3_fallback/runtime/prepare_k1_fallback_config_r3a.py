#!/usr/bin/env python3
"""Fail-closed K1 config preparation using the corrected r3a identity bindings."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/repository"))
from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file
from custom_nnunet.met_aug_k1245 import K1245Config, K1245Manifest


def main() -> None:
    r2_root = Path(sys.argv[1]).resolve()
    parent = Path(sys.argv[2]).resolve()
    child_plan = ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json"
    data = json.loads(child_plan.read_text())
    gate_path = ROOT / "evidence/F_FALLBACK_GATE_VALIDATION.json"
    gate = json.loads(gate_path.read_text())
    if data["training_authorization"] is not False or data["k5_authorization"] is not False:
        raise SystemExit("K1/K5 authorization must remain false")
    if gate.get("status") != "pass" or gate.get("F_STRICT_PREREG_GATE") is not False:
        raise SystemExit("fallback gate is not pass with strict result false")
    manifest_path = r2_root / "pools/rc27/component_manifest.json"
    manifest_file_sha = sha256_file(manifest_path)
    manifest = K1245Manifest.load(manifest_path)
    if manifest_file_sha != data["source_rc27_manifest_file_sha256"]:
        raise SystemExit("r2 RC27 manifest file SHA mismatch")
    if manifest.identity_sha256 != data["source_rc27_manifest_identity_sha256"]:
        raise SystemExit("r2 RC27 canonical manifest identity mismatch")
    r2_plan = json.loads((r2_root / "K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json").read_text())
    if sha256_file(r2_root / "K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json") != data["source_r2_plan_sha256"]:
        raise SystemExit("r2 plan SHA mismatch")
    route_path = parent / "calibration_rebuild_20260819_r10/FIX_V3_ROUTE_CONFIG_SCHEMA5.json"
    legacy_path = parent / "route_rebuild_20260812_r1/component_pool/component_manifest.json"
    calibration_path = parent / "calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
    if sha256_file(route_path) != data["source_legacy_route_config_sha256"]:
        raise SystemExit("legacy route config SHA mismatch")
    if sha256_file(legacy_path) != data["source_legacy_manifest_file_sha256"]:
        raise SystemExit("legacy manifest file SHA mismatch")
    legacy_payload = json.loads(legacy_path.read_text())
    if legacy_payload.get("manifest_sha256") != data["source_legacy_manifest_identity_sha256"]:
        raise SystemExit("legacy manifest identity mismatch")
    if sha256_file(calibration_path) != data["source_fix_v3_calibration_sha256"]:
        raise SystemExit("FixV3 calibration SHA mismatch")
    route = json.loads(route_path.read_text())
    calibration_identity = route.get("fix_v3", {}).get("calibration_sha256", "")
    if calibration_identity != data["source_fix_v3_calibration_sha256"]:
        raise SystemExit("FixV3 route does not bind frozen calibration")
    legacy_identity = legacy_payload["manifest_sha256"]
    evidence = json.loads((r2_root / "pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json").read_text())
    if evidence.get("status") != "blocked_f_gate" or evidence["empirical"].get("f_gate_pass") is not False:
        raise SystemExit("r2 strict failure semantics drifted")
    empirical = evidence["empirical"]
    config = {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "schema_version": 1,
        "route_id": "K1", "plan_sha256": sha256_file(ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"),
        "manifest_sha256": manifest.identity_sha256,
        "legacy_fix_v3_calibration_sha256": calibration_identity,
        "legacy_component_manifest_identity_sha256": legacy_identity,
        "legacy_component_manifest_file_sha256": sha256_file(legacy_path),
        "fallback_gate": "F_FALLBACK_ENGINEERING_ELIGIBILITY_GATE",
        "fallback_gate_sha256": sha256_file(gate_path), "F_STRICT_PREREG_GATE": False,
        "p_select": 0.20, "min_anchor_volume_mm3": 27.0, "max_bbox_mm": 56.0,
        "x1_rc_probability": 0.023, "second_tumour_probability": 0.0, "e_probability": 0.10,
        "features": {"b": True, "d": False, "e": True, "f": True, "x2": False, "x3": False},
        "empirical": {
            "e_shell_factor": empirical["e_shell_factor"],
            "e_volume_p10_mm3": empirical["e_volume_p10_mm3"],
            "e_volume_p90_mm3": empirical["e_volume_p90_mm3"],
            "burden_probabilities": gate["burden_probabilities_recomputed"],
            "target_counts": empirical["target_counts"],
        },
    }
    config["config_sha256"] = canonical_json_sha256(config, exclude=("config_sha256",))
    config_path = ROOT / "config/K1245_ROUTE_K1_FALLBACK.json"
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    K1245Config.load(config_path, manifest=manifest, plan_path=ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json")
    result = {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "route_id": "K1", "config_sha256": sha256_file(config_path),
        "child_plan_sha256": sha256_file(child_plan), "fallback_gate_sha256": sha256_file(gate_path),
        "manifest_file_sha256": manifest_file_sha, "manifest_identity_sha256": manifest.identity_sha256,
        "F_STRICT_PREREG_GATE": False, "scientific_gate_pass": False,
        "training_authorization": False, "k5_authorization": False,
    }
    out = ROOT / "evidence/K1_FALLBACK_CONFIG_PREPARATION.json"
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
