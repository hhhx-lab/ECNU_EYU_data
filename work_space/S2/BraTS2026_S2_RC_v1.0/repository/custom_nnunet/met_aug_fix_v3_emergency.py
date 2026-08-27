"""Explicit audit contract for the user-authorized unvalidated Fix-v3 route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .met_aug_core import (
        MetAugContractError,
        canonical_json_sha256,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        MetAugContractError,
        canonical_json_sha256,
        sha256_file,
    )


FIX_V3_EMERGENCY_DECISION_SCHEMA = 1
ORIGINAL_E_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
FIX_V2_FAILURE_SHA256 = "dd4afaad16359303e6d01f4961b04c235dbffa2a5f8504773f9229c2798e7e08"
FIX_V3_EMERGENCY_SKIPPED_STAGES = (
    "reference_increment",
    "development_96",
    "independent_holdout",
    "gate_0",
    "gate_1a_100000",
    "gate_1b_96_double_replay",
    "gate_2_120",
)
FIX_V3_EMERGENCY_SCOPE = (
    "128_step_training_smoke",
    "200_epoch_training",
    "fixed_103_model_comparison",
)


def _required_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _input_paths(
    *,
    component_manifest: str | Path,
    route_config: str | Path,
    valid_mask_manifest: str | Path,
    calibration: str | Path,
    original_e: str | Path,
    fix_v2_failure: str | Path,
) -> dict[str, Path]:
    return {
        "component_manifest": _required_file(
            component_manifest, label="component manifest"
        ),
        "route_config": _required_file(route_config, label="Fix-v3 route config"),
        "valid_mask_manifest": _required_file(
            valid_mask_manifest, label="valid-mask manifest"
        ),
        "calibration": _required_file(calibration, label="frozen Fix-v2 calibration"),
        "original_e": _required_file(original_e, label="original E checkpoint"),
        "fix_v2_failure": _required_file(
            fix_v2_failure, label="frozen Fix-v2 failure audit"
        ),
    }


def _expected_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["original_e"] != ORIGINAL_E_SHA256:
        raise MetAugContractError("emergency route does not bind the frozen original E")
    if hashes["fix_v2_failure"] != FIX_V2_FAILURE_SHA256:
        raise MetAugContractError("emergency route does not bind the frozen Fix-v2 failure")
    payload: dict[str, Any] = {
        "schema_version": FIX_V3_EMERGENCY_DECISION_SCHEMA,
        "status": "frozen_experimental_unvalidated",
        "decision": "allow_experimental_smoke_training_and_fixed_103_only",
        "formal_validation_status": "skipped",
        "experimental_training_authorized": True,
        "formal_training_eligible": False,
        "inference_eligible": False,
        "skipped_stages": list(FIX_V3_EMERGENCY_SKIPPED_STAGES),
        "authorized_scope": list(FIX_V3_EMERGENCY_SCOPE),
        "inputs": {f"{name}_sha256": value for name, value in sorted(hashes.items())},
        "training_contract": {
            "epochs": 200,
            "p_select": 0.20,
            "fixed_train_count": 1035,
            "fixed_validation_count": 103,
            "candidate_policy": "label_only_qc_v1",
            "processor_policy": "fix_v3_qc_v1",
        },
        "deployment_rule": {
            "original_e_fallback_required": True,
            "replacement_requires_fixed_103_superiority": True,
            "met_aug_forbidden_during_inference": True,
            "g1_g2_donor_forbidden_during_inference": True,
        },
    }
    payload["decision_audit_sha256"] = canonical_json_sha256(
        payload, exclude=("decision_audit_sha256",)
    )
    return payload


def make_fix_v3_emergency_decision(
    *,
    component_manifest: str | Path,
    route_config: str | Path,
    valid_mask_manifest: str | Path,
    calibration: str | Path,
    original_e: str | Path,
    fix_v2_failure: str | Path,
) -> str:
    paths = _input_paths(
        component_manifest=component_manifest,
        route_config=route_config,
        valid_mask_manifest=valid_mask_manifest,
        calibration=calibration,
        original_e=original_e,
        fix_v2_failure=fix_v2_failure,
    )
    return json.dumps(
        _expected_payload(paths),
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ) + "\n"


def validate_fix_v3_emergency_decision(
    decision_path: str | Path,
    *,
    component_manifest: str | Path,
    route_config: str | Path,
    valid_mask_manifest: str | Path,
    calibration: str | Path,
    original_e: str | Path,
    fix_v2_failure: str | Path,
) -> Mapping[str, Any]:
    resolved_decision = _required_file(decision_path, label="Fix-v3 emergency decision")
    try:
        observed = json.loads(resolved_decision.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetAugContractError("Fix-v3 emergency decision is unreadable") from exc
    paths = _input_paths(
        component_manifest=component_manifest,
        route_config=route_config,
        valid_mask_manifest=valid_mask_manifest,
        calibration=calibration,
        original_e=original_e,
        fix_v2_failure=fix_v2_failure,
    )
    expected = _expected_payload(paths)
    if observed != expected:
        changed = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        ) if isinstance(observed, Mapping) else ["root"]
        raise MetAugContractError(
            f"Fix-v3 emergency decision drifted: changed_keys={changed}"
        )
    return expected
