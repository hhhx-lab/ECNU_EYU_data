"""Route A approval contract shared by the offline gate and the S2 trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .met_aug_core import (
        ROUTE_A,
        VALID_MASK_MANIFEST_SCHEMA,
        ComponentManifest,
        MetAugContractError,
        RouteConfig,
        canonical_json_sha256,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        ROUTE_A,
        VALID_MASK_MANIFEST_SCHEMA,
        ComponentManifest,
        MetAugContractError,
        RouteConfig,
        canonical_json_sha256,
        sha256_file,
    )


ROUTE_A_APPROVAL_SCHEMA = 3
ROUTE_A_BASE_CANDIDATE = "E"
ROUTE_A_BASE_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
ROUTE_A_TRAINING_CONTRACT = {
    "stage": "second_stage_adaptation",
    "base_candidate": ROUTE_A_BASE_CANDIDATE,
    "base_checkpoint_sha256": ROUTE_A_BASE_CHECKPOINT_SHA256,
    "epochs": 200,
    "initial_lr": 0.001,
    "save_every": 25,
    "focal_gamma": 2.0,
    "rc_class_weight": 3.0,
    "training_seed": 20260724,
    "epoch_seed_policy": "training_seed_plus_epoch",
    "augmentation_workers": 0,
    "discarded_train_warmup_batches": 0,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "torch_compile": False,
}
ROUTE_A_RUNTIME_FILES = (
    "met_aug_core.py",
    "met_aug_data_loader.py",
    "met_aug_diffusion.py",
    "met_aug_gate.py",
    "met_aug_paired_training.py",
    "met_aug_transform.py",
    "nnUNetTrainerBraTS2026RC.py",
    "nnUNetTrainerBraTS2026RCCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.py",
    "online_diffusion_contract.py",
    "small_lesion_trainer_mixins.py",
    "small_lesion_variants.py",
)


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MetAugContractError(f"{label} must be a JSON object: {resolved}")
    return value


def route_a_runtime_code_snapshot(code_dir: str | Path) -> dict[str, Any]:
    root = Path(code_dir).expanduser().resolve()
    files: dict[str, str] = {}
    for filename in ROUTE_A_RUNTIME_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Route A runtime file is missing: {path}")
        files[filename] = sha256_file(path)
    return {
        "code_dir": str(root),
        "files": files,
        "sha256": canonical_json_sha256(files),
    }


def validate_route_a_training_contract(
    *,
    num_epochs: int,
    initial_lr: float,
    save_every: int,
    focal_gamma: float,
    training_seed: int = 20260724,
    augmentation_workers: int = 0,
    discarded_train_warmup_batches: int = 0,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    torch_compile: bool = False,
) -> None:
    observed = {
        "epochs": int(num_epochs),
        "initial_lr": float(initial_lr),
        "save_every": int(save_every),
        "focal_gamma": float(focal_gamma),
        "training_seed": int(training_seed),
        "augmentation_workers": int(augmentation_workers),
        "discarded_train_warmup_batches": int(discarded_train_warmup_batches),
        "cudnn_deterministic": bool(cudnn_deterministic),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "torch_compile": bool(torch_compile),
    }
    for key, value in observed.items():
        if value != ROUTE_A_TRAINING_CONTRACT[key]:
            raise MetAugContractError(
                f"Route A second-stage training contract drifted: {key}={value!r}, "
                f"expected={ROUTE_A_TRAINING_CONTRACT[key]!r}"
            )


def _validate_valid_mask_manifest(path: str | Path) -> None:
    payload = load_json_object(path, label="MET-AUG valid-mask manifest")
    if payload.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
        raise MetAugContractError("unsupported MET-AUG valid-mask manifest schema")
    if payload.get("manifest_sha256") != canonical_json_sha256(payload, exclude=("manifest_sha256",)):
        raise MetAugContractError("MET-AUG valid-mask manifest SHA256 mismatch")


def _validate_parent_contract(selection_path: str | Path, parent_gate_path: str | Path) -> None:
    selection = load_json_object(selection_path, label="G1 checkpoint selection")
    parent_gate = load_json_object(parent_gate_path, label="G2 parent gate")
    if parent_gate.get("decision") != "approve":
        raise MetAugContractError("G2 parent gate is not approved")
    if parent_gate.get("checkpoint_selection_sha256") != sha256_file(selection_path):
        raise MetAugContractError("G2 parent gate does not bind the supplied G1 checkpoint selection")
    expected = {
        "normalization": "zscore",
        "sampling_method": "edm_heun",
        "sampling_steps": 18,
        "crop_size": 64,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise MetAugContractError(f"G1 selection has unexpected {key}: {selection.get(key)!r}")


def _validate_gate_report(
    path: str | Path,
    *,
    label: str,
    manifest_sha256: str,
    route_config_sha256: str,
    valid_mask_manifest_sha256: str,
    require_smoke: bool,
    g1_checkpoint_selection_sha256: str | None = None,
    g2_parent_gate_sha256: str | None = None,
    g1_runtime_code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_json_object(path, label=label)
    if payload.get("status") != "pass" or payload.get("route_id") != ROUTE_A:
        raise MetAugContractError(f"{label} is not a passing Route A result")
    expected = {
        "component_manifest_sha256": manifest_sha256,
        "route_config_sha256": route_config_sha256,
        "valid_mask_manifest_sha256": valid_mask_manifest_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MetAugContractError(f"{label} does not bind {key}")
    if require_smoke:
        if payload.get("schema_version") != 2:
            raise MetAugContractError("unsupported Gate 2 final-report schema")
        if int(payload.get("smoke_count", 0)) < 24:
            raise MetAugContractError("Gate 2 did not cover the required 24 fixed smoke cases")
        if payload.get("manual_review_status") != "pass":
            raise MetAugContractError("Gate 2 manual review is not approved")
        if payload.get("gate2_report_sha256") != canonical_json_sha256(
            payload, exclude=("gate2_report_sha256",)
        ):
            raise MetAugContractError("Gate 2 final report SHA256 mismatch")
        for volume_bin in ("27_49", "50_275", "gt_275"):
            if int(payload.get("per_volume_bin", {}).get(volume_bin, 0)) < 8:
                raise MetAugContractError(
                    f"Gate 2 did not cover eight fixed smoke cases in {volume_bin}"
                )
        for key in (
            "smoke_manifest_sha256",
            "automatic_report_sha256",
            "review_decisions_sha256",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise MetAugContractError(f"Gate 2 final report does not bind {key}")
        external_expected = {
            "g1_checkpoint_selection_sha256": g1_checkpoint_selection_sha256,
            "g2_parent_gate_sha256": g2_parent_gate_sha256,
            "g1_runtime_code": g1_runtime_code,
        }
        for key, value in external_expected.items():
            if value is None or payload.get(key) != value:
                raise MetAugContractError(f"Gate 2 final report does not bind {key}")
    return payload


def build_route_a_approval(
    *,
    component_manifest_path: str | Path,
    route_config_path: str | Path,
    valid_mask_manifest_path: str | Path,
    gate1_report_path: str | Path,
    gate2_report_path: str | Path,
    g1_checkpoint_selection_path: str | Path,
    g2_parent_gate_path: str | Path,
    g1_code_dir: str | Path,
    code_dir: str | Path,
) -> dict[str, Any]:
    try:
        from .met_aug_diffusion import g1_runtime_code_snapshot
    except ImportError:
        from met_aug_diffusion import g1_runtime_code_snapshot  # type: ignore

    manifest = ComponentManifest.load(component_manifest_path)
    config = RouteConfig.load(route_config_path, manifest)
    _validate_valid_mask_manifest(valid_mask_manifest_path)
    _validate_parent_contract(g1_checkpoint_selection_path, g2_parent_gate_path)
    route_config_sha256 = sha256_file(config.path)
    valid_mask_manifest_sha256 = sha256_file(valid_mask_manifest_path)
    selection_sha256 = sha256_file(g1_checkpoint_selection_path)
    parent_gate_sha256 = sha256_file(g2_parent_gate_path)
    g1_runtime_code = g1_runtime_code_snapshot(g1_code_dir)
    _validate_gate_report(
        gate1_report_path,
        label="Gate 1 report",
        manifest_sha256=manifest.identity_sha256,
        route_config_sha256=route_config_sha256,
        valid_mask_manifest_sha256=valid_mask_manifest_sha256,
        require_smoke=False,
    )
    _validate_gate_report(
        gate2_report_path,
        label="Gate 2 report",
        manifest_sha256=manifest.identity_sha256,
        route_config_sha256=route_config_sha256,
        valid_mask_manifest_sha256=valid_mask_manifest_sha256,
        require_smoke=True,
        g1_checkpoint_selection_sha256=selection_sha256,
        g2_parent_gate_sha256=parent_gate_sha256,
        g1_runtime_code=g1_runtime_code,
    )
    approval = {
        "schema_version": ROUTE_A_APPROVAL_SCHEMA,
        "route_id": config.route_id,
        "decision": "approve",
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": route_config_sha256,
        "valid_mask_manifest_sha256": valid_mask_manifest_sha256,
        "gate1_report_sha256": sha256_file(gate1_report_path),
        "gate2_report_sha256": sha256_file(gate2_report_path),
        "g1_checkpoint_selection_sha256": selection_sha256,
        "g2_parent_gate_sha256": parent_gate_sha256,
        "g1_runtime_code": g1_runtime_code,
        "training_contract": ROUTE_A_TRAINING_CONTRACT,
        "runtime_code": route_a_runtime_code_snapshot(code_dir),
    }
    approval["approval_sha256"] = canonical_json_sha256(approval, exclude=("approval_sha256",))
    return approval


def validate_route_a_approval(
    approval_path: str | Path,
    *,
    component_manifest: ComponentManifest,
    route_config_path: str | Path,
    valid_mask_manifest_path: str | Path,
    g1_checkpoint_selection_path: str | Path,
    g2_parent_gate_path: str | Path,
    g1_code_dir: str | Path,
    code_dir: str | Path,
) -> Mapping[str, Any]:
    try:
        from .met_aug_diffusion import g1_runtime_code_snapshot
    except ImportError:
        from met_aug_diffusion import g1_runtime_code_snapshot  # type: ignore

    payload = load_json_object(approval_path, label="MET-AUG Route A approval")
    if payload.get("schema_version") != ROUTE_A_APPROVAL_SCHEMA:
        raise MetAugContractError("unsupported MET-AUG Route A approval schema")
    if payload.get("approval_sha256") != canonical_json_sha256(payload, exclude=("approval_sha256",)):
        raise MetAugContractError("MET-AUG Route A approval SHA256 mismatch")
    if payload.get("decision") != "approve" or payload.get("route_id") != ROUTE_A:
        raise MetAugContractError("MET-AUG Route A has not been approved")
    expected = {
        "component_manifest_sha256": component_manifest.identity_sha256,
        "route_config_sha256": sha256_file(route_config_path),
        "valid_mask_manifest_sha256": sha256_file(valid_mask_manifest_path),
        "g1_checkpoint_selection_sha256": sha256_file(g1_checkpoint_selection_path),
        "g2_parent_gate_sha256": sha256_file(g2_parent_gate_path),
        "training_contract": ROUTE_A_TRAINING_CONTRACT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MetAugContractError(f"MET-AUG Route A approval does not bind {key}")
    expected_g1_code = g1_runtime_code_snapshot(g1_code_dir)
    if payload.get("g1_runtime_code") != expected_g1_code:
        raise MetAugContractError("MET-AUG Route A approval does not match deployed G1 runtime code")
    expected_code = route_a_runtime_code_snapshot(code_dir)
    observed_code = payload.get("runtime_code")
    if not isinstance(observed_code, dict) or observed_code.get("sha256") != expected_code["sha256"]:
        raise MetAugContractError("MET-AUG Route A approval does not match deployed runtime code")
    return payload
