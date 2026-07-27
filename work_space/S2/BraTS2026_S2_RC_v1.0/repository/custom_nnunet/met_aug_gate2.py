"""Immutable Route A Gate 2 smoke-manifest and review contracts.

Gate 2 deliberately separates three actions:

* freeze a deterministic, stratified set of planned transactions;
* execute those exact transactions and run automatic four-modality checks;
* let a reviewer approve or reject the resulting renderings in a separate file.

Nothing in this module imports the G1 generator or nnU-Net.  That keeps manifest
construction and final review validation deterministic and safe to run on a CPU
login node.  The actual Diffusion invocation lives in the dedicated Gate 2 runner.
"""

from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .met_aug_core import (
        ComponentManifest,
        EventContext,
        MemoryAuditSink,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        VALID_MASK_MANIFEST_SCHEMA,
        canonical_json_sha256,
        component_record_support_counts,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        ComponentManifest,
        EventContext,
        MemoryAuditSink,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        VALID_MASK_MANIFEST_SCHEMA,
        canonical_json_sha256,
        component_record_support_counts,
        sha256_file,
    )


GATE2_MANIFEST_SCHEMA = 1
GATE2_AUTOMATIC_REPORT_SCHEMA = 3
GATE2_FINAL_REPORT_SCHEMA = 2
GATE2_VOLUME_BINS = ("27_49", "50_275", "gt_275")
MIN_SMOKE_PER_VOLUME_BIN = 8
GATE2_RUNTIME_FILES = (
    "scripts/18_run_met_aug_gate2_smoke.py",
    "custom_nnunet/met_aug_gate2.py",
    "custom_nnunet/met_aug_core.py",
    "custom_nnunet/met_aug_diffusion.py",
    "custom_nnunet/online_diffusion_contract.py",
)
REVIEW_DECISION_ACCEPT = "accept"
REVIEW_DECISION_REJECT = "reject"
REVIEW_TEMPLATE_FIELDS = (
    "smoke_id",
    "evidence_fingerprint",
    "target_case_id",
    "donor_component_id",
    "core_volume_bin",
    "artifact_path",
    "montage_path",
    "automatic_qc_status",
    "review_decision",
    "reviewer",
    "reviewed_at_utc",
    "notes",
)


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MetAugContractError(f"{label} must be a JSON object: {resolved}")
    return payload


def _canonical_file_sha256(path: str | Path) -> str:
    return sha256_file(Path(path).expanduser().resolve())


def gate2_runtime_code_snapshot(repository_root: str | Path) -> dict[str, Any]:
    """Hash the exact code that executes and validates the Gate 2 transaction."""
    root = Path(repository_root).expanduser().resolve()
    files: dict[str, str] = {}
    for relative in GATE2_RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Gate 2 runtime file is missing: {path}")
        files[relative] = sha256_file(path)
    return {
        "files": files,
        "sha256": canonical_json_sha256(files),
    }


def load_valid_mask_assets(
    path: str | Path,
    *,
    expected_ids: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load immutable full-case valid/foreground masks without nnU-Net imports."""
    manifest_path = Path(path).expanduser().resolve()
    payload = load_json_object(manifest_path, label="MET-AUG valid-mask manifest")
    if payload.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
        raise MetAugContractError("unsupported valid-mask manifest schema")
    if payload.get("manifest_sha256") != canonical_json_sha256(payload, exclude=("manifest_sha256",)):
        raise MetAugContractError("valid-mask manifest SHA256 mismatch")
    records_path = manifest_path.parent / str(payload.get("records_file", ""))
    if not records_path.is_file() or _canonical_file_sha256(records_path) != payload.get("records_sha256"):
        raise MetAugContractError("valid-mask records SHA256 mismatch")

    assets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["case_id"])
        asset_path = (manifest_path.parent / str(row["mask_path"])).resolve()
        if not asset_path.is_file() or _canonical_file_sha256(asset_path) != row.get("sha256"):
            raise MetAugContractError(f"valid-mask payload SHA256 mismatch: {case_id}")
        with np.load(asset_path, allow_pickle=False) as asset:
            valid = asset["valid_mask"].astype(bool, copy=True)
            foreground = asset["foreground_mask"].astype(bool, copy=True)
        expected_shape = tuple(int(value) for value in row["shape"])
        if valid.shape != expected_shape or foreground.shape != expected_shape:
            raise MetAugContractError(f"valid-mask payload shape mismatch: {case_id}")
        if not np.any(valid):
            raise MetAugContractError(f"valid-mask payload is empty: {case_id}")
        assets[case_id] = (valid, foreground)
    if set(assets) != expected_ids:
        raise MetAugContractError(
            "valid-mask IDs do not exactly match train targets: "
            f"missing={sorted(expected_ids - set(assets))[:10]}, "
            f"extra={sorted(set(assets) - expected_ids)[:10]}"
        )
    return assets


def _event_context_from_mapping(value: Mapping[str, Any]) -> EventContext:
    return EventContext(
        epoch=int(value["epoch"]),
        rank=int(value["rank"]),
        worker=int(value["worker"]),
        case_id=str(value["case_id"]),
        patch_index=int(value["patch_index"]),
        patch_origin=tuple(int(item) for item in value.get("patch_origin", (0, 0, 0))),
        full_shape=tuple(int(item) for item in value["full_shape"]),
    )


def _entry_fingerprint(entry: Mapping[str, Any]) -> str:
    payload = {
        "smoke_id": entry["smoke_id"],
        "event_id": entry["event_id"],
        "event_seed": entry["event_seed"],
        "target_case_id": entry["target_case_id"],
        "target_patient_group": entry["target_patient_group"],
        "donor_component_id": entry["donor_component_id"],
        "donor_patient_group": entry["donor_patient_group"],
        "core_volume_bin": entry["core_volume_bin"],
        "core_volume_mm3": entry["core_volume_mm3"],
        "classes_present": entry["classes_present"],
        "planned": entry["planned"],
    }
    for key in ("total_support_voxels", "core_voxels", "total_to_core_ratio"):
        if key in entry:
            payload[key] = entry[key]
    return canonical_json_sha256(payload)


def prepare_smoke_manifest(
    *,
    manifest: ComponentManifest,
    config: RouteConfig,
    valid_mask_manifest_path: str | Path,
    assets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    search_seed: int,
    per_volume_bin: int = MIN_SMOKE_PER_VOLUME_BIN,
    max_candidates: int = 100000,
) -> dict[str, Any]:
    """Pre-register the exact, non-replaceable Gate 2 Route A transactions.

    Donor components and target cases are both unique across the smoke set.  A
    component can therefore not be repeatedly selected merely because it happens
    to render well.  We refuse to create a manifest if any required size bin lacks
    enough distinct train-only components.
    """
    if per_volume_bin < MIN_SMOKE_PER_VOLUME_BIN:
        raise MetAugContractError(
            f"Gate 2 requires at least {MIN_SMOKE_PER_VOLUME_BIN} smoke cases per volume bin"
        )
    if max_candidates <= 0:
        raise MetAugContractError("Gate 2 candidate search must be positive")
    if set(assets) != set(manifest.target_groups):
        raise MetAugContractError("Gate 2 assets and component manifest target IDs differ")

    eligible_records = config.eligible_records(manifest)
    component_counts = Counter(record.stratum[1] for record in eligible_records)
    undersupplied = {
        volume_bin: component_counts[volume_bin]
        for volume_bin in GATE2_VOLUME_BINS
        if component_counts[volume_bin] < per_volume_bin
    }
    if undersupplied:
        raise MetAugContractError(
            "cannot pre-register Gate 2 without substituting size strata; "
            f"distinct components below quota: {undersupplied}"
        )

    engine = MetAugEngine(
        manifest=manifest,
        config=config,
        backend=None,
        audit_sink=MemoryAuditSink(),
    )
    candidate_rng = np.random.default_rng(int(search_seed))
    target_ids = np.asarray(sorted(assets), dtype=object)
    target_order = candidate_rng.permutation(target_ids)
    selected_by_bin: Counter[str] = Counter()
    selected_component_ids: set[str] = set()
    selected_target_ids: set[str] = set()
    skipped: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []

    for candidate_index in range(int(max_candidates)):
        if all(selected_by_bin[volume_bin] >= per_volume_bin for volume_bin in GATE2_VOLUME_BINS):
            break
        case_id = str(target_order[candidate_index % len(target_order)])
        valid, foreground = assets[case_id]
        context = EventContext(
            epoch=0,
            rank=0,
            worker=0,
            case_id=case_id,
            patch_index=candidate_index,
            patch_origin=(0, 0, 0),
            full_shape=tuple(int(value) for value in valid.shape),
        )
        # Placement only needs the existing foreground support. It does not infer
        # classes from it, so this remains a no-Diffusion and no-image operation.
        result = engine.plan(
            segmentation=(foreground.astype(np.int16) * 3)[None],
            valid_mask=valid,
            context=context,
        )
        if result.state != "PLACEMENT_VALID" or result.record is None or result.placement is None:
            skipped[result.reason or result.state] += 1
            continue
        record = result.record
        if not config.is_record_eligible(record):
            raise MetAugContractError(
                "ineligible compact-support donor escaped Gate 2 planner"
            )
        volume_bin = record.stratum[1]
        if volume_bin not in GATE2_VOLUME_BINS:
            skipped[f"unexpected_volume_bin:{volume_bin}"] += 1
            continue
        if selected_by_bin[volume_bin] >= per_volume_bin:
            skipped[f"quota_reached:{volume_bin}"] += 1
            continue
        if record.component_id in selected_component_ids:
            skipped["duplicate_component"] += 1
            continue
        if case_id in selected_target_ids:
            skipped["duplicate_target"] += 1
            continue
        if record.patient_group == manifest.target_groups[case_id]:
            raise MetAugContractError("same patient-group donor escaped Gate 2 planner")

        total_support_voxels, core_voxels = component_record_support_counts(record)
        entry = {
            "smoke_id": f"route-a-smoke-{len(entries) + 1:03d}",
            "target_case_id": case_id,
            "target_patient_group": manifest.target_groups[case_id],
            "event": {
                "epoch": context.epoch,
                "rank": context.rank,
                "worker": context.worker,
                "case_id": context.case_id,
                "patch_index": context.patch_index,
                "patch_origin": list(context.patch_origin),
                "full_shape": list(context.full_shape or ()),
            },
            "event_id": result.event_id,
            "event_seed": result.event_seed,
            "donor_component_id": record.component_id,
            "donor_source_case_id": record.source_case_id,
            "donor_patient_group": record.patient_group,
            "classes_present": list(record.classes_present),
            "core_volume_mm3": record.core_volume_mm3,
            "core_volume_bin": volume_bin,
            "total_support_voxels": total_support_voxels,
            "core_voxels": core_voxels,
            "total_to_core_ratio": total_support_voxels / core_voxels,
            "planned": {
                "crop_start": list(result.placement.crop_start),
                "placement_attempts": result.placement.attempts,
                "placement_strategy": result.placement.placement_strategy,
                "support_voxels": int(np.count_nonzero(result.placement.support)),
            },
        }
        entry["entry_sha256"] = _entry_fingerprint(entry)
        entries.append(entry)
        selected_by_bin[volume_bin] += 1
        selected_component_ids.add(record.component_id)
        selected_target_ids.add(case_id)

    missing = {
        volume_bin: per_volume_bin - selected_by_bin[volume_bin]
        for volume_bin in GATE2_VOLUME_BINS
        if selected_by_bin[volume_bin] < per_volume_bin
    }
    if missing:
        raise MetAugContractError(
            "Gate 2 pre-registration exhausted deterministic search without replacing cases; "
            f"missing={missing}, skipped={dict(sorted(skipped.items()))}"
        )

    payload: dict[str, Any] = {
        "schema_version": GATE2_MANIFEST_SCHEMA,
        "route_id": config.route_id,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": _canonical_file_sha256(config.path),
        "valid_mask_manifest_sha256": _canonical_file_sha256(valid_mask_manifest_path),
        "search_seed": int(search_seed),
        "per_volume_bin_quota": int(per_volume_bin),
        "max_candidates": int(max_candidates),
        "candidate_count_examined": min(int(max_candidates), candidate_index + 1),
        "skipped": dict(sorted(skipped.items())),
        "eligible_component_count": len(eligible_records),
        "eligible_per_volume_bin": {
            volume_bin: int(component_counts[volume_bin])
            for volume_bin in GATE2_VOLUME_BINS
        },
        "donor_eligibility": (
            config.donor_eligibility.as_mapping()
            if config.donor_eligibility is not None
            else None
        ),
        "smoke_count": len(entries),
        "per_volume_bin": {volume_bin: int(selected_by_bin[volume_bin]) for volume_bin in GATE2_VOLUME_BINS},
        "smoke_cases": entries,
    }
    payload["smoke_manifest_sha256"] = canonical_json_sha256(payload, exclude=("smoke_manifest_sha256",))
    return payload


def load_smoke_manifest(
    path: str | Path,
    *,
    manifest: ComponentManifest,
    config: RouteConfig,
    valid_mask_manifest_path: str | Path,
) -> dict[str, Any]:
    payload = load_json_object(path, label="Gate 2 smoke manifest")
    if payload.get("schema_version") != GATE2_MANIFEST_SCHEMA:
        raise MetAugContractError("unsupported Gate 2 smoke-manifest schema")
    if payload.get("smoke_manifest_sha256") != canonical_json_sha256(
        payload, exclude=("smoke_manifest_sha256",)
    ):
        raise MetAugContractError("Gate 2 smoke-manifest SHA256 mismatch")
    expected = {
        "route_id": config.route_id,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": _canonical_file_sha256(config.path),
        "valid_mask_manifest_sha256": _canonical_file_sha256(valid_mask_manifest_path),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MetAugContractError(f"Gate 2 smoke manifest does not bind {key}")
    eligible_records = config.eligible_records(manifest)
    expected_eligible_counts = Counter(
        record.stratum[1] for record in eligible_records
    )
    if config.donor_eligibility is not None:
        if payload.get("donor_eligibility") != config.donor_eligibility.as_mapping():
            raise MetAugContractError(
                "Gate 2 smoke manifest donor eligibility audit drifted"
            )
        if int(payload.get("eligible_component_count", -1)) != len(eligible_records):
            raise MetAugContractError(
                "Gate 2 smoke manifest eligible component count drifted"
            )
        expected_per_bin = {
            volume_bin: int(expected_eligible_counts[volume_bin])
            for volume_bin in GATE2_VOLUME_BINS
        }
        if payload.get("eligible_per_volume_bin") != expected_per_bin:
            raise MetAugContractError(
                "Gate 2 smoke manifest eligible stratum counts drifted"
            )
    if int(payload.get("per_volume_bin_quota", 0)) < MIN_SMOKE_PER_VOLUME_BIN:
        raise MetAugContractError("Gate 2 smoke-manifest quota is below the minimum")

    cases = payload.get("smoke_cases")
    if not isinstance(cases, list) or int(payload.get("smoke_count", -1)) != len(cases):
        raise MetAugContractError("Gate 2 smoke-manifest case count mismatch")
    smoke_ids: set[str] = set()
    event_ids: set[str] = set()
    target_ids: set[str] = set()
    component_ids: set[str] = set()
    observed_bins: Counter[str] = Counter()
    records = {record.component_id: record for record in manifest.records}
    for entry in cases:
        if not isinstance(entry, dict):
            raise MetAugContractError("Gate 2 smoke-manifest has a malformed case entry")
        if entry.get("entry_sha256") != _entry_fingerprint(entry):
            raise MetAugContractError(f"Gate 2 smoke entry SHA256 mismatch: {entry.get('smoke_id')}")
        smoke_id = str(entry.get("smoke_id", ""))
        event_id = str(entry.get("event_id", ""))
        target_case_id = str(entry.get("target_case_id", ""))
        component_id = str(entry.get("donor_component_id", ""))
        if not smoke_id or smoke_id in smoke_ids or not event_id or event_id in event_ids:
            raise MetAugContractError("Gate 2 smoke IDs or event IDs are duplicated")
        if target_case_id not in manifest.target_groups or target_case_id in target_ids:
            raise MetAugContractError("Gate 2 smoke target IDs are invalid or duplicated")
        record = records.get(component_id)
        if record is None or component_id in component_ids:
            raise MetAugContractError("Gate 2 smoke donor IDs are invalid or duplicated")
        if not config.is_record_eligible(record):
            raise MetAugContractError(
                "Gate 2 smoke manifest includes an ineligible compact-support donor"
            )
        if entry.get("target_patient_group") != manifest.target_groups[target_case_id]:
            raise MetAugContractError("Gate 2 smoke target patient group drifted")
        if entry.get("donor_patient_group") != record.patient_group:
            raise MetAugContractError("Gate 2 smoke donor patient group drifted")
        if record.patient_group == manifest.target_groups[target_case_id]:
            raise MetAugContractError("Gate 2 smoke includes a same patient-group donor")
        if entry.get("donor_source_case_id") != record.source_case_id:
            raise MetAugContractError("Gate 2 smoke donor source case drifted")
        if entry.get("classes_present") != list(record.classes_present):
            raise MetAugContractError("Gate 2 smoke class composition drifted")
        if not np.isclose(float(entry.get("core_volume_mm3", -1)), record.core_volume_mm3):
            raise MetAugContractError("Gate 2 smoke core volume drifted")
        if entry.get("core_volume_bin") != record.stratum[1]:
            raise MetAugContractError("Gate 2 smoke volume bin drifted")
        if config.donor_eligibility is not None:
            total_support_voxels, core_voxels = component_record_support_counts(record)
            if int(entry.get("total_support_voxels", -1)) != total_support_voxels:
                raise MetAugContractError("Gate 2 smoke total support drifted")
            if int(entry.get("core_voxels", -1)) != core_voxels:
                raise MetAugContractError("Gate 2 smoke core voxel count drifted")
            if not np.isclose(
                float(entry.get("total_to_core_ratio", -1)),
                total_support_voxels / core_voxels,
            ):
                raise MetAugContractError("Gate 2 smoke support/core ratio drifted")
        context = _event_context_from_mapping(entry["event"])
        if context.case_id != target_case_id or context.full_shape is None:
            raise MetAugContractError("Gate 2 smoke event context drifted")
        if len(context.patch_origin) != 3 or len(context.full_shape) != 3:
            raise MetAugContractError("Gate 2 smoke event geometry is malformed")
        if not isinstance(entry.get("planned"), dict):
            raise MetAugContractError("Gate 2 smoke placement is malformed")
        smoke_ids.add(smoke_id)
        event_ids.add(event_id)
        target_ids.add(target_case_id)
        component_ids.add(component_id)
        observed_bins[record.stratum[1]] += 1
    quota = int(payload["per_volume_bin_quota"])
    for volume_bin in GATE2_VOLUME_BINS:
        if observed_bins[volume_bin] < quota or int(payload.get("per_volume_bin", {}).get(volume_bin, -1)) != observed_bins[volume_bin]:
            raise MetAugContractError(f"Gate 2 smoke manifest lacks required {volume_bin} coverage")
    return payload


def build_review_template_row(case_result: Mapping[str, Any]) -> dict[str, str]:
    return {
        "smoke_id": str(case_result["smoke_id"]),
        "evidence_fingerprint": str(case_result["evidence_fingerprint"]),
        "target_case_id": str(case_result["target_case_id"]),
        "donor_component_id": str(case_result["donor_component_id"]),
        "core_volume_bin": str(case_result["core_volume_bin"]),
        "artifact_path": str(case_result["artifact_path"]),
        "montage_path": str(case_result["montage_path"]),
        "automatic_qc_status": str(case_result["automatic_qc_status"]),
        "review_decision": "pending",
        "reviewer": "",
        "reviewed_at_utc": "",
        "notes": "",
    }


def write_review_template(path: str | Path, case_results: list[Mapping[str, Any]]) -> None:
    destination = Path(path).expanduser().resolve()
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_TEMPLATE_FIELDS)
        writer.writeheader()
        for row in case_results:
            writer.writerow(build_review_template_row(row))


def validate_automatic_report(
    path: str | Path,
    *,
    smoke_manifest: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    payload = load_json_object(report_path, label="Gate 2 automatic report")
    if payload.get("schema_version") != GATE2_AUTOMATIC_REPORT_SCHEMA:
        raise MetAugContractError("unsupported Gate 2 automatic-report schema")
    if payload.get("automatic_report_sha256") != canonical_json_sha256(
        payload, exclude=("automatic_report_sha256",)
    ):
        raise MetAugContractError("Gate 2 automatic report SHA256 mismatch")
    expected = {
        "route_id": smoke_manifest["route_id"],
        "component_manifest_sha256": smoke_manifest["component_manifest_sha256"],
        "route_config_sha256": smoke_manifest["route_config_sha256"],
        "valid_mask_manifest_sha256": smoke_manifest["valid_mask_manifest_sha256"],
        "smoke_manifest_sha256": smoke_manifest["smoke_manifest_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MetAugContractError(f"Gate 2 automatic report does not bind {key}")
    if payload.get("status") != "hold_for_manual_review" or payload.get("automatic_status") != "pass":
        raise MetAugContractError("Gate 2 automatic report did not pass into manual-review hold")
    if payload.get("manual_review_status") != "hold_for_manual_review":
        raise MetAugContractError("Gate 2 automatic report has an invalid manual-review state")
    if payload.get("violations"):
        raise MetAugContractError("Gate 2 automatic report contains unresolved violations")
    if int(payload.get("smoke_count", 0)) != int(smoke_manifest["smoke_count"]):
        raise MetAugContractError("Gate 2 automatic report smoke count drifted")
    for volume_bin in GATE2_VOLUME_BINS:
        if int(payload.get("per_volume_bin", {}).get(volume_bin, 0)) < MIN_SMOKE_PER_VOLUME_BIN:
            raise MetAugContractError(f"Gate 2 automatic report lacks {volume_bin} coverage")
    expected_runtime = gate2_runtime_code_snapshot(repository_root)
    if payload.get("runtime_code") != expected_runtime:
        raise MetAugContractError("Gate 2 automatic report does not match the deployed Gate 2 runtime code")
    for key in ("g1_checkpoint_selection_sha256", "g2_parent_gate_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise MetAugContractError(f"Gate 2 automatic report does not bind {key}")
    g1_runtime = payload.get("g1_runtime_code")
    if not isinstance(g1_runtime, dict) or not isinstance(g1_runtime.get("files"), dict):
        raise MetAugContractError("Gate 2 automatic report does not bind the G1 runtime code")
    if g1_runtime.get("sha256") != canonical_json_sha256(g1_runtime["files"]):
        raise MetAugContractError("Gate 2 automatic report G1 runtime SHA256 mismatch")
    for path_key, sha_key in (
        ("case_results_file", "case_results_sha256"),
        ("manual_review_template", "manual_review_template_sha256"),
        ("event_audit_file", "event_audit_sha256"),
    ):
        relative = Path(str(payload.get(path_key, "")))
        if not relative.name or relative.is_absolute():
            raise MetAugContractError(f"Gate 2 automatic report has an invalid {path_key}")
        evidence_path = (report_path.parent / relative).resolve()
        if report_path.parent not in evidence_path.parents:
            raise MetAugContractError(f"Gate 2 automatic report {path_key} escapes its evidence directory")
        if not evidence_path.is_file() or sha256_file(evidence_path) != payload.get(sha_key):
            raise MetAugContractError(f"Gate 2 automatic report evidence drifted: {path_key}")
    return payload


def load_case_results_evidence(
    path: str | Path,
    *,
    evidence_root: str | Path,
    smoke_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reload every automatic case result and revalidate its immutable evidence."""
    results_path = Path(path).expanduser().resolve()
    root = Path(evidence_root).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_cases = {
        str(entry["smoke_id"]): entry for entry in smoke_manifest["smoke_cases"]
    }
    by_smoke: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise MetAugContractError("Gate 2 automatic case-results contains a malformed row")
        smoke_id = str(row.get("smoke_id", ""))
        if not smoke_id or smoke_id in by_smoke:
            raise MetAugContractError("Gate 2 automatic case-results has missing or duplicate smoke IDs")
        entry = expected_cases.get(smoke_id)
        if entry is None:
            raise MetAugContractError(f"Gate 2 automatic case-results has an unexpected smoke ID: {smoke_id}")
        fingerprint_payload = dict(row)
        observed_fingerprint = fingerprint_payload.pop("evidence_fingerprint", None)
        if observed_fingerprint != canonical_json_sha256(fingerprint_payload):
            raise MetAugContractError(f"Gate 2 evidence fingerprint mismatch: {smoke_id}")
        expected_fields = {
            "entry_sha256": entry["entry_sha256"],
            "event_id": entry["event_id"],
            "event_seed": entry["event_seed"],
            "target_case_id": entry["target_case_id"],
            "donor_component_id": entry["donor_component_id"],
            "core_volume_bin": entry["core_volume_bin"],
            "core_volume_mm3": entry["core_volume_mm3"],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise MetAugContractError(f"Gate 2 case-results field drifted ({field}): {smoke_id}")
        if (
            row.get("transaction_state") != "COMMITTED"
            or row.get("automatic_qc_status") != "pass"
            or row.get("violations")
        ):
            raise MetAugContractError(f"Gate 2 case result is not an automatically passing commit: {smoke_id}")
        for path_key, sha_key in (
            ("artifact_path", "artifact_sha256"),
            ("montage_path", "montage_sha256"),
        ):
            relative = Path(str(row.get(path_key, "")))
            if not relative.name or relative.is_absolute():
                raise MetAugContractError(f"Gate 2 case result has an invalid {path_key}: {smoke_id}")
            evidence_path = (root / relative).resolve()
            if root not in evidence_path.parents:
                raise MetAugContractError(f"Gate 2 case evidence escapes its output directory: {smoke_id}")
            if not evidence_path.is_file() or sha256_file(evidence_path) != row.get(sha_key):
                raise MetAugContractError(f"Gate 2 case evidence drifted ({path_key}): {smoke_id}")
        by_smoke[smoke_id] = row
    if set(by_smoke) != set(expected_cases):
        raise MetAugContractError(
            "Gate 2 automatic case-results does not cover the immutable smoke set: "
            f"missing={sorted(set(expected_cases) - set(by_smoke))}"
        )
    return by_smoke


def validate_manual_review(
    path: str | Path,
    *,
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decisions_path = Path(path).expanduser().resolve()
    if not decisions_path.is_file():
        raise FileNotFoundError(f"missing completed Gate 2 review decisions: {decisions_path}")
    with decisions_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(REVIEW_TEMPLATE_FIELDS) - set(rows[0]):
        raise MetAugContractError("Gate 2 manual review CSV lacks required columns")
    by_smoke: dict[str, dict[str, str]] = {}
    for row in rows:
        smoke_id = str(row.get("smoke_id", "")).strip()
        if not smoke_id or smoke_id in by_smoke:
            raise MetAugContractError("Gate 2 manual review has missing or duplicate smoke IDs")
        by_smoke[smoke_id] = {key: str(value or "") for key, value in row.items()}
    if set(by_smoke) != set(case_results):
        raise MetAugContractError(
            "Gate 2 manual review does not cover the immutable smoke set: "
            f"missing={sorted(set(case_results) - set(by_smoke))}, "
            f"extra={sorted(set(by_smoke) - set(case_results))}"
        )
    rejected: list[str] = []
    pending: list[str] = []
    for smoke_id, expected in case_results.items():
        row = by_smoke[smoke_id]
        for field in (
            "evidence_fingerprint",
            "target_case_id",
            "donor_component_id",
            "core_volume_bin",
            "artifact_path",
            "montage_path",
            "automatic_qc_status",
        ):
            if row.get(field) != str(expected[field]):
                raise MetAugContractError(f"Gate 2 manual review changed immutable field {field}: {smoke_id}")
        decision = row.get("review_decision", "").strip().lower()
        if decision == REVIEW_DECISION_REJECT:
            rejected.append(smoke_id)
        elif decision != REVIEW_DECISION_ACCEPT:
            pending.append(smoke_id)
        if decision == REVIEW_DECISION_ACCEPT and (
            not row.get("reviewer", "").strip() or not row.get("reviewed_at_utc", "").strip()
        ):
            pending.append(smoke_id)
    return {
        "decision_count": len(by_smoke),
        "accepted_count": len(by_smoke) - len(set(rejected) | set(pending)),
        "rejected_smoke_ids": sorted(set(rejected)),
        "pending_smoke_ids": sorted(set(pending)),
        "status": "pass" if not rejected and not pending else "fail",
    }
