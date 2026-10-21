#!/usr/bin/env python3
"""Strictly validate and freeze a failed Fix-v2 QC holdout without promotion."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    RouteConfig,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration  # noqa: E402
from custom_nnunet.met_aug_gate2 import load_smoke_manifest  # noqa: E402


EXPECTED_COUNT = 48
EXPECTED_BINS = ("27_49", "50_275", "gt_275")
EXPECTED_RATE_VIOLATIONS = {
    "generation_pass_rate_below_frozen_minimum",
    "effective_aug_rate_below_frozen_minimum",
}
EXPECTED_NO_OP_REASONS = {
    "RAW_GENERATION_QC_FAIL",
    "CANDIDATE_BOUNDARY_QC_FAIL",
    "CANDIDATE_CONTENT_QC_FAIL",
}
DECISION_FIELDS = {"review_decision", "reviewer", "reviewed_at_utc", "notes"}
NPZ_SCHEMA = {
    "original": ((4, 64, 64, 64), np.dtype("float32")),
    "original_segmentation": ((64, 64, 64), np.dtype("int16")),
    "inserted_label": ((64, 64, 64), np.dtype("int16")),
    "valid_mask": ((64, 64, 64), np.dtype("bool")),
    "raw_generation": ((4, 64, 64, 64), np.dtype("float32")),
    "preview_candidate": ((4, 64, 64, 64), np.dtype("float32")),
    "preview_segmentation": ((64, 64, 64), np.dtype("int16")),
    "strict_candidate": ((4, 64, 64, 64), np.dtype("float32")),
    "strict_segmentation": ((64, 64, 64), np.dtype("int16")),
    "label_support": ((64, 64, 64), np.dtype("bool")),
    "image_support": ((64, 64, 64), np.dtype("bool")),
    "event_json": ((), None),
    "transaction_json": ((), None),
    "preview_metadata_json": ((), None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--pre-qc-calibration", required=True)
    parser.add_argument("--measurement-calibration", required=True)
    parser.add_argument("--pre-qc-route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--review-decisions", required=True)
    parser.add_argument("--review-validation", required=True)
    parser.add_argument("--run-runtime-dir", required=True)
    parser.add_argument("--run-runtime-manifest", required=True)
    parser.add_argument("--review-runtime-dir", required=True)
    parser.add_argument("--review-runtime-manifest", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--freeze-at-utc", required=True)
    parser.add_argument("--files-manifest-output", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), [dict(row) for row in reader]


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{label} escapes its evidence root")
    return path


def _validate_runtime_manifest(runtime_dir: Path, manifest_path: Path) -> dict[str, str]:
    if not runtime_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("runtime directory or SHA manifest is missing")
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in entries:
            raise ValueError(f"malformed runtime manifest line: {line!r}")
        path = _safe_child(runtime_dir, relative, label="runtime manifest path")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"runtime evidence drifted: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("runtime manifest is empty")
    return entries


def _validate_png(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid montage PNG: {path.name}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"empty montage PNG: {path.name}")


def _parse_npz_json(value: np.ndarray, *, label: str) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} is not a scalar JSON string")
    parsed = json.loads(str(value.item()))
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} does not contain a JSON object")
    return parsed


def _require_labels(array: np.ndarray, allowed: set[int], *, label: str) -> None:
    observed = set(int(value) for value in np.unique(array))
    if not observed.issubset(allowed):
        raise ValueError(f"{label} contains illegal values: {sorted(observed - allowed)}")


def _validate_artifact(
    path: Path,
    *,
    row: Mapping[str, Any],
    event: Mapping[str, Any],
    audit: Mapping[str, Any],
    measurement_calibration_sha256: str,
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(NPZ_SCHEMA):
            raise ValueError(f"artifact array schema drifted: {path.name}")
        for key, (shape, dtype) in NPZ_SCHEMA.items():
            value = archive[key]
            if value.shape != shape or (dtype is not None and value.dtype != dtype):
                raise ValueError(
                    f"artifact array shape/dtype drifted ({key}): {path.name}"
                )
        for key in ("original", "raw_generation", "preview_candidate", "strict_candidate"):
            if not np.isfinite(archive[key]).all():
                raise ValueError(f"artifact contains nonfinite values ({key}): {path.name}")

        original = archive["original"]
        original_segmentation = archive["original_segmentation"]
        inserted_label = archive["inserted_label"]
        valid_mask = archive["valid_mask"]
        preview_candidate = archive["preview_candidate"]
        preview_segmentation = archive["preview_segmentation"]
        strict_candidate = archive["strict_candidate"]
        strict_segmentation = archive["strict_segmentation"]
        label_support = archive["label_support"]
        image_support = archive["image_support"]

        _require_labels(original_segmentation, {-1, 0, 1, 2, 3, 4}, label="original segmentation")
        _require_labels(inserted_label, {0, 1, 2, 3, 4}, label="inserted label")
        _require_labels(preview_segmentation, {-1, 0, 1, 2, 3, 4}, label="preview segmentation")
        _require_labels(strict_segmentation, {-1, 0, 1, 2, 3, 4}, label="strict segmentation")
        if not np.array_equal(label_support, inserted_label != 0):
            raise ValueError(f"label support does not match inserted labels: {path.name}")
        if not np.any(label_support) or not np.array_equal(image_support, label_support):
            raise ValueError(f"label-only image support contract drifted: {path.name}")
        if not np.all(valid_mask[label_support]):
            raise ValueError(f"inserted label leaves the valid mask: {path.name}")
        if not np.array_equal(preview_candidate[:, ~image_support], original[:, ~image_support]):
            raise ValueError(f"preview changed outside image support: {path.name}")
        if not np.array_equal(strict_candidate[:, ~image_support], original[:, ~image_support]):
            raise ValueError(f"strict candidate changed outside image support: {path.name}")
        expected_preview_segmentation = original_segmentation.copy()
        expected_preview_segmentation[label_support] = inserted_label[label_support]
        if not np.array_equal(preview_segmentation, expected_preview_segmentation):
            raise ValueError(f"preview segmentation insertion contract drifted: {path.name}")

        state = str(row["transaction_state"])
        if state == "COMMITTED":
            if not np.array_equal(strict_candidate, preview_candidate) or not np.array_equal(
                strict_segmentation, preview_segmentation
            ):
                raise ValueError(f"committed strict/preview replay drifted: {path.name}")
        elif state == "NO_OP":
            if not np.array_equal(strict_candidate, original) or not np.array_equal(
                strict_segmentation, original_segmentation
            ):
                raise ValueError(f"NO_OP changed the strict crop: {path.name}")
        else:
            raise ValueError(f"unexpected transaction state: {state}")

        artifact_event = _parse_npz_json(archive["event_json"], label="event_json")
        transaction = _parse_npz_json(
            archive["transaction_json"], label="transaction_json"
        )
        metadata = _parse_npz_json(
            archive["preview_metadata_json"], label="preview_metadata_json"
        )
        if artifact_event != event:
            raise ValueError(f"artifact event payload drifted: {path.name}")
        if transaction != audit:
            raise ValueError(f"artifact transaction audit drifted: {path.name}")
        expected_transaction = {
            "event_id": row["event_id"],
            "event_seed": row["event_seed"],
            "target_case_id": row["target_case_id"],
            "target_patient_group": row["target_patient_group"],
            "component_id": row["donor_component_id"],
            "donor_patient_group": row["donor_patient_group"],
            "state": state,
            "reason": row["transaction_reason"],
        }
        for key, expected in expected_transaction.items():
            if transaction.get(key) != expected:
                raise ValueError(f"transaction binding drifted ({key}): {path.name}")
        if (
            metadata.get("boundary_policy") != "label_only_qc_v1"
            or metadata.get("calibration_sha256") != measurement_calibration_sha256
            or metadata.get("harmonization", {}).get("policy") != "disabled"
            or metadata.get("raw_qc", {}).get("status") != "pass"
            or metadata.get("candidate_qc", {}).get("status") != "pass"
        ):
            raise ValueError(f"measurement preview metadata drifted: {path.name}")
        geometry = metadata.get("geometry", {})
        if (
            int(geometry.get("label_support_voxels", -1)) != int(label_support.sum())
            or int(geometry.get("image_support_voxels", -1)) != int(image_support.sum())
        ):
            raise ValueError(f"measurement geometry accounting drifted: {path.name}")


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _nested_counts(
    keys: list[str], decisions: Mapping[str, str], private_by_code: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for code in keys:
        key = str(private_by_code[code]["transaction_reason"] or "COMMITTED")
        decision = decisions[code]
        bucket = result.setdefault(key, {"accept": 0, "reject": 0})
        bucket[decision] += 1
    return dict(sorted(result.items()))


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    files_manifest_path = Path(args.files_manifest_output).expanduser().resolve()
    if output_path.exists() or files_manifest_path.exists():
        raise FileExistsError("QC failure report or files manifest already exists")

    manifest = ComponentManifest.load(args.component_manifest)
    strict_calibration_path = Path(args.pre_qc_calibration).expanduser().resolve()
    strict_calibration = FixV2Calibration.load(
        strict_calibration_path, expected_policy="label_only_qc_v1"
    )
    if strict_calibration.payload.get("calibration_role") != (
        "qc_holdout_candidate_not_gate_eligible"
    ):
        raise ValueError("strict calibration is not the pre-QC candidate")
    measurement_calibration_path = Path(args.measurement_calibration).expanduser().resolve()
    measurement_calibration = FixV2Calibration.load(
        measurement_calibration_path, expected_policy="label_only_qc_v1"
    )
    route_path = Path(args.pre_qc_route_config).expanduser().resolve()
    route = RouteConfig.load(route_path, manifest)
    if route.fix_v2 is None or route.fix_v2.calibration_sha256 != strict_calibration.sha256:
        raise ValueError("pre-QC route does not bind the strict calibration")

    event_manifest_path = Path(args.event_manifest).expanduser().resolve()
    event_manifest = load_smoke_manifest(
        event_manifest_path,
        manifest=manifest,
        config=route,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    events = list(event_manifest.get("smoke_cases", []))
    if (
        event_manifest.get("calibration_partition") != "qc_holdout"
        or int(event_manifest.get("smoke_count", -1)) != EXPECTED_COUNT
        or len(events) != EXPECTED_COUNT
        or any(
            int(event_manifest.get("per_volume_bin", {}).get(value, -1)) != 16
            for value in EXPECTED_BINS
        )
    ):
        raise ValueError("QC holdout event denominator or partition drifted")
    events_by_id = {str(entry["event_id"]): entry for entry in events}
    if len(events_by_id) != EXPECTED_COUNT:
        raise ValueError("QC event IDs are not unique")

    run_runtime_dir = Path(args.run_runtime_dir).expanduser().resolve()
    run_runtime_manifest_path = Path(args.run_runtime_manifest).expanduser().resolve()
    run_runtime = _validate_runtime_manifest(run_runtime_dir, run_runtime_manifest_path)
    runner_relative = "scripts/39_run_met_aug_fix_v2_fixed_events.py"
    if runner_relative not in run_runtime:
        raise ValueError("run runtime manifest lacks the fixed-event runner")
    review_runtime_dir = Path(args.review_runtime_dir).expanduser().resolve()
    review_runtime_manifest_path = Path(args.review_runtime_manifest).expanduser().resolve()
    review_runtime = _validate_runtime_manifest(
        review_runtime_dir, review_runtime_manifest_path
    )
    if "scripts/43_freeze_met_aug_fix_v2_blinded_review.py" not in review_runtime:
        raise ValueError("review runtime manifest lacks the blinded-review freezer")

    run_dir = Path(args.run_output).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"QC run output is missing: {run_dir}")
    report_path = run_dir / "FIXED_EVENT_REPORT.json"
    report = _read_json(report_path, "fixed-event report")
    if report.get("report_sha256") != canonical_json_sha256(
        report, exclude=("report_sha256",)
    ):
        raise ValueError("fixed-event report audit SHA256 drifted")
    expected_report = {
        "status": "fail",
        "stage": "qc_holdout",
        "calibration_partition": "qc_holdout",
        "attempt_count": EXPECTED_COUNT,
        "reviewable_count": EXPECTED_COUNT,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(route_path),
        "strict_calibration_sha256": strict_calibration.sha256,
        "measurement_calibration_sha256": measurement_calibration.sha256,
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "runner_sha256": run_runtime[runner_relative],
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"fixed-event report binding drifted: {key}")
    if set(report.get("violations", [])) != EXPECTED_RATE_VIOLATIONS:
        raise ValueError("QC report has unexpected or missing failure violations")

    results_path = _safe_child(run_dir, str(report["results_file"]), label="results file")
    template_path = _safe_child(
        run_dir, str(report["review_template"]), label="review template"
    )
    private_path = run_dir / "PRIVATE_BLINDING_MAP.json"
    audit_path = run_dir / "transaction_events.jsonl"
    for path, expected in (
        (results_path, report["results_sha256"]),
        (template_path, report["review_template_sha256"]),
        (private_path, report["private_blinding_map_sha256"]),
        (audit_path, report["transaction_audit_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"QC run evidence drifted: {path.name}")
    if stat.S_IMODE(private_path.stat().st_mode) != 0o400:
        raise ValueError("QC private blinding map is not mode 0400")

    rows = _read_jsonl(results_path, "fixed-event results")
    audit_rows = _read_jsonl(audit_path, "transaction audit")
    if len(rows) != EXPECTED_COUNT or len(audit_rows) != EXPECTED_COUNT:
        raise ValueError("QC result or transaction-audit denominator drifted")
    rows_by_code: dict[str, dict[str, Any]] = {}
    audit_by_event: dict[str, dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    bin_counts: Counter[str] = Counter()
    expected_files = {
        report_path.resolve(),
        results_path.resolve(),
        template_path.resolve(),
        private_path.resolve(),
        audit_path.resolve(),
    }
    for audit in audit_rows:
        event_id = str(audit.get("event_id", ""))
        if not event_id or event_id in audit_by_event:
            raise ValueError("transaction audit has missing or duplicate event IDs")
        audit_by_event[event_id] = audit
    for expected_index, row in enumerate(rows):
        code = str(row.get("blind_code", ""))
        event_id = str(row.get("event_id", ""))
        if not code or code in rows_by_code or event_id not in events_by_id:
            raise ValueError("fixed-event results have invalid code/event identity")
        if int(row.get("event_index", -1)) != expected_index:
            raise ValueError(f"fixed-event ordering drifted: {code}")
        if row.get("row_sha256") != canonical_json_sha256(
            row, exclude=("row_sha256",)
        ):
            raise ValueError(f"fixed-event row SHA256 drifted: {code}")
        event = events_by_id[event_id]
        expected_identity = {
            "smoke_id": event["smoke_id"],
            "event_seed": event["event_seed"],
            "target_case_id": event["target_case_id"],
            "target_patient_group": event["target_patient_group"],
            "donor_component_id": event["donor_component_id"],
            "donor_patient_group": event["donor_patient_group"],
            "core_volume_bin": event["core_volume_bin"],
        }
        for key, expected in expected_identity.items():
            if row.get(key) != expected:
                raise ValueError(f"fixed-event row identity drifted ({key}): {code}")
        if row.get("violations") != []:
            raise ValueError(f"fixed-event row contains a structural violation: {code}")
        state = str(row.get("transaction_state"))
        reason = row.get("transaction_reason")
        if state == "COMMITTED" and reason is not None:
            raise ValueError(f"committed row has a rejection reason: {code}")
        if state == "NO_OP" and reason not in EXPECTED_NO_OP_REASONS:
            raise ValueError(f"NO_OP row has an unexpected reason: {code}")
        if state not in {"COMMITTED", "NO_OP"}:
            raise ValueError(f"unexpected transaction state: {code}")
        artifact_path = _safe_child(
            run_dir, str(row["artifact_path"]), label="artifact path"
        )
        montage_path = _safe_child(
            run_dir, str(row["montage_path"]), label="montage path"
        )
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != row["artifact_sha256"]
            or not montage_path.is_file()
            or sha256_file(montage_path) != row["montage_sha256"]
        ):
            raise ValueError(f"artifact or montage SHA256 drifted: {code}")
        _validate_png(montage_path)
        if event_id not in audit_by_event:
            raise ValueError(f"transaction audit lacks event: {code}")
        _validate_artifact(
            artifact_path,
            row=row,
            event=event,
            audit=audit_by_event[event_id],
            measurement_calibration_sha256=measurement_calibration.sha256,
        )
        expected_files.update((artifact_path, montage_path))
        rows_by_code[code] = row
        state_counts[state] += 1
        reason_counts[str(reason or "COMMITTED")] += 1
        bin_counts[str(row["core_volume_bin"])] += 1
    if set(audit_by_event) != set(events_by_id):
        raise ValueError("transaction audit does not cover the event manifest")
    if state_counts != Counter({"COMMITTED": 24, "NO_OP": 24}):
        raise ValueError("QC transaction accounting drifted")
    if _counter_dict(state_counts) != report.get("state_counts"):
        raise ValueError("QC report state counts drifted")
    if _counter_dict(reason_counts) != report.get("reason_counts"):
        raise ValueError("QC report reason counts drifted")
    if any(bin_counts[value] != 16 for value in EXPECTED_BINS):
        raise ValueError("QC result volume-bin denominator drifted")
    generation_pass_rate = state_counts["COMMITTED"] / EXPECTED_COUNT
    effective_aug_rate = route.p_select * generation_pass_rate
    if (
        float(report["generation_pass_rate"]) != generation_pass_rate
        or float(report["effective_aug_rate"]) != effective_aug_rate
    ):
        raise ValueError("QC report rate accounting drifted")
    rate_contract = strict_calibration.payload["effective_rate_contract"]
    minimum_generation_pass_rate = float(
        rate_contract["minimum_generation_pass_rate"]
    )
    minimum_effective_aug_rate = float(rate_contract["minimum_effective_aug_rate"])
    if not (
        generation_pass_rate < minimum_generation_pass_rate
        and effective_aug_rate < minimum_effective_aug_rate
    ):
        raise ValueError("QC holdout did not actually miss both frozen rate floors")

    template_fields, template_rows = _read_csv(template_path)
    decisions_path = Path(args.review_decisions).expanduser().resolve()
    decision_fields, decision_rows = _read_csv(decisions_path)
    if (
        len(template_rows) != EXPECTED_COUNT
        or len(decision_rows) != EXPECTED_COUNT
        or template_fields != decision_fields
    ):
        raise ValueError("blinded-review denominator or CSV header drifted")
    template_by_code = {row["blind_code"]: row for row in template_rows}
    decisions_by_code = {row["blind_code"]: row for row in decision_rows}
    if (
        len(template_by_code) != EXPECTED_COUNT
        or len(decisions_by_code) != EXPECTED_COUNT
        or set(template_by_code) != set(decisions_by_code)
        or set(decisions_by_code) != set(rows_by_code)
    ):
        raise ValueError("blinded review does not cover the fixed set exactly once")
    decisions: dict[str, str] = {}
    for code, decision in decisions_by_code.items():
        template = template_by_code[code]
        for field, expected in template.items():
            if field not in DECISION_FIELDS and decision.get(field) != expected:
                raise ValueError(f"blinded review changed immutable field {field}: {code}")
        value = decision.get("review_decision", "").strip().lower()
        if value not in {"accept", "reject"}:
            raise ValueError(f"blinded review is pending or malformed: {code}")
        if not decision.get("reviewer", "").strip() or not decision.get(
            "reviewed_at_utc", ""
        ).strip():
            raise ValueError(f"blinded review lacks reviewer/timestamp: {code}")
        decisions[code] = value

    review_validation_path = Path(args.review_validation).expanduser().resolve()
    review_validation = _read_json(
        review_validation_path, "blinded-review validation"
    )
    if review_validation.get("validation_audit_sha256") != canonical_json_sha256(
        review_validation, exclude=("validation_audit_sha256",)
    ):
        raise ValueError("blinded-review validation audit SHA256 drifted")
    rejected_codes = sorted(code for code, value in decisions.items() if value == "reject")
    if (
        review_validation.get("status") != "pass"
        or review_validation.get("stage")
        != "qc_holdout_ai_assisted_blinded_visual_review"
        or review_validation.get("decision_file_sha256") != sha256_file(decisions_path)
        or review_validation.get("template_sha256") != sha256_file(template_path)
        or review_validation.get("rejected_blind_codes") != rejected_codes
        or int(review_validation.get("accept_count", -1))
        != EXPECTED_COUNT - len(rejected_codes)
        or int(review_validation.get("reject_count", -1)) != len(rejected_codes)
        or int(review_validation.get("pending_count", -1)) != 0
        or review_validation.get("private_blinding_map_accessed_before_decision_lock")
        is not False
    ):
        raise ValueError("blinded-review lock validation failed")

    # Access the identity map only after every visual decision and lock check passes.
    private = _read_json(private_path, "private blinding map")
    if private.get("stage") != "qc_holdout" or not isinstance(
        private.get("entries"), list
    ):
        raise ValueError("private blinding map schema drifted")
    private_by_code = {
        str(entry["blind_code"]): entry for entry in private["entries"]
    }
    if len(private_by_code) != EXPECTED_COUNT or set(private_by_code) != set(decisions):
        raise ValueError("private blinding map does not cover the decisions")
    for code, entry in private_by_code.items():
        row = rows_by_code[code]
        if (
            entry.get("event_id") != row["event_id"]
            or entry.get("transaction_state") != row["transaction_state"]
            or entry.get("transaction_reason") != row["transaction_reason"]
        ):
            raise ValueError(f"private map transaction binding drifted: {code}")

    committed_visual_rejects = sorted(
        code
        for code, value in decisions.items()
        if value == "reject"
        and private_by_code[code]["transaction_state"] == "COMMITTED"
    )
    automatic_false_rejects = sorted(
        code
        for code, value in decisions.items()
        if value == "accept"
        and private_by_code[code]["transaction_state"] == "NO_OP"
    )
    transaction_visual_counts = {
        state: {
            decision: sum(
                private_by_code[code]["transaction_state"] == state
                and decisions[code] == decision
                for code in decisions
            )
            for decision in ("accept", "reject")
        }
        for state in ("COMMITTED", "NO_OP")
    }
    reason_visual_counts = _nested_counts(
        sorted(decisions), decisions, private_by_code
    )

    actual_files = {path.resolve() for path in run_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        unexpected = sorted(
            str(path.relative_to(run_dir)) for path in actual_files - expected_files
        )
        missing = sorted(
            str(path.relative_to(run_dir)) for path in expected_files - actual_files
        )
        raise ValueError(
            f"QC run file set drifted (unexpected={unexpected}, missing={missing})"
        )
    file_manifest_bytes = "".join(
        f"{sha256_file(path)}  {path.relative_to(run_dir)}\n"
        for path in sorted(actual_files, key=lambda value: str(value.relative_to(run_dir)))
    ).encode("utf-8")
    files_manifest_sha256 = hashlib.sha256(file_manifest_bytes).hexdigest()

    run_log_path = Path(args.run_log).expanduser().resolve()
    if not run_log_path.is_file():
        raise FileNotFoundError("QC run log is missing")
    failure_report: dict[str, Any] = {
        "schema_version": 1,
        "status": "fail",
        "stage": "fix_v2_qc_holdout",
        "evidence_validation_status": "pass",
        "route_decision": "stop_before_gate0",
        "failure_reasons": sorted(EXPECTED_RATE_VIOLATIONS),
        "threshold_promotion_performed": False,
        "training_or_inference_eligible": False,
        "downstream_gates": {
            "gate0": "blocked",
            "gate1a_100000": "blocked",
            "gate1b_96_double_replay": "blocked",
            "gate2_120": "blocked",
        },
        "freeze_at_utc": args.freeze_at_utc,
        "decision_lock_at_utc": review_validation["decision_lock_at_utc"],
        "attempt_count": EXPECTED_COUNT,
        "artifact_validation_count": EXPECTED_COUNT,
        "montage_validation_count": EXPECTED_COUNT,
        "committed_count": int(state_counts["COMMITTED"]),
        "automatic_reject_count": int(state_counts["NO_OP"]),
        "generation_pass_rate": generation_pass_rate,
        "minimum_generation_pass_rate": minimum_generation_pass_rate,
        "effective_aug_rate": effective_aug_rate,
        "minimum_effective_aug_rate": minimum_effective_aug_rate,
        "visual_accept_count": EXPECTED_COUNT - len(rejected_codes),
        "visual_reject_count": len(rejected_codes),
        "committed_visual_reject_count": len(committed_visual_rejects),
        "automatic_false_reject_count": len(automatic_false_rejects),
        "committed_visual_rejects": committed_visual_rejects,
        "automatic_false_rejects": automatic_false_rejects,
        "visual_rejects": rejected_codes,
        "transaction_visual_counts": transaction_visual_counts,
        "reason_visual_counts": reason_visual_counts,
        "automatic_reason_counts": _counter_dict(reason_counts),
        "input_sha256": {
            "component_manifest_identity": manifest.identity_sha256,
            "pre_qc_calibration": strict_calibration.sha256,
            "measurement_calibration": measurement_calibration.sha256,
            "pre_qc_route_config": sha256_file(route_path),
            "valid_mask_manifest": sha256_file(args.valid_mask_manifest),
            "event_manifest": sha256_file(event_manifest_path),
            "fixed_event_report_file": sha256_file(report_path),
            "fixed_event_report_audit": report["report_sha256"],
            "review_decisions": sha256_file(decisions_path),
            "review_validation": sha256_file(review_validation_path),
            "run_runtime_manifest": sha256_file(run_runtime_manifest_path),
            "review_runtime_manifest": sha256_file(review_runtime_manifest_path),
            "run_log": sha256_file(run_log_path),
        },
        "run_files_manifest_sha256": files_manifest_sha256,
        "run_file_count": len(actual_files),
    }
    failure_report["failure_audit_sha256"] = canonical_json_sha256(
        failure_report, exclude=("failure_audit_sha256",)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(files_manifest_path, file_manifest_bytes)
    _write_exclusive(output_path, _encoded(failure_report))

    for path in actual_files:
        path.chmod(0o400 if path == private_path else 0o444)
    for directory in sorted(
        (path for path in run_dir.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    run_dir.chmod(0o555)
    run_log_path.chmod(0o444)
    print(json.dumps(failure_report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
