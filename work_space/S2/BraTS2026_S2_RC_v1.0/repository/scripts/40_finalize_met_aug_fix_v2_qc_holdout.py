#!/usr/bin/env python3
"""Validate blinded QC holdout and promote unchanged thresholds to gate eligibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

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


DECISION_FIELDS = {"review_decision", "reviewer", "reviewed_at_utc", "notes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--pre-qc-calibration", required=True)
    parser.add_argument("--pre-qc-route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--review-decisions", required=True)
    parser.add_argument("--review-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _encoded(payload: dict[str, Any]) -> bytes:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"QC freeze output already exists: {output_dir}")
    manifest = ComponentManifest.load(args.component_manifest)
    pre_qc_calibration_path = Path(args.pre_qc_calibration).expanduser().resolve()
    pre_qc_calibration = FixV2Calibration.load(
        pre_qc_calibration_path, expected_policy="label_only_qc_v1"
    )
    if pre_qc_calibration.payload.get("calibration_role") != (
        "qc_holdout_candidate_not_gate_eligible"
    ):
        raise ValueError("input calibration is not the frozen pre-QC candidate")
    pre_qc_route_path = Path(args.pre_qc_route_config).expanduser().resolve()
    pre_qc_route = RouteConfig.load(pre_qc_route_path, manifest)
    if pre_qc_route.fix_v2 is None or pre_qc_route.fix_v2.calibration_sha256 != (
        pre_qc_calibration.sha256
    ):
        raise ValueError("pre-QC route config does not bind the calibration candidate")
    event_manifest_path = Path(args.event_manifest).expanduser().resolve()
    event_manifest = load_smoke_manifest(
        event_manifest_path,
        manifest=manifest,
        config=pre_qc_route,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    if (
        event_manifest.get("calibration_partition") != "qc_holdout"
        or int(event_manifest.get("smoke_count", -1)) != 48
        or any(
            int(event_manifest.get("per_volume_bin", {}).get(value, -1)) != 16
            for value in ("27_49", "50_275", "gt_275")
        )
    ):
        raise ValueError("QC holdout manifest denominator or partition drifted")

    run_dir = Path(args.run_output).expanduser().resolve()
    report_path = run_dir / "FIXED_EVENT_REPORT.json"
    report = _read_json(report_path, "fixed-event report")
    if report.get("report_sha256") != canonical_json_sha256(
        report, exclude=("report_sha256",)
    ):
        raise ValueError("fixed-event report audit SHA256 drifted")
    expected_report = {
        "status": "hold_for_blinded_manual_review",
        "stage": "qc_holdout",
        "calibration_partition": "qc_holdout",
        "attempt_count": 48,
        "reviewable_count": 48,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(pre_qc_route_path),
        "strict_calibration_sha256": sha256_file(pre_qc_calibration_path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "event_manifest_sha256": sha256_file(event_manifest_path),
    }
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise ValueError(f"fixed-event report does not bind {key}")
    if report.get("violations"):
        raise ValueError("QC holdout run contains automatic violations")
    if int(report.get("committed_count", -1)) + int(report.get("rejected_count", -1)) != 48:
        raise ValueError("QC holdout transaction accounting drifted")
    rate_contract = pre_qc_calibration.payload["effective_rate_contract"]
    if float(report["generation_pass_rate"]) < float(
        rate_contract["minimum_generation_pass_rate"]
    ) or float(report["effective_aug_rate"]) < float(
        rate_contract["minimum_effective_aug_rate"]
    ):
        raise ValueError("QC holdout missed a frozen rate floor")

    results_path = run_dir / report["results_file"]
    template_path = run_dir / report["review_template"]
    private_path = run_dir / "PRIVATE_BLINDING_MAP.json"
    for path, expected in (
        (results_path, report["results_sha256"]),
        (template_path, report["review_template_sha256"]),
        (private_path, report["private_blinding_map_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"QC holdout evidence drifted: {path.name}")
    if stat.S_IMODE(private_path.stat().st_mode) != 0o400:
        raise ValueError("QC private blinding map is not mode 0400")
    result_rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(result_rows) != 48:
        raise ValueError("QC fixed-event result count drifted")
    results_by_code: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        code = str(row.get("blind_code", ""))
        if not code or code in results_by_code:
            raise ValueError("QC fixed-event results contain duplicate blind codes")
        if row.get("row_sha256") != canonical_json_sha256(
            row, exclude=("row_sha256",)
        ):
            raise ValueError(f"QC fixed-event row SHA256 drifted: {code}")
        for path_key, sha_key in (
            ("artifact_path", "artifact_sha256"),
            ("montage_path", "montage_sha256"),
        ):
            evidence_path = (run_dir / row[path_key]).resolve()
            if run_dir not in evidence_path.parents or sha256_file(evidence_path) != row[sha_key]:
                raise ValueError(f"QC evidence file drifted ({path_key}): {code}")
        results_by_code[code] = row

    template_rows = _read_csv(template_path)
    decisions_path = Path(args.review_decisions).expanduser().resolve()
    decision_rows = _read_csv(decisions_path)
    if len(template_rows) != 48 or len(decision_rows) != 48:
        raise ValueError("QC review template or decision denominator drifted")
    template_by_code = {row["blind_code"]: row for row in template_rows}
    decisions_by_code = {row["blind_code"]: row for row in decision_rows}
    if (
        len(template_by_code) != 48
        or len(decisions_by_code) != 48
        or set(template_by_code) != set(decisions_by_code)
        or set(decisions_by_code) != set(results_by_code)
    ):
        raise ValueError("QC blinded review does not cover the fixed set exactly once")
    for code, decision in decisions_by_code.items():
        template = template_by_code[code]
        for field, expected in template.items():
            if field not in DECISION_FIELDS and decision.get(field) != expected:
                raise ValueError(f"QC review changed immutable field {field}: {code}")
        if decision.get("review_decision", "").strip().lower() not in {
            "accept",
            "reject",
        }:
            raise ValueError(f"QC review is pending or malformed: {code}")
        if not decision.get("reviewer", "").strip() or not decision.get(
            "reviewed_at_utc", ""
        ).strip():
            raise ValueError(f"QC review lacks reviewer/timestamp: {code}")

    review_validation_path = Path(args.review_validation).expanduser().resolve()
    review_validation = _read_json(review_validation_path, "QC review validation")
    if (
        review_validation.get("status") != "pass"
        or review_validation.get("decision_file_sha256") != sha256_file(decisions_path)
        or review_validation.get("template_sha256") != sha256_file(template_path)
        or review_validation.get("private_blinding_map_accessed_before_decision_lock")
        is not False
    ):
        raise ValueError("QC blinded-review lock validation failed")

    # The private mapping is read only after every visual decision and its lock audit pass.
    private = _read_json(private_path, "QC private blinding map")
    private_by_code = {entry["blind_code"]: entry for entry in private["entries"]}
    if len(private_by_code) != 48 or set(private_by_code) != set(decisions_by_code):
        raise ValueError("QC private map does not cover the blinded decisions")
    committed_rejects: list[str] = []
    automatic_false_rejects: list[str] = []
    visual_rejects: list[str] = []
    for code, decision in decisions_by_code.items():
        visual = decision["review_decision"].strip().lower()
        transaction = private_by_code[code]["transaction_state"]
        if visual == "reject":
            visual_rejects.append(code)
        if transaction == "COMMITTED" and visual == "reject":
            committed_rejects.append(code)
        if transaction == "NO_OP" and visual == "accept":
            automatic_false_rejects.append(code)
    if committed_rejects:
        raise RuntimeError(
            f"QC holdout has manually rejected automatically released candidates: {committed_rejects}"
        )

    pre_threshold_sha = pre_qc_calibration.payload["threshold_derivation"][
        "threshold_payload_sha256"
    ]
    qc_holdout_audit = {
        "schema_version": 1,
        "status": "pass",
        "fixed_event_report_sha256": sha256_file(report_path),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "review_decisions_sha256": sha256_file(decisions_path),
        "review_validation_sha256": sha256_file(review_validation_path),
        "attempt_count": 48,
        "committed_count": int(report["committed_count"]),
        "automatic_reject_count": int(report["rejected_count"]),
        "visual_reject_count": len(visual_rejects),
        "committed_visual_reject_count": 0,
        "automatic_false_reject_count": len(automatic_false_rejects),
        "generation_pass_rate": float(report["generation_pass_rate"]),
        "effective_aug_rate": float(report["effective_aug_rate"]),
        "threshold_payload_sha256": pre_threshold_sha,
    }
    # Promotion never rewrites a threshold-bearing byte. Gate eligibility is
    # conferred only by the external .ok artifact that binds this exact file to
    # the one-time holdout evidence.
    calibration_bytes = pre_qc_calibration_path.read_bytes()
    route_bytes = pre_qc_route_path.read_bytes()
    calibration_sha256 = sha256_file(pre_qc_calibration_path)
    route_sha256 = sha256_file(pre_qc_route_path)
    final_report = {
        "schema_version": 1,
        "status": "pass",
        "selected_candidate_id": "A_label_only",
        "boundary_policy": "label_only_qc_v1",
        "threshold_payload_sha256": pre_threshold_sha,
        "pre_qc_calibration_sha256": pre_qc_calibration.sha256,
        "frozen_calibration_sha256": calibration_sha256,
        "frozen_route_config_sha256": route_sha256,
        "qc_holdout": qc_holdout_audit,
        "committed_visual_rejects": committed_rejects,
        "automatic_false_rejects": sorted(automatic_false_rejects),
        "visual_rejects": sorted(visual_rejects),
    }
    final_report["finalization_audit_sha256"] = canonical_json_sha256(
        final_report, exclude=("finalization_audit_sha256",)
    )
    report_bytes = _encoded(final_report)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    ok_payload = {
        "status": "pass",
        "stage": "fix_v2_qc_holdout_frozen",
        "calibration_sha256": calibration_sha256,
        "route_config_sha256": route_sha256,
        "final_report_sha256": report_sha256,
        "threshold_payload_sha256": pre_threshold_sha,
        "promotion_policy": "immutable_calibration_bytes_plus_external_qc_approval_v1",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_exclusive(output_dir / "FROZEN_FIX_V2_CALIBRATION.json", calibration_bytes)
    _write_exclusive(output_dir / "FROZEN_FIX_V2_ROUTE_CONFIG.json", route_bytes)
    _write_exclusive(output_dir / "QC_HOLDOUT_FINAL_REPORT.json", report_bytes)
    _write_exclusive(output_dir / "FROZEN_FIX_V2_CALIBRATION.ok", _encoded(ok_payload))
    output_dir.chmod(0o555)
    print(json.dumps(final_report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
