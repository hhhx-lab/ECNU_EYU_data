#!/usr/bin/env python3
"""Select Fix-v2 A/B/C and derive a pre-QC A calibration from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    S2_MODALITIES,
    canonical_json_sha256,
    make_fix_v2_route_a_config,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2 import (  # noqa: E402
    FixV2Calibration,
    FixV2CandidateProcessor,
)
from custom_nnunet.met_aug_fix_v2_selection import derive_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--reference-cdf", required=True)
    parser.add_argument("--reference-validation", required=True)
    parser.add_argument("--measurement-config-index", required=True)
    parser.add_argument("--development-output", required=True)
    parser.add_argument("--development-validation", required=True)
    parser.add_argument("--development-files-manifest", required=True)
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


def _encoded(payload: dict[str, Any], *, compact: bool = False) -> bytes:
    options = {"ensure_ascii": True, "sort_keys": True, "allow_nan": False}
    if compact:
        text = json.dumps(payload, separators=(",", ":"), **options) + "\n"
    else:
        text = json.dumps(payload, indent=2, **options) + "\n"
    return text.encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive(path: Path, content: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(
        row.get("row_sha256")
        != canonical_json_sha256(row, exclude=("row_sha256",))
        for row in rows
    ):
        raise ValueError("Development measurement row SHA256 drifted")
    return rows


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


class _ReplayBackend:
    def __init__(self, generated: np.ndarray):
        self.generated = np.asarray(generated, dtype=np.float32)

    def generate(self, image, label, *, seed, inpaint_support=None):
        support = label != 0 if inpaint_support is None else np.asarray(inpaint_support, dtype=bool)
        if self.generated.shape != image.shape or np.any(self.generated[:, ~support] != image[:, ~support]):
            raise ValueError("Development raw replay violates the known-region contract")
        return self.generated.copy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"selection/derivation output already exists: {output_dir}")

    manifest = ComponentManifest.load(args.component_manifest)
    reference_path = Path(args.reference_cdf).expanduser().resolve()
    reference_validation_path = Path(args.reference_validation).expanduser().resolve()
    reference = _read_json(reference_path, "Reference CDF")
    reference_validation = _read_json(reference_validation_path, "Reference validation")
    if reference_validation.get("status") != "pass" or reference_validation.get(
        "reference_file_sha256"
    ) != sha256_file(reference_path):
        raise ValueError("Reference validation is not passing or binds another file")

    development_dir = Path(args.development_output).expanduser().resolve()
    development_report_path = development_dir / "DEVELOPMENT_REPORT.json"
    development_report = _read_json(development_report_path, "Development report")
    if {
        "status": development_report.get("status"),
        "event_count": development_report.get("event_count"),
        "candidate_count": development_report.get("candidate_count"),
        "attempt_count": development_report.get("attempt_count"),
        "reviewable_count": development_report.get("reviewable_count"),
    } != {
        "status": "hold_for_blinded_manual_review",
        "event_count": 48,
        "candidate_count": 9,
        "attempt_count": 432,
        "reviewable_count": 240,
    }:
        raise ValueError("Development report counts or state drifted")
    if development_report.get("report_sha256") != canonical_json_sha256(
        development_report, exclude=("report_sha256",)
    ):
        raise ValueError("Development report audit SHA256 drifted")

    development_validation_path = Path(args.development_validation).expanduser().resolve()
    files_manifest_path = Path(args.development_files_manifest).expanduser().resolve()
    development_validation = _read_json(development_validation_path, "Development validation")
    if development_validation.get("status") != "pass":
        raise ValueError("Development strict validation is not passing")

    review_path = Path(args.review_decisions).expanduser().resolve()
    review_validation_path = Path(args.review_validation).expanduser().resolve()
    review_validation = _read_json(review_validation_path, "review validation")
    if review_validation.get("status") != "pass":
        raise ValueError("Development blinded review validation is not passing")
    if review_validation.get("decision_file_sha256") != sha256_file(review_path):
        raise ValueError("review validation binds another decision CSV")
    if review_validation.get("development_validation_sha256") != sha256_file(
        development_validation_path
    ) or review_validation.get("development_files_manifest_sha256") != sha256_file(
        files_manifest_path
    ):
        raise ValueError("review validation Development evidence drifted")
    if review_validation.get("private_blinding_map_accessed_before_decision_lock") is not False:
        raise ValueError("blinded review was not locked before unblinding")

    with review_path.open("r", encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    if len(decisions) != 240 or {row["review_decision"] for row in decisions} != {
        "accept",
        "reject",
    }:
        raise ValueError("Development decisions do not contain the frozen 240 binary rows")
    decisions_by_code = {row["blind_code"]: row for row in decisions}
    if len(decisions_by_code) != len(decisions):
        raise ValueError("Development decisions contain duplicate blind codes")

    private_map_path = development_dir / "PRIVATE_BLINDING_MAP.json"
    if sha256_file(private_map_path) != development_report["private_blinding_map_sha256"]:
        raise ValueError("private blinding map drifted")
    private_map = _read_json(private_map_path, "private blinding map")
    identity_by_code = {entry["blind_code"]: entry for entry in private_map["entries"]}
    if len(identity_by_code) != 432:
        raise ValueError("private blinding map does not cover all attempts")

    measurements_path = development_dir / development_report["measurements_file"]
    if sha256_file(measurements_path) != development_report["measurements_sha256"]:
        raise ValueError("Development measurement file drifted")
    rows = _load_rows(measurements_path)
    if len(rows) != 432:
        raise ValueError("Development measurements do not contain 432 rows")
    for row in rows:
        identity = identity_by_code.get(row["blind_code"])
        if identity is None or identity["event_id"] != row["event_id"] or identity[
            "candidate_id"
        ] != row["candidate_id"]:
            raise ValueError("private blinding map and measurement rows disagree")

    candidate_summary: dict[str, dict[str, int]] = {}
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        reviewable = [row for row in candidate_rows if row["blind_code"] in decisions_by_code]
        candidate_summary[candidate_id] = {
            "attempted": len(candidate_rows),
            "measured": sum(row["status"] == "measured" for row in candidate_rows),
            "reviewable": len(reviewable),
            "accepted": sum(
                decisions_by_code[row["blind_code"]]["review_decision"] == "accept"
                for row in reviewable
            ),
            "rejected": sum(
                decisions_by_code[row["blind_code"]]["review_decision"] == "reject"
                for row in reviewable
            ),
        }
    expected_tie = {
        "A_label_only",
        "B_halo_1p5mm",
        "B_halo_2mm",
        "B_halo_3mm",
        "B_halo_4mm",
    }
    if any(
        candidate_summary[candidate] != {
            "attempted": 48,
            "measured": 48,
            "reviewable": 48,
            "accepted": 46,
            "rejected": 2,
        }
        for candidate in expected_tie
    ):
        raise ValueError("A/B Development tie or fixed denominator drifted")
    c_candidates = set(candidate_summary) - expected_tie
    if any(
        candidate_summary[candidate] != {
            "attempted": 48,
            "measured": 0,
            "reviewable": 0,
            "accepted": 0,
            "rejected": 0,
        }
        for candidate in c_candidates
    ):
        raise ValueError("C harmonization failure accounting drifted")

    selection_audit = {
        "schema_version": 1,
        "status": "pass",
        "stage": "development_candidate_selection_after_locked_blinded_review",
        "selection_rule": [
            "zero_manual_reject_among_automatically_released_holdout_candidates",
            "frozen_generation_and_effective_rate_floors",
            "residual_cross_modal_and_structure_contracts",
            "least_new_code_and_highest_throughput_on_tie",
        ],
        "candidate_summary": candidate_summary,
        "quality_tie": sorted(expected_tie),
        "selected_candidate_id": "A_label_only",
        "selected_boundary_policy": "label_only_qc_v1",
        "selected_halo_radius_mm": 0.0,
        "selection_reason": "A and all B radii tied 46/48; A is the pre-registered simpler and faster tie-break winner; all C radii failed harmonization 48/48",
        "review_decisions_sha256": sha256_file(review_path),
        "review_validation_sha256": sha256_file(review_validation_path),
        "private_blinding_map_sha256": sha256_file(private_map_path),
    }
    selection_audit["selection_audit_sha256"] = canonical_json_sha256(
        selection_audit, exclude=("selection_audit_sha256",)
    )
    selection_bytes = _encoded(selection_audit)

    index_path = Path(args.measurement_config_index).expanduser().resolve()
    index = _read_json(index_path, "measurement config index")
    if index.get("status") != "measurement_only_not_gate_eligible":
        raise ValueError("measurement config index role drifted")
    a_index = [value for value in index["candidates"] if value["candidate_id"] == "A_label_only"]
    if len(a_index) != 1:
        raise ValueError("measurement config index lacks exactly one A candidate")
    measurement_calibration_path = index_path.parent / a_index[0]["calibration_file"]
    if sha256_file(measurement_calibration_path) != a_index[0]["calibration_sha256"]:
        raise ValueError("A measurement calibration drifted")
    measurement_calibration = _read_json(
        measurement_calibration_path, "A measurement calibration"
    )

    records = {record.component_id: record for record in manifest.records}
    a_rows = [row for row in rows if row["candidate_id"] == "A_label_only"]
    accepted_rows = [
        row
        for row in a_rows
        if decisions_by_code[row["blind_code"]]["review_decision"] == "accept"
    ]
    rejected_rows = [row for row in a_rows if row not in accepted_rows]
    raw_max = {modality: 0.0 for modality in S2_MODALITIES}
    for row in accepted_rows:
        artifact_path = development_dir / row["artifact_path"]
        if sha256_file(artifact_path) != row["artifact_sha256"]:
            raise ValueError(f"Development artifact drifted: {row['blind_code']}")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            original = np.asarray(artifact["original"], dtype=np.float32)
            raw = np.asarray(artifact["raw_generation"], dtype=np.float32)
            label = np.asarray(artifact["inserted_label"]) != 0
            ring = np.asarray(artifact["reference_ring"], dtype=bool)
            for channel, modality in enumerate(S2_MODALITIES):
                scale = _mad(original[channel][ring].astype(np.float64))
                if scale <= 0:
                    raise ValueError(f"{row['blind_code']}: nonpositive local MAD")
                value = float(np.max(np.abs(raw[channel][label] - original[channel][label]) / scale))
                raw_max[modality] = max(raw_max[modality], value)

    source_bindings = {
        "reference_validation_sha256": sha256_file(reference_validation_path),
        "measurement_config_index_sha256": sha256_file(index_path),
        "development_report_sha256": sha256_file(development_report_path),
        "development_validation_sha256": sha256_file(development_validation_path),
        "development_files_manifest_sha256": sha256_file(files_manifest_path),
        "development_review_decisions_sha256": sha256_file(review_path),
        "development_review_validation_sha256": sha256_file(review_validation_path),
        "candidate_selection_file_sha256": _bytes_sha256(selection_bytes),
    }
    calibration_payload = derive_calibration(
        measurement_calibration=measurement_calibration,
        reference=reference,
        accepted_metadata=[row["metadata"] for row in accepted_rows],
        accepted_raw_max_abs_z=raw_max,
        source_bindings=source_bindings,
    )
    calibration_bytes = _encoded(calibration_payload)
    calibration_sha256 = _bytes_sha256(calibration_bytes)
    processor = FixV2CandidateProcessor(
        FixV2Calibration(
            path=Path("<in-memory-derived-calibration>"),
            sha256=calibration_sha256,
            payload=calibration_payload,
        )
    )
    replay_results: list[dict[str, Any]] = []
    for row in a_rows:
        artifact_path = development_dir / row["artifact_path"]
        record = records[row["donor_component_id"]]
        with np.load(artifact_path, allow_pickle=False) as artifact:
            raw = np.asarray(artifact["raw_generation"], dtype=np.float32)
            result = processor.process(
                original_image=np.asarray(artifact["original"], dtype=np.float32),
                original_segmentation=np.asarray(artifact["original_segmentation"], dtype=np.int16),
                label_cube=np.asarray(artifact["inserted_label"], dtype=np.int16),
                valid_mask=np.asarray(artifact["valid_mask"], dtype=bool),
                spacing_mm=record.spacing_mm,
                core_volume_mm3=record.core_volume_mm3,
                seed=int(row["event_seed"]),
                backend=_ReplayBackend(raw),
            )
        decision = decisions_by_code[row["blind_code"]]["review_decision"]
        replay_results.append(
            {
                "blind_code": row["blind_code"],
                "event_id": row["event_id"],
                "review_decision": decision,
                "strict_result": "pass" if result.reason is None else "reject",
                "strict_reason": result.reason,
            }
        )
    false_rejects = [value for value in replay_results if value["review_decision"] == "accept" and value["strict_result"] != "pass"]
    missed_rejects = [value for value in replay_results if value["review_decision"] == "reject" and value["strict_result"] != "reject"]
    if false_rejects or missed_rejects:
        raise RuntimeError(
            f"derived thresholds do not reproduce locked Development review: false_rejects={false_rejects}, missed_rejects={missed_rejects}"
        )

    route_payload = make_fix_v2_route_a_config(
        manifest,
        boundary_policy="label_only_qc_v1",
        calibration_sha256=calibration_sha256,
        seed=args.seed,
    )
    route_bytes = _encoded(route_payload)
    derivation_report = {
        "schema_version": 1,
        "status": "pass",
        "selected_candidate_id": "A_label_only",
        "accepted_development_count": len(accepted_rows),
        "rejected_development_count": len(rejected_rows),
        "strict_replay_pass_count": sum(value["strict_result"] == "pass" for value in replay_results),
        "strict_replay_reject_count": sum(value["strict_result"] == "reject" for value in replay_results),
        "strict_replay_reasons": {
            reason: sum(value["strict_reason"] == reason for value in replay_results)
            for reason in sorted({value["strict_reason"] for value in replay_results if value["strict_reason"]})
        },
        "selection_audit_sha256": _bytes_sha256(selection_bytes),
        "calibration_candidate_sha256": calibration_sha256,
        "qc_route_config_sha256": _bytes_sha256(route_bytes),
        "threshold_payload_sha256": calibration_payload["threshold_derivation"]["threshold_payload_sha256"],
        "source_bindings": source_bindings,
    }
    derivation_report["derivation_audit_sha256"] = canonical_json_sha256(
        derivation_report, exclude=("derivation_audit_sha256",)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_exclusive(output_dir / "CANDIDATE_SELECTION_AUDIT.json", selection_bytes)
    _write_exclusive(output_dir / "FIX_V2_CALIBRATION_CANDIDATE_BEFORE_QC.json", calibration_bytes)
    _write_exclusive(output_dir / "FIX_V2_QC_HOLDOUT_ROUTE_CONFIG.json", route_bytes)
    _write_exclusive(output_dir / "SELECTION_AND_DERIVATION_VALIDATION.json", _encoded(derivation_report))
    output_dir.chmod(0o555)
    print(json.dumps(derivation_report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
