#!/usr/bin/env python3
"""Validate two isolated 96-event Fix-v2 replays down to every array payload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-a", required=True)
    parser.add_argument("--replay-b", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(
        left, right
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Gate-1B validation output already exists: {output}")
    roots = [Path(value).expanduser().resolve() for value in (args.replay_a, args.replay_b)]
    reports = [_json(root / "FIXED_EVENT_REPORT.json") for root in roots]
    for index, report in enumerate(reports):
        if report.get("report_sha256") != canonical_json_sha256(
            report, exclude=("report_sha256",)
        ):
            raise ValueError(f"replay {index} report audit SHA256 drifted")
        if (
            report.get("status") != "hold_for_blinded_manual_review"
            or report.get("stage") != "gate1b_replay"
            or int(report.get("attempt_count", -1)) != 96
            or int(report.get("reviewable_count", -1)) != 96
            or report.get("violations")
        ):
            raise ValueError(f"replay {index} did not complete the fixed 96 events")
    identity_keys = (
        "component_manifest_sha256",
        "route_config_sha256",
        "strict_calibration_sha256",
        "measurement_calibration_sha256",
        "valid_mask_manifest_sha256",
        "event_manifest_sha256",
        "g1_checkpoint_selection_sha256",
        "g2_parent_gate_sha256",
        "g1_runtime_code",
        "runner_sha256",
        "state_counts",
        "reason_counts",
        "committed_count",
        "rejected_count",
        "generation_pass_rate",
        "effective_aug_rate",
        "per_volume_bin",
    )
    for key in identity_keys:
        if reports[0].get(key) != reports[1].get(key):
            raise ValueError(f"Gate-1B replay report differs at {key}")
    result_paths = [root / report["results_file"] for root, report in zip(roots, reports)]
    for path, report in zip(result_paths, reports):
        if sha256_file(path) != report["results_sha256"]:
            raise ValueError(f"Gate-1B result file drifted: {path}")
    rows = [_rows(path) for path in result_paths]
    if len(rows[0]) != 96 or len(rows[1]) != 96:
        raise ValueError("Gate-1B replay result denominator drifted")
    by_event = [
        {str(row["event_id"]): row for row in replay_rows} for replay_rows in rows
    ]
    if len(by_event[0]) != 96 or set(by_event[0]) != set(by_event[1]):
        raise ValueError("Gate-1B replay event identities differ")
    compared_arrays = 0
    compared_events = 0
    for event_id in sorted(by_event[0]):
        left_row = by_event[0][event_id]
        right_row = by_event[1][event_id]
        for key in (
            "event_index",
            "smoke_id",
            "event_id",
            "event_seed",
            "target_case_id",
            "target_patient_group",
            "donor_component_id",
            "donor_patient_group",
            "core_volume_bin",
            "blind_code",
            "transaction_state",
            "transaction_reason",
            "violations",
        ):
            if left_row.get(key) != right_row.get(key):
                raise ValueError(f"Gate-1B event {event_id} differs at {key}")
        artifacts = [root / row["artifact_path"] for root, row in zip(roots, (left_row, right_row))]
        for artifact, row in zip(artifacts, (left_row, right_row)):
            if sha256_file(artifact) != row["artifact_sha256"]:
                raise ValueError(f"Gate-1B artifact SHA drifted: {artifact}")
        with np.load(artifacts[0], allow_pickle=False) as left, np.load(
            artifacts[1], allow_pickle=False
        ) as right:
            if set(left.files) != set(right.files):
                raise ValueError(f"Gate-1B artifact keys differ: {event_id}")
            for key in sorted(left.files):
                if not _array_equal(left[key], right[key]):
                    raise ValueError(f"Gate-1B array differs: {event_id}:{key}")
                compared_arrays += 1
        compared_events += 1
    audit_paths = [root / "transaction_events.jsonl" for root in roots]
    if audit_paths[0].read_bytes() != audit_paths[1].read_bytes():
        raise ValueError("Gate-1B transaction audit JSONL bytes differ")
    validation = {
        "schema_version": 1,
        "status": "pass",
        "event_count": compared_events,
        "array_payload_count": compared_arrays,
        "transaction_audit_bytes_identical": True,
        "replay_a_report_sha256": sha256_file(roots[0] / "FIXED_EVENT_REPORT.json"),
        "replay_b_report_sha256": sha256_file(roots[1] / "FIXED_EVENT_REPORT.json"),
        "event_manifest_sha256": reports[0]["event_manifest_sha256"],
        "route_config_sha256": reports[0]["route_config_sha256"],
        "strict_calibration_sha256": reports[0]["strict_calibration_sha256"],
        "state_counts": reports[0]["state_counts"],
        "reason_counts": reports[0]["reason_counts"],
    }
    validation["validation_audit_sha256"] = canonical_json_sha256(
        validation, exclude=("validation_audit_sha256",)
    )
    encoded = (
        json.dumps(validation, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(validation, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
