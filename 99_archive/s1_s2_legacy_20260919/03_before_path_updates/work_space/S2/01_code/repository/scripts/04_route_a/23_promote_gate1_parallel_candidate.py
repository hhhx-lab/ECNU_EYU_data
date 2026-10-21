#!/usr/bin/env python3
"""Promote a verified parallel Gate 1 result while preserving serial evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", required=True)
    parser.add_argument("--equivalence-report", required=True)
    parser.add_argument("--expected-events", type=int, default=100000)
    parser.add_argument("--require-stopped-pid", type=int, action="append", default=[])
    return parser.parse_args()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                return count
            count += chunk.count(b"\n")


def assert_prefix_identical(*, prefix_path: Path, complete_path: Path) -> None:
    with prefix_path.open("rb") as prefix, complete_path.open("rb") as complete:
        for index, expected in enumerate(prefix, start=1):
            observed = complete.readline()
            if observed != expected:
                raise RuntimeError(f"parallel Gate 1 differs from serial evidence at event {index}")


def assert_pids_stopped(pids: Iterable[int]) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify stopped PID {pid}") from exc
        raise RuntimeError(f"refusing Gate 1 promotion while PID {pid} is still alive")


def promote_gate1_candidate(
    *,
    route_root: str | Path,
    equivalence_report_path: str | Path,
    expected_events: int,
    stopped_pids: Iterable[int],
) -> dict:
    root = Path(route_root).expanduser().resolve()
    serial_dir = root / "gate1"
    candidate_dir = root / "gate1_parallel_candidate"
    archive_dir = root / "gate1_serial_partial"
    migration_dir = root / "gate1_parallel_migration"
    audit_path = migration_dir / "PROMOTION_AUDIT.json"
    gate2_manifest = root / "gate2_smoke_manifest.json"
    prepare_marker = root / "PREPARE_COMPLETE.ok"
    if not serial_dir.is_dir() or not candidate_dir.is_dir():
        raise FileNotFoundError("serial or parallel Gate 1 directory is missing")
    for forbidden in (archive_dir, audit_path, gate2_manifest, prepare_marker):
        if forbidden.exists():
            raise FileExistsError(f"promotion refuses to overwrite existing evidence: {forbidden}")
    if (candidate_dir / ".gate1_shards").exists():
        raise RuntimeError("parallel Gate 1 candidate still has active shard state")
    assert_pids_stopped(stopped_pids)

    serial_events = serial_dir / "gate1_events.jsonl"
    candidate_events = candidate_dir / "gate1_events.jsonl"
    candidate_report_path = candidate_dir / "gate1_report.json"
    for required in (serial_events, candidate_events, candidate_report_path):
        if not required.is_file():
            raise FileNotFoundError(f"required Gate 1 evidence is missing: {required}")
    serial_count = line_count(serial_events)
    candidate_count = line_count(candidate_events)
    if serial_count < 1 or serial_count >= expected_events:
        raise RuntimeError(f"unexpected serial partial event count: {serial_count}")
    if candidate_count != expected_events:
        raise RuntimeError(
            f"parallel Gate 1 has {candidate_count} events, expected {expected_events}"
        )
    assert_prefix_identical(prefix_path=serial_events, complete_path=candidate_events)

    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (
        candidate_report.get("status") != "pass"
        or int(candidate_report.get("event_count", 0)) != expected_events
        or candidate_report.get("violations") != []
    ):
        raise RuntimeError("parallel Gate 1 candidate report is not an accepted full result")
    component_manifest_path = root / "component_pool" / "component_manifest.json"
    route_config_path = root / "route_a_config.json"
    valid_mask_manifest_path = root / "valid_masks" / "valid_mask_manifest.json"
    component_manifest = json.loads(component_manifest_path.read_text(encoding="utf-8"))
    expected_bindings = {
        "component_manifest_sha256": component_manifest.get("manifest_sha256"),
        "route_config_sha256": sha256_file(route_config_path),
        "valid_mask_manifest_sha256": sha256_file(valid_mask_manifest_path),
    }
    for key, expected in expected_bindings.items():
        if candidate_report.get(key) != expected:
            raise RuntimeError(f"parallel Gate 1 candidate does not bind {key}")

    equivalence_path = Path(equivalence_report_path).expanduser().resolve()
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if (
        equivalence.get("status") != "pass"
        or equivalence.get("serial_parallel_jsonl_byte_identical") is not True
        or equivalence.get("serial_parallel_report_byte_identical") is not True
        or equivalence.get("legacy_reference_prefix_byte_identical") is not True
    ):
        raise RuntimeError("Gate 1 serial/parallel equivalence evidence is not passing")

    serial_events_sha256 = sha256_file(serial_events)
    candidate_events_sha256 = sha256_file(candidate_events)
    candidate_report_sha256 = sha256_file(candidate_report_path)
    serial_dir.rename(archive_dir)
    candidate_dir.rename(serial_dir)

    payload = {
        "schema_version": 1,
        "status": "pass",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "expected_events": expected_events,
        "serial_partial": {
            "path": str(archive_dir),
            "event_count": serial_count,
            "events_sha256": serial_events_sha256,
        },
        "parallel_result": {
            "path": str(serial_dir),
            "event_count": candidate_count,
            "events_sha256": candidate_events_sha256,
            "report_sha256": candidate_report_sha256,
        },
        "serial_prefix_matches_parallel": True,
        "equivalence_report": str(equivalence_path),
        "equivalence_report_sha256": sha256_file(equivalence_path),
        "bindings": expected_bindings,
        "legacy_runner_sha256": sha256_file(migration_dir / "legacy_source" / "15_run_met_aug_gate1.py"),
        "legacy_core_sha256": sha256_file(migration_dir / "legacy_source" / "met_aug_core.py"),
        "parallel_runner_sha256": sha256_file(REPOSITORY_ROOT / "scripts" / "15_run_met_aug_gate1.py"),
        "parallel_module_sha256": sha256_file(REPOSITORY_ROOT / "custom_nnunet" / "met_aug_gate1.py"),
        "current_core_sha256": sha256_file(REPOSITORY_ROOT / "custom_nnunet" / "met_aug_core.py"),
    }
    payload["promotion_sha256"] = canonical_json_sha256(
        payload, exclude=("promotion_sha256",)
    )
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = promote_gate1_candidate(
        route_root=args.route_root,
        equivalence_report_path=args.equivalence_report,
        expected_events=args.expected_events,
        stopped_pids=args.require_stopped_pid,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
