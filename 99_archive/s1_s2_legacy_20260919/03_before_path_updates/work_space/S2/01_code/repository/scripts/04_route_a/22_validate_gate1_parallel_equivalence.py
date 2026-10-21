#!/usr/bin/env python3
"""Validate Gate 1 fork-parallel output against serial and legacy-prefix evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import sha256_file
from custom_nnunet.met_aug_gate1 import run_gate1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-jsonl")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target-seed", type=int, default=20260725)
    return parser.parse_args()


def read_prefix(path: Path, line_count: int) -> bytes:
    lines: list[bytes] = []
    with path.open("rb") as handle:
        for _ in range(line_count):
            line = handle.readline()
            if not line:
                raise RuntimeError(f"reference JSONL has fewer than {line_count} events: {path}")
            lines.append(line)
    return b"".join(lines)


def run_one(args: argparse.Namespace, *, output_dir: Path, workers: int) -> float:
    started = time.monotonic()
    run_gate1(
        component_manifest_path=args.component_manifest,
        route_config_path=args.route_config,
        valid_mask_manifest_path=args.valid_mask_manifest,
        output_dir=output_dir,
        events=args.events,
        target_seed=args.target_seed,
        workers=workers,
        minimum_events=1,
        enforce_acceptance=False,
    )
    return time.monotonic() - started


def main() -> None:
    args = parse_args()
    if args.events < 1:
        raise ValueError("equivalence validation requires at least one event")
    if args.workers < 2:
        raise ValueError("equivalence validation requires at least two parallel workers")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"equivalence output is immutable and already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    serial_dir = output_dir / "serial"
    parallel_dir = output_dir / "parallel"

    serial_seconds = run_one(args, output_dir=serial_dir, workers=1)
    parallel_seconds = run_one(args, output_dir=parallel_dir, workers=args.workers)
    serial_events = serial_dir / "gate1_events.jsonl"
    parallel_events = parallel_dir / "gate1_events.jsonl"
    serial_report = serial_dir / "gate1_report.json"
    parallel_report = parallel_dir / "gate1_report.json"
    if serial_events.read_bytes() != parallel_events.read_bytes():
        raise RuntimeError("parallel Gate 1 JSONL is not byte-identical to serial output")
    if serial_report.read_bytes() != parallel_report.read_bytes():
        raise RuntimeError("parallel Gate 1 report is not byte-identical to serial output")

    reference_matches = None
    reference_path = None
    if args.reference_jsonl:
        reference_path = Path(args.reference_jsonl).expanduser().resolve()
        reference_matches = read_prefix(reference_path, args.events) == serial_events.read_bytes()
        if not reference_matches:
            raise RuntimeError("new serial Gate 1 output differs from the legacy serial JSONL prefix")

    payload = {
        "schema_version": 1,
        "status": "pass",
        "events": args.events,
        "workers": args.workers,
        "target_seed": args.target_seed,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": serial_seconds / parallel_seconds if parallel_seconds else None,
        "events_sha256": sha256_file(serial_events),
        "report_sha256": sha256_file(serial_report),
        "serial_parallel_jsonl_byte_identical": True,
        "serial_parallel_report_byte_identical": True,
        "legacy_reference_jsonl": str(reference_path) if reference_path else None,
        "legacy_reference_prefix_byte_identical": reference_matches,
    }
    report_path = output_dir / "parallel_equivalence_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
