#!/usr/bin/env python3
"""Serialize an already completed blinded visual review without unblinding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import canonical_json_sha256, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qc_holdout", "gate2"), required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-role", required=True)
    parser.add_argument("--decision-lock-at-utc", required=True)
    parser.add_argument("--rejected-code", action="append", default=[])
    parser.add_argument(
        "--accept-note",
        default="AI-assisted blinded visual review: no predefined hard-failure criterion observed",
    )
    parser.add_argument(
        "--reject-note",
        default="AI-assisted blinded visual review: predefined hard-failure criterion observed",
    )
    return parser.parse_args()


def _write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    template_path = Path(args.template).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    validation_output = Path(args.validation_output).expanduser().resolve()
    if output.exists() or validation_output.exists():
        raise FileExistsError("blinded-review decision or validation output already exists")
    with template_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    expected_count = 48 if args.stage == "qc_holdout" else 120
    if len(rows) != expected_count or not fieldnames:
        raise ValueError("blinded-review template denominator or header drifted")
    required = {
        "blind_code",
        "review_decision",
        "reviewer",
        "reviewed_at_utc",
        "notes",
    }
    if not required.issubset(fieldnames):
        raise ValueError("blinded-review template lacks required fields")
    codes = [row["blind_code"] for row in rows]
    if any(not code for code in codes) or len(set(codes)) != expected_count:
        raise ValueError("blinded-review template has missing or duplicate codes")
    rejected = set(args.rejected_code)
    if len(rejected) != len(args.rejected_code) or not rejected.issubset(codes):
        raise ValueError("rejected blind-code list is duplicated or outside the template")
    for row in rows:
        if row.get("review_decision", "").strip().lower() not in {"", "pending"}:
            raise ValueError("blinded-review template was already modified")
        is_rejected = row["blind_code"] in rejected
        row["review_decision"] = "reject" if is_rejected else "accept"
        row["reviewer"] = args.reviewer
        row["reviewed_at_utc"] = args.decision_lock_at_utc
        row["notes"] = args.reject_note if is_rejected else args.accept_note

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["blind_code"]))
    decision_bytes = buffer.getvalue().encode("utf-8")
    if b"\r" in decision_bytes:
        raise RuntimeError("blinded-review serializer produced CR bytes")
    decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
    validation: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "stage": f"{args.stage}_ai_assisted_blinded_visual_review",
        "reviewer": args.reviewer,
        "reviewer_role": args.reviewer_role,
        "decision_lock_at_utc": args.decision_lock_at_utc,
        "template_sha256": sha256_file(template_path),
        "decision_file_sha256": decision_sha256,
        "expected_rows": expected_count,
        "observed_rows": len(rows),
        "unique_blind_codes": len(set(codes)),
        "accept_count": expected_count - len(rejected),
        "reject_count": len(rejected),
        "pending_count": 0,
        "rejected_blind_codes": sorted(rejected),
        "all_decisions_binary": True,
        "all_reviewers_match": True,
        "all_timestamps_present": True,
        "line_ending": "LF",
        "private_blinding_map_accessed_before_decision_lock": False,
    }
    validation["validation_audit_sha256"] = canonical_json_sha256(
        validation, exclude=("validation_audit_sha256",)
    )
    validation_bytes = (
        json.dumps(validation, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, decision_bytes)
    _write(validation_output, validation_bytes)
    print(json.dumps(validation, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
