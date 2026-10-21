#!/usr/bin/env python3
"""Delete only the previous imported source, explicitly requested by the user."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "PREVIOUS_SOURCE_DELETION_RECEIPT.json"
PREVIOUS = HERE / "previous_source_20261009"


def manifest(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise SystemExit("Refuse deletion of a source tree containing symlinks")
    return {p.relative_to(root).as_posix(): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                               bytes=p.stat().st_size)
            for p in root.rglob("*") if p.is_file()}


def main():
    if RECEIPT.exists():
        raise SystemExit("Deletion receipt already exists; refuse repeated operation")
    replacement = json.loads((HERE / "REPLACEMENT_RECEIPT.json").read_text())
    active = Path(replacement["active_root"])
    if not PREVIOUS.is_dir() or PREVIOUS.resolve() != Path(replacement["preserved_previous_root"]):
        raise SystemExit("Previous source path mismatch")
    previous = manifest(PREVIOUS)
    current = manifest(active)
    if previous != replacement["previous"] or current != replacement["revision"]:
        raise SystemExit("Source changed since replacement; refuse deletion")
    receipt = dict(artifact_status="experimental_unvalidated", operator_approved=False,
                   formal_gate_status="not_run_not_passed",
                   user_instruction="上一版删除，只保留这一版",
                   deleted_path=str(PREVIOUS), files=len(previous),
                   bytes=sum(p["bytes"] for p in previous.values()), deleted_file_manifest=previous,
                   active_path=str(active), historical_audits_retained=True,
                   other_diffusion_models_untouched=True, status="deletion_authorized")
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    shutil.rmtree(PREVIOUS)
    receipt.update(status="deleted", deleted_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   previous_path_absent=not PREVIOUS.exists(), active_sha_unchanged=manifest(active) == current)
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "deleted_file_manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
