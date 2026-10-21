#!/usr/bin/env python3
"""Freeze and snapshot Fix-v3 when checkpoint current_epoch reaches 125."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
from typing import Any


ARTIFACT_STATUS = "experimental_unvalidated"
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
EVENT_STRUCT = struct.Struct("iIII")


class StopControllerError(RuntimeError):
    """Raised when the epoch-125 stop contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StopControllerError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def copy_file_exclusive(source: Path, destination: Path) -> str:
    require(source.is_file() and source.stat().st_size > 0, f"missing source: {source}")
    require(not destination.exists(), f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    require(not temporary.exists(), f"stale temporary destination: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(destination)


def read_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def process_record(pid: int) -> dict[str, Any]:
    process_root = Path(f"/proc/{pid}")
    require(process_root.is_dir(), f"process is not alive: {pid}")
    status: dict[str, str] = {}
    for line in (process_root / "status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    command = (process_root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
        "utf-8", "replace"
    ).strip()
    return {
        "pid": pid,
        "ppid": int(status.get("PPid", "-1")),
        "state": status.get("State", "unknown"),
        "command": command,
    }


def validate_processes(
    trainer_pid: int,
    wrapper_pid: int,
    *,
    expected_boot_id: str,
) -> dict[str, Any]:
    boot_id = read_boot_id()
    require(boot_id == expected_boot_id, f"boot ID changed: {boot_id}")
    trainer = process_record(trainer_pid)
    wrapper = process_record(wrapper_pid)
    require(trainer["ppid"] == wrapper_pid, "trainer/wrapper parent relationship changed")
    require("nnUNetv2_train" in trainer["command"], "trainer command identity changed")
    require("train.sh" in wrapper["command"], "wrapper command identity changed")
    return {"boot_id": boot_id, "trainer": trainer, "wrapper": wrapper}


def iter_tensors(value: Any) -> Iterable[Any]:
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def validate_tensor_tree(value: Any, name: str, *, require_nonempty: bool = True) -> int:
    import torch

    count = 0
    for tensor in iter_tensors(value):
        count += 1
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all().item()), f"non-finite tensor in {name}")
    if require_nonempty:
        require(count > 0, f"no tensors found in {name}")
    return count


def validate_checkpoint(path: Path, target_epoch: int) -> dict[str, Any]:
    import torch

    require(path.is_file() and path.stat().st_size > 0, f"checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "checkpoint root is not a mapping")
    require(checkpoint.get("current_epoch") == target_epoch, "checkpoint epoch is not 125")
    network = checkpoint.get("network_weights")
    optimizer = checkpoint.get("optimizer_state")
    scaler = checkpoint.get("grad_scaler_state")
    require(isinstance(network, Mapping), "network_weights is incomplete")
    require(isinstance(optimizer, Mapping), "optimizer_state is incomplete")
    require(isinstance(scaler, Mapping), "grad_scaler_state is incomplete")
    result = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "current_epoch": int(checkpoint["current_epoch"]),
        "trainer_name": str(checkpoint.get("trainer_name", "")),
        "network_tensor_count": validate_tensor_tree(network, "network_weights"),
        "optimizer_tensor_count": validate_tensor_tree(optimizer, "optimizer_state"),
        "grad_scaler_tensor_count": validate_tensor_tree(
            scaler, "grad_scaler_state", require_nonempty=False
        ),
    }
    del checkpoint
    return result


def read_checkpoint_epoch(path: Path) -> int:
    import torch

    require(path.is_file() and path.stat().st_size > 0, f"checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "checkpoint root is not a mapping")
    value = checkpoint.get("current_epoch")
    require(isinstance(value, int), "checkpoint current_epoch is invalid")
    return value


def validate_audit(path: Path, target_epoch: int, patches_per_epoch: int) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"audit is missing: {path}")
    counts: Counter[int] = Counter()
    patches: dict[int, set[int]] = defaultdict(set)
    event_ids: set[str] = set()
    malformed = 0
    duplicate_event_ids = 0
    total_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_rows += 1
            try:
                event = json.loads(line)
                require(isinstance(event, dict), f"audit row {line_number} is not an object")
                epoch = event.get("epoch")
                patch = event.get("patch_index")
                event_id = event.get("event_id")
                require(isinstance(epoch, int), f"invalid epoch at audit row {line_number}")
                require(isinstance(patch, int), f"invalid patch at audit row {line_number}")
                require(isinstance(event_id, str) and event_id, f"invalid event ID at row {line_number}")
            except (json.JSONDecodeError, StopControllerError):
                malformed += 1
                continue
            counts[epoch] += 1
            patches[epoch].add(patch)
            if event_id in event_ids:
                duplicate_event_ids += 1
            event_ids.add(event_id)

    require(malformed == 0, f"malformed audit rows: {malformed}")
    require(duplicate_event_ids == 0, f"duplicate audit event IDs: {duplicate_event_ids}")
    expected_patches = set(range(patches_per_epoch))
    for epoch in range(target_epoch):
        require(counts[epoch] == patches_per_epoch, f"epoch {epoch} audit count is {counts[epoch]}")
        require(patches[epoch] == expected_patches, f"epoch {epoch} patch coverage is invalid")
    extra_epochs = sorted(epoch for epoch in counts if epoch > target_epoch)
    require(not extra_epochs, f"audit already advanced beyond epoch {target_epoch}: {extra_epochs}")
    partial_patches = patches.get(target_epoch, set())
    if partial_patches:
        require(
            partial_patches == set(range(max(partial_patches) + 1)),
            f"partial epoch {target_epoch} patches are not contiguous",
        )
    closed_rows = target_epoch * patches_per_epoch
    require(sum(counts[epoch] for epoch in range(target_epoch)) == closed_rows, "closed audit mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "total_rows": total_rows,
        "closed_rows": closed_rows,
        "completed_epoch_count": target_epoch,
        "completed_epoch_range": [0, target_epoch - 1],
        "partial_epoch": target_epoch if counts[target_epoch] else None,
        "partial_epoch_rows": counts[target_epoch],
        "malformed_rows": malformed,
        "duplicate_event_ids": duplicate_event_ids,
        "unique_event_ids": len(event_ids),
    }


def stop_and_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    process_before = validate_processes(
        args.trainer_pid,
        args.wrapper_pid,
        expected_boot_id=args.expected_boot_id,
    )
    os.kill(args.trainer_pid, signal.SIGSTOP)
    checkpoint = validate_checkpoint(args.checkpoint, args.target_epoch)
    audit = validate_audit(args.audit, args.target_epoch, args.patches_per_epoch)
    process_stopped = process_record(args.trainer_pid)
    require(process_stopped["state"].startswith(("T", "t")), "trainer did not enter stopped state")

    require(not args.snapshot_root.exists(), f"snapshot root already exists: {args.snapshot_root}")
    args.snapshot_root.mkdir(parents=False, mode=0o755)
    checkpoint_copy = args.snapshot_root / "checkpoint" / "checkpoint_epoch125.pth"
    audit_copy = args.snapshot_root / "evidence" / "met_aug_events_at_stop.jsonl"
    log_copy = args.snapshot_root / "evidence" / "training_log_at_stop.log"
    copied = {
        "checkpoint/checkpoint_epoch125.pth": copy_file_exclusive(args.checkpoint, checkpoint_copy),
        "evidence/met_aug_events_at_stop.jsonl": copy_file_exclusive(args.audit, audit_copy),
        "evidence/training_log_at_stop.log": copy_file_exclusive(args.training_log, log_copy),
    }
    require(copied["checkpoint/checkpoint_epoch125.pth"] == checkpoint["sha256"], "checkpoint copy drift")
    require(copied["evidence/met_aug_events_at_stop.jsonl"] == audit["sha256"], "audit copy drift")

    decision = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": ARTIFACT_STATUS,
        "created_at_utc": utc_now(),
        "user_decision": "stop_after_125_completed_epochs_then_run_fixed_103",
        "early_stop_complete": True,
        "training_complete": False,
        "completed_epochs": 125,
        "completed_epoch_indices": [0, 124],
        "checkpoint": checkpoint,
        "audit": audit,
        "process_before_stop": process_before,
        "process_after_sigstop": process_stopped,
        "copied_artifacts_sha256": copied,
        "termination": {
            "trainer_pid": args.trainer_pid,
            "signals": ["SIGSTOP", "SIGTERM", "SIGCONT"],
            "exit_confirmation_required_by_next_snapshot": True,
        },
        "old_200_epoch_validation": {
            "status": "not_run_inapplicable_after_user_early_stop",
            "must_not_be_claimed_passed": True,
        },
        "skipped_gates_claimed_passed": False,
        "zip_created": False,
        "synapse_uploaded": False,
    }
    decision_path = args.snapshot_root / "EARLY_STOP_EPOCH125_SNAPSHOT.json"
    write_json_exclusive(decision_path, decision)
    copied["EARLY_STOP_EPOCH125_SNAPSHOT.json"] = sha256_file(decision_path)
    manifest_path = args.snapshot_root / "ARTIFACT_SHA256SUMS.txt"
    manifest_text = "".join(f"{digest}  {name}\n" for name, digest in sorted(copied.items()))
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(manifest_text)
        handle.flush()
        os.fsync(handle.fileno())

    os.kill(args.trainer_pid, signal.SIGTERM)
    os.kill(args.trainer_pid, signal.SIGCONT)
    print(
        "S2_FIX_V3_EPOCH125_STOP_SIGNAL_SENT "
        f"status={ARTIFACT_STATUS} checkpoint_sha256={checkpoint['sha256']}"
    )
    return decision


def inotify_fd(directory: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC)
    if descriptor < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    watch = libc.inotify_add_watch(
        descriptor,
        os.fsencode(directory),
        IN_CLOSE_WRITE | IN_MOVED_TO,
    )
    if watch < 0:
        errno_value = ctypes.get_errno()
        os.close(descriptor)
        raise OSError(errno_value, os.strerror(errno_value))
    return descriptor


def watch_until_target(args: argparse.Namespace) -> None:
    current_epoch = read_checkpoint_epoch(args.checkpoint)
    require(current_epoch <= args.target_epoch, f"checkpoint already advanced to {current_epoch}")
    if current_epoch == args.target_epoch:
        stop_and_snapshot(args)
        return

    descriptor = inotify_fd(args.checkpoint.parent)
    try:
        # Close the race between the initial checkpoint read and installing the watch.
        current_epoch = read_checkpoint_epoch(args.checkpoint)
        if current_epoch == args.target_epoch:
            stop_and_snapshot(args)
            return
        require(current_epoch < args.target_epoch, f"checkpoint advanced to {current_epoch}")
        print(
            "S2_FIX_V3_EPOCH125_WATCH_ARMED "
            f"status={ARTIFACT_STATUS} current_epoch={current_epoch}",
            flush=True,
        )
        while True:
            payload = os.read(descriptor, 65536)
            offset = 0
            while offset + EVENT_STRUCT.size <= len(payload):
                _, mask, _, name_length = EVENT_STRUCT.unpack_from(payload, offset)
                offset += EVENT_STRUCT.size
                name_bytes = payload[offset : offset + name_length]
                offset += name_length
                name = name_bytes.rstrip(b"\0").decode("utf-8", "replace")
                require(not (mask & IN_Q_OVERFLOW), "inotify queue overflow")
                require(not (mask & IN_IGNORED), "inotify watch was removed")
                if name != args.checkpoint.name or not (mask & (IN_CLOSE_WRITE | IN_MOVED_TO)):
                    continue
                current_epoch = read_checkpoint_epoch(args.checkpoint)
                require(current_epoch <= args.target_epoch, f"checkpoint advanced to {current_epoch}")
                if current_epoch == args.target_epoch:
                    stop_and_snapshot(args)
                    return
    finally:
        os.close(descriptor)


def write_failure(path: Path, exc: BaseException) -> None:
    failure = {
        "schema_version": 1,
        "status": "fail",
        "artifact_status": ARTIFACT_STATUS,
        "failed_at_utc": utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "training_may_be_sigstopped": True,
        "operator_review_required": True,
        "zip_created": False,
        "synapse_uploaded": False,
    }
    try:
        write_json_exclusive(path, failure)
    except FileExistsError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--failure-report", required=True, type=Path)
    parser.add_argument("--trainer-pid", required=True, type=int)
    parser.add_argument("--wrapper-pid", required=True, type=int)
    parser.add_argument("--expected-boot-id", required=True)
    parser.add_argument("--target-epoch", type=int, default=125)
    parser.add_argument("--patches-per-epoch", type=int, default=500)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--watch", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        process = validate_processes(
            args.trainer_pid,
            args.wrapper_pid,
            expected_boot_id=args.expected_boot_id,
        )
        checkpoint_epoch = read_checkpoint_epoch(args.checkpoint)
        require(checkpoint_epoch <= args.target_epoch, f"checkpoint already advanced to {checkpoint_epoch}")
        if args.preflight:
            print(
                json.dumps(
                    {
                        "status": "preflight_pass",
                        "artifact_status": ARTIFACT_STATUS,
                        "checkpoint_current_epoch": checkpoint_epoch,
                        "target_epoch": args.target_epoch,
                        "process": process,
                        "snapshot_root_exists": args.snapshot_root.exists(),
                        "failure_report_exists": args.failure_report.exists(),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            return 0
        watch_until_target(args)
        return 0
    except BaseException as exc:
        write_failure(args.failure_report, exc)
        print(f"S2_FIX_V3_EPOCH125_STOP_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
