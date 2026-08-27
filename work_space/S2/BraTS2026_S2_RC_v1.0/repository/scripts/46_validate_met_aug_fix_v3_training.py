#!/usr/bin/env python3
"""Strictly validate and freeze the completed experimental Fix-v3 training run."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


EXPECTED_TRAINER = (
    "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune"
)
EXPECTED_ROUTE_STATUS = "experimental_unvalidated"
EXPECTED_STATES = frozenset(("COMMITTED", "NO_OP"))


class TrainingValidationError(RuntimeError):
    """Raised when completed Fix-v3 training evidence violates its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingValidationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path, *, allow_nan: bool = False) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing JSON: {path}")

    def reject_constant(value: str) -> None:
        raise TrainingValidationError(f"non-finite JSON constant {value}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=None if allow_nan else reject_constant,
    )
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_ids(path: Path, *, expected_count: int) -> list[str]:
    require(path.is_file() and path.stat().st_size > 0, f"missing ID list: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    require(len(values) == expected_count, f"ID count mismatch: {path}")
    require(len(set(values)) == expected_count, f"duplicate IDs: {path}")
    return values


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return all(_all_finite(item) for item in value)
    return True


def validate_training_log(path: Path, *, expected_epochs: int) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing training log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    red_flags = re.findall(
        r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|\bnan\b|\binf\b",
        text,
        flags=re.IGNORECASE,
    )
    require(not red_flags, f"training log red flags: {red_flags[:5]}")
    require("Training done." in text, "training completion line is missing")
    require("Validation complete" in text, "final validation completion line is missing")

    epochs = [int(value) for value in re.findall(r"^.*: Epoch ([0-9]+)\s*$", text, re.MULTILINE)]
    require(epochs == list(range(expected_epochs)), "training epoch sequence is incomplete")

    train_losses = [float(value) for value in re.findall(r"train_loss\s+([^\s]+)", text)]
    val_losses = [float(value) for value in re.findall(r"val_loss\s+([^\s]+)", text)]
    epoch_times = [float(value) for value in re.findall(r"Epoch time:\s+([^\s]+)\s+s", text)]
    dice_rows = [
        ast.literal_eval(value)
        for value in re.findall(r"Pseudo dice\s+(\[[^\n]+\])", text)
    ]
    for label, values in (
        ("train losses", train_losses),
        ("validation losses", val_losses),
        ("epoch times", epoch_times),
        ("pseudo Dice", dice_rows),
    ):
        require(len(values) == expected_epochs, f"{label} count mismatch")
        require(_all_finite(values), f"{label} contain non-finite values")
    require(all(value > 0 for value in epoch_times), "epoch times must be positive")

    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch_count": len(epochs),
        "last_train_loss": train_losses[-1],
        "last_validation_loss": val_losses[-1],
        "last_pseudo_dice": dice_rows[-1],
        "median_epoch_seconds": sorted(epoch_times)[len(epoch_times) // 2],
        "validation_complete": True,
    }


def _default_checkpoint_reader(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in the remote runtime
        raise TrainingValidationError("torch is required to validate the checkpoint") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "checkpoint payload is not a dictionary")
    network = checkpoint.get("network_weights")
    require(isinstance(network, Mapping) and network, "checkpoint lacks network weights")
    tensor_count = 0
    element_count = 0
    for name, tensor in network.items():
        require(hasattr(tensor, "numel"), f"invalid network tensor: {name}")
        tensor_count += 1
        element_count += int(tensor.numel())
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            require(bool(torch.isfinite(tensor).all()), f"non-finite network tensor: {name}")
    checkpoint["__network_tensor_count"] = tensor_count
    checkpoint["__network_element_count"] = element_count
    return checkpoint


def validate_checkpoint(
    path: Path,
    *,
    expected_epochs: int,
    reader: Callable[[Path], Mapping[str, Any]] = _default_checkpoint_reader,
) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing checkpoint: {path}")
    checkpoint = reader(path)
    require(checkpoint.get("current_epoch") == expected_epochs, "checkpoint epoch mismatch")
    require(checkpoint.get("trainer_name") == EXPECTED_TRAINER, "checkpoint trainer drift")
    logging = checkpoint.get("logging")
    require(isinstance(logging, Mapping), "checkpoint logging is missing")
    required_series = (
        "train_losses",
        "val_losses",
        "dice_per_class_or_region",
        "mean_fg_dice",
        "epoch_start_timestamps",
        "epoch_end_timestamps",
    )
    for name in required_series:
        values = logging.get(name)
        require(isinstance(values, Sequence), f"checkpoint logging lacks {name}")
        require(len(values) == expected_epochs, f"checkpoint logging length drift: {name}")
        require(_all_finite(values), f"checkpoint logging contains non-finite values: {name}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "current_epoch": checkpoint["current_epoch"],
        "trainer_name": checkpoint["trainer_name"],
        "network_tensor_count": int(checkpoint.get("__network_tensor_count", 0)),
        "network_element_count": int(checkpoint.get("__network_element_count", 0)),
    }


def validate_audit(
    path: Path,
    *,
    train_ids: set[str],
    expected_epochs: int,
    expected_rows: int,
) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing MET-AUG audit: {path}")
    require(expected_rows % expected_epochs == 0, "audit rows must divide evenly by epochs")
    rows_per_epoch = expected_rows // expected_epochs
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    epoch_counts: Counter[int] = Counter()
    patch_indices: dict[int, set[int]] = defaultdict(set)
    event_ids: set[str] = set()
    digest = hashlib.sha256()
    row_count = 0

    def reject_constant(value: str) -> None:
        raise TrainingValidationError(f"non-finite audit constant: {value}")

    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            require(raw_line.endswith(b"\n"), f"audit row lacks newline: {row_count + 1}")
            row_count += 1
            try:
                row = json.loads(raw_line, parse_constant=reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TrainingValidationError(f"invalid audit row {row_count}") from exc
            require(isinstance(row, dict), f"audit row {row_count} is not an object")
            require(_all_finite(row), f"audit row {row_count} contains non-finite values")
            state = str(row.get("state", ""))
            require(state in EXPECTED_STATES, f"invalid audit state at row {row_count}: {state}")
            epoch = row.get("epoch")
            patch_index = row.get("patch_index")
            require(isinstance(epoch, int) and 0 <= epoch < expected_epochs, "invalid audit epoch")
            require(
                isinstance(patch_index, int) and 0 <= patch_index < rows_per_epoch,
                "invalid audit patch index",
            )
            event_id = str(row.get("event_id", ""))
            require(bool(re.fullmatch(r"[0-9a-f]{24}", event_id)), "invalid audit event ID")
            require(event_id not in event_ids, f"duplicate audit event ID: {event_id}")
            require(row.get("route_id") == "MET-AUG-A", "audit route drift")
            require(row.get("rank") == 0 and row.get("worker") == 0, "audit worker drift")
            require(row.get("target_case_id") in train_ids, "audit target is outside fixed train")
            if state == "COMMITTED":
                require(row.get("reason") is None, "committed audit row has a failure reason")
                fix_v3 = row.get("fix_v3")
                require(isinstance(fix_v3, dict), "committed audit row lacks Fix-v3 evidence")
                require(fix_v3.get("status") == "pass", "committed Fix-v3 evidence did not pass")
                require(
                    row.get("target_patient_group") != row.get("donor_patient_group"),
                    "committed event reuses the target patient group",
                )
            else:
                require(bool(row.get("reason")), "NO_OP audit row lacks a reason")
            states[state] += 1
            reasons[str(row.get("reason"))] += 1
            epoch_counts[epoch] += 1
            patch_indices[epoch].add(patch_index)
            event_ids.add(event_id)

    require(row_count == expected_rows, f"audit row count mismatch: {row_count}")
    require(set(epoch_counts) == set(range(expected_epochs)), "audit epoch coverage mismatch")
    expected_indices = set(range(rows_per_epoch))
    for epoch in range(expected_epochs):
        require(epoch_counts[epoch] == rows_per_epoch, f"audit row count drift at epoch {epoch}")
        require(patch_indices[epoch] == expected_indices, f"audit patch coverage drift at epoch {epoch}")
    require(sum(states.values()) == row_count, "audit state accounting mismatch")

    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "row_count": row_count,
        "rows_per_epoch": rows_per_epoch,
        "state_counts": dict(sorted(states.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "event_ids_unique": True,
        "epoch_patch_cartesian_complete": True,
    }


def _default_prediction_inspector(prediction: Path, reference: Path) -> dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in the remote runtime
        raise TrainingValidationError("nibabel and numpy are required to validate predictions") from exc

    image = nib.load(str(prediction))
    reference_image = nib.load(str(reference))
    values = np.asanyarray(image.dataobj)
    require(values.ndim == 3, f"prediction is not 3D: {prediction}")
    require(bool(np.isfinite(values).all()), f"prediction contains non-finite values: {prediction}")
    labels = {int(value) for value in np.unique(values)}
    require(all(float(value).is_integer() for value in np.unique(values)), f"non-integer labels: {prediction}")
    require(labels <= {0, 1, 2, 3, 4}, f"illegal labels in {prediction}: {sorted(labels)}")
    require(image.shape == reference_image.shape, f"prediction/reference shape drift: {prediction.name}")
    require(
        bool(np.allclose(image.affine, reference_image.affine, rtol=0.0, atol=1e-5)),
        f"prediction/reference affine drift: {prediction.name}",
    )
    return {"shape": list(image.shape), "labels": sorted(labels)}


def _finite_aggregate_metrics(summary: Mapping[str, Any]) -> None:
    mean = summary.get("mean")
    foreground = summary.get("foreground_mean")
    require(isinstance(mean, Mapping) and set(mean) == {"1", "2", "3", "4"}, "summary mean drift")
    require(isinstance(foreground, Mapping), "summary foreground mean is missing")
    require(_all_finite(mean), "summary aggregate class metrics contain non-finite values")
    require(_all_finite(foreground), "summary aggregate foreground metrics contain non-finite values")


def validate_predictions(
    validation_root: Path,
    reference_root: Path,
    *,
    val_ids: Sequence[str],
    inspector: Callable[[Path, Path], Mapping[str, Any]] = _default_prediction_inspector,
) -> dict[str, Any]:
    require(validation_root.is_dir(), f"missing validation directory: {validation_root}")
    require(reference_root.is_dir(), f"missing reference directory: {reference_root}")
    expected_names = {f"{case_id}.nii.gz" for case_id in val_ids}
    predictions = sorted(validation_root.glob("*.nii.gz"))
    require({path.name for path in predictions} == expected_names, "validation prediction coverage mismatch")

    manifest_rows: list[str] = []
    total_bytes = 0
    for prediction in predictions:
        require(prediction.stat().st_size > 0, f"empty validation prediction: {prediction}")
        reference = reference_root / prediction.name
        require(reference.is_file() and reference.stat().st_size > 0, f"missing reference: {reference}")
        inspector(prediction, reference)
        size = prediction.stat().st_size
        manifest_rows.append(f"{prediction.name}\t{size}\t{sha256_file(prediction)}")
        total_bytes += size
    prediction_manifest = hashlib.sha256(("\n".join(manifest_rows) + "\n").encode()).hexdigest()

    summary_path = validation_root / "summary.json"
    summary = read_json(summary_path, allow_nan=True)
    metric_per_case = summary.get("metric_per_case")
    require(isinstance(metric_per_case, list), "validation summary lacks per-case metrics")
    require(len(metric_per_case) == len(val_ids), "validation summary case count mismatch")
    summary_ids = {
        Path(str(row.get("prediction_file", ""))).name.removesuffix(".nii.gz")
        for row in metric_per_case
        if isinstance(row, Mapping)
    }
    require(summary_ids == set(val_ids), "validation summary subject coverage mismatch")
    _finite_aggregate_metrics(summary)
    return {
        "root": str(validation_root),
        "prediction_count": len(predictions),
        "prediction_manifest_sha256": prediction_manifest,
        "prediction_total_bytes": total_bytes,
        "summary_sha256": sha256_file(summary_path),
        "aggregate_metrics": {
            "mean": summary["mean"],
            "foreground_mean": summary["foreground_mean"],
        },
    }


def _binding_paths(preflight_path: Path, preflight: Mapping[str, Any], fold: Path) -> dict[str, Path]:
    environment = preflight.get("environment")
    require(isinstance(environment, Mapping), "preflight environment is missing")
    route_root = preflight_path.parent.parent
    preprocessed = Path(str(environment["nnUNet_preprocessed"]))
    dataset_name = fold.parent.parent.name
    return {
        "component_manifest": Path(str(environment["S2_MET_AUG_COMPONENT_MANIFEST"])),
        "dataset_fingerprint": preprocessed / dataset_name / "dataset_fingerprint.json",
        "emergency_decision": Path(str(environment["S2_MET_AUG_EMERGENCY_DECISION"])),
        "fix_v2_failure_audit": Path(str(environment["S2_MET_AUG_FIX_V2_FAILURE_AUDIT"])),
        "fix_v3_calibration": Path(str(environment["S2_MET_AUG_FIX_V3_CALIBRATION"])),
        "fix_v3_freeze": route_root / "config" / "FIX_V3_FREEZE_AUDIT.json",
        "g1_checkpoint_selection": Path(str(environment["S2_MET_AUG_G1_CHECKPOINT_SELECTION"])),
        "g2_parent_gate": Path(str(environment["S2_MET_AUG_G2_QC_GATE"])),
        "original_e_checkpoint": Path(str(environment["S2_MET_AUG_ORIGINAL_E_CHECKPOINT"])),
        "overlay_audit": preprocessed / "PREPROCESSED_OVERLAY_AUDIT.json",
        "route_config": Path(str(environment["S2_MET_AUG_ROUTE_CONFIG"])),
        "smoke_validation": route_root / "config" / "TRAINING_SMOKE_ATTEMPT_03_VALIDATION.json",
        "source_runtime_manifest": route_root / "config" / "SOURCE_RUNTIME.sha256",
        "source_train_sh": route_root / "runtime" / "source" / "train.sh",
        "valid_mask_manifest": Path(str(environment["S2_MET_AUG_VALID_MASK_MANIFEST"])),
    }


def validate_bindings(
    preflight_path: Path,
    preflight: Mapping[str, Any],
    fold: Path,
) -> dict[str, Any]:
    expected = preflight.get("input_sha256")
    require(isinstance(expected, Mapping), "preflight input SHA map is missing")
    paths = _binding_paths(preflight_path, preflight, fold)
    require(set(paths) == set(expected), "preflight input SHA key drift")
    evidence: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        require(path.is_file() and path.stat().st_size > 0, f"missing frozen input {name}: {path}")
        digest = sha256_file(path)
        require(digest == expected[name], f"frozen input SHA drift: {name}")
        evidence[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return evidence


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _artifact_paths(
    fold: Path,
    supporting_paths: Sequence[Path],
) -> list[Path]:
    paths = {path.resolve() for path in fold.rglob("*") if path.is_file()}
    paths.update(path.resolve() for path in supporting_paths if path.is_file())
    return sorted(paths, key=str)


def validate_completed_training(
    *,
    preflight_path: Path,
    fold: Path,
    val_list: Path,
    pid_file: Path,
    output: Path,
    artifact_manifest: Path,
    expected_epochs: int,
    expected_validation_count: int,
    expected_audit_rows: int,
    checkpoint_reader: Callable[[Path], Mapping[str, Any]] = _default_checkpoint_reader,
    prediction_inspector: Callable[[Path, Path], Mapping[str, Any]] = _default_prediction_inspector,
) -> dict[str, Any]:
    for target in (output, artifact_manifest):
        require(not target.exists(), f"refusing to overwrite validation output: {target}")
    preflight_path = preflight_path.expanduser().resolve()
    fold = fold.expanduser().resolve()
    val_list = val_list.expanduser().resolve()
    pid_file = pid_file.expanduser().resolve()
    preflight = read_json(preflight_path)
    require(preflight.get("status") == "pass", "training preflight did not pass")
    require(preflight.get("route_status") == EXPECTED_ROUTE_STATUS, "training route status drift")
    require(preflight.get("training_epochs") == expected_epochs, "preflight epoch count drift")
    require(Path(str(preflight.get("training_result_root"))).resolve() == fold, "training fold drift")
    require(preflight.get("warm_start_policy") == "original_E_only", "warm-start policy drift")
    require(
        preflight.get("inference_policy") == "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
        "inference policy drift",
    )

    require(pid_file.is_file(), f"missing training PID file: {pid_file}")
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    require(not pid_is_alive(pid), f"training wrapper is still alive: {pid}")

    environment = preflight["environment"]
    split_root = Path(str(environment["BRATS_SPLIT_DIR"]))
    train_list = split_root / "train_fixed.txt"
    train_ids = set(read_ids(train_list, expected_count=1035))
    val_ids = read_ids(val_list, expected_count=expected_validation_count)
    require(train_ids.isdisjoint(val_ids), "fixed train/validation IDs overlap")
    split_audit_path = split_root / "fixed_split_cache_audit.json"
    split_audit = read_json(split_audit_path)
    require(split_audit.get("status") == "pass", "fixed split cache audit did not pass")
    require(split_audit.get("train_count") == len(train_ids), "fixed train count drift")
    require(split_audit.get("validation_count") == len(val_ids), "fixed validation count drift")

    bindings = validate_bindings(preflight_path, preflight, fold)
    logs = sorted(fold.glob("training_log_*.txt"))
    require(len(logs) == 1, f"expected exactly one formal training log, found {len(logs)}")
    log_evidence = validate_training_log(logs[0], expected_epochs=expected_epochs)
    checkpoint_path = fold / "checkpoint_final.pth"
    checkpoint_evidence = validate_checkpoint(
        checkpoint_path,
        expected_epochs=expected_epochs,
        reader=checkpoint_reader,
    )
    best_checkpoint = fold / "checkpoint_best.pth"
    require(best_checkpoint.is_file() and best_checkpoint.stat().st_size > 0, "best checkpoint is missing")
    audit_evidence = validate_audit(
        fold / "met_aug_events.jsonl",
        train_ids=train_ids,
        expected_epochs=expected_epochs,
        expected_rows=expected_audit_rows,
    )
    dataset_name = fold.parent.parent.name
    reference_root = (
        Path(str(environment["nnUNet_preprocessed"]))
        / dataset_name
        / "gt_segmentations"
    )
    prediction_evidence = validate_predictions(
        fold / "validation",
        reference_root,
        val_ids=val_ids,
        inspector=prediction_inspector,
    )

    supporting_paths = [
        preflight_path,
        pid_file,
        train_list,
        val_list,
        split_audit_path,
        *(Path(item["path"]) for item in bindings.values()),
    ]
    artifacts = _artifact_paths(fold, supporting_paths)
    manifest_text = "".join(
        f"{sha256_file(path)}  {path.stat().st_size}  {path}\n" for path in artifacts
    )
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "route_status": EXPECTED_ROUTE_STATUS,
        "generated_at_utc": utc_now(),
        "attempt": 1,
        "training_complete": True,
        "training_wrapper_pid": pid,
        "training_wrapper_stopped": True,
        "expected_epochs": expected_epochs,
        "fixed_train_count": len(train_ids),
        "fixed_validation_count": len(val_ids),
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "preflight_audit_sha256": preflight.get("preflight_audit_sha256"),
        },
        "fixed_split": {
            "train_sha256": sha256_file(train_list),
            "validation_sha256": sha256_file(val_list),
            "cache_audit_sha256": sha256_file(split_audit_path),
            "patient_sets_disjoint": True,
        },
        "frozen_inputs": bindings,
        "checkpoint_final": checkpoint_evidence,
        "checkpoint_best": {
            "path": str(best_checkpoint),
            "sha256": sha256_file(best_checkpoint),
            "bytes": best_checkpoint.stat().st_size,
        },
        "training_log": log_evidence,
        "met_aug_audit": audit_evidence,
        "validation": prediction_evidence,
        "artifact_manifest": {
            "path": str(artifact_manifest),
            "sha256": manifest_sha,
            "file_count": len(artifacts),
        },
        "inference_contract": "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
        "skipped_validation_stages": [
            "reference_increment",
            "development_96",
            "independent_holdout",
            "gate_0",
            "gate_1a_100000",
            "gate_1b_96_double_replay",
            "gate_2_120",
        ],
        "official_179_started": False,
        "zip_created": False,
        "synapse_uploaded": False,
        "validator_sha256": sha256_file(Path(__file__)),
    }
    summary["validation_audit_sha256"] = canonical_sha256(summary)
    encoded = json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for target, content in ((artifact_manifest, manifest_text), (output, encoded)):
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            created.append(target)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--expected-epochs", type=int, default=200)
    parser.add_argument("--expected-validation-count", type=int, default=103)
    parser.add_argument("--expected-audit-rows", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_completed_training(
        preflight_path=args.preflight,
        fold=args.fold,
        val_list=args.val_list,
        pid_file=args.pid_file,
        output=args.output,
        artifact_manifest=args.artifact_manifest,
        expected_epochs=args.expected_epochs,
        expected_validation_count=args.expected_validation_count,
        expected_audit_rows=args.expected_audit_rows,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
