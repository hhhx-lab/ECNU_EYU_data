#!/usr/bin/env python3
"""Merge and validate 179 official predictions without creating a ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_SELECTION_SHA256 = "380467ce9157ffc31b58baaa6d1182e7882cd4c68555ee1a0048304cc708be5f"
SHARDS = ("gpu0", "gpu1")


class FinalizationError(RuntimeError):
    """Raised when official prediction evidence is incomplete or invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        require("=" in line, f"Malformed marker line: {path}: {line}")
        key, value = line.split("=", 1)
        values[key] = value
    return values


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    repository = args.repository.resolve()
    contract_path = run_root / "OFFICIAL_INFERENCE_CONTRACT.json"
    predictions_root = run_root / "predictions"
    manifest_path = run_root / "official_prediction_manifest.csv"
    summary_path = run_root / "OFFICIAL_INFERENCE_VALIDATION.json"
    marker_path = run_root / "OFFICIAL_INFERENCE_COMPLETE.ok"

    require(run_root.is_dir(), f"Missing run root: {run_root}")
    for path in (predictions_root, manifest_path, summary_path, marker_path):
        require(not path.exists(), f"Refusing to overwrite final inference artifact: {path}")
    require(contract_path.is_file(), f"Missing inference contract: {contract_path}")
    contract = load_json(contract_path)
    require(contract.get("status") == "prepared", "Inference contract is not prepared")
    require(contract.get("contract_sha256") == args.expected_contract_id, "Contract identity drift")
    require(contract.get("expected_case_count") == args.expected_count, "Contract case count drift")
    require(contract.get("model", {}).get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "Checkpoint contract drift")
    require(contract.get("selection", {}).get("model_selection_sha256") == EXPECTED_SELECTION_SHA256, "Selection contract drift")
    require(contract.get("packaging") == {"zip_allowed": False, "synapse_upload_allowed": False}, "Packaging boundary drift")

    selection_path = Path(contract["selection"]["root"]) / "MODEL_SELECTION.json"
    require(sha256_file(selection_path) == EXPECTED_SELECTION_SHA256, "Model selection evidence drift")
    selection = load_json(selection_path)
    require(selection.get("selected_model") == "E", "Selected model is not original E")
    require(selection.get("selected_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "Selected checkpoint drift")
    require(
        sha256_file(Path(contract["model"]["checkpoint_path"])) == EXPECTED_CHECKPOINT_SHA256,
        "Checkpoint file drift",
    )

    expected_ids: set[str] = set()
    shard_files: dict[str, list[Path]] = {}
    shard_evidence: dict[str, Any] = {}
    for shard in SHARDS:
        complete_marker = run_root / f"{shard}_COMPLETE.ok"
        failure_marker = run_root / f"{shard}_FAILED.ok"
        pid_path = run_root / f"{shard}.pid"
        log_path = run_root / "logs" / f"{shard}.log"
        output_root = run_root / f"output_{shard}"
        output_manifest = run_root / f"{shard}_output_manifest.tsv"
        for path in (complete_marker, pid_path, log_path, output_manifest):
            require(path.is_file() and path.stat().st_size > 0, f"Missing shard evidence: {path}")
        require(not failure_marker.exists(), f"Shard failure marker exists: {failure_marker}")
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        require(not pid_alive(pid), f"Shard process is still alive: {shard} pid={pid}")
        marker = parse_marker(complete_marker)
        expected_shard_count = int(contract["shards"][shard]["case_count"])
        require(marker.get("status") == "pass", f"Shard marker failed: {shard}")
        require(int(marker.get("predictions", "-1")) == expected_shard_count, f"Shard marker count drift: {shard}")
        require(marker.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, f"Shard checkpoint drift: {shard}")
        require(marker.get("output_manifest_sha256") == sha256_file(output_manifest), f"Shard manifest SHA drift: {shard}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        red_flags = re.findall(
            r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|worker failure",
            log_text,
            flags=re.IGNORECASE,
        )
        require(not red_flags, f"Shard log errors: {shard}: {red_flags[:5]}")
        require("S2_OFFICIAL_INFERENCE_SHARD_PASS" in log_text, f"Shard pass log missing: {shard}")
        outputs = sorted(output_root.glob("*.nii.gz"))
        require(len(outputs) == expected_shard_count, f"Shard output count mismatch: {shard}")
        ids = {path.name.removesuffix(".nii.gz") for path in outputs}
        require(not (expected_ids & ids), f"Shard output overlap: {shard}")
        expected_ids.update(ids)
        shard_files[shard] = outputs
        shard_evidence[shard] = {
            "pid": pid,
            "pid_stopped": True,
            "prediction_count": len(outputs),
            "completion_marker_sha256": sha256_file(complete_marker),
            "output_manifest_sha256": sha256_file(output_manifest),
            "log_sha256": sha256_file(log_path),
        }
    require(len(expected_ids) == args.expected_count, f"Merged case count mismatch: {len(expected_ids)}")

    source_root = Path(contract["source_root"])
    validation_script = repository / "scripts/07_package_official_submission.py"
    require(validation_script.is_file(), f"Missing validation implementation: {validation_script}")
    validation_module = load_module(validation_script, "official_prediction_validation_locked")
    case_dirs = validation_module.discover_source_cases(source_root, args.expected_count)
    source_ids = {path.name for path in case_dirs}
    require(expected_ids == source_ids, "Prediction/source ID coverage mismatch")

    predictions_root.mkdir()
    for shard in SHARDS:
        for source in shard_files[shard]:
            target = predictions_root / source.name
            target.hardlink_to(source)
            require(os.path.samefile(source, target), f"Merged output is not a hardlink: {target}")

    rows = [
        validation_module.validate_case(
            case_dir,
            predictions_root / f"{case_dir.name}.nii.gz",
        )
        for case_dir in case_dirs
    ]
    require(len(rows) == args.expected_count, "Validated row count mismatch")
    require({str(row["case_id"]) for row in rows} == source_ids, "Validated case coverage mismatch")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    empty_cases = [str(row["case_id"]) for row in rows if row["empty_prediction"]]
    prediction_payload = "".join(
        f"{row['prediction_filename']}\t{row['file_size_bytes']}\t{row['sha256']}\n"
        for row in rows
    )
    prediction_manifest_sha = hashlib.sha256(prediction_payload.encode("utf-8")).hexdigest()
    zip_files = list(run_root.rglob("*.zip"))
    require(not zip_files, f"ZIP creation is forbidden, found: {zip_files}")

    summary = {
        "schema_version": 1,
        "status": "pass",
        "generated_at_utc": utc_now(),
        "role": "BraTS_2026_S2_Task1_official_179_predictions_technical_validation",
        "run_root": str(run_root),
        "source_root": str(source_root),
        "prediction_root": str(predictions_root),
        "prediction_count": len(rows),
        "expected_case_count": args.expected_count,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_selection_sha256": EXPECTED_SELECTION_SHA256,
        "contract_identity": args.expected_contract_id,
        "contract_file_sha256": sha256_file(contract_path),
        "validation_script": str(validation_script),
        "validation_script_sha256": sha256_file(validation_script),
        "finalizer_script": str(Path(__file__).resolve()),
        "finalizer_script_sha256": sha256_file(Path(__file__).resolve()),
        "prediction_manifest_sha256": prediction_manifest_sha,
        "official_prediction_manifest_sha256": sha256_file(manifest_path),
        "empty_prediction_count": len(empty_cases),
        "empty_prediction_case_ids": empty_cases,
        "allowed_labels": [0, 1, 2, 3, 4],
        "geometry_checks": [
            "array_dimensions",
            "voxel_spacing",
            "image_origin",
            "spatial_orientation",
            "affine",
        ],
        "shards": shard_evidence,
        "zip_created": False,
        "synapse_uploaded": False,
    }
    write_json(summary_path, summary)
    marker_path.write_text(
        "status=pass\n"
        f"completed_at_utc={utc_now()}\n"
        f"predictions={len(rows)}\n"
        f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}\n"
        f"prediction_manifest_sha256={prediction_manifest_sha}\n"
        f"validation_sha256={sha256_file(summary_path)}\n"
        f"official_prediction_manifest_sha256={sha256_file(manifest_path)}\n"
        "zip_created=false\n"
        "synapse_uploaded=false\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "prediction_count": len(rows),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "prediction_manifest_sha256": prediction_manifest_sha,
        "validation_sha256": sha256_file(summary_path),
        "completion_marker_sha256": sha256_file(marker_path),
        "empty_prediction_count": len(empty_cases),
        "zip_created": False,
        "synapse_uploaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--expected-contract-id", required=True)
    parser.add_argument("--expected-count", type=int, default=179)
    return parser


def main() -> None:
    try:
        result = finalize(build_parser().parse_args())
    except FinalizationError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
