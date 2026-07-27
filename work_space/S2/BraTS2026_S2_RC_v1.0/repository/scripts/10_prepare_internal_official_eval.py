#!/usr/bin/env python3
"""Prepare a fixed-validation prediction/reference view for official-style evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_id_list(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in fixed validation list: {path}")
    return values


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    source_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Invalid mapping row {line_number}: {row}")
            nnunet_id, source_id = (value.strip() for value in row)
            if nnunet_id in mapping:
                raise ValueError(f"Duplicate nnU-Net ID in mapping: {nnunet_id}")
            if source_id in source_ids:
                raise ValueError(f"Duplicate source ID in mapping: {source_id}")
            mapping[nnunet_id] = source_id
            source_ids.add(source_id)
    return mapping


def discover_nifti_ids(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {root}")
    files = sorted(root.glob("*.nii.gz"))
    empty = [path.name for path in files if path.stat().st_size <= 0]
    if empty:
        raise ValueError(f"Empty NIfTI files in {root}: {empty[:10]}")
    return {path.name[: -len(".nii.gz")] for path in files}


def materialize(source: Path, target: Path, mode: str) -> None:
    if mode == "hardlink":
        target.hardlink_to(source)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")


def prepare_internal_eval(
    prediction_root: Path,
    reference_root: Path,
    mapping_path: Path,
    val_list_path: Path,
    output_root: Path,
    checkpoint_path: Path,
    *,
    expected_count: int = 103,
    mode: str = "hardlink",
) -> dict[str, object]:
    prediction_root = prediction_root.expanduser().resolve()
    reference_root = reference_root.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve()
    val_list_path = val_list_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint missing or empty: {checkpoint_path}")

    fixed_val_ids = read_id_list(val_list_path)
    if len(fixed_val_ids) != expected_count:
        raise ValueError(
            f"Fixed validation count mismatch: expected={expected_count}, "
            f"actual={len(fixed_val_ids)}"
        )
    fixed_val_set = set(fixed_val_ids)
    prediction_ids = discover_nifti_ids(prediction_root)
    if prediction_ids != fixed_val_set:
        raise ValueError(
            "prediction/fixed-val ID mismatch: "
            f"missing={sorted(fixed_val_set - prediction_ids)[:10]}, "
            f"extra={sorted(prediction_ids - fixed_val_set)[:10]}"
        )

    mapping = read_mapping(mapping_path)
    missing_mapping = sorted(fixed_val_set - mapping.keys())
    if missing_mapping:
        raise ValueError(f"Missing fixed-val mapping IDs: {missing_mapping[:10]}")
    source_ids = [mapping[nnunet_id] for nnunet_id in fixed_val_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Fixed validation maps to duplicate source IDs")

    reference_ids = discover_nifti_ids(reference_root)
    missing_references = sorted(fixed_val_set - reference_ids)
    if missing_references:
        raise ValueError(f"Missing fixed-val references: {missing_references[:10]}")

    if output_root.exists() and any(output_root.rglob("*.nii.gz")):
        raise ValueError(f"Output already contains NIfTI files: {output_root}")
    prediction_output = output_root / "prediction"
    reference_output = output_root / "reference"
    prediction_output.mkdir(parents=True, exist_ok=True)
    reference_output.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for nnunet_id in fixed_val_ids:
        source_id = mapping[nnunet_id]
        prediction_source = prediction_root / f"{nnunet_id}.nii.gz"
        reference_source = reference_root / f"{nnunet_id}.nii.gz"
        prediction_target = prediction_output / f"{source_id}.nii.gz"
        reference_target = reference_output / f"{source_id}.nii.gz"
        materialize(prediction_source, prediction_target, mode)
        materialize(reference_source, reference_target, mode)
        manifest_rows.append(
            {
                "nnunet_id": nnunet_id,
                "source_case_id": source_id,
                "prediction_source": str(prediction_source),
                "reference_source": str(reference_source),
                "prediction_eval_path": str(prediction_target),
                "reference_eval_path": str(reference_target),
            }
        )

    manifest_path = output_root / "nnunet_to_source_id.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary: dict[str, object] = {
        "status": "pass",
        "dataset_role": "fixed_internal_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(fixed_val_ids),
        "prediction_count": len(list(prediction_output.glob("*.nii.gz"))),
        "reference_count": len(list(reference_output.glob("*.nii.gz"))),
        "mapping_count": len(manifest_rows),
        "materialization_mode": mode,
        "prediction_source_root": str(prediction_root),
        "reference_source_root": str(reference_root),
        "fixed_val_list": str(val_list_path),
        "mapping_source": str(mapping_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_config": "mets",
        "vol_threshold": 27,
        "overlap_threshold": 0.2,
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare source-ID views for fixed internal BraTS evaluation."
    )
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=103)
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_internal_eval(
        args.prediction_root,
        args.reference_root,
        args.mapping,
        args.val_list,
        args.output_root,
        args.checkpoint,
        expected_count=args.expected_count,
        mode=args.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
