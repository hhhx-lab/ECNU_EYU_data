#!/usr/bin/env python3
"""Validate and materialize the unlabeled Task 1 validation set for nnU-Net."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


CASE_ID_PATTERN = re.compile(r"^BraTS-MET-[0-9]{5}-[0-9]{3}$")
CHANNELS = (
    ("t1n", "0000"),
    ("t1c", "0001"),
    ("t2w", "0002"),
    ("t2f", "0003"),
)


def discover_cases(source_root: Path, expected_count: int) -> list[dict[str, Path | str]]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Official validation directory does not exist: {source_root}")
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")

    case_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    unexpected_dirs = [path.name for path in case_dirs if not CASE_ID_PATTERN.fullmatch(path.name)]
    if unexpected_dirs:
        raise ValueError(
            "Official validation contains unexpected case directories: "
            f"{unexpected_dirs[:10]}"
        )
    if len(case_dirs) != expected_count:
        raise ValueError(
            "Official validation case-count mismatch: "
            f"expected={expected_count}, actual={len(case_dirs)}"
        )

    cases: list[dict[str, Path | str]] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        expected_nifti = {
            f"{case_id}-{modality}.nii.gz" for modality, _ in CHANNELS
        }
        actual_nifti = {
            path.name
            for path in case_dir.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        }
        missing = sorted(expected_nifti - actual_nifti)
        unexpected = sorted(actual_nifti - expected_nifti)
        if missing or unexpected:
            raise ValueError(
                f"Invalid official validation case {case_id}: "
                f"missing={missing}, unexpected_nifti={unexpected}"
            )

        row: dict[str, Path | str] = {"case_id": case_id, "case_dir": case_dir}
        for modality, _ in CHANNELS:
            source = case_dir / f"{case_id}-{modality}.nii.gz"
            if source.stat().st_size <= 0:
                raise ValueError(f"Empty official validation volume: {source}")
            row[modality] = source
        cases.append(row)
    return cases


def case_id_sha256(case_ids: list[str]) -> str:
    payload = "".join(f"{case_id}\n" for case_id in case_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def materialize_file(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def prepare_official_validation(
    source_root: Path,
    input_root: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    expected_count: int = 179,
    mode: str = "symlink",
    clean: bool = False,
) -> dict[str, object]:
    if mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported materialization mode: {mode}")

    source_root = source_root.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    if source_root == input_root:
        raise ValueError("Source and nnU-Net input directories must be different")

    cases = discover_cases(source_root, expected_count)
    input_root.mkdir(parents=True, exist_ok=True)

    expected_input_names = {
        f"{row['case_id']}_{channel}.nii.gz"
        for row in cases
        for _, channel in CHANNELS
    }
    existing_nifti = {
        path.name
        for path in input_root.iterdir()
        if (path.is_file() or path.is_symlink()) and path.name.endswith(".nii.gz")
    }
    stale_nifti = sorted(existing_nifti - expected_input_names)
    if stale_nifti and not clean:
        raise ValueError(
            "nnU-Net input contains stale NIfTI files; rerun with --clean: "
            f"{stale_nifti[:10]}"
        )
    if clean:
        for path in input_root.iterdir():
            if (path.is_file() or path.is_symlink()) and path.name.endswith(".nii.gz"):
                path.unlink()

    manifest_rows: list[dict[str, str]] = []
    for row in cases:
        case_id = str(row["case_id"])
        manifest_row = {"case_id": case_id}
        for modality, channel in CHANNELS:
            source = Path(row[modality])
            target = input_root / f"{case_id}_{channel}.nii.gz"
            materialize_file(source, target, mode)
            manifest_row[f"{modality}_source_path"] = str(source)
            manifest_row[f"nnunet_{channel}_path"] = str(target)
        manifest_rows.append(manifest_row)

    actual_input_names = {
        path.name
        for path in input_root.iterdir()
        if (path.is_file() or path.is_symlink()) and path.name.endswith(".nii.gz")
    }
    if actual_input_names != expected_input_names:
        raise RuntimeError(
            "Official validation materialization mismatch: "
            f"expected={len(expected_input_names)}, actual={len(actual_input_names)}"
        )

    manifest_path = manifest_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id"]
    for modality, channel in CHANNELS:
        fieldnames.extend((f"{modality}_source_path", f"nnunet_{channel}_path"))
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    case_ids = [str(row["case_id"]) for row in cases]
    summary: dict[str, object] = {
        "status": "pass",
        "dataset_role": "official_unlabeled_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "nnunet_input_root": str(input_root),
        "manifest_path": str(manifest_path),
        "expected_case_count": expected_count,
        "case_count": len(cases),
        "nifti_count": len(actual_input_names),
        "segmentation_count": 0,
        "channel_order": {
            channel: modality for modality, channel in CHANNELS
        },
        "materialization_mode": mode,
        "case_id_sha256": case_id_sha256(case_ids),
        "first_case_id": case_ids[0],
        "last_case_id": case_ids[-1],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the 179-case unlabeled Task 1 validation set for nnU-Net inference."
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-count", type=int, default=179)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest or args.dst.parent / "official_validation_manifest.csv"
    summary_path = args.summary or args.dst.parent / "official_validation_preparation.json"
    summary = prepare_official_validation(
        args.src,
        args.dst,
        manifest_path,
        summary_path,
        expected_count=args.expected_count,
        mode=args.mode,
        clean=args.clean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
