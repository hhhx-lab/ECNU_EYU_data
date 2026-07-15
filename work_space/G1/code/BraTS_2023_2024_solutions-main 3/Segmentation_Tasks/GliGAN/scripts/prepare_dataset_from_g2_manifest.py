#!/usr/bin/env python3
"""Materialize the Diffusion V3 train/val view as symlinks from G2 master data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


MODALITY_COLUMNS = {
    "t1n": "t1n_path",
    "t1c": "t1c_path",
    "t2w": "t2w_path",
    "t2f": "t2f_path",
    "seg": "seg_path",
}
CASE_PATTERN = re.compile(r"^BraTS-MET-\d{5}-\d{3}$")
MARKER_NAME = ".g1_diffusion_dataset"


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_case_id",
            "patient_group",
            "split",
            "t2w_status",
            "allowed_as_v2_source",
            *MODALITY_COLUMNS.values(),
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        return list(reader)


def resolve_source(project_root: Path, value: str) -> Path:
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = project_root / source
    return source.resolve()


def first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return None


def resolve_server_source(
    raw_root: Path,
    corrected_label_roots: list[Path],
    row: dict[str, str],
    modality: str,
) -> Path:
    """Resolve the flattened ECNU layout without weakening manifest policy."""
    case_id = row["source_case_id"].strip()
    case_dir = raw_root / case_id
    if modality == "seg" and row.get("label_source", "raw").strip().lower() == "corrected":
        corrected_candidates = []
        for root in corrected_label_roots:
            corrected_candidates.extend([
                root / f"{case_id}-seg.nii.gz",
                root / case_id / f"{case_id}-seg.nii.gz",
                root / case_id / "seg.nii.gz",
            ])
        corrected = first_existing(corrected_candidates)
        if corrected is None:
            raise FileNotFoundError(
                f"Manifest requires a corrected segmentation for {case_id}, but none "
                f"was found under: {corrected_label_roots}")
        return corrected

    source = first_existing([
        case_dir / f"{case_id}-{modality}.nii.gz",
        case_dir / f"{modality}.nii.gz",
    ])
    if source is None:
        raise FileNotFoundError(
            f"Missing {case_id}/{modality} under flattened raw root: {case_dir}")
    return source


def prepare_output(output_dir: Path, clean: bool, dry_run: bool) -> None:
    if output_dir == Path("/") or len(output_dir.parts) < 3:
        raise ValueError(f"Unsafe output directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        marker = output_dir / MARKER_NAME
        if not clean:
            raise FileExistsError(
                f"Output is not empty: {output_dir}. Re-run with --clean after inspection.")
        if not marker.is_file():
            raise RuntimeError(
                f"Refusing to clean unmarked directory: {output_dir}. "
                f"Expected marker {marker}")
        if not dry_run:
            shutil.rmtree(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / MARKER_NAME).write_text(
            "Managed by prepare_dataset_from_g2_manifest.py\n", encoding="ascii")


def write_membership(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_case_id", "patient_group", "split", "label_source",
        "t1n_path", "t1c_path", "t2w_path", "t2f_path", "seg_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an authentic-T2W train/val DataSet view from the G2 manifest")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("DataSet"), type=Path)
    parser.add_argument("--split-dir", default=Path("splits/current"), type=Path)
    parser.add_argument(
        "--raw-root", type=Path,
        help="Optional flattened server root: <root>/<case_id>/<modality>.nii.gz")
    parser.add_argument(
        "--corrected-label-root", action="append", default=[], type=Path,
        help="Root containing corrected <case_id>-seg.nii.gz files; repeatable")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    manifest = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    corrected_label_roots = [
        root.expanduser().resolve() for root in args.corrected_label_root
    ]
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest}")
    if raw_root is not None and not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    missing_corrected_roots = [
        root for root in corrected_label_roots if not root.is_dir()
    ]
    if missing_corrected_roots:
        raise FileNotFoundError(
            f"Corrected-label roots not found: {missing_corrected_roots}")

    rows = read_manifest(manifest)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["patient_group"]].add(row["split"].strip().lower())
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Patient groups cross splits: {leaking_groups[:10]}")

    selected = []
    for row in rows:
        case_id = row["source_case_id"].strip()
        split = row["split"].strip().lower()
        if split not in {"train", "val"}:
            continue
        if row["t2w_status"].strip().lower() != "authentic":
            continue
        if split == "train" and not parse_bool(row["allowed_as_v2_source"]):
            raise ValueError(f"Authentic train case is not approved as V2 source: {case_id}")
        if not CASE_PATTERN.fullmatch(case_id):
            raise ValueError(f"Unexpected case ID: {case_id}")

        resolved = {}
        for modality, column in MODALITY_COLUMNS.items():
            if raw_root is None:
                source = resolve_source(project_root, row[column])
            else:
                source = resolve_server_source(
                    raw_root, corrected_label_roots, row, modality)
            if not source.is_file():
                raise FileNotFoundError(f"Missing {case_id}/{modality}: {source}")
            resolved[modality] = source
        selected.append((row, resolved))

    if not selected:
        raise RuntimeError("No authentic train/val cases were selected")

    prepare_output(output_dir, args.clean, args.dry_run)
    if not args.dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    membership_rows = []
    for row, resolved in sorted(selected, key=lambda item: item[0]["source_case_id"]):
        case_id = row["source_case_id"].strip()
        case_dir = output_dir / case_id
        if not args.dry_run:
            case_dir.mkdir()
        membership = dict(row)
        for modality, source in resolved.items():
            destination = case_dir / f"{case_id}-{modality}.nii.gz"
            if not args.dry_run:
                os.symlink(source, destination)
            membership[MODALITY_COLUMNS[modality]] = str(source)
        membership_rows.append(membership)

    val_suffixes = [
        row["source_case_id"][-9:]
        for row in membership_rows if row["split"].strip().lower() == "val"
    ]
    counts = Counter(row["split"].strip().lower() for row in membership_rows)
    summary = {
        "source_manifest": str(manifest),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "case_counts": dict(sorted(counts.items())),
        "total_cases": len(membership_rows),
        "patient_group_overlap": 0,
        "copy_policy": "symlink_only",
        "source_layout": "flattened_raw_root" if raw_root else "manifest_paths",
        "raw_root": str(raw_root) if raw_root else "",
        "corrected_label_roots": [str(root) for root in corrected_label_roots],
    }

    if not args.dry_run:
        write_membership(split_dir / "dataset_membership.csv", membership_rows)
        (split_dir / "val_patient_suffixes.txt").write_text(
            "".join(f"{value}\n" for value in val_suffixes), encoding="ascii")
        (split_dir / "val_patient_suffixes_one_line.txt").write_text(
            ",".join(val_suffixes) + "\n", encoding="ascii")
        (split_dir / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("dry_run=true; no files were created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
