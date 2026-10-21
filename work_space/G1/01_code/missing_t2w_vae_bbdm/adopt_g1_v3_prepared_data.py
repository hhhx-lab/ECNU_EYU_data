#!/usr/bin/env python3
"""Adopt a verified G1 data placement without duplicating the NIfTI files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


COPIED_FILES = {
    "data_csv.csv": "data_csv.csv",
    "g1_v2_data_placement_manifest.csv": "g1_v3_data_placement_manifest.csv",
    "g1_v2_label_filter_report.csv": "g1_v3_label_filter_report.csv",
    "g1_v2_label_filter_summary.json": "g1_v3_label_filter_summary.json",
    "g1_split_summary.json": "g1_split_summary.json",
    "g1_split_membership.csv": "g1_split_membership.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        required=True,
        help="Verified data directory containing input/, input_inference/, and split files.",
    )
    parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=Path("data"),
        help="V3-local data directory. Default: data",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace existing V3 input links and copied metadata.",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def count_case_dirs(path: Path) -> int:
    return sum(1 for child in path.iterdir() if child.is_dir())


def main() -> None:
    args = parse_args()
    source = args.source_data_dir.resolve(strict=True)
    target = args.target_data_dir.resolve()
    if source == target:
        raise ValueError("Source and target data directories must be different.")

    required_dirs = ("input", "input_inference")
    missing = [name for name in required_dirs if not (source / name).is_dir()]
    missing.extend(name for name in COPIED_FILES if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Prepared source is incomplete; missing: {sorted(missing)}"
        )

    target.mkdir(parents=True, exist_ok=True)
    for name in required_dirs:
        destination = target / name
        if destination.exists() or destination.is_symlink():
            if not args.clean:
                raise FileExistsError(
                    f"Target already exists: {destination}; pass --clean to replace it."
                )
            remove_path(destination)
        destination.symlink_to((source / name).resolve(), target_is_directory=True)

    for source_name, target_name in COPIED_FILES.items():
        destination = target / target_name
        if destination.exists() and not args.clean:
            raise FileExistsError(
                f"Target already exists: {destination}; pass --clean to replace it."
            )
        atomic_copy(source / source_name, destination)

    for runtime_dir in ("output", "evaluation", "latents", "attention_masks", "lesion_weights"):
        (target / runtime_dir).mkdir(parents=True, exist_ok=True)

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": str(source),
        "target_data_dir": str(target),
        "input_cases": count_case_dirs(target / "input"),
        "inference_cases": count_case_dirs(target / "input_inference"),
        "input_link": str((target / "input").resolve()),
        "input_inference_link": str((target / "input_inference").resolve()),
        "copied_files": COPIED_FILES,
    }
    output = target / "g1_v3_data_adoption.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
