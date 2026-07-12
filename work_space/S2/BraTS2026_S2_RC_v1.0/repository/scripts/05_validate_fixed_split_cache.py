#!/usr/bin/env python3
"""Verify that the S2 fixed split, raw dataset, and preprocessed cache agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file is missing: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Split file is empty or contains duplicate IDs: {path}")
    return values


def raw_dataset_ids(dataset_dir: Path) -> set[str]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"S2 raw imagesTr/labelsTr is incomplete: {dataset_dir}")

    suffix = "_0000.nii.gz"
    case_ids = {
        path.name[: -len(suffix)]
        for path in images_dir.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    }
    if not case_ids:
        raise ValueError(f"S2 raw dataset contains no cases: {dataset_dir}")

    missing = []
    for case_id in sorted(case_ids):
        for channel in range(4):
            path = images_dir / f"{case_id}_{channel:04d}.nii.gz"
            if not path.is_file():
                missing.append(str(path))
        label = labels_dir / f"{case_id}.nii.gz"
        if not label.is_file():
            missing.append(str(label))
    if missing:
        raise FileNotFoundError(f"S2 raw dataset has missing or broken files: {missing[:10]}")
    return case_ids


def preprocessed_cache_ids(preprocessed_dir: Path) -> tuple[set[str], str]:
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed configuration directory is missing: {preprocessed_dir}")

    b2nd_ids = {
        path.name[: -len(".b2nd")]
        for path in preprocessed_dir.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    }
    npz_ids = {path.stem for path in preprocessed_dir.glob("*.npz")}
    npy_ids = {
        path.stem
        for path in preprocessed_dir.glob("*.npy")
        if not path.name.endswith("_seg.npy")
    }
    if b2nd_ids:
        cache_format, case_ids = "b2nd", b2nd_ids
    elif npz_ids:
        cache_format, case_ids = "npz", npz_ids
    elif npy_ids:
        cache_format, case_ids = "npy", npy_ids
    else:
        raise ValueError(
            f"No supported nnU-Net cache files were found in {preprocessed_dir}"
        )
    missing = []
    for case_id in sorted(case_ids):
        if cache_format == "b2nd":
            companion = preprocessed_dir / f"{case_id}_seg.b2nd"
        else:
            companion = preprocessed_dir / f"{case_id}.pkl"
        if not companion.is_file():
            missing.append(str(companion))
    if missing:
        raise FileNotFoundError(f"Preprocessed cases have missing companion files: {missing[:10]}")
    return case_ids, cache_format


def check_dataset_json(path: Path, expected: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"dataset.json is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    actual = data.get("numTraining")
    if actual != expected:
        raise ValueError(
            f"dataset.json numTraining mismatch at {path}: expected={expected}, actual={actual}"
        )


def validate_fixed_split_cache(
    train_file: Path,
    val_file: Path,
    dataset_dir: Path,
    preprocessed_dir: Path,
) -> dict:
    train_ids = read_ids(train_file)
    val_ids = read_ids(val_file)
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Fixed train/validation overlap: {overlap[:10]}")
    expected_ids = set(train_ids) | set(val_ids)

    raw_ids = raw_dataset_ids(dataset_dir)
    cache_ids, cache_format = preprocessed_cache_ids(preprocessed_dir)
    if raw_ids != expected_ids:
        raise ValueError(
            "Raw dataset ID space differs from the fixed split: "
            f"missing_raw={sorted(expected_ids - raw_ids)[:10]}, "
            f"extra_raw={sorted(raw_ids - expected_ids)[:10]}"
        )
    if cache_ids != expected_ids:
        raise ValueError(
            "Preprocessed cache ID space differs from the fixed split: "
            f"missing_cache={sorted(expected_ids - cache_ids)[:10]}, "
            f"extra_cache={sorted(cache_ids - expected_ids)[:10]}"
        )

    check_dataset_json(dataset_dir / "dataset.json", len(expected_ids))
    check_dataset_json(preprocessed_dir.parent / "dataset.json", len(expected_ids))
    return {
        "status": "pass",
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "dataset_count": len(raw_ids),
        "preprocessed_count": len(cache_ids),
        "cache_format": cache_format,
        "id_sets_equal": True,
    }


def main() -> None:
    args = parse_args()
    result = validate_fixed_split_cache(
        Path(args.train_file),
        Path(args.val_file),
        Path(args.dataset_dir),
        Path(args.preprocessed_dir),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
