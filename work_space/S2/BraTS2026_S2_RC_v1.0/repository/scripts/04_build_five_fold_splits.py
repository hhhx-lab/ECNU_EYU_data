#!/usr/bin/env python3
"""Build deterministic S2 five-fold splits while preserving the completed fold 0.

The source G2 split already contains train, validation, and locked-test sets.
Fold 0 keeps that validation set unchanged. The original training set is
partitioned into four deterministic validation sets for folds 1-4. This gives
every train+validation case exactly one validation appearance without touching
the internal locked test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = "BraTS2026-S2-realonly-five-fold-v1"
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fold-specific S2 split files from the fixed G2 split."
    )
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainval-mapping-csv", required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def load_source_split(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    split = data[0] if isinstance(data, list) else data
    if not isinstance(split, dict):
        raise ValueError(f"Expected a split object in {path}")
    for key in ("train", "val"):
        if key not in split or not isinstance(split[key], list):
            raise ValueError(f"Split JSON is missing list field '{key}': {path}")
    split.setdefault("test", [])
    if not isinstance(split["test"], list):
        raise ValueError(f"Split JSON field 'test' must be a list: {path}")
    return split


def normalized_ids(values: Iterable[object], field: str) -> list[str]:
    result = [str(value).strip() for value in values if str(value).strip()]
    duplicates = sorted(case_id for case_id, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate IDs in {field}: {duplicates[:10]}")
    return result


def stable_key(case_id: str, seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return digest, case_id


def balanced_chunks(values: list[str], count: int) -> list[list[str]]:
    quotient, remainder = divmod(len(values), count)
    chunks: list[list[str]] = []
    start = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        chunks.append(values[start:start + size])
        start += size
    if start != len(values):
        raise AssertionError("Internal split chunking error")
    return chunks


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(values) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "nnunet_case_id" not in reader.fieldnames:
            raise ValueError(f"Mapping CSV lacks nnunet_case_id: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    ids = [row["nnunet_case_id"].strip() for row in rows]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate nnunet_case_id values in mapping: {duplicates[:10]}")
    return rows, fieldnames


def write_trainval_mapping(
    path: Path,
    rows_by_id: dict[str, dict[str, str]],
    fieldnames: list[str],
    pool_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case_id in pool_ids:
            writer.writerow(rows_by_id[case_id])
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    split_path = Path(args.split_json).expanduser().resolve()
    mapping_path = Path(args.mapping_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    trainval_mapping_path = Path(args.trainval_mapping_csv).expanduser().resolve()

    source = load_source_split(split_path)
    source_train = normalized_ids(source["train"], "train")
    source_val = normalized_ids(source["val"], "val")
    locked_test = normalized_ids(source.get("test", []), "test")

    train_set = set(source_train)
    val_set = set(source_val)
    test_set = set(locked_test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Source train, val, and locked test sets must be disjoint")

    cv_pool = source_train + source_val
    cv_pool_set = set(cv_pool)
    if len(cv_pool_set) != len(cv_pool):
        raise ValueError("Duplicate IDs exist across source train and val")
    if len(cv_pool) < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} cases for five-fold CV")

    mapping_rows, fieldnames = load_mapping(mapping_path)
    rows_by_id = {row["nnunet_case_id"].strip(): row for row in mapping_rows}
    missing_from_mapping = sorted((cv_pool_set | test_set) - set(rows_by_id))
    if missing_from_mapping:
        raise ValueError(
            f"Split IDs missing from mapping ({len(missing_from_mapping)}): "
            f"{missing_from_mapping[:10]}"
        )

    remaining_ordered = sorted(source_train, key=lambda case_id: stable_key(case_id, args.seed))
    fold_validation = [source_val] + balanced_chunks(remaining_ordered, N_FOLDS - 1)

    validation_counts = Counter(case_id for fold_ids in fold_validation for case_id in fold_ids)
    invalid_coverage = sorted(
        case_id for case_id in cv_pool if validation_counts.get(case_id, 0) != 1
    )
    if invalid_coverage:
        raise AssertionError(
            "Each CV-pool case must appear in exactly one validation fold: "
            f"{invalid_coverage[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    for fold, validation_ids in enumerate(fold_validation):
        validation_set = set(validation_ids)
        training_ids = [case_id for case_id in cv_pool if case_id not in validation_set]
        if set(training_ids) & validation_set:
            raise AssertionError(f"Fold {fold} train/val overlap")
        if set(training_ids) | validation_set != cv_pool_set:
            raise AssertionError(f"Fold {fold} does not cover the complete CV pool")

        train_path = output_dir / f"train_fold{fold}.txt"
        val_path = output_dir / f"val_fold{fold}.txt"
        write_lines(train_path, training_ids)
        write_lines(val_path, validation_ids)
        fold_summaries.append(
            {
                "fold": fold,
                "train_count": len(training_ids),
                "val_count": len(validation_ids),
                "train_file": train_path.name,
                "val_file": val_path.name,
                "train_sha256": file_sha256(train_path),
                "val_sha256": file_sha256(val_path),
            }
        )

    # Keep legacy names as exact aliases of fold 0 for old checkpoints and docs.
    write_lines(
        output_dir / "train_full.txt",
        [case_id for case_id in cv_pool if case_id not in set(fold_validation[0])],
    )
    write_lines(output_dir / "val_full.txt", fold_validation[0])
    write_lines(output_dir / "test_internal_locked.txt", locked_test)
    write_trainval_mapping(
        trainval_mapping_path,
        rows_by_id,
        fieldnames,
        cv_pool,
    )

    membership_path = output_dir / "five_fold_membership.csv"
    with membership_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nnunet_case_id", "validation_fold", "locked_test"],
            lineterminator="\n",
        )
        writer.writeheader()
        fold_by_id = {
            case_id: fold
            for fold, fold_ids in enumerate(fold_validation)
            for case_id in fold_ids
        }
        for case_id in cv_pool:
            writer.writerow(
                {
                    "nnunet_case_id": case_id,
                    "validation_fold": fold_by_id[case_id],
                    "locked_test": False,
                }
            )
        for case_id in locked_test:
            writer.writerow(
                {
                    "nnunet_case_id": case_id,
                    "validation_fold": "",
                    "locked_test": True,
                }
            )

    summary = {
        "schema_version": "1.0",
        "strategy": "preserve_source_val_as_fold0_partition_source_train_into_folds1_to4",
        "seed": args.seed,
        "source_split_json": str(split_path),
        "source_mapping_csv": str(mapping_path),
        "source_split_sha256": file_sha256(split_path),
        "source_mapping_sha256": file_sha256(mapping_path),
        "cv_pool_count": len(cv_pool),
        "locked_test_count": len(locked_test),
        "validation_coverage_count": len(validation_counts),
        "folds": fold_summaries,
        "membership_csv": membership_path.name,
        "trainval_mapping_csv": str(trainval_mapping_path),
    }
    summary_path = output_dir / "five_fold_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)

    print(f"cv_pool={len(cv_pool)}")
    print(f"locked_test={len(locked_test)}")
    for item in fold_summaries:
        print(f"fold{item['fold']}: train={item['train_count']} val={item['val_count']}")
    print(f"summary={summary_path}")
    print(f"trainval_mapping={trainval_mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
