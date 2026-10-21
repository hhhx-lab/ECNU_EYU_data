#!/usr/bin/env python3
"""Materialize the single locked S2 train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainval-mapping-csv", required=True)
    parser.add_argument(
        "--existing-dataset-dir",
        help="Existing Dataset260 raw directory used to recover the checkpoint ID space",
    )
    parser.add_argument(
        "--existing-validation-dir",
        help="Existing fold_0 validation directory used to recover the locked validation IDs",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Recover the completed fixed split from Dataset260 and fold_0 validation outputs",
    )
    parser.add_argument(
        "--baseline-excluded-source-id",
        action="append",
        default=[],
        help="Source case excluded from the completed baseline; may be repeated",
    )
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-val-count", type=int)
    parser.add_argument("--expected-test-count", type=int)
    parser.add_argument("--require-patient-group-disjoint", action="store_true")
    return parser.parse_args()


def load_split(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    split = data[0] if isinstance(data, list) else data
    if not isinstance(split, dict):
        raise ValueError(f"Expected one split object in {path}.")
    result = {}
    for name in ("train", "val", "test"):
        values = split.get(name, [])
        if not isinstance(values, list):
            raise ValueError(f"Split field '{name}' must be a list in {path}.")
        normalized = [str(value).strip() for value in values if str(value).strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Split field '{name}' contains duplicate IDs.")
        result[name] = normalized
    all_ids = result["train"] + result["val"] + result["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Train, validation, and locked-test IDs overlap.")
    if not result["train"] or not result["val"] or not result["test"]:
        raise ValueError("Fixed train, validation, and locked-test splits must all be non-empty.")
    return result


def load_mapping(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"nnunet_case_id", "source_case_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Mapping lacks required columns {sorted(required)}: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    ids = [row["nnunet_case_id"].strip() for row in rows]
    source_ids = [row["source_case_id"].strip() for row in rows]
    if not rows or any(not value for value in ids + source_ids):
        raise ValueError("Mapping is empty or contains an empty case ID.")
    if len(ids) != len(set(ids)) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Mapping contains duplicate nnU-Net or source case IDs.")
    return rows, fieldnames


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_group(source_case_id: str) -> str:
    prefix, separator, suffix = source_case_id.rpartition("-")
    return prefix if separator and suffix.isdigit() else source_case_id


def existing_dataset_case_ids(existing_dataset_dir: Path) -> list[str]:
    images_dir = existing_dataset_dir / "imagesTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Existing Dataset260 imagesTr is missing: {images_dir}")
    suffix = "_0000.nii.gz"
    case_ids = sorted(
        path.name[: -len(suffix)]
        for path in images_dir.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    )
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Existing Dataset260 has no unique case IDs: {images_dir}")
    return case_ids


def validation_case_ids(existing_validation_dir: Path) -> list[str]:
    if not existing_validation_dir.is_dir():
        raise FileNotFoundError(
            f"Existing fold_0 validation directory is missing: {existing_validation_dir}"
        )

    prediction_ids = {
        path.name[: -len(".nii.gz")]
        for path in existing_validation_dir.glob("*.nii.gz")
        if path.name.endswith(".nii.gz")
    }
    summary_ids: set[str] = set()
    summary_path = existing_validation_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entries = summary.get("metric_per_case", [])
        if not isinstance(entries, list):
            raise ValueError(f"metric_per_case must be a list: {summary_path}")
        for entry in entries:
            prediction_file = entry.get("prediction_file", "") if isinstance(entry, dict) else ""
            if prediction_file:
                name = Path(prediction_file).name
                if not name.endswith(".nii.gz"):
                    raise ValueError(
                        f"Unexpected validation prediction filename in {summary_path}: {name}"
                    )
                summary_ids.add(name[: -len(".nii.gz")])

    if prediction_ids and summary_ids and prediction_ids != summary_ids:
        raise ValueError(
            "Existing validation predictions and summary.json use different ID sets: "
            f"predictions={len(prediction_ids)}, summary={len(summary_ids)}"
        )
    recovered = sorted(prediction_ids or summary_ids)
    if not recovered:
        raise ValueError(
            f"Cannot recover validation IDs from predictions or summary.json: {existing_validation_dir}"
        )
    return recovered


def reconstruct_existing_dataset_mapping(
    case_ids: list[str],
    rows: list[dict[str, str]],
    existing_dataset_dir: Path,
) -> list[dict[str, str]]:
    """Bind old nnU-Net IDs to source cases using Dataset260 symlink targets."""
    by_source_case_id: dict[str, dict[str, str]] = {}
    for row in rows:
        source_case_id = row.get("source_case_id", "").strip()
        if not source_case_id:
            raise ValueError("Mapping contains an empty source_case_id.")
        if source_case_id in by_source_case_id:
            raise ValueError(f"Mapping contains duplicate source_case_id: {source_case_id}")
        by_source_case_id[source_case_id] = row

    reconstructed = []
    for nnunet_case_id in case_ids:
        image_path = (
            existing_dataset_dir / "imagesTr" / f"{nnunet_case_id}_0000.nii.gz"
        )
        if not image_path.is_symlink():
            raise ValueError(
                f"{nnunet_case_id}: expected a source-traceable symlink at {image_path}"
            )

        resolved = image_path.resolve()
        candidates = [
            source_case_id
            for source_case_id in by_source_case_id
            if source_case_id in resolved.parts
            or resolved.name == f"{source_case_id}-t1n.nii.gz"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{nnunet_case_id}: could not uniquely identify the source case from "
                f"symlink target {resolved}; candidates={candidates[:10]}"
            )

        row = dict(by_source_case_id[candidates[0]])
        row["nnunet_case_id"] = nnunet_case_id
        reconstructed.append(row)

    return reconstructed


def build_fixed_split(
    split_json: Path,
    mapping_csv: Path,
    output_dir: Path,
    trainval_mapping_csv: Path,
    reuse_existing: bool = False,
    existing_dataset_dir: Path | None = None,
    existing_validation_dir: Path | None = None,
    baseline_excluded_source_ids: set[str] | None = None,
    expected_train_count: int | None = None,
    expected_val_count: int | None = None,
    expected_test_count: int | None = None,
    require_patient_group_disjoint: bool = False,
) -> dict:
    source = load_split(split_json)
    rows, fieldnames = load_mapping(mapping_csv)
    by_id = {row["nnunet_case_id"].strip(): row for row in rows}
    selected_rows: list[dict[str, str]]

    missing_val = [case_id for case_id in source["val"] if case_id not in by_id]
    missing_test = [case_id for case_id in source["test"] if case_id not in by_id]
    if not reuse_existing and (missing_val or missing_test):
        raise ValueError(
            "Fixed validation or locked-test cases are missing from the current mapping: "
            f"val={missing_val[:10]}, test={missing_test[:10]}"
        )

    split_source = "g2_split_json"
    if reuse_existing:
        split_source = "existing_dataset260_and_fold0_validation"
        if existing_dataset_dir is None:
            raise ValueError("--existing-dataset-dir is required with --reuse-existing.")
        if existing_validation_dir is None:
            raise ValueError("--existing-validation-dir is required with --reuse-existing.")
        dataset_ids = existing_dataset_case_ids(existing_dataset_dir)
        val_ids = validation_case_ids(existing_validation_dir)
        unknown_val = sorted(set(val_ids) - set(dataset_ids))
        if unknown_val:
            raise ValueError(
                "Existing fold_0 validation IDs are absent from Dataset260: "
                f"{unknown_val[:10]}"
            )
        train_ids = sorted(set(dataset_ids) - set(val_ids))
        test_ids: list[str] = []
        missing_train = []
    else:
        train_ids = [case_id for case_id in source["train"] if case_id in by_id]
        val_ids = list(source["val"])
        test_ids = list(source["test"])
        missing_train = [case_id for case_id in source["train"] if case_id not in by_id]

    if not train_ids or not val_ids or (not reuse_existing and not test_ids):
        raise ValueError("Effective train, validation, and locked test must all be non-empty.")
    if set(train_ids) & set(val_ids):
        raise ValueError("Effective training and validation IDs overlap.")

    dataset_mapping_checked = False
    mapping_reconstructed = False
    validation_recovered = False
    if reuse_existing:
        assert existing_dataset_dir is not None
        selected_rows = reconstruct_existing_dataset_mapping(
            train_ids + val_ids, rows, existing_dataset_dir
        )
        dataset_mapping_checked = True
        mapping_reconstructed = True
        validation_recovered = True
    else:
        selected_rows = [by_id[case_id] for case_id in train_ids + val_ids]

    train_rows = selected_rows[: len(train_ids)]
    val_rows = selected_rows[len(train_ids) :]
    train_source_ids = {row["source_case_id"].strip() for row in train_rows}
    val_source_ids = {row["source_case_id"].strip() for row in val_rows}
    by_source_case_id = {row["source_case_id"].strip(): row for row in rows}
    excluded_source_ids = {
        value.strip() for value in (baseline_excluded_source_ids or set()) if value.strip()
    }
    if reuse_existing:
        baseline_source_universe = set(by_source_case_id) - excluded_source_ids
        test_source_ids = baseline_source_universe - train_source_ids - val_source_ids
        recovered_test_count = expected_test_count or len(source["test"])
        if len(test_source_ids) != recovered_test_count:
            raise ValueError(
                "Cannot recover the completed baseline locked test as the source-case "
                "complement of Dataset260: "
                f"expected={recovered_test_count}, recovered={len(test_source_ids)}, "
                f"excluded={sorted(excluded_source_ids)}"
            )
        test_rows = [by_source_case_id[source_id] for source_id in sorted(test_source_ids)]
        test_ids = [row["nnunet_case_id"].strip() for row in test_rows]
    else:
        test_rows = [by_id[case_id] for case_id in test_ids]
        test_source_ids = {row["source_case_id"].strip() for row in test_rows}
    if not test_ids or not test_source_ids:
        raise ValueError("Recovered locked test is empty.")

    actual_counts = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test_internal_locked": len(test_ids),
    }
    expected_counts = {
        "train": expected_train_count,
        "val": expected_val_count,
        "test_internal_locked": expected_test_count,
    }
    count_mismatches = {
        name: {"expected": expected, "actual": actual_counts[name]}
        for name, expected in expected_counts.items()
        if expected is not None and actual_counts[name] != expected
    }
    if count_mismatches:
        raise ValueError(f"Fixed split count contract failed: {count_mismatches}")
    source_overlap = (
        (train_source_ids & val_source_ids)
        | (train_source_ids & test_source_ids)
        | (val_source_ids & test_source_ids)
    )
    if source_overlap:
        raise ValueError(
            "Train, validation, and locked test overlap in source-case identity: "
            f"{sorted(source_overlap)[:10]}"
        )

    train_groups = {patient_group(source_id) for source_id in train_source_ids}
    val_groups = {patient_group(source_id) for source_id in val_source_ids}
    test_groups = {patient_group(source_id) for source_id in test_source_ids}
    patient_group_overlap = (
        (train_groups & val_groups)
        | (train_groups & test_groups)
        | (val_groups & test_groups)
    )
    if require_patient_group_disjoint and patient_group_overlap:
        raise ValueError(
            "Patient groups cross train/validation/locked-test boundaries: "
            f"{sorted(patient_group_overlap)[:10]}"
        )

    if not reuse_existing:
        expected_val_source_ids = {
            by_id[case_id]["source_case_id"].strip() for case_id in source["val"]
        }
        if val_source_ids != expected_val_source_ids:
            raise ValueError("Fresh fixed validation source identities are inconsistent.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_fixed.txt"
    val_file = output_dir / "val_fixed.txt"
    test_file = output_dir / "test_internal_locked.txt"
    test_source_file = output_dir / "test_internal_locked_source_ids.txt"
    write_lines(train_file, train_ids)
    write_lines(val_file, val_ids)
    write_lines(test_file, test_ids)
    write_lines(test_source_file, sorted(test_source_ids))

    # Keep these aliases for existing server checkpoints and adjacent scripts.
    write_lines(output_dir / "train_full.txt", train_ids)
    write_lines(output_dir / "val_full.txt", val_ids)

    trainval_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with trainval_mapping_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    membership_path = output_dir / "fixed_split_membership.csv"
    trainval_id_space = "dataset260_historical" if reuse_existing else "current_g2_mapping"
    with membership_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("nnunet_case_id", "source_case_id", "split", "id_space"),
        )
        writer.writeheader()
        for split_name, values, split_rows, id_space in (
            ("train", train_ids, train_rows, trainval_id_space),
            ("val", val_ids, val_rows, trainval_id_space),
            ("test_internal_locked", test_ids, test_rows, "current_g2_mapping"),
        ):
            writer.writerows(
                {
                    "nnunet_case_id": case_id,
                    "source_case_id": row["source_case_id"].strip(),
                    "split": split_name,
                    "id_space": id_space,
                }
                for case_id, row in zip(values, split_rows)
            )

    summary = {
        "strategy": "single_fixed_train_val_with_locked_internal_test",
        "split_source": split_source,
        "existing_dataset_mapping_checked": dataset_mapping_checked,
        "mapping_reconstructed_from_dataset_symlinks": mapping_reconstructed,
        "validation_recovered_from_fold0_outputs": validation_recovered,
        "locked_test_recovered_as_source_complement": reuse_existing,
        "baseline_excluded_source_ids": sorted(excluded_source_ids),
        "source_split_json": str(split_json.resolve()),
        "mapping_csv": str(mapping_csv.resolve()),
        "source_mapping_sha256": sha256(mapping_csv),
        "trainval_mapping_sha256": sha256(trainval_mapping_csv),
        "source_counts": {name: len(values) for name, values in source.items()},
        "expected_counts": expected_counts,
        "effective_counts": actual_counts,
        "missing_train_ids_excluded_by_mapping": missing_train,
        "source_identity_disjoint": True,
        "patient_group_disjoint": not patient_group_overlap,
        "patient_group_overlap": sorted(patient_group_overlap),
        "files": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (
                train_file,
                val_file,
                test_file,
                test_source_file,
                trainval_mapping_csv,
                membership_path,
            )
        },
    }
    summary_path = output_dir / "fixed_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_fixed_split(
        Path(args.split_json),
        Path(args.mapping_csv),
        Path(args.output_dir),
        Path(args.trainval_mapping_csv),
        args.reuse_existing,
        Path(args.existing_dataset_dir) if args.existing_dataset_dir else None,
        Path(args.existing_validation_dir) if args.existing_validation_dir else None,
        set(args.baseline_excluded_source_id),
        args.expected_train_count,
        args.expected_val_count,
        args.expected_test_count,
        args.require_patient_group_disjoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
