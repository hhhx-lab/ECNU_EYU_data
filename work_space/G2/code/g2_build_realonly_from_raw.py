#!/usr/bin/env python3
"""Build real-only G2 mapping/split artifacts directly from raw BraTS-MET data.

This is not a QC gate. It only creates the lightweight artifacts needed by
S1/S2/S3 from raw case folders and skips cases that are incomplete, especially
cases without an available T2W file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g2_create_train_val_test_split import (
    DEFAULT_SEED,
    create_train_val_test_split,
    write_split_outputs,
)


REQUIRED_SUFFIXES = {
    "t1n": "t1n_source_path",
    "t1c": "t1c_source_path",
    "t2w": "t2w_source_path",
    "t2f": "t2f_source_path",
    "seg": "seg_source_path",
}

ALLOWED_LABEL_VALUES = {0, 1, 2, 3, 4}

MAPPING_FIELDNAMES = [
    "nnunet_case_id",
    "source_case_id",
    "t1n_source_path",
    "t1c_source_path",
    "t2w_source_path",
    "t2f_source_path",
    "seg_source_path",
    "label_source",
    "materialization_status",
    "raw_data_root",
]

SKIPPED_FIELDNAMES = [
    "source_case_id",
    "case_dir",
    "reason",
    "missing_files",
    "raw_data_root",
]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    results_root = project_root / "work_space" / "G2" / "results"
    default_data_root = project_root / "work_space" / "G1" / "data" / "raw" / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
    default_corrected_root = (
        project_root
        / "work_space"
        / "G1"
        / "data"
        / "raw"
        / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
    )

    parser = argparse.ArgumentParser(
        description="Scan raw BraTS-MET data and write G2 real-only mapping/split artifacts."
    )
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Raw data root containing BraTS-MET case folders. Can be passed more than once.",
    )
    parser.add_argument(
        "--corrected-labels-root",
        action="append",
        default=[],
        help="Directory containing corrected <case_id>-seg.nii.gz files. Can be passed more than once.",
    )
    parser.add_argument("--results-root", default=str(results_root))
    parser.add_argument("--mapping-csv", default="")
    parser.add_argument("--skipped-csv", default="")
    parser.add_argument("--split-json", default="")
    parser.add_argument("--membership-csv", default="")
    parser.add_argument("--val-fraction-of-train-pool", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--fail-if-no-valid-cases",
        action="store_true",
        help="Exit non-zero when no complete cases are found.",
    )
    args = parser.parse_args()
    if not args.data_root:
        args.data_root = [str(default_data_root)]
    if not args.corrected_labels_root:
        args.corrected_labels_root = [str(default_corrected_root)]
    return args


def display_path(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def iter_case_dirs(data_roots: list[Path]) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for root in data_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            path = Path(dirpath)
            if path.name.startswith("BraTS-MET-"):
                found.append((path, root))
                dirnames[:] = []
    return sorted(found, key=lambda item: item[0].name)


def inspect_case(case_dir: Path) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    missing: list[str] = []
    for suffix in REQUIRED_SUFFIXES:
        file_path = case_dir / f"{case_dir.name}-{suffix}.nii.gz"
        if file_path.exists():
            files[suffix] = file_path
        else:
            missing.append(file_path.name)
    return files, missing


def normalized_label_values(values: object) -> set[int | float]:
    normalized: set[int | float] = set()
    for value in values:  # type: ignore[union-attr]
        numeric = float(value)
        if numeric.is_integer():
            normalized.add(int(numeric))
        else:
            normalized.add(numeric)
    return normalized


def read_label_values(path: Path) -> set[int | float]:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        arr = np.asanyarray(nib.load(str(path)).dataobj)
        return normalized_label_values(np.unique(arr))
    except ModuleNotFoundError:
        try:
            import SimpleITK as sitk  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            return normalized_label_values(np.unique(arr))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Label value checking requires nibabel or SimpleITK in the active environment."
            ) from exc


def illegal_label_values(values: set[int | float]) -> set[int | float]:
    return set(values) - ALLOWED_LABEL_VALUES


def format_label_values(values: set[int | float]) -> str:
    return ";".join(str(value) for value in sorted(values))


def find_corrected_seg(case_id: str, corrected_label_roots: list[Path]) -> Path | None:
    file_name = f"{case_id}-seg.nii.gz"
    for root in corrected_label_roots:
        direct = root / file_name
        if direct.exists():
            return direct
        nested = root / case_id / file_name
        if nested.exists():
            return nested
    return None


def select_seg_source(
    case_id: str,
    raw_seg_path: Path,
    corrected_label_roots: list[Path],
    label_value_reader=read_label_values,
) -> tuple[Path | None, str, str]:
    corrected_seg = find_corrected_seg(case_id, corrected_label_roots)
    if corrected_seg is not None:
        corrected_values = label_value_reader(corrected_seg)
        corrected_illegal = illegal_label_values(corrected_values)
        if not corrected_illegal:
            return corrected_seg, "corrected", ""

    raw_values = label_value_reader(raw_seg_path)
    raw_illegal = illegal_label_values(raw_values)
    if not raw_illegal:
        return raw_seg_path, "raw", ""

    details = f"illegal_label_values:{format_label_values(raw_illegal)}"
    if corrected_seg is not None:
        corrected_values = label_value_reader(corrected_seg)
        corrected_illegal = illegal_label_values(corrected_values)
        if corrected_illegal:
            details += f";corrected_illegal_label_values:{format_label_values(corrected_illegal)}"
    return None, "", details


def build_mapping_rows(
    data_roots: list[Path],
    project_root: Path,
    corrected_label_roots: list[Path] | None = None,
    label_value_reader=read_label_values,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    seen_cases: set[str] = set()
    valid_cases: list[tuple[str, Path, Path, dict[str, Path], Path, str]] = []
    corrected_label_roots = corrected_label_roots or []

    for case_dir, raw_root in iter_case_dirs(data_roots):
        case_id = case_dir.name
        if case_id in seen_cases:
            skipped.append({
                "source_case_id": case_id,
                "case_dir": display_path(case_dir, project_root),
                "reason": "duplicate_case_id",
                "missing_files": "",
                "raw_data_root": display_path(raw_root, project_root),
            })
            continue
        seen_cases.add(case_id)
        files, missing = inspect_case(case_dir)
        if missing:
            skipped.append({
                "source_case_id": case_id,
                "case_dir": display_path(case_dir, project_root),
                "reason": "missing_required_files",
                "missing_files": ";".join(missing),
                "raw_data_root": display_path(raw_root, project_root),
            })
            continue
        seg_source_path, label_source, skip_reason = select_seg_source(
            case_id,
            files["seg"],
            corrected_label_roots,
            label_value_reader=label_value_reader,
        )
        if seg_source_path is None:
            skipped.append({
                "source_case_id": case_id,
                "case_dir": display_path(case_dir, project_root),
                "reason": "illegal_label_values",
                "missing_files": skip_reason,
                "raw_data_root": display_path(raw_root, project_root),
            })
            continue
        files["seg"] = seg_source_path
        valid_cases.append((case_id, case_dir, raw_root, files, seg_source_path, label_source))

    for idx, (case_id, _case_dir, raw_root, files, _seg_source_path, label_source) in enumerate(sorted(valid_cases), start=1):
        nnunet_case_id = f"BraTSMET_{idx:06d}"
        row = {
            "nnunet_case_id": nnunet_case_id,
            "source_case_id": case_id,
            "label_source": label_source,
            "materialization_status": "deferred_symlink_on_training_machine",
            "raw_data_root": display_path(raw_root, project_root),
        }
        for suffix, column in REQUIRED_SUFFIXES.items():
            row[column] = display_path(files[suffix], project_root)
        rows.append(row)

    return rows, skipped


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    data_roots = [Path(root).expanduser().resolve() for root in args.data_root]
    corrected_label_roots = [Path(root).expanduser().resolve() for root in args.corrected_labels_root]

    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    skipped_csv = Path(args.skipped_csv) if args.skipped_csv else results_root / "manifests" / "realonly_skipped_incomplete_cases.csv"
    split_json = Path(args.split_json) if args.split_json else results_root / "splits" / "splits_final_train_val_test.json"
    membership_csv = Path(args.membership_csv) if args.membership_csv else results_root / "splits" / "splits_final_train_val_test_membership.csv"

    missing_roots = [root for root in data_roots if not root.exists()]
    for root in missing_roots:
        print(f"warning: raw data root does not exist and will be ignored: {root}", file=sys.stderr)
    missing_corrected_roots = [root for root in corrected_label_roots if not root.exists()]
    for root in missing_corrected_roots:
        print(f"warning: corrected labels root does not exist and will be ignored: {root}", file=sys.stderr)
    corrected_label_roots = [root for root in corrected_label_roots if root.exists()]

    mapping_rows, skipped_rows = build_mapping_rows(data_roots, project_root, corrected_label_roots)
    if not mapping_rows:
        message = "no complete BraTS-MET cases found; expected t1n/t1c/t2w/t2f/seg for each training case"
        if args.fail_if_no_valid_cases:
            raise SystemExit(message)
        print(f"warning: {message}", file=sys.stderr)

    write_csv(mapping_csv, MAPPING_FIELDNAMES, mapping_rows)
    write_csv(skipped_csv, SKIPPED_FIELDNAMES, skipped_rows)

    if mapping_rows:
        split = create_train_val_test_split(
            mapping_rows,
            base_split=None,
            val_fraction_of_train_pool=args.val_fraction_of_train_pool,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        split["source_split_json"] = ""
        split["mapping_csv"] = display_path(mapping_csv, results_root)
        write_split_outputs(split, mapping_rows, split_json, membership_csv)
        counts = split["counts"]
        print(f"train={counts['train']}")  # type: ignore[index]
        print(f"val={counts['val']}")  # type: ignore[index]
        print(f"test={counts['test']}")  # type: ignore[index]

    print(f"valid_cases={len(mapping_rows)}")
    print(f"skipped_cases={len(skipped_rows)}")
    print(f"mapping_csv={mapping_csv}")
    print(f"skipped_csv={skipped_csv}")
    print(f"split_json={split_json}")
    print(f"membership_csv={membership_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
