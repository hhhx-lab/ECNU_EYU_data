#!/usr/bin/env python3
"""Build G2 master and real-only mappings directly from raw BraTS-MET data.

The master mapping retains physically complete fake/broken-T2W cases because
G1 V3 must repair them. The derived real-only mapping contains authentic T2W
cases only and is the safe baseline input for S1-S5.
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
    filter_split,
    patient_group,
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
    "patient_group",
    "t2w_status",
    "eligible_for_realonly",
    "completion_required",
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

V2_SOURCE_FIELDNAMES = [
    "source_case_id",
    "patient_group",
    "nnunet_case_id",
    "split",
    "t2w_status",
    "allowed_as_v2_source",
    "t1n_path",
    "t1c_path",
    "t2w_path",
    "t2f_path",
    "seg_path",
    "label_source",
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
    default_fake_t2w = results_root / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"

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
    parser.add_argument("--mapping-csv", default="", help="Derived authentic-T2W real-only mapping.")
    parser.add_argument("--master-mapping-csv", default="", help="All physically complete cases, including completion targets.")
    parser.add_argument("--skipped-csv", default="")
    parser.add_argument("--fake-t2w-cases", default=str(default_fake_t2w))
    parser.add_argument(
        "--allow-missing-fake-t2w-list",
        action="store_true",
        help="Diagnostic only. Without the official fake/broken list every T2W is treated as authentic.",
    )
    parser.add_argument(
        "--exclude-ids",
        action="append",
        default=[],
        help="Source case_id(s) to exclude (full folder name, e.g. BraTS-MET-01094-002). Repeatable.",
    )
    parser.add_argument("--split-json", default="", help="Derived real-only split JSON.")
    parser.add_argument("--membership-csv", default="", help="Derived real-only membership CSV.")
    parser.add_argument("--master-split-json", default="")
    parser.add_argument("--master-membership-csv", default="")
    parser.add_argument("--v2-source-manifest", default="")
    parser.add_argument("--val-fraction-of-train-pool", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
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


def load_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("case_id") or row.get("source_case_id") or row.get("id") or "").strip()
            for row in rows
            if str(row.get("case_id") or row.get("source_case_id") or row.get("id") or "").strip()
        }


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
    exclude_ids: set[str] | None = None,
    fake_t2w_case_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    exclude_ids = exclude_ids or set()
    fake_t2w_case_ids = fake_t2w_case_ids or set()
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
        if case_id in exclude_ids:
            skipped.append({
                "source_case_id": case_id,
                "case_dir": display_path(case_dir, project_root),
                "reason": "excluded_by_request",
                "missing_files": "",
                "raw_data_root": display_path(raw_root, project_root),
            })
            continue
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
            "patient_group": patient_group(case_id),
            "t2w_status": "fake_or_broken" if case_id in fake_t2w_case_ids else "authentic",
            "eligible_for_realonly": str(case_id not in fake_t2w_case_ids),
            "completion_required": str(case_id in fake_t2w_case_ids),
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


def build_v2_source_rows(
    mapping_rows: list[dict[str, str]],
    master_split: dict[str, object],
) -> list[dict[str, str]]:
    split_by_id = {
        str(nnunet_id): split_name
        for split_name in ("train", "val", "test")
        for nnunet_id in master_split[split_name]  # type: ignore[index]
    }
    rows: list[dict[str, str]] = []
    for row in mapping_rows:
        split_name = split_by_id[row["nnunet_case_id"]]
        allowed = split_name == "train" and row["eligible_for_realonly"] == "True"
        rows.append({
            "source_case_id": row["source_case_id"],
            "patient_group": row["patient_group"],
            "nnunet_case_id": row["nnunet_case_id"],
            "split": split_name,
            "t2w_status": row["t2w_status"],
            "allowed_as_v2_source": str(allowed),
            "t1n_path": row["t1n_source_path"],
            "t1c_path": row["t1c_source_path"],
            "t2w_path": row["t2w_source_path"],
            "t2f_path": row["t2f_source_path"],
            "seg_path": row["seg_source_path"],
            "label_source": row["label_source"],
        })
    return rows


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    data_roots = [Path(root).expanduser().resolve() for root in args.data_root]
    corrected_label_roots = [Path(root).expanduser().resolve() for root in args.corrected_labels_root]

    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    master_mapping_csv = Path(args.master_mapping_csv) if args.master_mapping_csv else results_root / "manifests" / "nnunet_case_mapping_master.csv"
    skipped_csv = Path(args.skipped_csv) if args.skipped_csv else results_root / "manifests" / "realonly_skipped_incomplete_cases.csv"
    split_json = Path(args.split_json) if args.split_json else results_root / "splits" / "splits_final_train_val_test.json"
    membership_csv = Path(args.membership_csv) if args.membership_csv else results_root / "splits" / "splits_final_train_val_test_membership.csv"
    master_split_json = Path(args.master_split_json) if args.master_split_json else results_root / "splits" / "splits_master_train_val_test.json"
    master_membership_csv = Path(args.master_membership_csv) if args.master_membership_csv else results_root / "splits" / "splits_master_train_val_test_membership.csv"
    v2_source_manifest = Path(args.v2_source_manifest) if args.v2_source_manifest else results_root / "manifests" / "g1_v2_source_manifest.csv"
    fake_t2w_path = Path(args.fake_t2w_cases).expanduser().resolve()
    if not fake_t2w_path.exists() and not args.allow_missing_fake_t2w_list:
        raise SystemExit(
            f"fake/broken T2W list not found: {fake_t2w_path}. "
            "Restore the official G2 list or use --allow-missing-fake-t2w-list for diagnostics only."
        )
    fake_t2w_case_ids = load_case_ids(fake_t2w_path)

    missing_roots = [root for root in data_roots if not root.exists()]
    for root in missing_roots:
        print(f"warning: raw data root does not exist and will be ignored: {root}", file=sys.stderr)
    missing_corrected_roots = [root for root in corrected_label_roots if not root.exists()]
    for root in missing_corrected_roots:
        print(f"warning: corrected labels root does not exist and will be ignored: {root}", file=sys.stderr)
    corrected_label_roots = [root for root in corrected_label_roots if root.exists()]

    mapping_rows, skipped_rows = build_mapping_rows(
        data_roots,
        project_root,
        corrected_label_roots,
        exclude_ids=set(args.exclude_ids),
        fake_t2w_case_ids=fake_t2w_case_ids,
    )
    if not mapping_rows:
        message = "no complete BraTS-MET cases found; expected t1n/t1c/t2w/t2f/seg for each training case"
        if args.fail_if_no_valid_cases:
            raise SystemExit(message)
        print(f"warning: {message}", file=sys.stderr)

    eligible_rows = [row for row in mapping_rows if row["eligible_for_realonly"] == "True"]
    write_csv(master_mapping_csv, MAPPING_FIELDNAMES, mapping_rows)
    write_csv(mapping_csv, MAPPING_FIELDNAMES, eligible_rows)
    write_csv(skipped_csv, SKIPPED_FIELDNAMES, skipped_rows)

    if mapping_rows:
        authentic_ids = {row["source_case_id"] for row in eligible_rows}
        master_split = create_train_val_test_split(
            mapping_rows,
            base_split=None,
            val_fraction_of_train_pool=args.val_fraction_of_train_pool,
            test_fraction=args.test_fraction,
            seed=args.seed,
            anchor_case_ids=authentic_ids,
        )
        master_split["mapping_csv"] = display_path(master_mapping_csv, results_root)
        write_split_outputs(master_split, mapping_rows, master_split_json, master_membership_csv)
        write_csv(v2_source_manifest, V2_SOURCE_FIELDNAMES, build_v2_source_rows(mapping_rows, master_split))

        eligible_nnunet_ids = {row["nnunet_case_id"] for row in eligible_rows}
        split = filter_split(master_split, eligible_nnunet_ids, "realonly_patient_group_train_val_test")
        split["mapping_csv"] = display_path(mapping_csv, results_root)
        write_split_outputs(split, eligible_rows, split_json, membership_csv)
        counts = split["counts"]
        print(f"train={counts['train']}")  # type: ignore[index]
        print(f"val={counts['val']}")  # type: ignore[index]
        print(f"test={counts['test']}")  # type: ignore[index]
        print(f"master_counts={master_split['counts']}")

    print(f"master_cases={len(mapping_rows)}")
    print(f"realonly_eligible_cases={len(eligible_rows)}")
    print(f"fake_or_broken_t2w_cases={len(mapping_rows) - len(eligible_rows)}")
    print(f"skipped_cases={len(skipped_rows)}")
    print(f"mapping_csv={mapping_csv}")
    print(f"master_mapping_csv={master_mapping_csv}")
    print(f"skipped_csv={skipped_csv}")
    print(f"split_json={split_json}")
    print(f"membership_csv={membership_csv}")
    print(f"master_split_json={master_split_json}")
    print(f"master_membership_csv={master_membership_csv}")
    print(f"v2_source_manifest={v2_source_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
