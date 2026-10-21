#!/usr/bin/env python3
"""Materialize approved G2 data for nnU-Net and case-folder consumers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_FAKE_T2W = DEFAULT_RESULTS_ROOT / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"
CHANNEL_ORDERS = {"g2_official": ["t1n", "t1c", "t2w", "t2f"]}
LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_fake_t2w_cases(path: Path) -> set[str]:
    cases = set()
    for row in read_csv(path):
        case_id = row.get("case_id") or row.get("source_case_id") or row.get("id")
        if case_id:
            cases.add(case_id.strip())
    return cases


def load_split(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    split = data[0] if isinstance(data, list) else data
    if not isinstance(split, dict) or not all(name in split for name in ("train", "val", "test")):
        raise ValueError(f"invalid master split: {path}")
    return split


def source_for(row: dict[str, str], modality: str) -> str | None:
    if modality == "seg":
        candidates = ("seg_source_path", "normalized_seg_path", "raw_seg_path")
    else:
        candidates = (
            f"{modality}_source_path",
            f"normalized_{modality}_path",
            f"raw_{modality}_path",
        )
    values = [row.get(column, "") for column in candidates if row.get(column, "")]
    for value in values:
        resolved = resolve_path(value)
        if resolved is not None and resolved.exists():
            return value
    return values[0] if values else None


def is_completion_row(row: dict[str, str]) -> bool:
    return boolish(row.get("source_completion_mode", "")) or row.get("label_kind") == "completion"


def approved_for_training(row: dict[str, str]) -> bool:
    return boolish(row.get("accepted_for_training", ""))


def approved_for_evaluation(row: dict[str, str]) -> bool:
    return boolish(row.get("accepted_for_evaluation", ""))


def read_synthetic_manifests(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_run_raw_ids: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_csv(path):
            raw_id = row.get("synthetic_raw_id", "").strip()
            if not raw_id:
                raise ValueError(f"synthetic manifest contains an empty synthetic_raw_id: {path}")
            run_id = row.get("generation_run_id", "").strip()
            if not run_id:
                raise ValueError(f"synthetic manifest row is missing generation_run_id: {path}:{raw_id}")
            run_raw_id = (run_id, raw_id)
            if run_raw_id in seen_run_raw_ids:
                raise ValueError(f"duplicate synthetic row across manifests: {run_id}/{raw_id}")
            seen_run_raw_ids.add(run_raw_id)
            row["g2_input_manifest"] = str(path)
            rows.append(row)
    return rows


def apply_completion_root(
    rows: list[dict[str, str]],
    completion_root: Path | None,
) -> int:
    """Override completion T2W paths with a server-local V3 output root."""
    if completion_root is None:
        return 0
    root = completion_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"completion root does not exist: {root}")

    updated = 0
    for row in rows:
        if not is_completion_row(row):
            continue
        source_case_id = row.get("source_case_id", "").strip()
        if not source_case_id:
            raise ValueError("completion manifest row is missing source_case_id")
        expected = root / source_case_id / f"{source_case_id}-t2w.nii.gz"
        if not expected.is_file():
            raise FileNotFoundError(
                f"completion T2W is missing under --completion-root: {expected}"
            )
        row["t2w_source_path"] = str(expected)
        updated += 1
    return updated


def apply_real_data_root(
    rows: list[dict[str, str]],
    real_data_root: Path | None,
) -> int:
    """Override real-case paths for the ECNU flat per-case data layout."""
    if real_data_root is None:
        return 0
    root = real_data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"real data root does not exist: {root}")

    updated = 0
    for row in rows:
        source_case_id = row.get("source_case_id", "").strip()
        if not source_case_id:
            raise ValueError("real mapping row is missing source_case_id")
        case_dir = root / source_case_id
        for modality in CHANNEL_ORDERS["g2_official"]:
            candidates = (
                case_dir / f"{modality}.nii.gz",
                case_dir / f"{source_case_id}-{modality}.nii.gz",
            )
            expected = next((path for path in candidates if path.is_file()), None)
            if expected is None:
                raise FileNotFoundError(
                    f"real modality is missing under --real-data-root: {candidates}"
                )
            row[f"{modality}_source_path"] = str(expected)
        if row.get("label_source", "").strip().lower() != "corrected":
            seg_candidates = (
                case_dir / "seg.nii.gz",
                case_dir / f"{source_case_id}-seg.nii.gz",
            )
            expected_seg = next(
                (path for path in seg_candidates if path.is_file()), None
            )
            if expected_seg is None:
                raise FileNotFoundError(
                    f"real segmentation is missing under --real-data-root: {seg_candidates}"
                )
            row["seg_source_path"] = str(expected_seg)
        updated += 1
    return updated


def select_completion_replacements(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    replacements: dict[str, dict[str, str]] = {}
    for row in rows:
        if not is_completion_row(row):
            continue
        if not (approved_for_training(row) or approved_for_evaluation(row)):
            continue
        source_case_id = row.get("source_case_id", "").strip()
        if not source_case_id or not source_for(row, "t2w"):
            continue
        if source_case_id in replacements:
            raise ValueError(f"multiple approved completion T2W files for {source_case_id}")
        replacements[source_case_id] = row
    return replacements


def output_is_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_output(path: Path, clean: bool) -> None:
    if output_is_nonempty(path):
        if not clean:
            raise FileExistsError(f"output is not empty; pass --clean-output explicitly: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, mode: str) -> str:
    if not src.is_file():
        raise FileNotFoundError(src)
    if mode == "manifest-only":
        return "planned"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported materialization mode: {mode}")
    return mode


def selected_path(
    real_row: dict[str, str],
    modality: str,
    completion_row: dict[str, str] | None,
) -> str | None:
    if modality == "t2w" and completion_row is not None:
        return source_for(completion_row, "t2w")
    return source_for(real_row, modality)


def build_case_specs(
    real_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    fake_t2w_cases: set[str],
    profile: str,
    allow_incomplete_completion: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    replacements = select_completion_replacements(synthetic_rows)
    specs: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    used_ids: set[str] = set()

    for row in real_rows:
        nnunet_id = (row.get("nnunet_case_id") or "").strip()
        source_case_id = (row.get("source_case_id") or "").strip()
        if not nnunet_id or not source_case_id:
            raise ValueError("real mapping contains an empty nnunet_case_id/source_case_id")
        if nnunet_id in used_ids:
            raise ValueError(f"duplicate nnU-Net ID in real mapping: {nnunet_id}")
        replacement = replacements.get(source_case_id)
        is_fake = source_case_id in fake_t2w_cases or row.get("t2w_status") == "fake_or_broken"
        if is_fake and profile == "real-only":
            stats["skipped_fake_realonly"] += 1
            continue
        if is_fake and replacement is None:
            if not allow_incomplete_completion:
                raise ValueError(f"fake/broken T2W has no approved completion: {source_case_id}")
            stats["skipped_missing_completion"] += 1
            continue
        paths = {
            modality: selected_path(row, modality, replacement)
            for modality in (*CHANNEL_ORDERS["g2_official"], "seg")
        }
        missing = [modality for modality, value in paths.items() if not value]
        if missing:
            raise ValueError(f"real case {source_case_id} is missing source paths: {missing}")
        specs.append({
            "nnunet_case_id": nnunet_id,
            "case_folder_id": source_case_id,
            "source_case_id": source_case_id,
            "row_type": "real_with_completion_t2w" if replacement else "real",
            "paths": paths,
            "completion_raw_id": replacement.get("synthetic_raw_id", "") if replacement else "",
            "completion_approved_for_training": approved_for_training(replacement) if replacement else False,
            "completion_approved_for_evaluation": approved_for_evaluation(replacement) if replacement else False,
        })
        used_ids.add(nnunet_id)
        stats["real_cases"] += 1
        if replacement:
            stats["completion_replacements"] += 1

    if profile == "real-synth":
        for row in synthetic_rows:
            if is_completion_row(row) or not approved_for_training(row):
                continue
            if row.get("source_split") != "train":
                raise ValueError(f"approved augmentation source is not train: {row.get('synthetic_raw_id')}")
            nnunet_id = (row.get("nnunet_case_id") or "").strip()
            final_id = (row.get("synthetic_final_id") or "").strip()
            source_case_id = (row.get("source_case_id") or "").strip()
            if not nnunet_id or not final_id or not source_case_id:
                raise ValueError("approved augmentation is missing stable IDs/source_case_id")
            if nnunet_id in used_ids:
                raise ValueError(f"synthetic nnU-Net ID collides with another case: {nnunet_id}")
            paths = {
                modality: source_for(row, modality)
                for modality in (*CHANNEL_ORDERS["g2_official"], "seg")
            }
            missing = [modality for modality, value in paths.items() if not value]
            if missing:
                raise ValueError(f"synthetic case {final_id} is missing paths: {missing}")
            specs.append({
                "nnunet_case_id": nnunet_id,
                "case_folder_id": final_id,
                "source_case_id": source_case_id,
                "row_type": "synthetic_augmentation",
                "paths": paths,
                "completion_raw_id": "",
                "completion_approved_for_training": False,
                "completion_approved_for_evaluation": False,
            })
            used_ids.add(nnunet_id)
            stats["synthetic_augmentation_cases"] += 1

    stats["included_cases"] = len(specs)
    return specs, dict(stats)


def assign_spec_splits(
    specs: list[dict[str, object]],
    master: dict[str, object],
) -> None:
    split_by_id = {
        str(case_id): split_name
        for split_name in ("train", "val", "test")
        for case_id in master[split_name]  # type: ignore[index]
    }
    source_split_by_case: dict[str, str] = {}
    for spec in specs:
        if spec["row_type"] == "synthetic_augmentation":
            continue
        nnunet_id = str(spec["nnunet_case_id"])
        if nnunet_id not in split_by_id:
            raise ValueError(f"real case is absent from master split: {nnunet_id}")
        split_name = split_by_id[nnunet_id]
        spec["split"] = split_name
        source_split_by_case[str(spec["source_case_id"])] = split_name
        if spec["row_type"] == "real_with_completion_t2w":
            if split_name == "train" and not bool(spec["completion_approved_for_training"]):
                raise ValueError(
                    f"train completion lacks training approval: {spec['source_case_id']}"
                )
            if split_name in {"val", "test"} and not bool(spec["completion_approved_for_evaluation"]):
                raise ValueError(
                    f"{split_name} completion lacks evaluation approval: {spec['source_case_id']}"
                )

    for spec in specs:
        if spec["row_type"] != "synthetic_augmentation":
            continue
        source_case_id = str(spec["source_case_id"])
        actual_source_split = source_split_by_case.get(source_case_id)
        if actual_source_split != "train":
            raise ValueError(
                f"synthetic augmentation source is not master train: "
                f"{source_case_id} ({actual_source_split or 'missing'})"
            )
        spec["split"] = "train"


def materialize_specs(
    specs: list[dict[str, object]],
    dataset_dir: Path,
    case_folder_root: Path,
    mode: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    modalities = CHANNEL_ORDERS["g2_official"]
    for spec in specs:
        nnunet_id = str(spec["nnunet_case_id"])
        folder_id = str(spec["case_folder_id"])
        split_name = str(spec.get("split", ""))
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"case has no valid fixed split: {folder_id}")
        image_partition = "imagesTs" if split_name == "test" else "imagesTr"
        label_partition = "labelsTs" if split_name == "test" else "labelsTr"
        paths = spec["paths"]  # type: ignore[assignment]
        for channel, modality in enumerate(modalities):
            source_value = paths[modality]
            source = resolve_path(source_value)
            if source is None:
                raise FileNotFoundError(f"empty source path for {folder_id}/{modality}")
            nnunet_target = dataset_dir / image_partition / f"{nnunet_id}_{channel:04d}.nii.gz"
            folder_target = case_folder_root / split_name / folder_id / f"{folder_id}-{modality}.nii.gz"
            action = materialize_file(source, nnunet_target, mode)
            materialize_file(source, folder_target, mode)
            records.append({
                "nnunet_case_id": nnunet_id,
                "case_folder_id": folder_id,
                "source_case_id": spec["source_case_id"],
                "row_type": spec["row_type"],
                "split": split_name,
                "nnunet_partition": image_partition,
                "modality": modality,
                "nnunet_channel": f"{channel:04d}",
                "source_path": str(source),
                "nnunet_target_path": str(nnunet_target),
                "case_folder_target_path": str(folder_target),
                "action": action,
                "completion_raw_id": spec["completion_raw_id"],
            })
        source_value = paths["seg"]
        source = resolve_path(source_value)
        if source is None:
            raise FileNotFoundError(f"empty seg path for {folder_id}")
        nnunet_target = dataset_dir / label_partition / f"{nnunet_id}.nii.gz"
        folder_target = case_folder_root / split_name / folder_id / f"{folder_id}-seg.nii.gz"
        action = materialize_file(source, nnunet_target, mode)
        materialize_file(source, folder_target, mode)
        records.append({
            "nnunet_case_id": nnunet_id,
            "case_folder_id": folder_id,
            "source_case_id": spec["source_case_id"],
            "row_type": spec["row_type"],
            "split": split_name,
            "nnunet_partition": label_partition,
            "modality": "seg",
            "nnunet_channel": "label",
            "source_path": str(source),
            "nnunet_target_path": str(nnunet_target),
            "case_folder_target_path": str(folder_target),
            "action": action,
            "completion_raw_id": spec["completion_raw_id"],
        })
    return records


def build_output_split(master: dict[str, object], specs: list[dict[str, object]]) -> dict[str, object]:
    real_ids = {
        str(spec["nnunet_case_id"])
        for spec in specs
        if spec["row_type"] != "synthetic_augmentation"
    }
    synthetic_ids = sorted(
        str(spec["nnunet_case_id"])
        for spec in specs
        if spec["row_type"] == "synthetic_augmentation"
    )
    output = {
        "name": "g2_materialized_fixed_split",
        "derived_from": master.get("name", "master_patient_group_train_val_test"),
        "train": [case_id for case_id in master["train"] if case_id in real_ids] + synthetic_ids,  # type: ignore[index]
        "val": [case_id for case_id in master["val"] if case_id in real_ids],  # type: ignore[index]
        "test": [case_id for case_id in master["test"] if case_id in real_ids],  # type: ignore[index]
        "synthetic_train_ids": synthetic_ids,
    }
    output["counts"] = {name: len(output[name]) for name in ("train", "val", "test")}
    if (set(output["train"]) & set(output["val"])) or (set(output["train"]) & set(output["test"])) or (set(output["val"]) & set(output["test"])):
        raise ValueError("materialized fixed split overlaps")
    return output


def verify_materialized_dataset(
    dataset_dir: Path,
    specs: list[dict[str, object]],
    mode: str,
) -> dict[str, object]:
    expected_ids = {str(spec["nnunet_case_id"]) for spec in specs}
    report: dict[str, object] = {
        "mode": mode,
        "expected_cases": len(expected_ids),
        "checked_cases": 0,
        "errors": [],
        "passed": True,
    }
    if mode == "manifest-only":
        report["status"] = "planned_only_sources_preflighted"
        return report
    errors: list[str] = []
    expected_trainval = {
        str(spec["nnunet_case_id"])
        for spec in specs
        if spec.get("split") in {"train", "val"}
    }
    expected_test = {
        str(spec["nnunet_case_id"])
        for spec in specs
        if spec.get("split") == "test"
    }
    actual_trainval = {path.name.removesuffix(".nii.gz") for path in (dataset_dir / "labelsTr").glob("*.nii.gz")}
    actual_test = {path.name.removesuffix(".nii.gz") for path in (dataset_dir / "labelsTs").glob("*.nii.gz")}
    if actual_trainval != expected_trainval:
        errors.append(f"labelsTr IDs mismatch: expected={len(expected_trainval)} actual={len(actual_trainval)}")
    if actual_test != expected_test:
        errors.append(f"labelsTs IDs mismatch: expected={len(expected_test)} actual={len(actual_test)}")
    spec_by_id = {str(spec["nnunet_case_id"]): spec for spec in specs}
    for case_id in sorted(expected_ids):
        split_name = str(spec_by_id[case_id]["split"])
        image_partition = "imagesTs" if split_name == "test" else "imagesTr"
        label_partition = "labelsTs" if split_name == "test" else "labelsTr"
        paths = [dataset_dir / image_partition / f"{case_id}_{index:04d}.nii.gz" for index in range(4)]
        label_path = dataset_dir / label_partition / f"{case_id}.nii.gz"
        missing = [str(path) for path in [*paths, label_path] if not path.is_file()]
        if missing:
            errors.append(f"{case_id}:missing:{missing}")
            continue
        images = [nib.load(str(path)) for path in paths]
        label_image = nib.load(str(label_path))
        geometry_images = [*images, label_image]
        reference_shape = geometry_images[0].shape[:3]
        reference_affine = np.asarray(geometry_images[0].affine, dtype=np.float64)
        if any(
            image.shape[:3] != reference_shape
            or not np.allclose(
                np.asarray(image.affine, dtype=np.float64),
                reference_affine,
                rtol=0.0,
                atol=1e-6,
                equal_nan=False,
            )
            for image in geometry_images[1:]
        ):
            errors.append(f"{case_id}:geometry_mismatch")
        labels = np.unique(np.asanyarray(label_image.dataobj))
        if not set(labels.tolist()).issubset({0, 1, 2, 3, 4}):
            errors.append(f"{case_id}:illegal_labels:{labels.tolist()}")
        report["checked_cases"] = int(report["checked_cases"]) + 1
    report["errors"] = errors
    report["passed"] = not errors
    report["status"] = "pass" if not errors else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--real-mapping", default="")
    parser.add_argument(
        "--real-data-root",
        default="",
        help=(
            "Optional server-local real-case root using "
            "<root>/<source_case_id>/<modality>.nii.gz. Corrected seg paths in "
            "the mapping remain authoritative."
        ),
    )
    parser.add_argument("--master-split", default="")
    parser.add_argument(
        "--synthetic-accepted-manifest",
        action="append",
        default=[],
        help="Approved training or evaluation manifest. Repeat for multiple G1 runs.",
    )
    parser.add_argument(
        "--completion-root",
        default="",
        help=(
            "Optional server-local G1 V3 run root. Completion T2W is resolved as "
            "<root>/<source_case_id>/<source_case_id>-t2w.nii.gz."
        ),
    )
    parser.add_argument("--fake-t2w-cases", default=str(DEFAULT_FAKE_T2W))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case-folder-root", default="")
    parser.add_argument("--dataset-id", default="261")
    parser.add_argument("--dataset-name", default="BraTS2026_MET_RealSynth_G1")
    parser.add_argument("--dataset-profile", choices=["auto", "real-only", "completion", "real-synth"], default="auto")
    parser.add_argument("--channel-order", choices=sorted(CHANNEL_ORDERS), default="g2_official")
    parser.add_argument("--mode", choices=["manifest-only", "symlink", "hardlink", "copy"], default="manifest-only")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--allow-incomplete-completion", action="store_true")
    parser.add_argument("--run-nnunet-integrity", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    manifest_paths = [Path(value).expanduser().resolve() for value in args.synthetic_accepted_manifest]
    profile = args.dataset_profile
    if profile == "auto":
        profile = "real-synth" if manifest_paths else "real-only"
    if args.real_mapping:
        real_mapping = Path(args.real_mapping).expanduser().resolve()
    elif profile == "real-only":
        real_mapping = results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    else:
        real_mapping = results_root / "manifests" / "nnunet_case_mapping_master.csv"
    master_split_path = Path(args.master_split).expanduser().resolve() if args.master_split else results_root / "splits" / "splits_master_train_val_test.json"
    fake_t2w_path = Path(args.fake_t2w_cases).expanduser().resolve()
    for required in [real_mapping, master_split_path, fake_t2w_path, *manifest_paths]:
        if not required.is_file():
            raise SystemExit(f"required input not found: {required}")

    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / f"Dataset{args.dataset_id}_{args.dataset_name}"
    case_folder_root = Path(args.case_folder_root).expanduser().resolve() if args.case_folder_root else output_root / f"{args.dataset_name}_case_folders"
    prepare_output(dataset_dir, args.clean_output)
    prepare_output(case_folder_root, args.clean_output)
    (dataset_dir / "imagesTr").mkdir()
    (dataset_dir / "labelsTr").mkdir()
    (dataset_dir / "imagesTs").mkdir()
    (dataset_dir / "labelsTs").mkdir()

    real_rows = read_csv(real_mapping)
    real_data_root = (
        Path(args.real_data_root).expanduser().resolve()
        if args.real_data_root
        else None
    )
    real_paths_overridden = apply_real_data_root(real_rows, real_data_root)
    synthetic_rows = read_synthetic_manifests(manifest_paths)
    completion_root = (
        Path(args.completion_root).expanduser().resolve()
        if args.completion_root
        else None
    )
    completion_paths_overridden = apply_completion_root(
        synthetic_rows, completion_root
    )
    fake_t2w_cases = load_fake_t2w_cases(fake_t2w_path)
    specs, stats = build_case_specs(
        real_rows,
        synthetic_rows,
        fake_t2w_cases,
        profile,
        args.allow_incomplete_completion,
    )
    master_split = load_split(master_split_path)
    assign_spec_splits(specs, master_split)
    records = materialize_specs(specs, dataset_dir, case_folder_root, args.mode)
    output_split = build_output_split(master_split, specs)

    dataset_json = {
        "channel_names": {str(index): modality for index, modality in enumerate(CHANNEL_ORDERS[args.channel_order])},
        "labels": LABELS,
        "numTraining": sum(spec.get("split") in {"train", "val"} for spec in specs),
        "numTest": sum(spec.get("split") == "test" for spec in specs),
        "file_ending": ".nii.gz",
        "g2_dataset_profile": profile,
        "g2_channel_order": args.channel_order,
        "g2_real_mapping": str(real_mapping),
        "g2_real_data_root": str(real_data_root) if real_data_root else "",
        "g2_real_paths_overridden": real_paths_overridden,
        "g2_synthetic_manifests": [str(path) for path in manifest_paths],
        "g2_completion_root": str(completion_root) if completion_root else "",
        "g2_completion_paths_overridden": completion_paths_overridden,
        "g2_master_split": str(master_split_path),
        "g2_materialization_stats": stats,
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dataset_dir / "g2_fixed_split.json").write_text(json.dumps([output_split], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dataset_dir / "splits_final.json").write_text(
        json.dumps([{"train": output_split["train"], "val": output_split["val"]}], indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        dataset_dir / "g2_materialization_manifest.csv",
        records,
        [
            "nnunet_case_id",
            "case_folder_id",
            "source_case_id",
            "row_type",
            "split",
            "nnunet_partition",
            "modality",
            "nnunet_channel",
            "source_path",
            "nnunet_target_path",
            "case_folder_target_path",
            "action",
            "completion_raw_id",
        ],
    )

    integrity = verify_materialized_dataset(dataset_dir, specs, args.mode)
    integrity_path = dataset_dir / "g2_integrity_report.json"
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not integrity["passed"]:
        raise SystemExit(f"G2 built-in integrity check failed: {integrity_path}")
    if args.run_nnunet_integrity:
        if args.mode == "manifest-only":
            raise SystemExit("--run-nnunet-integrity requires symlink/hardlink/copy mode")
        env = os.environ.copy()
        env["nnUNet_raw"] = str(output_root)
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id), "--verify_dataset_integrity"],
            check=True,
            env=env,
        )

    print(f"dataset_dir={dataset_dir}")
    print(f"case_folder_root={case_folder_root}")
    print(f"profile={profile}")
    print(f"included_cases={len(specs)}")
    print(f"real_paths_overridden={real_paths_overridden}")
    print(f"completion_paths_overridden={completion_paths_overridden}")
    print(f"split_counts={output_split['counts']}")
    print(f"integrity={integrity['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
