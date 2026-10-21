#!/usr/bin/env python3
"""Build the immutable Dataset264 train-only MET-AUG single-lesion pool.

Components are extracted from the exact nnU-Net preprocessed segmentation
space used by S2.  The raw tree remains the source of immutable image, label,
affine, and spacing provenance.  The script refuses an existing output
directory so a route can never silently replace its donor pool.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (
    ALLOWED_LABELS,
    COMPONENT_MANIFEST_SCHEMA,
    ComponentRecord,
    MetAugContractError,
    S2_MODALITIES,
    canonical_json_sha256,
    extract_met_components,
    patient_group,
    sha256_file,
)


PREPROCESSED_IGNORE_LABEL = -1
PREPROCESSED_LABEL_NORMALIZATION_POLICY = (
    "replace_nnunet_ignore_minus_one_with_background_for_donor_component_extraction_only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset264 nnUNet_raw directory")
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Dataset264 nnUNetPlans_3d_fullres directory used by S2",
    )
    parser.add_argument("--train-file", required=True, help="Frozen 1035-case train_fixed.txt")
    parser.add_argument("--mapping-csv", required=True, help="G2 nnU-Net/source identity mapping")
    parser.add_argument(
        "--preprocessed-coordinate-audit",
        required=True,
        help="Passing immutable coordinate-contract JSON for the exact b2nd cache",
    )
    parser.add_argument("--output-dir", required=True, help="New immutable component-pool directory")
    parser.add_argument("--manifest-version", default="met_aug_component_pool_v1")
    parser.add_argument("--min-core-volume-mm3", type=float, default=27.0)
    parser.add_argument("--max-bbox-mm", type=float, default=56.0)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"train split is empty or has duplicate IDs: {path}")
    return values


def read_mapping(path: Path, train_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"nnunet_case_id", "source_case_id", "patient_group"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"mapping lacks fields {sorted(required)}: {path}")
        mapping: dict[str, str] = {}
        groups: dict[str, str] = {}
        for row in reader:
            case_id = row["nnunet_case_id"].strip()
            if not case_id:
                continue
            source_case_id = row["source_case_id"].strip()
            group = row["patient_group"].strip()
            if not source_case_id or not group:
                raise ValueError(f"mapping has empty source_case_id/patient_group for {case_id}")
            if case_id in mapping:
                raise ValueError(f"mapping has duplicate nnunet_case_id: {case_id}")
            if group != patient_group(source_case_id):
                raise ValueError(
                    f"mapping patient_group does not match source_case_id: {case_id} -> {source_case_id}/{group}"
                )
            mapping[case_id] = source_case_id
            groups[case_id] = group
    missing = sorted(train_ids - set(mapping))
    if missing:
        raise ValueError(f"train split IDs are missing from mapping: {missing[:10]}")
    return (
        {case: mapping[case] for case in train_ids},
        {case: groups[case] for case in train_ids},
    )


def load_label(path: Path) -> tuple[np.ndarray, tuple[float, float, float], str]:
    image = nib.load(str(path))
    label = np.asanyarray(image.dataobj).astype(np.int16, copy=True)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    affine_sha = __import__("hashlib").sha256(np.asarray(image.affine, dtype=np.float64).tobytes()).hexdigest()
    return label, spacing, affine_sha


def validate_source_modalities(
    images_dir: Path,
    case_id: str,
    label_shape: tuple[int, int, int],
    label_spacing: tuple[float, float, float],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index, modality in enumerate(S2_MODALITIES):
        path = images_dir / f"{case_id}_{index:04d}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {modality} modality: {path}")
        image = nib.load(str(path))
        if tuple(image.shape[:3]) != label_shape:
            raise ValueError(f"{case_id}/{modality} shape differs from segmentation")
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(spacing, label_spacing, atol=1e-5):
            raise ValueError(f"{case_id}/{modality} spacing differs from segmentation")
        hashes[modality] = sha256_file(path)
    return hashes


def load_preprocessed_contract(preprocessed_dir: Path) -> tuple[tuple[float, float, float], Path]:
    plans_path = preprocessed_dir.parent / "nnUNetPlans.json"
    if not plans_path.is_file():
        raise FileNotFoundError(f"missing nnU-Net plans: {plans_path}")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    matches = [
        value
        for value in plans.get("configurations", {}).values()
        if value.get("data_identifier") == preprocessed_dir.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot bind preprocessed directory to one nnU-Net configuration: {preprocessed_dir}"
        )
    spacing = tuple(float(value) for value in matches[0].get("spacing", ()))
    if len(spacing) != 3 or not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-5):
        raise ValueError(f"Route A requires 1 mm 3D preprocessing, got {spacing}")
    return spacing, plans_path


def load_preprocessed_label(dataset, case_id: str) -> np.ndarray:
    data, segmentation, previous_stage_segmentation, _properties = dataset.load_case(case_id)
    if previous_stage_segmentation is not None:
        raise ValueError(f"{case_id}: MET-AUG does not support cascaded preprocessed data")
    if data.ndim != 4 or data.shape[0] != 4:
        raise ValueError(f"{case_id}: expected preprocessed four-channel image, got {data.shape}")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise ValueError(f"{case_id}: expected preprocessed one-channel segmentation, got {segmentation.shape}")
    if segmentation.shape[1:] != data.shape[1:]:
        raise ValueError(f"{case_id}: preprocessed image/segmentation shapes differ")
    return np.asarray(segmentation[0], dtype=np.int16).copy()


def normalize_preprocessed_label_for_component_extraction(
    label: np.ndarray,
    *,
    case_id: str,
) -> tuple[np.ndarray, int]:
    """Return a donor-only view while preserving nnU-Net's training label."""
    if label.ndim != 3:
        raise MetAugContractError(f"{case_id}: preprocessed label must be 3D, got {label.shape}")
    values = set(int(value) for value in np.unique(label))
    allowed_source_labels = ALLOWED_LABELS | {PREPROCESSED_IGNORE_LABEL}
    unsupported = values - allowed_source_labels
    if unsupported:
        raise MetAugContractError(
            f"{case_id}: preprocessed label contains unsupported classes: {sorted(unsupported)}"
        )
    normalized = np.asarray(label, dtype=np.int16).copy()
    ignore_voxels = int(np.count_nonzero(normalized == PREPROCESSED_IGNORE_LABEL))
    normalized[normalized == PREPROCESSED_IGNORE_LABEL] = 0
    return normalized, ignore_voxels


def build_preprocessed_label_normalization_audit(
    ignore_voxels_by_case: dict[str, int],
) -> dict[str, object]:
    ordered_counts = {
        case_id: int(ignore_voxels_by_case[case_id])
        for case_id in sorted(ignore_voxels_by_case)
    }
    return {
        "policy": PREPROCESSED_LABEL_NORMALIZATION_POLICY,
        "scope": "donor_component_extraction_view_only",
        "source_ignore_label": PREPROCESSED_IGNORE_LABEL,
        "replacement_label": 0,
        "allowed_source_labels": sorted(ALLOWED_LABELS | {PREPROCESSED_IGNORE_LABEL}),
        "normalized_label_classes": sorted(ALLOWED_LABELS),
        "case_ignore_voxel_counts": ordered_counts,
        "cases_with_ignore_voxels": sum(value > 0 for value in ordered_counts.values()),
        "total_ignore_voxels": sum(ordered_counts.values()),
    }


def build_raw_source_geometry_audit(
    source_geometries: dict[str, tuple[tuple[int, int, int], tuple[float, float, float]]],
) -> dict[str, object]:
    """Record native NIfTI geometry without confusing it with the 1 mm pool space."""
    if not source_geometries:
        raise ValueError("raw source geometry audit is empty")
    spacing_counts = Counter(spacing for _shape, spacing in source_geometries.values())
    shape_counts = Counter(shape for shape, _spacing in source_geometries.values())
    return {
        "role": "native_nifti_provenance_and_raw_label_to_modality_alignment_only",
        "component_coordinate_space": "nnUNetPlans_3d_fullres_preprocessed",
        "component_spacing_mm": [1.0, 1.0, 1.0],
        "raw_label_to_each_modality_geometry_match_required": True,
        "case_count": len(source_geometries),
        "native_spacing_unique_count": len(spacing_counts),
        "native_shape_unique_count": len(shape_counts),
        "native_one_mm_case_count": sum(
            int(case_count)
            for spacing, case_count in spacing_counts.items()
            if np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-5)
        ),
        "native_spacing_case_counts": [
            {
                "spacing_mm": [float(value) for value in spacing],
                "case_count": int(case_count),
            }
            for spacing, case_count in sorted(spacing_counts.items())
        ],
    }


def validate_preprocessed_coordinate_audit(
    path: Path,
    *,
    preprocessed_dir: Path,
    plans_path: Path,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing preprocessed coordinate audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise MetAugContractError("preprocessed coordinate audit did not pass")
    if payload.get("preprocessed_dir") != str(preprocessed_dir):
        raise MetAugContractError("coordinate audit is bound to a different preprocessed directory")
    if payload.get("plans_sha256") != sha256_file(plans_path):
        raise MetAugContractError("coordinate audit plans SHA256 does not match the active cache")
    if payload.get("configuration_spacing_mm") != [1.0, 1.0, 1.0]:
        raise MetAugContractError("coordinate audit does not establish the required 1 mm cache")
    if payload.get("required_spacing_mm") != [1.0, 1.0, 1.0]:
        raise MetAugContractError("coordinate audit required spacing has drifted")
    if payload.get("missing_count") != 0 or payload.get("mismatch_count") != 0:
        raise MetAugContractError("coordinate audit has missing or shape-mismatched cache entries")
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    labels_dir = dataset_dir / "labelsTr"
    images_dir = dataset_dir / "imagesTr"
    train_file = Path(args.train_file).expanduser().resolve()
    mapping_csv = Path(args.mapping_csv).expanduser().resolve()
    coordinate_audit_path = Path(args.preprocessed_coordinate_audit).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"component-pool output already exists and is immutable: {output_dir}"
        )
    if not labels_dir.is_dir() or not images_dir.is_dir() or not preprocessed_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset264 raw/preprocessed inputs are incomplete: {dataset_dir} / {preprocessed_dir}"
        )
    train_ids = read_ids(train_file)
    mapping, target_groups = read_mapping(mapping_csv, set(train_ids))
    preprocessed_spacing, plans_path = load_preprocessed_contract(preprocessed_dir)
    coordinate_audit_sha256 = validate_preprocessed_coordinate_audit(
        coordinate_audit_path,
        preprocessed_dir=preprocessed_dir,
        plans_path=plans_path,
    )
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(str(preprocessed_dir))
    preprocessed_dataset = dataset_class(str(preprocessed_dir), train_ids)
    output_dir.mkdir(parents=True, exist_ok=False)
    component_dir = output_dir / "components"
    component_dir.mkdir()

    records: list[ComponentRecord] = []
    exclusions: Counter[str] = Counter()
    source_affines: dict[str, str] = {}
    source_geometries: dict[str, tuple[tuple[int, int, int], tuple[float, float, float]]] = {}
    ignore_voxels_by_case: dict[str, int] = {}
    for case_id in train_ids:
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing train segmentation: {label_path}")
        raw_label, raw_spacing, affine_sha = load_label(label_path)
        source_modalities_sha256 = validate_source_modalities(
            images_dir, case_id, tuple(raw_label.shape), raw_spacing
        )
        preprocessed_label = load_preprocessed_label(preprocessed_dataset, case_id)
        label, ignore_voxels = normalize_preprocessed_label_for_component_extraction(
            preprocessed_label,
            case_id=case_id,
        )
        ignore_voxels_by_case[case_id] = ignore_voxels
        source_affines[case_id] = affine_sha
        source_geometries[case_id] = (tuple(raw_label.shape), raw_spacing)
        source_label_sha = sha256_file(label_path)
        payloads, dropped = extract_met_components(
            label,
            preprocessed_spacing,
            min_core_volume_mm3=args.min_core_volume_mm3,
            max_bbox_mm=args.max_bbox_mm,
        )
        exclusions.update(dropped)
        source_case_id = mapping[case_id]
        for index, payload in enumerate(payloads, start=1):
            component_id = __import__("hashlib").sha256(
                f"{args.manifest_version}|{case_id}|{index}".encode("utf-8")
            ).hexdigest()[:24]
            component_path = Path("components") / f"{component_id}.npz"
            destination = output_dir / component_path
            np.savez_compressed(destination, label=payload["label"])
            stats = payload["stats"]
            record = ComponentRecord(
                component_id=component_id,
                manifest_version=args.manifest_version,
                source_case_id=source_case_id,
                patient_group=target_groups[case_id],
                split="train",
                component_path=str(component_path),
                label_sha256=sha256_file(destination),
                source_label_sha256=source_label_sha,
                source_modalities_sha256=source_modalities_sha256,
                source_affine_sha256=affine_sha,
                spacing_mm=preprocessed_spacing,
                core_volume_mm3=float(stats["core_volume_mm3"]),
                total_volume_mm3=float(stats["total_volume_mm3"]),
                bbox_mm=tuple(float(value) for value in stats["bbox_mm"]),
                bbox_voxels=tuple(int(value) for value in stats["bbox_voxels"]),
                class_counts=stats["class_counts"],
                classes_present=tuple(int(value) for value in stats["classes_present"]),
                core_centroid_norm=tuple(float(value) for value in payload["core_centroid_norm"]),
            )
            records.append(record)

    if not records:
        raise RuntimeError("component pool contains no eligible train-only lesions")
    records_path = output_dir / "components.jsonl"
    rows = []
    for record in records:
        rows.append(record.as_mapping())
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    groups_payload = {
        "schema_version": 1,
        "case_to_patient_group": {
            case_id: target_groups[case_id] for case_id in sorted(train_ids)
        },
    }
    groups_path = output_dir / "target_case_groups.json"
    groups_path.write_text(json.dumps(groups_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": COMPONENT_MANIFEST_SCHEMA,
        "manifest_version": args.manifest_version,
        "dataset_dir": str(dataset_dir),
        "preprocessed_dir": str(preprocessed_dir),
        "builder_code_sha256": sha256_file(Path(__file__)),
        "component_core_sha256": sha256_file(
            REPOSITORY_ROOT / "custom_nnunet" / "met_aug_core.py"
        ),
        "nnunet_plans_sha256": sha256_file(plans_path),
        "preprocessed_coordinate_audit": str(coordinate_audit_path),
        "preprocessed_coordinate_audit_sha256": coordinate_audit_sha256,
        "coordinate_space": "nnUNetPlans_3d_fullres_preprocessed",
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "mapping_csv": str(mapping_csv),
        "mapping_csv_sha256": sha256_file(mapping_csv),
        "train_count": len(train_ids),
        "component_count": len(records),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "target_groups_file": groups_path.name,
        "target_groups_sha256": sha256_file(groups_path),
        "source_affine_sha256": source_affines,
        "raw_source_geometry": build_raw_source_geometry_audit(source_geometries),
        "preprocessed_label_normalization": build_preprocessed_label_normalization_audit(
            ignore_voxels_by_case
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "constraints": {
            "spacing_mm": [1.0, 1.0, 1.0],
            "min_core_volume_mm3": float(args.min_core_volume_mm3),
            "max_bbox_mm": float(args.max_bbox_mm),
            "rc_policy": "exclude_source_case",
            "snfh_policy": "attach_only_if_adjacent_to_one_core",
        },
    }
    summary["manifest_sha256"] = canonical_json_sha256(summary, exclude=("manifest_sha256",))
    manifest_path = output_dir / "component_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": summary["manifest_sha256"],
        "component_count": len(records),
        "train_count": len(train_ids),
        "exclusions": dict(sorted(exclusions.items())),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
