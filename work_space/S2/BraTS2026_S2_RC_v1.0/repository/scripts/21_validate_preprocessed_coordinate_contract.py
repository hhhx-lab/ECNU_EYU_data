#!/usr/bin/env python3
"""Verify that Dataset264 b2nd arrays actually match the locked nnU-Net plan.

The fixed-split cache gate establishes case identity and file presence. This
additional gate establishes the coordinate contract needed by physical lesion
constraints and the 64-cube diffusion adapter: every cached array must have
the shape produced by its own stored source geometry and ``nnUNetPlans``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import blosc2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset264 nnUNet_raw directory")
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Dataset264 nnUNetPlans_3d_fullres cache directory",
    )
    parser.add_argument("--output", required=True, help="New immutable coordinate-audit JSON")
    parser.add_argument(
        "--required-spacing-mm",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
    )
    return parser.parse_args()


def _configuration_contract(
    preprocessed_dir: Path,
) -> tuple[tuple[int, int, int], tuple[float, float, float], str, Path]:
    plans_path = preprocessed_dir.parent / "nnUNetPlans.json"
    if not plans_path.is_file():
        raise FileNotFoundError(f"missing nnU-Net plans: {plans_path}")
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    transpose = tuple(int(value) for value in payload.get("transpose_forward", (0, 1, 2)))
    if sorted(transpose) != [0, 1, 2]:
        raise ValueError(f"unsupported transpose_forward in {plans_path}: {transpose}")
    matches = [
        name
        for name, configuration in payload.get("configurations", {}).items()
        if configuration.get("data_identifier") == preprocessed_dir.name
    ]
    if len(matches) != 1:
        raise ValueError(
            "cannot bind preprocessed cache to one nn-U-Net configuration: "
            f"{preprocessed_dir}"
        )
    spacing = tuple(float(value) for value in payload["configurations"][matches[0]].get("spacing", ()))
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"invalid configuration spacing in {plans_path}: {spacing}")
    return transpose, spacing, matches[0], plans_path


def expected_shape_from_properties(
    properties: dict,
    *,
    transpose_forward: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    try:
        cropped_shape = tuple(
            int(value) for value in properties["shape_after_cropping_and_before_resampling"]
        )
        source_spacing = tuple(float(value) for value in properties["spacing"])
    except KeyError as exc:
        raise ValueError(f"preprocessed properties lack {exc.args[0]}") from exc
    if len(cropped_shape) != 3 or any(value <= 0 for value in cropped_shape):
        raise ValueError(f"invalid cropped shape in properties: {cropped_shape}")
    if len(source_spacing) != 3 or any(value <= 0 for value in source_spacing):
        raise ValueError(f"invalid source spacing in properties: {source_spacing}")
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

    current_spacing = tuple(source_spacing[axis] for axis in transpose_forward)
    return tuple(
        int(value)
        for value in compute_new_shape(cropped_shape, current_spacing, target_spacing)
    )


def coordinate_case_record(
    *,
    case_id: str,
    properties: dict,
    data_shape: tuple[int, ...],
    segmentation_shape: tuple[int, ...],
    transpose_forward: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> dict[str, object]:
    expected_shape = expected_shape_from_properties(
        properties,
        transpose_forward=transpose_forward,
        target_spacing=target_spacing,
    )
    observed_data = tuple(int(value) for value in data_shape)
    observed_segmentation = tuple(int(value) for value in segmentation_shape)
    data_matches = len(observed_data) == 4 and observed_data[0] == 4 and observed_data[1:] == expected_shape
    segmentation_matches = (
        len(observed_segmentation) == 4
        and observed_segmentation[0] == 1
        and observed_segmentation[1:] == expected_shape
    )
    return {
        "case_id": case_id,
        "expected_spatial_shape": list(expected_shape),
        "observed_data_shape": list(observed_data),
        "observed_segmentation_shape": list(observed_segmentation),
        "matches": bool(data_matches and segmentation_matches),
    }


def _case_ids_from_labels(dataset_dir: Path) -> list[str]:
    labels_dir = dataset_dir / "labelsTr"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"missing labelsTr: {labels_dir}")
    case_ids = sorted(path.name.removesuffix(".nii.gz") for path in labels_dir.glob("*.nii.gz"))
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"labelsTr case IDs are empty or duplicated: {labels_dir}")
    return case_ids


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"coordinate audit is immutable and already exists: {output_path}")
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"missing preprocessed directory: {preprocessed_dir}")
    transpose_forward, target_spacing, configuration_name, plans_path = _configuration_contract(
        preprocessed_dir
    )
    required_spacing = tuple(float(value) for value in args.required_spacing_mm)
    case_ids = _case_ids_from_labels(dataset_dir)
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, object]] = []
    mismatch_count = 0
    for case_id in case_ids:
        pkl_path = preprocessed_dir / f"{case_id}.pkl"
        data_path = preprocessed_dir / f"{case_id}.b2nd"
        segmentation_path = preprocessed_dir / f"{case_id}_seg.b2nd"
        if not pkl_path.is_file() or not data_path.is_file() or not segmentation_path.is_file():
            missing.append(
                {
                    "case_id": case_id,
                    "pkl": str(pkl_path.is_file()),
                    "data_b2nd": str(data_path.is_file()),
                    "seg_b2nd": str(segmentation_path.is_file()),
                }
            )
            continue
        from batchgenerators.utilities.file_and_folder_operations import load_pickle

        properties = load_pickle(str(pkl_path))
        data = blosc2.open(urlpath=str(data_path), mode="r", dparams={"nthreads": 1}, mmap_mode="r")
        segmentation = blosc2.open(
            urlpath=str(segmentation_path), mode="r", dparams={"nthreads": 1}, mmap_mode="r"
        )
        record = coordinate_case_record(
            case_id=case_id,
            properties=properties,
            data_shape=tuple(data.shape),
            segmentation_shape=tuple(segmentation.shape),
            transpose_forward=transpose_forward,
            target_spacing=target_spacing,
        )
        if not record["matches"]:
            mismatch_count += 1
            if len(mismatches) < 25:
                mismatches.append(record)
    spacing_matches_requirement = all(
        abs(observed - required) <= 1e-5
        for observed, required in zip(target_spacing, required_spacing)
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if not missing and mismatch_count == 0 and spacing_matches_requirement else "fail",
        "dataset_dir": str(dataset_dir),
        "preprocessed_dir": str(preprocessed_dir),
        "plans_path": str(plans_path),
        "plans_sha256": sha256_file(plans_path),
        "configuration": configuration_name,
        "transpose_forward": list(transpose_forward),
        "configuration_spacing_mm": list(target_spacing),
        "required_spacing_mm": list(required_spacing),
        "spacing_matches_requirement": spacing_matches_requirement,
        "case_count": len(case_ids),
        "missing_count": len(missing),
        "missing": missing[:25],
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))
    if payload["status"] != "pass":
        raise SystemExit("preprocessed coordinate contract failed")


if __name__ == "__main__":
    main()
