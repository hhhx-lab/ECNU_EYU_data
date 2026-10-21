#!/usr/bin/env python3
"""Technical and paired-quality QC for four-modality Diffusion support outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


MODALITIES = ("t1c", "t1n", "t2w", "t2f")
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_geometry(images: dict[str, nib.spatialimages.SpatialImage]) -> None:
    first_name, first = next(iter(images.items()))
    first_spacing = np.asarray(first.header.get_zooms()[:3], dtype=float)
    for name, image in images.items():
        if image.shape != first.shape:
            raise ValueError(
                f"shape mismatch: {first_name}={first.shape}, {name}={image.shape}"
            )
        if not np.allclose(first.affine, image.affine, atol=1e-4, rtol=0.0):
            raise ValueError(f"affine mismatch: {first_name} vs {name}")
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.allclose(first_spacing, spacing, atol=1e-5, rtol=0.0):
            raise ValueError(f"spacing mismatch: {first_name} vs {name}")


def crop_to_mask(mask: np.ndarray, padding: int = 4) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("metric mask is empty")
    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + padding + 1, np.asarray(mask.shape))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))


def masked_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    *,
    data_range: float = 6.0,
) -> dict[str, float | int | None]:
    region = np.asarray(mask, dtype=bool)
    voxel_count = int(region.sum())
    if voxel_count == 0:
        return {
            "voxel_count": 0,
            "mse": None,
            "mae": None,
            "psnr": None,
            "ssim": None,
            "contrast_reference": None,
            "contrast_generated": None,
            "contrast_abs_error": None,
        }
    difference = generated[region].astype(np.float64) - reference[region].astype(
        np.float64
    )
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = 100.0 if mse == 0 else float(10.0 * math.log10(data_range**2 / mse))
    roi = crop_to_mask(region)
    local_mask = region[roi].astype(np.float32)
    ref_roi = reference[roi] * local_mask
    gen_roi = generated[roi] * local_mask
    minimum = min(ref_roi.shape)
    window = min(7, minimum if minimum % 2 == 1 else minimum - 1)
    ssim = None
    if window >= 3:
        ssim = float(
            structural_similarity(
                ref_roi,
                gen_roi,
                data_range=data_range,
                win_size=window,
            )
        )
    background = ~region
    background_reference = float(np.mean(reference[background])) if background.any() else 0.0
    background_generated = float(np.mean(generated[background])) if background.any() else 0.0
    ref_contrast = float(np.mean(reference[region])) - background_reference
    gen_contrast = float(np.mean(generated[region])) - background_generated
    return {
        "voxel_count": voxel_count,
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "contrast_reference": ref_contrast,
        "contrast_generated": gen_contrast,
        "contrast_abs_error": abs(gen_contrast - ref_contrast),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-8:
        return None
    return float(numerator / denominator)


def _largest_component_fraction(mask: np.ndarray, denominator: int) -> float:
    if denominator <= 0 or not mask.any():
        return 0.0
    components, count = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(3, 3)
    )
    if count == 0:
        return 0.0
    sizes = np.bincount(components.ravel())[1:]
    return float(sizes.max() / denominator)


def _adjacent_z_metrics(array: np.ndarray, support: np.ndarray) -> tuple[float, int]:
    differences: list[float] = []
    repeated = 0
    for index in range(1, array.shape[2]):
        pair_support = support[:, :, index - 1] | support[:, :, index]
        if not pair_support.any():
            continue
        previous = array[:, :, index - 1]
        current = array[:, :, index]
        differences.append(float(np.mean(np.abs(current - previous)[pair_support])))
        if np.array_equal(
            previous[pair_support], current[pair_support]
        ) and np.array_equal(support[:, :, index - 1], support[:, :, index]):
            repeated += 1
    return (float(np.mean(differences)) if differences else 0.0, repeated)


def artifact_metrics(
    reference: np.ndarray, generated: np.ndarray, support: np.ndarray
) -> dict[str, Any]:
    """Measure boundary, void, high-frequency and z-continuity warning signals."""
    support_count = int(support.sum())
    generated_zero = support & np.isclose(generated, 0.0, atol=1e-7)
    reference_zero = support & np.isclose(reference, 0.0, atol=1e-7)
    boundary = support & ~ndimage.binary_erosion(
        support, structure=ndimage.generate_binary_structure(3, 1)
    )

    def gradient_mean(values: np.ndarray, region: np.ndarray) -> float:
        gradients = [ndimage.sobel(values, axis=axis, mode="nearest") for axis in range(3)]
        magnitude = np.sqrt(sum(component.astype(np.float64) ** 2 for component in gradients))
        return float(np.mean(magnitude[region])) if region.any() else 0.0

    generated_boundary_gradient = gradient_mean(generated, boundary)
    reference_boundary_gradient = gradient_mean(reference, boundary)
    generated_laplacian = float(
        np.mean(np.abs(ndimage.laplace(generated, mode="nearest"))[support])
    )
    reference_laplacian = float(
        np.mean(np.abs(ndimage.laplace(reference, mode="nearest"))[support])
    )
    generated_z_gradient, generated_repeated = _adjacent_z_metrics(generated, support)
    reference_z_gradient, reference_repeated = _adjacent_z_metrics(reference, support)
    generated_zero_fraction = float(generated_zero.sum() / max(support_count, 1))
    reference_zero_fraction = float(reference_zero.sum() / max(support_count, 1))
    generated_largest_zero = _largest_component_fraction(generated_zero, support_count)
    reference_largest_zero = _largest_component_fraction(reference_zero, support_count)
    boundary_ratio = _safe_ratio(generated_boundary_gradient, reference_boundary_gradient)
    laplacian_ratio = _safe_ratio(generated_laplacian, reference_laplacian)
    z_gradient_ratio = _safe_ratio(generated_z_gradient, reference_z_gradient)
    extreme_fraction = float(np.count_nonzero(np.abs(generated[support]) > 8.0) / max(support_count, 1))

    flags: list[str] = []
    if generated_largest_zero - reference_largest_zero > 0.10:
        flags.append("large_generated_zero_block")
    if boundary_ratio is not None and (boundary_ratio > 3.0 or boundary_ratio < 1 / 3):
        flags.append("boundary_gradient_shift")
    if laplacian_ratio is not None and (laplacian_ratio > 3.0 or laplacian_ratio < 1 / 3):
        flags.append("high_frequency_shift")
    if z_gradient_ratio is not None and (z_gradient_ratio > 3.0 or z_gradient_ratio < 1 / 3):
        flags.append("z_continuity_shift")
    if generated_repeated > reference_repeated + 1:
        flags.append("repeated_adjacent_z_slices")
    if extreme_fraction > 0.001:
        flags.append("extreme_zscore_signal")
    if any(np.any(support.take((0, -1), axis=axis)) for axis in range(3)):
        flags.append("support_touches_volume_edge")
    support_shape = np.ptp(np.argwhere(support), axis=0) + 1
    if np.any(support_shape > 64):
        flags.append("large_tiled_support")
    return {
        "generated_zero_fraction": generated_zero_fraction,
        "reference_zero_fraction": reference_zero_fraction,
        "generated_largest_zero_component_fraction": generated_largest_zero,
        "reference_largest_zero_component_fraction": reference_largest_zero,
        "generated_boundary_gradient": generated_boundary_gradient,
        "reference_boundary_gradient": reference_boundary_gradient,
        "boundary_gradient_ratio": boundary_ratio,
        "generated_laplacian": generated_laplacian,
        "reference_laplacian": reference_laplacian,
        "laplacian_ratio": laplacian_ratio,
        "generated_z_gradient": generated_z_gradient,
        "reference_z_gradient": reference_z_gradient,
        "z_gradient_ratio": z_gradient_ratio,
        "generated_repeated_adjacent_z_slices": generated_repeated,
        "reference_repeated_adjacent_z_slices": reference_repeated,
        "generated_extreme_zscore_fraction": extreme_fraction,
        "support_size_x": int(support_shape[0]),
        "support_size_y": int(support_shape[1]),
        "support_size_z": int(support_shape[2]),
        "artifact_flag_count": len(flags),
        "artifact_flags": ";".join(flags),
    }


def lesion_profile(label: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    components, count = ndimage.label(
        label > 0, structure=ndimage.generate_binary_structure(3, 3)
    )
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    volumes = [
        int(np.count_nonzero(components == component_id)) * voxel_volume
        for component_id in range(1, count + 1)
    ]
    return {
        "lesion_count": count,
        "tiny_count": sum(volume < 27.0 for volume in volumes),
        "small_count": sum(27.0 <= volume <= 275.0 for volume in volumes),
        "large_count": sum(volume > 275.0 for volume in volumes),
        "has_rc": bool(np.any(label == 4)),
    }


def focus_coordinate(label: np.ndarray) -> tuple[int, int, int]:
    mask = label == 4
    if not mask.any():
        mask = label > 0
    if not mask.any():
        return tuple(int(value // 2) for value in label.shape)
    return tuple(int(round(value)) for value in ndimage.center_of_mass(mask))


def plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(array[index, :, :])
    if axis == 1:
        return np.rot90(array[:, index, :])
    return np.rot90(array[:, :, index])


def support_projection_crop(
    support: np.ndarray, axis: int, padding: int = 8
) -> tuple[slice, slice]:
    projection = np.rot90(np.any(support, axis=axis))
    coordinates = np.argwhere(projection)
    if coordinates.size == 0:
        return slice(0, projection.shape[0]), slice(0, projection.shape[1])
    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + padding + 1, projection.shape)
    return slice(int(lower[0]), int(upper[0])), slice(int(lower[1]), int(upper[1]))


def display_limits(reference: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    values = reference[support]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def render_montage(
    case_id: str,
    case_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    support: np.ndarray,
    label: np.ndarray,
    output_path: Path,
) -> None:
    focus = focus_coordinate(label)
    figure, axes = plt.subplots(len(MODALITIES), 7, figsize=(21, 11))
    orientations = ((0, focus[0]), (1, focus[1]), (2, focus[2]))
    crops = {
        axis: support_projection_crop(support, axis=axis, padding=8)
        for axis, _ in orientations
    }
    for row_index, modality in enumerate(MODALITIES):
        reference, generated = case_pairs[modality]
        low, high = display_limits(reference, support)
        def view(values: np.ndarray, orientation_index: int) -> np.ndarray:
            axis, index = orientations[orientation_index]
            return plane(values, axis, index)[crops[axis]]

        panels = [
            (view(reference, 0), "ax ref", "gray", low, high),
            (view(generated, 0), "ax gen", "gray", low, high),
            (
                view(np.abs(generated - reference), 0),
                "ax abs",
                "magma",
                0.0,
                None,
            ),
            (view(reference, 1), "cor ref", "gray", low, high),
            (view(generated, 1), "cor gen", "gray", low, high),
            (view(reference, 2), "sag ref", "gray", low, high),
            (view(generated, 2), "sag gen", "gray", low, high),
        ]
        for column, (image, title, cmap, vmin, vmax) in enumerate(panels):
            axes[row_index, column].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            if column in (1, 4, 6):
                orientation_index = {1: 0, 4: 1, 6: 2}[column]
                axis, index = orientations[orientation_index]
                overlay = plane(label > 0, axis, index)[crops[axis]]
                axes[row_index, column].contour(
                    overlay.astype(float), levels=[0.5], colors="lime", linewidths=0.6
                )
            axes[row_index, column].set_title(f"{modality} {title}", fontsize=8)
            axes[row_index, column].axis("off")
    figure.suptitle(case_id, fontsize=12)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110, facecolor="white")
    plt.close(figure)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_qc(
    manifest_path: Path,
    selection_path: Path,
    inventory_path: Path,
    output_root: Path,
    *,
    expected_cases: int = 20,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    selection_path = Path(selection_path).expanduser().resolve()
    inventory_path = Path(inventory_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"QC output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    inventory_rows = read_csv(inventory_path)
    expected_ids = set(selection.get("selected_source_case_ids", []))
    hard_failures: list[str] = []
    expected_row_count = expected_cases * len(MODALITIES)
    if len(expected_ids) != expected_cases:
        hard_failures.append(
            f"selection case count {len(expected_ids)} != expected {expected_cases}"
        )
    if len(manifest_rows) != expected_row_count:
        hard_failures.append(
            f"manifest row count {len(manifest_rows)} != expected {expected_row_count}"
        )

    inventory_by_modality = {
        str(row.get("modality")): row
        for row in inventory_rows
        if str(row.get("step")) == "150000" and row.get("checksum_verified") == "yes"
    }
    inventory_final = set(inventory_by_modality)
    if inventory_final != set(MODALITIES):
        hard_failures.append(
            f"verified 150000 inventory modalities {sorted(inventory_final)} != {list(MODALITIES)}"
        )

    checkpoint_hashes: dict[str, str] = {}
    for modality in MODALITIES:
        paths = {
            Path(row.get("checkpoint_path", "")).expanduser().resolve()
            for row in manifest_rows
            if row.get("modality") == modality
        }
        if len(paths) != 1:
            hard_failures.append(
                f"{modality}: manifest checkpoint paths are not unique: {sorted(map(str, paths))}"
            )
            continue
        checkpoint_path = next(iter(paths))
        if not checkpoint_path.is_file():
            hard_failures.append(f"{modality}: checkpoint is missing: {checkpoint_path}")
            continue
        actual_hash = sha256_file(checkpoint_path)
        checkpoint_hashes[modality] = actual_hash
        inventory_row = inventory_by_modality.get(modality, {})
        if actual_hash != inventory_row.get("sha256"):
            hard_failures.append(f"{modality}: checkpoint SHA256 differs from inventory")
        if int(checkpoint_path.stat().st_size) != int(inventory_row.get("bytes", -1)):
            hard_failures.append(f"{modality}: checkpoint size differs from inventory")

    rows_by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in manifest_rows:
        case_id = row.get("source_case_id", "")
        modality = row.get("modality", "")
        if modality in rows_by_case[case_id]:
            hard_failures.append(f"{case_id}: duplicate modality {modality}")
        rows_by_case[case_id][modality] = row
    if set(rows_by_case) != expected_ids:
        hard_failures.append(
            "manifest/selection IDs differ: "
            f"missing={sorted(expected_ids - set(rows_by_case))[:10]} "
            f"extra={sorted(set(rows_by_case) - expected_ids)[:10]}"
        )

    modality_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    valid_case_count = 0

    for case_id in sorted(rows_by_case):
        modality_map = rows_by_case[case_id]
        if set(modality_map) != set(MODALITIES):
            hard_failures.append(
                f"{case_id}: modalities {sorted(modality_map)} != {list(MODALITIES)}"
            )
            continue
        case_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        shared_support = None
        shared_label = None
        shared_spacing = None
        case_valid = True
        case_metric_rows = []
        for modality in MODALITIES:
            row = modality_map[modality]
            try:
                if row.get("normalization") != "per_crop_or_tile_brain_zscore":
                    raise ValueError(f"unexpected normalization {row.get('normalization')}")
                if str(row.get("checkpoint_step")) != "150000":
                    raise ValueError(f"unexpected checkpoint step {row.get('checkpoint_step')}")
                paths = {
                    "generated": Path(row["generated_zscore_path"]),
                    "reference": Path(row["reference_zscore_path"]),
                    "support": Path(row["support_path"]),
                    "label": Path(row["label_path"]),
                }
                missing = [name for name, path in paths.items() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing files {missing}")
                images = {name: nib.load(str(path)) for name, path in paths.items()}
                validate_geometry(images)
                generated = images["generated"].get_fdata(dtype=np.float32)
                reference = images["reference"].get_fdata(dtype=np.float32)
                support_raw = images["support"].get_fdata(dtype=np.float32)
                label_raw = images["label"].get_fdata(dtype=np.float32)
                if not all(
                    np.isfinite(values).all()
                    for values in (generated, reference, support_raw, label_raw)
                ):
                    raise ValueError("non-finite NIfTI values")
                support_values = set(np.unique(support_raw).tolist())
                if not support_values <= {0.0, 1.0}:
                    raise ValueError(f"non-binary support values {sorted(support_values)}")
                support = support_raw > 0
                label = np.rint(label_raw).astype(np.int16)
                illegal = sorted(set(np.unique(label).tolist()) - {0, 1, 2, 3, 4})
                if illegal:
                    raise ValueError(f"illegal labels {illegal}")
                generated_outside = int(np.count_nonzero(np.abs(generated[~support]) > 1e-6))
                reference_outside = int(np.count_nonzero(np.abs(reference[~support]) > 1e-6))
                if generated_outside:
                    raise ValueError(
                        f"{generated_outside} generated voxels changed outside support"
                    )
                if reference_outside:
                    raise ValueError(
                        f"{reference_outside} reference voxels changed outside support"
                    )
                tumour_outside = int(np.count_nonzero((label > 0) & ~support))
                if tumour_outside:
                    raise ValueError(f"{tumour_outside} tumour voxels outside support")
                if int(row.get("tumour_outside_support", "-1")) != 0:
                    raise ValueError("manifest reports tumour outside support")
                if int(support.sum()) != int(row["support_voxels"]):
                    raise ValueError("support voxel count differs from manifest")
                if int(np.count_nonzero(label)) != int(row["tumour_voxels"]):
                    raise ValueError("tumour voxel count differs from manifest")
                if float(np.std(generated[support])) <= 0:
                    raise ValueError("generated support is constant")
                if float(np.std(reference[support])) <= 0:
                    raise ValueError("reference support is constant")
                if shared_support is None:
                    shared_support = support
                    shared_label = label
                    shared_spacing = tuple(images["generated"].header.get_zooms()[:3])
                else:
                    if not np.array_equal(shared_support, support):
                        raise ValueError("support differs across modalities")
                    if not np.array_equal(shared_label, label):
                        raise ValueError("label differs across modalities")
                case_pairs[modality] = (reference, generated)

                region_masks = {
                    "support": support,
                    "tumour_all": label > 0,
                    **{name: label == value for value, name in LABEL_NAMES.items()},
                }
                region_metric_values = {
                    region_name: masked_metrics(reference, generated, region_mask)
                    for region_name, region_mask in region_masks.items()
                }
                support_metrics = region_metric_values["support"]
                tumour_metrics = region_metric_values["tumour_all"]
                artifact_metric_values = artifact_metrics(reference, generated, support)
                modality_row = {
                    "source_case_id": case_id,
                    "modality": modality,
                    "case_seed": int(row["case_seed"]),
                    "support_voxels": int(support.sum()),
                    "tumour_voxels": int(np.count_nonzero(label)),
                    **{f"support_{key}": value for key, value in support_metrics.items()},
                    **{f"tumour_{key}": value for key, value in tumour_metrics.items()},
                    "artifact_flag_count": artifact_metric_values["artifact_flag_count"],
                    "artifact_flags": artifact_metric_values["artifact_flags"],
                }
                modality_rows.append(modality_row)
                case_metric_rows.append(modality_row)
                artifact_rows.append(
                    {
                        "source_case_id": case_id,
                        "modality": modality,
                        **artifact_metric_values,
                    }
                )
                for region_name, region_metrics in region_metric_values.items():
                    region_rows.append(
                        {
                            "source_case_id": case_id,
                            "modality": modality,
                            "region": region_name,
                            **region_metrics,
                        }
                    )
            except Exception as exc:
                hard_failures.append(f"{case_id}/{modality}: {exc}")
                case_valid = False

        if case_valid and shared_support is not None and shared_label is not None:
            profile = lesion_profile(shared_label, shared_spacing)
            min_tumour_ssim = min(
                float(row["tumour_ssim"])
                for row in case_metric_rows
                if row.get("tumour_ssim") is not None
            )
            mean_support_ssim = float(
                np.mean([row["support_ssim"] for row in case_metric_rows])
            )
            case_artifact_rows = [
                row for row in artifact_rows if row["source_case_id"] == case_id
            ]
            case_artifact_flags = sorted(
                {
                    flag
                    for row in case_artifact_rows
                    for flag in str(row.get("artifact_flags", "")).split(";")
                    if flag
                }
            )
            review_rows.append(
                {
                    "source_case_id": case_id,
                    **profile,
                    "min_tumour_ssim": min_tumour_ssim,
                    "mean_support_ssim": mean_support_ssim,
                    "artifact_flag_count": len(case_artifact_flags),
                    "artifact_flags": ";".join(case_artifact_flags),
                    "review_priority": (
                        "high"
                        if (
                            profile["has_rc"]
                            or profile["tiny_count"] > 0
                            or profile["large_count"] > 0
                            or case_artifact_flags
                            or min_tumour_ssim < 0.5
                        )
                        else "routine"
                    ),
                }
            )
            render_montage(
                case_id,
                case_pairs,
                shared_support,
                shared_label,
                output_root / "montages" / f"{case_id}.png",
            )
            valid_case_count += 1

    review_rows.sort(
        key=lambda row: (
            0 if row["review_priority"] == "high" else 1,
            float(row["min_tumour_ssim"]),
            str(row["source_case_id"]),
        )
    )
    write_csv(output_root / "modality_metrics.csv", modality_rows)
    write_csv(output_root / "region_metrics.csv", region_rows)
    write_csv(output_root / "artifact_metrics.csv", artifact_rows)
    write_csv(output_root / "review_index.csv", review_rows)
    (output_root / "hard_failures.txt").write_text(
        "\n".join(hard_failures) + ("\n" if hard_failures else ""),
        encoding="utf-8",
    )
    summary = {
        "technical_gate": "pass" if not hard_failures else "fail",
        "quality_gate": "hold_for_manual_review",
        "case_count": valid_case_count,
        "expected_case_count": expected_cases,
        "modality_row_count": len(modality_rows),
        "region_row_count": len(region_rows),
        "artifact_row_count": len(artifact_rows),
        "montage_count": len(list((output_root / "montages").glob("*.png"))),
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "rc_case_count": sum(bool(row["has_rc"]) for row in review_rows),
        "tiny_lesion_case_count": sum(int(row["tiny_count"]) > 0 for row in review_rows),
        "small_lesion_case_count": sum(int(row["small_count"]) > 0 for row in review_rows),
        "artifact_flagged_case_count": sum(
            int(row["artifact_flag_count"]) > 0 for row in review_rows
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {
            "generation_manifest": sha256_file(manifest_path),
            "selection": sha256_file(selection_path),
            "checkpoint_inventory": sha256_file(inventory_path),
        },
        "metrics_by_modality": {
            modality: {
                metric: numeric_summary(
                    [row for row in modality_rows if row["modality"] == modality],
                    metric,
                )
                for metric in ("support_ssim", "tumour_ssim", "tumour_mae", "tumour_psnr")
            }
            for modality in MODALITIES
        },
    }
    summary = finite_json(summary)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# G2 Diffusion Checkpoint QC",
        "",
        f"- technical gate: `{summary['technical_gate']}`",
        f"- quality gate: `{summary['quality_gate']}`",
        f"- cases: {summary['case_count']}/{expected_cases}",
        f"- modality rows: {summary['modality_row_count']}",
        f"- artifact rows: {summary['artifact_row_count']}",
        f"- artifact-flagged cases: {summary['artifact_flagged_case_count']}",
        f"- montages: {summary['montage_count']}",
        f"- hard failures: {summary['hard_failure_count']}",
        "",
        "Manual review remains required before checkpoint selection is frozen.",
    ]
    (output_root / "QC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_qc(
        args.generation_manifest,
        args.selection_json,
        args.checkpoint_inventory,
        args.output_root,
        expected_cases=args.expected_cases,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
