#!/usr/bin/env python3
"""Paired image- and lesion-level QC for G1 V3 validation completions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
CORE_LABELS = (1, 3, 4)
MONTAGE_COLORS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.95, 0.20, 0.20, 0.65),
    (0.20, 0.85, 0.30, 0.55),
    (1.00, 0.85, 0.10, 0.70),
    (0.10, 0.75, 1.00, 0.75),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", required=True, help="G1 V3 data/input directory")
    parser.add_argument("--synthetic-root", required=True, help="Stage-5 synthesized T2W directory")
    parser.add_argument("--stage5-metrics", required=True, help="Stage-5 metrics.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-cases", type=int, default=103)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robust_normalize(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    selection = finite
    if mask is not None:
        selection = selection & np.asarray(mask, dtype=bool)
    selected = values[selection]
    if selected.size == 0:
        raise ValueError("normalization mask has no finite voxels")
    lower = max(float(selected.min()), 0.0)
    upper = float(selected.max())
    if upper <= lower:
        raise ValueError("image is constant after masking")
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32, copy=False)


def validate_geometry(
    reference: nib.spatialimages.SpatialImage,
    generated: nib.spatialimages.SpatialImage,
    segmentation: nib.spatialimages.SpatialImage,
) -> None:
    if reference.shape != generated.shape or reference.shape != segmentation.shape:
        raise ValueError(
            "shape mismatch: "
            f"reference={reference.shape}, generated={generated.shape}, seg={segmentation.shape}"
        )
    if not np.allclose(reference.affine, generated.affine, atol=1e-4, rtol=0.0):
        raise ValueError("affine mismatch between reference and generated T2W")
    if not np.allclose(reference.affine, segmentation.affine, atol=1e-4, rtol=0.0):
        raise ValueError("affine mismatch between reference T2W and segmentation")
    ref_spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    for name, image in (("generated", generated), ("seg", segmentation)):
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.allclose(ref_spacing, spacing, atol=1e-5, rtol=0.0):
            raise ValueError(f"spacing mismatch between reference and {name}")


def _crop_to_mask(mask: np.ndarray, padding: int = 4) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("metric mask is empty")
    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + padding + 1, np.asarray(mask.shape))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _masked_ssim(reference: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> float:
    roi = _crop_to_mask(mask)
    local_mask = mask[roi].astype(np.float32)
    ref_roi = reference[roi] * local_mask
    gen_roi = generated[roi] * local_mask
    minimum_dimension = min(ref_roi.shape)
    window = min(7, minimum_dimension if minimum_dimension % 2 == 1 else minimum_dimension - 1)
    if window < 3:
        return float("nan")
    return float(
        structural_similarity(
            ref_roi,
            gen_roi,
            data_range=1.0,
            win_size=window,
        )
    )


def compute_masked_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    region = np.asarray(mask, dtype=bool)
    voxel_count = int(region.sum())
    if voxel_count == 0:
        return {
            "voxel_count": 0,
            "ssim": float("nan"),
            "psnr": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
        }
    difference = generated[region].astype(np.float64) - reference[region].astype(np.float64)
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    return {
        "voxel_count": voxel_count,
        "ssim": _masked_ssim(reference, generated, region),
        "psnr": psnr,
        "mse": mse,
        "mae": mae,
    }


def build_region_masks(segmentation: np.ndarray) -> dict[str, np.ndarray]:
    seg = np.asarray(segmentation)
    return {
        "tumor_all": seg > 0,
        **{name: seg == value for value, name in LABEL_NAMES.items()},
    }


def lesion_size_class(volume_mm3: float) -> str:
    if volume_mm3 < 27.0:
        return "tiny"
    if volume_mm3 <= 275.0:
        return "small"
    return "large"


def extract_lesions(
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
) -> list[dict[str, Any]]:
    seg = np.asarray(segmentation)
    core = np.isin(seg, CORE_LABELS)
    components, component_count = ndimage.label(
        core,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    lesions: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        mask = components == component_id
        voxel_count = int(mask.sum())
        volume_mm3 = voxel_count * voxel_volume
        labels = sorted(int(value) for value in np.unique(seg[mask]) if int(value) in CORE_LABELS)
        labels_present = "+".join(LABEL_NAMES[value] for value in labels)
        centroid = tuple(int(round(value)) for value in ndimage.center_of_mass(mask))
        lesions.append(
            {
                "lesion_id": component_id,
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
                "size_class": lesion_size_class(volume_mm3),
                "labels_present": labels_present,
                "centroid": centroid,
                "_mask": mask,
            }
        )
    return lesions


def choose_review_focus(
    segmentation: np.ndarray,
    lesions: list[dict[str, Any]],
) -> tuple[tuple[int, int, int], str]:
    seg = np.asarray(segmentation)
    rc = seg == 4
    if rc.any():
        components, count = ndimage.label(rc, structure=ndimage.generate_binary_structure(3, 3))
        component_id = max(range(1, count + 1), key=lambda value: int((components == value).sum()))
        focus = tuple(int(round(value)) for value in ndimage.center_of_mass(components == component_id))
        return focus, "RC"
    if lesions:
        small_candidates = [row for row in lesions if row["size_class"] in {"tiny", "small"}]
        selected = min(
            small_candidates or lesions,
            key=lambda row: float(row["volume_mm3"]),
        )
        return tuple(selected["centroid"]), str(selected["size_class"])
    tumor = seg > 0
    if tumor.any():
        focus = tuple(int(round(value)) for value in ndimage.center_of_mass(tumor))
        return focus, "tumor_all"
    return tuple(int(value // 2) for value in seg.shape), "volume_center"


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gradients = np.gradient(np.asarray(image, dtype=np.float32))
    return np.sqrt(sum(component**2 for component in gradients), dtype=np.float32)


def region_contrast(
    image: np.ndarray,
    region: np.ndarray,
    brain_mask: np.ndarray,
) -> float:
    mask = np.asarray(region, dtype=bool)
    if not mask.any():
        return float("nan")
    inner = ndimage.binary_dilation(mask, iterations=2)
    outer = ndimage.binary_dilation(mask, iterations=5)
    shell = outer & ~inner & np.asarray(brain_mask, dtype=bool)
    if not shell.any():
        return float("nan")
    return float(np.mean(image[mask]) - np.mean(image[shell]))


def compute_artifact_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    brain_mask: np.ndarray,
    lesion_mask: np.ndarray,
) -> dict[str, float | int | bool]:
    brain = np.asarray(brain_mask, dtype=bool)
    lesion = np.asarray(lesion_mask, dtype=bool)
    if not brain.any():
        raise ValueError("brain mask is empty")
    ref_void = float(np.mean(reference[brain] <= 0.01))
    gen_void = float(np.mean(generated[brain] <= 0.01))
    lesion_void = float(np.mean(generated[lesion] <= 0.01)) if lesion.any() else float("nan")
    protected_external = ~ndimage.binary_dilation(brain, iterations=2)
    external_signal = (
        float(np.mean(generated[protected_external] > 0.02))
        if protected_external.any()
        else 0.0
    )

    reference_gradient = _gradient_magnitude(reference)
    generated_gradient = _gradient_magnitude(generated)
    evaluation_region = lesion if lesion.any() else brain
    ref_gradient_mean = float(np.mean(reference_gradient[evaluation_region]))
    gen_gradient_mean = float(np.mean(generated_gradient[evaluation_region]))
    blur_ratio = gen_gradient_mean / max(ref_gradient_mean, 1e-8)
    boundary = (
        ndimage.binary_dilation(evaluation_region, iterations=2)
        ^ ndimage.binary_erosion(evaluation_region, iterations=1)
    ) & brain
    boundary_gradient_mae = (
        float(np.mean(np.abs(generated_gradient[boundary] - reference_gradient[boundary])))
        if boundary.any()
        else float("nan")
    )
    ref_contrast = region_contrast(reference, evaluation_region, brain)
    gen_contrast = region_contrast(generated, evaluation_region, brain)
    contrast_error = (
        abs(gen_contrast - ref_contrast)
        if np.isfinite(ref_contrast) and np.isfinite(gen_contrast)
        else float("nan")
    )

    flags = {
        "flag_brain_void": gen_void - ref_void > 0.05,
        "flag_external_signal": external_signal > 0.001,
        "flag_blur": blur_ratio < 0.5,
        "flag_oversharpen": blur_ratio > 2.0,
        "flag_boundary_break": np.isfinite(boundary_gradient_mae) and boundary_gradient_mae > 0.15,
    }
    return {
        "brain_void_reference": ref_void,
        "brain_void_generated": gen_void,
        "brain_void_excess": max(gen_void - ref_void, 0.0),
        "lesion_void_fraction": lesion_void,
        "external_signal_fraction": external_signal,
        "lesion_gradient_reference": ref_gradient_mean,
        "lesion_gradient_generated": gen_gradient_mean,
        "lesion_blur_ratio": blur_ratio,
        "boundary_gradient_mae": boundary_gradient_mae,
        "lesion_contrast_reference": ref_contrast,
        "lesion_contrast_generated": gen_contrast,
        "lesion_contrast_abs_error": contrast_error,
        **flags,
        "artifact_flag_count": sum(bool(value) for value in flags.values()),
    }


def _plane(array: np.ndarray, orientation: str, focus: tuple[int, int, int]) -> np.ndarray:
    x, y, z = focus
    if orientation == "axial":
        plane = array[:, :, z]
    elif orientation == "coronal":
        plane = array[:, y, :]
    elif orientation == "sagittal":
        plane = array[x, :, :]
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return np.rot90(plane)


def render_montage(
    reference: np.ndarray,
    generated: np.ndarray,
    segmentation: np.ndarray,
    focus: tuple[int, int, int],
    case_id: str,
    annotations: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    difference = np.abs(generated - reference)
    nonzero = reference[reference > 0]
    vmax = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
    error_vmax = max(float(np.percentile(difference, 99.0)), 0.05)
    orientations = ("axial", "coronal", "sagittal")
    titles = ("Real T2W", "Generated T2W", "Absolute error", "Generated + seg")
    figure, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    seg_cmap = ListedColormap(MONTAGE_COLORS)

    for row, orientation in enumerate(orientations):
        ref_plane = _plane(reference, orientation, focus)
        gen_plane = _plane(generated, orientation, focus)
        diff_plane = _plane(difference, orientation, focus)
        seg_plane = _plane(segmentation, orientation, focus)
        planes = (ref_plane, gen_plane, diff_plane, gen_plane)
        for column, plane in enumerate(planes):
            axis = axes[row, column]
            if column == 2:
                axis.imshow(plane, cmap="magma", vmin=0.0, vmax=error_vmax)
            else:
                axis.imshow(plane, cmap="gray", vmin=0.0, vmax=vmax)
            if column == 3:
                overlay = np.ma.masked_where(seg_plane == 0, seg_plane)
                axis.imshow(overlay, cmap=seg_cmap, vmin=0, vmax=4, interpolation="nearest")
            if row == 0:
                axis.set_title(titles[column], fontsize=11)
            if column == 0:
                axis.set_ylabel(orientation.capitalize(), fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

    annotation_text = " | ".join(f"{key}={value}" for key, value in annotations.items())
    figure.suptitle(f"{case_id} | focus={focus} | {annotation_text}", fontsize=12)
    figure.savefig(output_path, dpi=120, facecolor="white")
    plt.close(figure)


def read_stage5_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "subject" not in rows[0]:
        raise ValueError(f"invalid stage-5 metrics CSV: {path}")
    identifiers = [row["subject"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("stage-5 metrics contains duplicate subjects")
    return rows


def find_case_file(case_dir: Path, case_id: str, suffix: str) -> Path:
    exact = case_dir / f"{case_id}-{suffix}.nii.gz"
    if exact.is_file():
        return exact
    alternatives = sorted(case_dir.glob(f"*{suffix}.nii*"))
    if len(alternatives) != 1:
        raise FileNotFoundError(f"expected one {suffix} file in {case_dir}, found {len(alternatives)}")
    return alternatives[0]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite_or_none(row.get(key, "")) for key in fieldnames})


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(row[key]) for row in rows if key in row and row[key] not in (None, "")],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "montages").mkdir()


def build_review_index(case_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(case_rows, key=lambda row: float(row["stage5_whole_ssim"]))
    worst_ids = {row["case_id"] for row in ordered[:10]}
    median_pool = ordered[max(0, len(ordered) * 4 // 10): max(1, len(ordered) * 6 // 10)]
    high_pool = ordered[max(0, len(ordered) * 8 // 10):]
    rng = np.random.default_rng(seed)

    def sample_ids(pool: list[dict[str, Any]], count: int) -> set[str]:
        if not pool:
            return set()
        selected = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        return {pool[int(index)]["case_id"] for index in np.atleast_1d(selected)}

    median_ids = sample_ids(median_pool, 5)
    high_ids = sample_ids(high_pool, 5)
    review_rows: list[dict[str, Any]] = []
    for row in case_rows:
        reasons: list[str] = []
        case_id = str(row["case_id"])
        if case_id in worst_ids:
            reasons.append("lowest_stage5_ssim")
        if bool(row["has_rc"]):
            reasons.append("RC")
        if int(row["tiny_lesion_count"]) > 0:
            reasons.append("tiny_lesion")
        if int(row["small_lesion_count"]) > 0:
            reasons.append("small_lesion")
        if int(row["artifact_flag_count"]) > 0:
            reasons.append("artifact_screen")
        if case_id in median_ids:
            reasons.append("random_median")
        if case_id in high_ids:
            reasons.append("random_high")
        priority = "high" if any(
            reason in reasons
            for reason in ("lowest_stage5_ssim", "artifact_screen", "RC")
        ) else "medium" if reasons else "routine"
        review_rows.append(
            {
                "case_id": case_id,
                "review_priority": priority,
                "review_reasons": ";".join(reasons) if reasons else "routine_all_case_review",
                "montage_path": f"montages/{case_id}.png",
                "stage5_whole_ssim": row["stage5_whole_ssim"],
                "saved_t2w_tumor_ssim": row["saved_t2w_tumor_ssim"],
                "artifact_flag_count": row["artifact_flag_count"],
            }
        )
    return sorted(
        review_rows,
        key=lambda row: (
            {"high": 0, "medium": 1, "routine": 2}[str(row["review_priority"])],
            float(row["stage5_whole_ssim"]),
        ),
    )


def build_report(summary: dict[str, Any], review_rows: list[dict[str, Any]]) -> str:
    high_priority = [row for row in review_rows if row["review_priority"] == "high"]
    lines = [
        "# G1 V3 103 例配对影像与病灶 QC 报告",
        "",
        "## 结论边界",
        "",
        "本报告只评价真实 T2W 与阶段 5 保存的生成 T2W 的影像、肿瘤区域和连通病灶质量。",
        "不包含冻结分割模型或官方 lesionwise 指标，不自动批准阶段 6。",
        "",
        "## 完整性",
        "",
        f"- 病例数：{summary['case_count']}",
        f"- montage 数：{summary['montage_count']}",
        f"- 区域记录数：{summary['region_row_count']}",
        f"- 连通病灶记录数：{summary['lesion_row_count']}",
        f"- 高优先级人工复核：{len(high_priority)}",
        "",
        "## 核心指标",
        "",
        f"- saved T2W tumor SSIM：{summary['metrics']['saved_t2w_tumor_ssim']}",
        f"- saved T2W tumor MAE：{summary['metrics']['saved_t2w_tumor_mae']}",
        f"- lesion contrast absolute error：{summary['metrics']['lesion_contrast_abs_error']}",
        f"- lesion blur ratio：{summary['metrics']['lesion_blur_ratio']}",
        "",
        "## 人工复核",
        "",
        "所有 103 例均有 montage。先按 `review_index.csv` 的 high、medium、routine 顺序检查，",
        "重点确认低 SSIM、RC、tiny/small lesion、空洞、脑外信号、模糊和边界异常筛查病例。",
        "",
        "## 阶段 6 Gate",
        "",
        "当前状态：`hold_for_review`。操作者完成人工复核后，才能改为 `approve_stage6` 或 `reject_and_retune`。",
        "最终训练价值由 real-only 与 real+V3 的固定 split 成对消融决定。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    real_root = Path(args.real_root).expanduser().resolve()
    synthetic_root = Path(args.synthetic_root).expanduser().resolve()
    stage5_path = Path(args.stage5_metrics).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for path, description in (
        (real_root, "real root"),
        (synthetic_root, "synthetic root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not stage5_path.is_file():
        raise FileNotFoundError(f"stage-5 metrics not found: {stage5_path}")

    stage5_rows = read_stage5_metrics(stage5_path)
    if len(stage5_rows) != args.expected_cases:
        raise ValueError(
            f"stage-5 case count {len(stage5_rows)} != expected {args.expected_cases}"
        )
    prepare_output_root(output_root, args.overwrite)

    case_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []

    for index, stage5 in enumerate(stage5_rows, start=1):
        case_id = stage5["subject"]
        case_dir = real_root / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"real case directory not found: {case_dir}")
        reference_path = find_case_file(case_dir, case_id, "t2w")
        seg_path = find_case_file(case_dir, case_id, "seg")
        generated_path = synthetic_root / f"{case_id}-t2w.nii.gz"
        if not generated_path.is_file():
            raise FileNotFoundError(f"generated T2W not found: {generated_path}")

        reference_image = nib.load(str(reference_path))
        generated_image = nib.load(str(generated_path))
        segmentation_image = nib.load(str(seg_path))
        validate_geometry(reference_image, generated_image, segmentation_image)
        reference_raw = reference_image.get_fdata(dtype=np.float32)
        generated_raw = generated_image.get_fdata(dtype=np.float32)
        segmentation = segmentation_image.get_fdata(dtype=np.float32)
        if not np.isfinite(reference_raw).all() or not np.isfinite(generated_raw).all():
            raise ValueError(f"{case_id}: T2W contains NaN/Inf")
        rounded_seg = np.rint(segmentation).astype(np.int16)
        if not np.allclose(segmentation, rounded_seg, atol=1e-6):
            raise ValueError(f"{case_id}: segmentation is not integer-valued")
        invalid_labels = sorted(set(np.unique(rounded_seg)) - {0, 1, 2, 3, 4})
        if invalid_labels:
            raise ValueError(f"{case_id}: invalid segmentation labels {invalid_labels}")

        reference = robust_normalize(reference_raw)
        generated = robust_normalize(generated_raw)
        brain = ndimage.binary_fill_holes(reference_raw > 0)
        regions = build_region_masks(rounded_seg)
        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        lesions = extract_lesions(rounded_seg, spacing)
        focus, focus_reason = choose_review_focus(rounded_seg, lesions)
        artifacts = compute_artifact_metrics(reference, generated, brain, regions["tumor_all"])

        whole_metrics = compute_masked_metrics(reference, generated, np.ones_like(brain))
        brain_metrics = compute_masked_metrics(reference, generated, brain)
        metrics_by_region: dict[str, dict[str, Any]] = {}
        for region_name, mask in regions.items():
            metrics = compute_masked_metrics(reference, generated, mask)
            ref_contrast = region_contrast(reference, mask, brain)
            gen_contrast = region_contrast(generated, mask, brain)
            contrast_error = (
                abs(gen_contrast - ref_contrast)
                if np.isfinite(ref_contrast) and np.isfinite(gen_contrast)
                else float("nan")
            )
            metrics_by_region[region_name] = metrics
            region_rows.append(
                {
                    "case_id": case_id,
                    "region": region_name,
                    "voxel_count": metrics["voxel_count"],
                    "volume_mm3": int(metrics["voxel_count"]) * voxel_volume,
                    **metrics,
                    "contrast_reference": ref_contrast,
                    "contrast_generated": gen_contrast,
                    "contrast_abs_error": contrast_error,
                }
            )

        for lesion in lesions:
            lesion_metrics = compute_masked_metrics(reference, generated, lesion["_mask"])
            ref_contrast = region_contrast(reference, lesion["_mask"], brain)
            gen_contrast = region_contrast(generated, lesion["_mask"], brain)
            lesion_rows.append(
                {
                    "case_id": case_id,
                    "lesion_id": lesion["lesion_id"],
                    "size_class": lesion["size_class"],
                    "voxel_count": lesion["voxel_count"],
                    "volume_mm3": lesion["volume_mm3"],
                    "labels_present": lesion["labels_present"],
                    "centroid_x": lesion["centroid"][0],
                    "centroid_y": lesion["centroid"][1],
                    "centroid_z": lesion["centroid"][2],
                    **lesion_metrics,
                    "contrast_reference": ref_contrast,
                    "contrast_generated": gen_contrast,
                    "contrast_abs_error": (
                        abs(gen_contrast - ref_contrast)
                        if np.isfinite(ref_contrast) and np.isfinite(gen_contrast)
                        else float("nan")
                    ),
                }
            )

        size_counts = {
            size: sum(row["size_class"] == size for row in lesions)
            for size in ("tiny", "small", "large")
        }
        tumor_metrics = metrics_by_region["tumor_all"]
        case_row = {
            "case_id": case_id,
            "case_index": index,
            "stage5_whole_ssim": float(stage5["whole_SSIM"]),
            "stage5_whole_psnr": float(stage5["whole_PSNR"]),
            "stage5_brain_ssim": float(stage5["brain_SSIM"]),
            "stage5_brain_psnr": float(stage5["brain_PSNR"]),
            "saved_t2w_whole_ssim": whole_metrics["ssim"],
            "saved_t2w_whole_psnr": whole_metrics["psnr"],
            "saved_t2w_whole_mae": whole_metrics["mae"],
            "saved_t2w_brain_ssim": brain_metrics["ssim"],
            "saved_t2w_brain_psnr": brain_metrics["psnr"],
            "saved_t2w_brain_mae": brain_metrics["mae"],
            "saved_t2w_tumor_ssim": tumor_metrics["ssim"],
            "saved_t2w_tumor_psnr": tumor_metrics["psnr"],
            "saved_t2w_tumor_mae": tumor_metrics["mae"],
            "lesion_count": len(lesions),
            "tiny_lesion_count": size_counts["tiny"],
            "small_lesion_count": size_counts["small"],
            "large_lesion_count": size_counts["large"],
            "has_rc": bool(regions["RC"].any()),
            "focus_x": focus[0],
            "focus_y": focus[1],
            "focus_z": focus[2],
            "focus_reason": focus_reason,
            **artifacts,
        }
        case_rows.append(case_row)
        render_montage(
            reference,
            generated,
            rounded_seg,
            focus,
            case_id,
            {
                "stage5_whole_ssim": f"{case_row['stage5_whole_ssim']:.3f}",
                "tumor_ssim": f"{float(case_row['saved_t2w_tumor_ssim']):.3f}",
                "focus_reason": focus_reason,
            },
            output_root / "montages" / f"{case_id}.png",
        )
        print(f"[{index}/{len(stage5_rows)}] {case_id}: lesions={len(lesions)} focus={focus_reason}")

    review_rows = build_review_index(case_rows, args.seed)
    write_csv(output_root / "case_metrics.csv", case_rows)
    write_csv(output_root / "region_metrics.csv", region_rows)
    write_csv(output_root / "lesion_metrics.csv", lesion_rows)
    write_csv(output_root / "review_index.csv", review_rows)

    summary = {
        "case_count": len(case_rows),
        "montage_count": len(list((output_root / "montages").glob("*.png"))),
        "region_row_count": len(region_rows),
        "lesion_row_count": len(lesion_rows),
        "rc_case_count": sum(bool(row["has_rc"]) for row in case_rows),
        "tiny_lesion_case_count": sum(int(row["tiny_lesion_count"]) > 0 for row in case_rows),
        "small_lesion_case_count": sum(int(row["small_lesion_count"]) > 0 for row in case_rows),
        "artifact_flag_case_count": sum(int(row["artifact_flag_count"]) > 0 for row in case_rows),
        "review_priority_counts": {
            priority: sum(row["review_priority"] == priority for row in review_rows)
            for priority in ("high", "medium", "routine")
        },
        "metrics": {
            key: numeric_summary(case_rows, key)
            for key in (
                "saved_t2w_whole_ssim",
                "saved_t2w_brain_ssim",
                "saved_t2w_tumor_ssim",
                "saved_t2w_tumor_mae",
                "lesion_contrast_abs_error",
                "lesion_blur_ratio",
                "boundary_gradient_mae",
            )
        },
        "screening_thresholds": {
            "brain_void_excess": 0.05,
            "external_signal_fraction": 0.001,
            "lesion_blur_ratio_low": 0.5,
            "lesion_blur_ratio_high": 2.0,
            "boundary_gradient_mae": 0.15,
        },
        "stage6_gate": "hold_for_review",
    }
    (output_root / "summary.json").write_text(
        json.dumps(_finite_or_none(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "QC_REPORT.md").write_text(
        build_report(summary, review_rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_finite_or_none(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
