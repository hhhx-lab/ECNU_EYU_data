#!/usr/bin/env python3
"""Build reproducible visual-review montages for flagged G1 V3 completions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CORE_LABELS = (1, 3, 4)
SEG_CMAP = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.95, 0.20, 0.20, 0.72),
        (0.20, 0.85, 0.30, 0.60),
        (1.00, 0.85, 0.10, 0.78),
        (0.10, 0.75, 1.00, 0.78),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--qc-metrics", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-lesions", type=int, default=8)
    return parser.parse_args()


def normalize_volume(array: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    selected = values[finite & brain_mask]
    if selected.size == 0:
        raise ValueError("normalization mask has no finite voxels")
    lower, upper = np.percentile(selected, [1.0, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("image has no usable intensity range")
    output = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    output[~finite] = 0.0
    return output


def plane(array: np.ndarray, orientation: str, focus: tuple[int, int, int]) -> np.ndarray:
    x, y, z = focus
    if orientation == "axial":
        result = array[:, :, z]
    elif orientation == "coronal":
        result = array[:, y, :]
    elif orientation == "sagittal":
        result = array[x, :, :]
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return np.rot90(result)


def crop_cube(
    array: np.ndarray, focus: tuple[int, int, int], width: int = 72
) -> tuple[np.ndarray, tuple[int, int, int]]:
    starts = []
    stops = []
    local = []
    for coordinate, size in zip(focus, array.shape):
        start = max(0, min(int(coordinate) - width // 2, max(0, size - width)))
        stop = min(size, start + width)
        starts.append(start)
        stops.append(stop)
        local.append(int(coordinate) - start)
    cropped = array[
        starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
    ]
    return cropped, tuple(local)


def component_rows(
    segmentation: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    lesion_mask = np.isin(segmentation, CORE_LABELS)
    if not lesion_mask.any():
        lesion_mask = segmentation > 0
    components, count = ndimage.label(
        lesion_mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    rows: list[dict[str, object]] = []
    for component_id in range(1, count + 1):
        mask = components == component_id
        voxels = int(mask.sum())
        if voxels == 0:
            continue
        volume = voxels * voxel_volume
        centroid = tuple(
            int(round(value)) for value in ndimage.center_of_mass(mask)
        )
        rows.append(
            {
                "component_id": component_id,
                "voxels": voxels,
                "volume_mm3": volume,
                "tiny": volume < 27.0,
                "centroid": centroid,
                "z_extent": int(np.ptp(np.argwhere(mask)[:, 2]) + 1),
            }
        )
    return components, rows


def select_components(
    rows: list[dict[str, object]], max_lesions: int
) -> list[dict[str, object]]:
    if len(rows) <= max_lesions:
        return sorted(rows, key=lambda row: float(row["volume_mm3"]))
    smallest = sorted(rows, key=lambda row: float(row["volume_mm3"]))[
        : max(1, max_lesions - 2)
    ]
    largest = sorted(
        rows, key=lambda row: float(row["volume_mm3"]), reverse=True
    )[:2]
    selected: list[dict[str, object]] = []
    seen: set[int] = set()
    for row in [*smallest, *largest]:
        component_id = int(row["component_id"])
        if component_id not in seen:
            selected.append(row)
            seen.add(component_id)
    return selected[:max_lesions]


def choose_focus(
    rows: list[dict[str, object]], tiny_flag: bool, z_flag: bool, shape: tuple[int, ...]
) -> tuple[int, int, int]:
    if not rows:
        return tuple(int(size // 2) for size in shape[:3])
    if tiny_flag:
        selected = min(rows, key=lambda row: float(row["volume_mm3"]))
    elif z_flag:
        selected = max(rows, key=lambda row: int(row["z_extent"]))
    else:
        selected = max(rows, key=lambda row: float(row["volume_mm3"]))
    return tuple(int(value) for value in selected["centroid"])


def show_image(
    axis,
    image: np.ndarray,
    segmentation: np.ndarray | None = None,
    title: str = "",
) -> None:
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    if segmentation is not None:
        overlay = np.ma.masked_where(segmentation == 0, segmentation)
        axis.imshow(
            overlay,
            cmap=SEG_CMAP,
            vmin=0,
            vmax=4,
            interpolation="nearest",
        )
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def render_overview(
    case_id: str,
    images: dict[str, np.ndarray],
    segmentation: np.ndarray,
    focus: tuple[int, int, int],
    flags: str,
    output_path: Path,
) -> None:
    orientations = ("axial", "coronal", "sagittal")
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, orientation in enumerate(orientations):
        seg_plane = plane(segmentation, orientation, focus)
        for column, modality in enumerate(("t1c", "t2w", "t2f")):
            show_image(
                axes[row, column],
                plane(images[modality], orientation, focus),
                seg_plane,
                f"{orientation} {modality}",
            )
        show_image(
            axes[row, 3],
            plane(images["t2w"], orientation, focus),
            seg_plane,
            f"{orientation} generated T2W + seg",
        )
    figure.suptitle(f"{case_id} | {flags} | focus={focus}", fontsize=13)
    figure.savefig(output_path, dpi=130, facecolor="white")
    plt.close(figure)


def render_lesions(
    case_id: str,
    images: dict[str, np.ndarray],
    segmentation: np.ndarray,
    components: np.ndarray,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 5, figsize=(15, max(3.0, 2.8 * len(rows))), squeeze=False,
        constrained_layout=True,
    )
    for row_index, lesion in enumerate(rows):
        focus = tuple(int(value) for value in lesion["centroid"])
        crops = {}
        local_focus = None
        for modality in ("t1c", "t2w", "t2f"):
            crops[modality], local_focus = crop_cube(images[modality], focus)
        seg_crop, local_focus = crop_cube(segmentation, focus)
        assert local_focus is not None
        axial_seg = plane(seg_crop, "axial", local_focus)
        show_image(
            axes[row_index, 0], plane(crops["t1c"], "axial", local_focus),
            axial_seg, "t1c axial",
        )
        show_image(
            axes[row_index, 1], plane(crops["t2w"], "axial", local_focus),
            axial_seg, "T2W axial",
        )
        show_image(
            axes[row_index, 2], plane(crops["t2w"], "coronal", local_focus),
            plane(seg_crop, "coronal", local_focus), "T2W coronal",
        )
        show_image(
            axes[row_index, 3], plane(crops["t2w"], "sagittal", local_focus),
            plane(seg_crop, "sagittal", local_focus), "T2W sagittal",
        )
        show_image(
            axes[row_index, 4], plane(crops["t2f"], "axial", local_focus),
            axial_seg, "t2f axial",
        )
        axes[row_index, 0].set_ylabel(
            f"cc{lesion['component_id']}\n{float(lesion['volume_mm3']):.1f} mm3",
            fontsize=9,
        )
    figure.suptitle(f"{case_id} lesion-focused review", fontsize=13)
    figure.savefig(output_path, dpi=130, facecolor="white")
    plt.close(figure)


def render_z_stack(
    case_id: str,
    images: dict[str, np.ndarray],
    segmentation: np.ndarray,
    focus: tuple[int, int, int],
    output_path: Path,
) -> None:
    offsets = (-4, -2, 0, 2, 4)
    figure, axes = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
    for row, modality in enumerate(("t1c", "t2w", "t2f")):
        for column, offset in enumerate(offsets):
            z = min(max(0, focus[2] + offset), segmentation.shape[2] - 1)
            current_focus = (focus[0], focus[1], z)
            show_image(
                axes[row, column],
                plane(images[modality], "axial", current_focus),
                plane(segmentation, "axial", current_focus),
                f"{modality} z={z} ({offset:+d})",
            )
    figure.suptitle(f"{case_id} adjacent-slice continuity", fontsize=13)
    figure.savefig(output_path, dpi=130, facecolor="white")
    plt.close(figure)


def render_contact_panel(
    case_id: str,
    flags: str,
    images: dict[str, np.ndarray],
    segmentation: np.ndarray,
    focus: tuple[int, int, int],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    cropped_images = {}
    local_focus = None
    for modality in ("t1c", "t2w", "t2f"):
        cropped_images[modality], local_focus = crop_cube(images[modality], focus)
    seg_crop, local_focus = crop_cube(segmentation, focus)
    assert local_focus is not None
    panels = [
        ("t1c axial", "t1c", "axial"),
        ("T2W axial", "t2w", "axial"),
        ("T2W coronal", "t2w", "coronal"),
        ("T2W sagittal", "t2w", "sagittal"),
        ("t2f axial", "t2f", "axial"),
    ]
    for column, (title, modality, orientation) in enumerate(panels):
        show_image(
            axes[column],
            plane(cropped_images[modality], orientation, local_focus),
            plane(seg_crop, orientation, local_focus),
            title,
        )
    figure.suptitle(f"{case_id} | {flags} | focus={focus}", fontsize=12)
    figure.savefig(output_path, dpi=140, facecolor="white")
    plt.close(figure)


def build_contact_sheets(
    panels: list[dict[str, object]], output_dir: Path, cases_per_sheet: int = 4
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_index in range(math.ceil(len(panels) / cases_per_sheet)):
        page = panels[page_index * cases_per_sheet : (page_index + 1) * cases_per_sheet]
        figure, axes = plt.subplots(
            len(page), 1, figsize=(17, 3.7 * len(page)), squeeze=False,
            constrained_layout=True,
        )
        for row_index, panel in enumerate(page):
            panel_path = Path(str(panel["panel_path"]))
            axes[row_index, 0].imshow(plt.imread(panel_path))
            axes[row_index, 0].set_axis_off()
        output_path = output_dir / f"review_{page_index + 1:02d}.png"
        figure.savefig(output_path, dpi=140, facecolor="white")
        plt.close(figure)
        paths.append(output_path)
    return paths


def load_case(run_root: Path, case_id: str):
    case_dir = run_root / case_id
    raw_images: dict[str, np.ndarray] = {}
    reference_image = None
    for modality in MODALITIES:
        image = nib.load(str(case_dir / f"{case_id}-{modality}.nii.gz"))
        raw_images[modality] = np.asanyarray(image.dataobj).astype(np.float32)
        if reference_image is None:
            reference_image = image
    seg_image = nib.load(str(case_dir / f"{case_id}-seg.nii.gz"))
    segmentation = np.asanyarray(seg_image.dataobj).astype(np.int16)
    shapes = {array.shape for array in [*raw_images.values(), segmentation]}
    if len(shapes) != 1:
        raise ValueError(f"{case_id}: shape mismatch {shapes}")
    brain_mask = np.logical_or.reduce(
        [np.abs(array) > 1e-6 for array in raw_images.values()]
    )
    images = {
        modality: normalize_volume(array, brain_mask)
        for modality, array in raw_images.items()
    }
    spacing = tuple(float(value) for value in seg_image.header.get_zooms()[:3])
    return images, segmentation, spacing


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    qc_path = args.qc_metrics.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    montage_dir = output_root / "montages"
    lesion_dir = output_root / "lesions"
    z_stack_dir = output_root / "z_stacks"
    contact_panel_dir = output_root / "contact_panels"
    for path in (montage_dir, lesion_dir, z_stack_dir, contact_panel_dir):
        path.mkdir()

    qc = pd.read_csv(qc_path, keep_default_na=False)
    flagged = qc[
        qc["manual_review_reason"].str.contains(
            "tiny_ratio_high|z_discontinuity", regex=True
        )
    ].copy()
    if flagged.empty:
        raise SystemExit("no tiny-ratio or z-continuity review cases were found")

    review_rows: list[dict[str, object]] = []
    contact_panels: list[dict[str, object]] = []
    for _, qc_row in flagged.sort_values("synthetic_raw_id").iterrows():
        case_id = str(qc_row["synthetic_raw_id"])
        reason = str(qc_row["manual_review_reason"])
        tiny_flag = "tiny_ratio_high" in reason
        z_flag = "z_discontinuity" in reason
        flags = "+".join(
            name for name, enabled in (("tiny", tiny_flag), ("z", z_flag)) if enabled
        )
        images, segmentation, spacing = load_case(run_root, case_id)
        components, lesions = component_rows(segmentation, spacing)
        selected = select_components(lesions, args.max_lesions)
        focus = choose_focus(lesions, tiny_flag, z_flag, segmentation.shape)
        overview_path = montage_dir / f"{case_id}.png"
        lesion_path = lesion_dir / f"{case_id}.png"
        render_overview(case_id, images, segmentation, focus, flags, overview_path)
        render_lesions(
            case_id, images, segmentation, components, selected, lesion_path
        )
        contact_panel_path = contact_panel_dir / f"{case_id}.png"
        render_contact_panel(
            case_id,
            flags,
            images,
            segmentation,
            focus,
            contact_panel_path,
        )
        z_stack_path = ""
        if z_flag:
            z_path = z_stack_dir / f"{case_id}.png"
            render_z_stack(case_id, images, segmentation, focus, z_path)
            z_stack_path = str(z_path.relative_to(output_root))
        review_rows.append(
            {
                "case_id": case_id,
                "flags": flags,
                "lesion_count": len(lesions),
                "tiny_lesion_count": sum(bool(row["tiny"]) for row in lesions),
                "focus_x": focus[0],
                "focus_y": focus[1],
                "focus_z": focus[2],
                "overview_path": str(overview_path.relative_to(output_root)),
                "lesion_path": str(lesion_path.relative_to(output_root)),
                "z_stack_path": z_stack_path,
                "contact_panel_path": str(contact_panel_path.relative_to(output_root)),
                "visual_decision": "pending",
                "visual_notes": "",
            }
        )
        contact_panels.append(
            {
                "case_id": case_id,
                "flags": flags,
                "panel_path": contact_panel_path,
            }
        )
        print(f"RENDERED {case_id} flags={flags} lesions={len(lesions)}", flush=True)

    contact_paths = build_contact_sheets(
        contact_panels, output_root / "contact_sheets"
    )
    fields = list(review_rows[0])
    with (output_root / "review_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    summary = {
        "run_root": str(run_root),
        "qc_metrics": str(qc_path),
        "case_count": len(review_rows),
        "tiny_flag_count": sum("tiny" in str(row["flags"]) for row in review_rows),
        "z_flag_count": sum("z" in str(row["flags"]) for row in review_rows),
        "overlap_count": sum(row["flags"] == "tiny+z" for row in review_rows),
        "overview_count": len(list(montage_dir.glob("*.png"))),
        "lesion_sheet_count": len(list(lesion_dir.glob("*.png"))),
        "z_stack_count": len(list(z_stack_dir.glob("*.png"))),
        "contact_panel_count": len(list(contact_panel_dir.glob("*.png"))),
        "contact_sheet_count": len(contact_paths),
        "review_status": "pending_manual_visual_review",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
