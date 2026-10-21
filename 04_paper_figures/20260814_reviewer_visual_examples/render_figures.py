#!/usr/bin/env python3
"""Render auditable paired qualitative examples for the manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "inputs"
G1_METRICS = (
    ROOT.parent.parent
    / "G1/results/experimental_unvalidated_single_model_eval_20260812_r4/ensemble_r4/metrics.csv"
)

SEGMENTATION_CASES = (
    "BraTS-MET-00025-000",  # Three labels; macro label Dice 0.928.
    "BraTS-MET-01060-001",  # Distinct peripheral morphology; macro label Dice 0.904.
)

HIGH_AGREEMENT_SEGMENTATION_CASES = (
    "BraTS-MET-00532-000",  # Highest fixed-103 macro label Dice: 0.971.
    "BraTS-MET-00552-000",  # Second-highest fixed-103 macro label Dice: 0.951.
)

SEGMENTATION_OPTIONS = {
    "A": HIGH_AGREEMENT_SEGMENTATION_CASES,
    "B": SEGMENTATION_CASES,
    "C": ("BraTS-MET-00532-000", "BraTS-MET-00025-000"),
    "D": ("BraTS-MET-00552-000", "BraTS-MET-01060-001"),
}

SLICE_OVERRIDES = {
    # Its maximum-area slice shows SNFH alone; this slice also contains ET.
    "BraTS-MET-00552-000": 115,
}

T2W_CASES = (
    "BraTS-MET-00009-000",  # Clear lesion on T2W; brain SSIM 0.798.
    "BraTS-MET-00807-000",  # Different morphology; upper-quartile r4 brain SSIM.
)

LABELS = {
    1: ("NETC", "#D33F3F"),
    2: ("SNFH", "#4CAF50"),
    3: ("ET", "#2F6DA8"),
    4: ("RC", "#E4B62A"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"Invalid 3D finite NIfTI: {path}")
    return data, image


def display_slice(volume: np.ndarray, z_index: int) -> np.ndarray:
    return np.rot90(np.asarray(volume[:, :, z_index]))


def intensity_window(image: np.ndarray) -> tuple[float, float]:
    foreground = image[np.isfinite(image) & (image > 0)]
    if foreground.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(foreground, (1.0, 99.5))
    if not upper > lower:
        return float(lower), float(lower + 1.0)
    return float(lower), float(upper)


def robust_normalize_volume(image: np.ndarray) -> np.ndarray:
    """Match the project's default robust_normalize(..., percentile=(0, 100))."""
    image = np.asarray(image, dtype=np.float32)
    lower, upper = np.percentile(image, (0.0, 100.0))
    lower = max(float(lower), 0.0)
    upper = float(upper)
    if not upper > lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def normalize_with_window(image: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lower, upper = window
    return np.clip((image.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0)


def square_crop(mask: np.ndarray, minimum: int, expansion: float, shape: tuple[int, int]):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0, shape[0], 0, shape[1])
    lower = coords.min(axis=0)
    upper = coords.max(axis=0) + 1
    center = (lower + upper) / 2.0
    side = max(minimum, int(np.ceil(max(upper - lower) * expansion)))
    side = min(side, min(shape))
    row0 = int(round(center[0] - side / 2))
    col0 = int(round(center[1] - side / 2))
    row0 = min(max(row0, 0), shape[0] - side)
    col0 = min(max(col0, 0), shape[1] - side)
    return (row0, row0 + side, col0, col0 + side)


def crop(array: np.ndarray, bounds):
    row0, row1, col0, col1 = bounds
    return array[row0:row1, col0:col1]


def macro_label_dice(reference: np.ndarray, prediction: np.ndarray):
    values = {}
    for label in LABELS:
        ref = reference == label
        pred = prediction == label
        denominator = int(ref.sum() + pred.sum())
        if denominator:
            values[str(label)] = float(2 * np.count_nonzero(ref & pred) / denominator)
    return values, float(np.mean(list(values.values())))


def overlay_segmentation(axis, base, segmentation, bounds):
    base_crop = crop(base, bounds)
    seg_crop = crop(segmentation, bounds)
    axis.imshow(base_crop, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")

    rgba = np.zeros((*seg_crop.shape, 4), dtype=np.float32)
    for label, (_, color) in LABELS.items():
        rgb = mpl.colors.to_rgb(color)
        selected = seg_crop == label
        rgba[selected, :3] = rgb
        rgba[selected, 3] = 0.62
        if selected.any():
            axis.contour(
                selected.astype(np.uint8),
                levels=[0.5],
                colors=[color],
                linewidths=1.25,
            )
    axis.imshow(rgba, interpolation="nearest")


def read_g1_metrics():
    with G1_METRICS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_case = {row["subject"]: row for row in rows}
    ordered = sorted(rows, key=lambda row: float(row["brain_SSIM"]))
    rank = {row["subject"]: index + 1 for index, row in enumerate(ordered)}
    return by_case, rank, len(rows)


def prepare_cases():
    metrics, ranks, cohort_size = read_g1_metrics()
    prepared = []
    case_ids = tuple(
        dict.fromkeys(
            (*SEGMENTATION_CASES, *HIGH_AGREEMENT_SEGMENTATION_CASES, *T2W_CASES)
        )
    )
    for case_id in case_ids:
        case_root = INPUT_ROOT / case_id
        paths = {"seg_gt": case_root / f"{case_id}-seg.nii.gz"}
        if case_id in (*SEGMENTATION_CASES, *HIGH_AGREEMENT_SEGMENTATION_CASES):
            paths.update(
                {
                    "t1c": case_root / f"{case_id}-t1c.nii.gz",
                    "seg_prediction": case_root / f"{case_id}-seg-pred-original-E.nii.gz",
                }
            )
        if case_id in T2W_CASES:
            paths.update(
                {
                    "t2w_real": case_root / f"{case_id}-t2w.nii.gz",
                    "t2w_ensemble": case_root / f"{case_id}-t2w-ensemble-r4.nii.gz",
                }
            )
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)

        arrays = {}
        images = {}
        for key, path in paths.items():
            arrays[key], images[key] = load_canonical(path)
        shapes = {key: value.shape for key, value in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Shape mismatch for {case_id}: {shapes}")
        reference_affine = images["seg_gt"].affine
        if any(
            not np.allclose(image.affine, reference_affine, atol=1e-4, rtol=0.0)
            for image in images.values()
        ):
            raise ValueError(f"Affine mismatch for {case_id}")

        lesion_per_slice = (arrays["seg_gt"] > 0).sum(axis=(0, 1))
        z_index = SLICE_OVERRIDES.get(case_id, int(np.argmax(lesion_per_slice)))
        gt_slice = display_slice(arrays["seg_gt"], z_index).astype(np.int16)
        case = {
            "case_id": case_id,
            "figure_roles": [
                role
                for role, selected in (
                    ("segmentation", SEGMENTATION_CASES),
                    ("segmentation_high_agreement", HIGH_AGREEMENT_SEGMENTATION_CASES),
                    ("t2w_reconstruction", T2W_CASES),
                )
                if case_id in selected
            ],
            "paths": paths,
            "arrays": arrays,
            "z_index_canonical": z_index,
            "lesion_voxels_on_slice": int(lesion_per_slice[z_index]),
            "gt_slice": gt_slice,
            "shape": list(arrays["seg_gt"].shape),
            "affine": reference_affine.tolist(),
        }

        if case_id in (*SEGMENTATION_CASES, *HIGH_AGREEMENT_SEGMENTATION_CASES):
            pred_slice = display_slice(arrays["seg_prediction"], z_index).astype(np.int16)
            t1c_slice = display_slice(
                robust_normalize_volume(arrays["t1c"]), z_index
            ).astype(np.float32)
            t1c_window = intensity_window(t1c_slice)
            case.update(
                {
                    "t1c_display": normalize_with_window(t1c_slice, t1c_window),
                    "pred_slice": pred_slice,
                    "lesion_bounds": square_crop(
                        (gt_slice > 0) | (pred_slice > 0),
                        minimum=max(80, int(round(min(gt_slice.shape) * 0.38))),
                        expansion=2.2,
                        shape=gt_slice.shape,
                    ),
                    "t1c_window": t1c_window,
                }
            )
            case["dice_by_label"], case["macro_label_dice"] = macro_label_dice(
                arrays["seg_gt"], arrays["seg_prediction"]
            )

        if case_id in T2W_CASES:
            real_t2w_slice = display_slice(
                robust_normalize_volume(arrays["t2w_real"]), z_index
            ).astype(np.float32)
            ensemble_t2w_slice = display_slice(
                np.clip(arrays["t2w_ensemble"].astype(np.float32), 0.0, 1.0),
                z_index,
            ).astype(np.float32)
            t2w_window = intensity_window(real_t2w_slice)
            case.update(
                {
                    "real_t2w_display": normalize_with_window(real_t2w_slice, t2w_window),
                    "ensemble_t2w_display": normalize_with_window(ensemble_t2w_slice, t2w_window),
                    "brain_bounds": square_crop(
                        real_t2w_slice > t2w_window[0],
                        minimum=max(128, int(round(min(gt_slice.shape) * 0.75))),
                        expansion=1.08,
                        shape=gt_slice.shape,
                    ),
                    "t2w_window_from_real": t2w_window,
                    "g1_metrics": {
                        key: float(value)
                        for key, value in metrics[case_id].items()
                        if key != "subject"
                    },
                    "brain_ssim_rank_ascending": ranks[case_id],
                    "cohort_size": cohort_size,
                }
            )

        prepared.append(case)
    return prepared


def style_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#C9CED6")
        spine.set_linewidth(0.8)


def add_inside_label(axis, text, *, align="left"):
    x = 0.025 if align == "left" else 0.975
    axis.text(
        x,
        0.975,
        text,
        transform=axis.transAxes,
        ha=align,
        va="top",
        fontsize=8.6,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.08",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.64,
        },
        zorder=10,
    )


def render_segmentation(
    cases,
    *,
    output_stem="segmentation_gt_vs_prediction",
    title="Qualitative segmentation results",
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.35), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.095, top=0.875, wspace=0.035, hspace=0.07)
    fig.suptitle(title, y=0.982, fontsize=14, fontweight="bold")
    axes[0, 0].set_title("Ground truth", fontsize=11.5, fontweight="bold", pad=7)
    axes[0, 1].set_title("nnU-Net prediction (model E)", fontsize=11.5, fontweight="bold", pad=7)

    for row, case in enumerate(cases):
        overlay_segmentation(
            axes[row, 0], case["t1c_display"], case["gt_slice"], case["lesion_bounds"]
        )
        overlay_segmentation(
            axes[row, 1], case["t1c_display"], case["pred_slice"], case["lesion_bounds"]
        )
        for axis in axes[row]:
            style_axis(axis)
        add_inside_label(axes[row, 0], f"Example {row + 1}  |  {case['case_id']}")
        add_inside_label(
            axes[row, 1],
            f"whole-volume macro label Dice: {case['macro_label_dice']:.3f}",
            align="right",
        )

    legend_handles = [
        Patch(facecolor=color, edgecolor=color, label=name)
        for _, (name, color) in LABELS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9.5,
        handlelength=1.3,
    )
    fig.text(
        0.5,
        0.008,
        "T1-weighted contrast-enhanced background; identical slice, crop, and intensity window within each row.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#59616B",
    )
    for extension in ("png", "pdf", "svg"):
        path = ROOT / f"{output_stem}.{extension}"
        fig.savefig(path, dpi=600 if extension == "png" else None, facecolor="white")
    plt.close(fig)


def render_t2w(cases):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.065, top=0.875, wspace=0.035, hspace=0.07)
    fig.suptitle("Qualitative missing-modality reconstruction", y=0.975, fontsize=14, fontweight="bold")
    axes[0, 0].set_title("Real T2W", fontsize=11.5, fontweight="bold", pad=7)
    axes[0, 1].set_title("Ensemble reconstruction", fontsize=11.5, fontweight="bold", pad=7)

    for row, case in enumerate(cases):
        bounds = case["brain_bounds"]
        axes[row, 0].imshow(
            crop(case["real_t2w_display"], bounds),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        axes[row, 1].imshow(
            crop(case["ensemble_t2w_display"], bounds),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        lesion = crop(case["gt_slice"] > 0, bounds)
        for axis in axes[row]:
            if lesion.any():
                axis.contour(lesion.astype(np.uint8), levels=[0.5], colors=["#E4B62A"], linewidths=0.8)
            style_axis(axis)
        add_inside_label(axes[row, 0], f"Example {row + 1}  |  {case['case_id']}")
        add_inside_label(
            axes[row, 1],
            f"brain SSIM: {case['g1_metrics']['brain_SSIM']:.3f}  |  PSNR: {case['g1_metrics']['brain_PSNR']:.2f} dB",
            align="right",
        )

    fig.text(
        0.5,
        0.018,
        "Yellow contour denotes the ground-truth tumor region. Each pair shares the real-T2W intensity window.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#59616B",
    )
    for extension in ("png", "pdf", "svg"):
        path = ROOT / f"t2w_real_vs_ensemble.{extension}"
        fig.savefig(path, dpi=600 if extension == "png" else None, facecolor="white")
    plt.close(fig)


def render_option_contact_sheet():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.035, top=0.955, wspace=0.04, hspace=0.08)
    for axis, option in zip(axes.flat, SEGMENTATION_OPTIONS):
        image = plt.imread(ROOT / f"segmentation_option_{option}.png")
        axis.imshow(image)
        axis.set_title(f"Option {option}", fontsize=14, fontweight="bold", pad=6)
        axis.axis("off")
    path = ROOT / "segmentation_options_contact_sheet.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def write_manifest(cases):
    output_paths = [
        ROOT / f"{stem}.{extension}"
        for stem in (
            "segmentation_gt_vs_prediction",
            "segmentation_high_agreement_gt_vs_prediction",
            *[f"segmentation_option_{option}" for option in SEGMENTATION_OPTIONS],
            "t2w_real_vs_ensemble",
        )
        for extension in ("png", "pdf", "svg")
    ]
    serializable = []
    for case in cases:
        serializable.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"arrays", "t1c_display", "real_t2w_display", "ensemble_t2w_display", "gt_slice", "pred_slice"}
            }
        )
        serializable[-1]["paths"] = {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in case["paths"].items()
        }
    manifest = {
        "status": "experimental_unvalidated",
        "operator_approved": False,
        "dataset_role": "fixed_internal_validation_103",
        "selection_rules": {
            "segmentation": (
                "Two illustrative fixed-103 validation cases were selected for clear tumor visibility, "
                "three visible annotation labels, strong original-E whole-volume overlap, and distinct "
                "lesion morphologies. They are not claimed to be random or statistically representative."
            ),
            "segmentation_high_agreement": (
                "The two fixed-103 cases with the highest original-E whole-volume macro label Dice "
                "were selected as best-case, high-agreement examples. This deterministic selection "
                "is not representative of cohort-level performance and does not establish statistical significance."
            ),
            "t2w_reconstruction": (
                "Two illustrative fixed-103 validation cases were selected for clear tumor visibility "
                "and distinct lesion morphologies; one case is near the upper quartile of G1 r4 Ensemble "
                "brain SSIM. They are not claimed to be random or statistically representative."
            ),
        },
        "figure_case_ids": {
            "segmentation": list(SEGMENTATION_CASES),
            "segmentation_high_agreement": list(HIGH_AGREEMENT_SEGMENTATION_CASES),
            "segmentation_options": {
                option: list(case_ids)
                for option, case_ids in SEGMENTATION_OPTIONS.items()
            },
            "t2w_reconstruction": list(T2W_CASES),
        },
        "segmentation_model": {
            "name": "original E",
            "architecture": "nnU-Net",
            "checkpoint_sha256": "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267",
            "selection_evidence": (
                "S2/results/s2_official_validation_inference_20260727_r1/"
                "selection_evidence/MODEL_SELECTION.json"
            ),
        },
        "t2w_model": {
            "name": "G1 r4 Ensemble (EncDec + BBDM)",
            "synthesis_type_as_recorded": "ensamble",
            "result_contract": str(G1_METRICS.relative_to(ROOT.parent.parent)),
            "interpretation_boundary": (
                "Qualitative and reconstruction-metric evidence only; this figure does not "
                "establish downstream nnU-Net segmentation benefit."
            ),
        },
        "display_contract": {
            "orientation": "nibabel closest-canonical, axial slice rotated 90 degrees for display",
            "slice_rule": (
                "argmax GT non-background voxel count over axial slices, except the documented "
                "BraTS-MET-00552-000 override to slice 115 so both SNFH and ET are visible"
            ),
            "segmentation_background": "real T1C; shared window/crop for GT and prediction",
            "t2w_window": "1st-99.5th positive-voxel percentiles from real T2W, shared with reconstruction",
            "no_independent_panel_normalization": True,
        },
        "output_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths
        },
        "cases": serializable,
    }
    with (ROOT / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def write_captions(cases):
    labels = ", ".join(f"{value[0]} (label {label})" for label, value in LABELS.items())
    caption = f"""# Suggested figure captions

## Segmentation

**Qualitative segmentation results for two illustrative cases from the fixed internal validation cohort.** Ground-truth annotations (left) and predictions from the selected nnU-Net model E (right) are overlaid on the same axial T1-weighted contrast-enhanced slice using an identical crop and intensity window. Colors denote {labels}. The cases show distinct lesion morphologies and strong segmentation overlap; the displayed slice was selected automatically as the axial slice with the largest ground-truth tumor area. These illustrative cases are not presented as a random or statistically representative sample; cohort-level performance is reported separately for all 103 validation cases.

## High-agreement segmentation

**Selected high-agreement segmentation examples from the fixed internal validation cohort.** The two cases with the highest whole-volume macro label Dice for model E are shown: BraTS-MET-00532-000 (0.971) and BraTS-MET-00552-000 (0.951). Ground-truth annotations (left) and nnU-Net predictions (right) are overlaid on identical axial T1-weighted contrast-enhanced images. Colors denote {labels}. These deliberately selected best-case examples illustrate the attainable agreement and must not be interpreted as random, cohort-representative, or statistical-significance evidence; aggregate performance is reported over all 103 validation cases.

## Missing-modality reconstruction

**Qualitative missing-T2W reconstruction for two illustrative cases from the fixed internal validation cohort.** Real T2W images (left) are compared with G1 r4 Ensemble reconstructions (right). The yellow contour denotes the ground-truth tumor region. Each pair uses the same axial slice, crop, and intensity window derived from the real T2W image; no panel-wise normalization was applied. Brain-region SSIM/PSNR values are shown for each case. These illustrative examples assess reconstruction fidelity and do not by themselves establish downstream segmentation benefit or replace the full 103-case quantitative evaluation.
"""
    (ROOT / "suggested_captions.md").write_text(caption, encoding="utf-8")


def main():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    cases = prepare_cases()
    by_id = {case["case_id"]: case for case in cases}
    render_segmentation([by_id[case_id] for case_id in SEGMENTATION_CASES])
    render_segmentation(
        [by_id[case_id] for case_id in HIGH_AGREEMENT_SEGMENTATION_CASES],
        output_stem="segmentation_high_agreement_gt_vs_prediction",
    )
    for option, case_ids in SEGMENTATION_OPTIONS.items():
        render_segmentation(
            [by_id[case_id] for case_id in case_ids],
            output_stem=f"segmentation_option_{option}",
        )
    render_option_contact_sheet()
    render_t2w([by_id[case_id] for case_id in T2W_CASES])
    write_manifest(cases)
    write_captions(cases)

    outputs = [
        ROOT / f"{stem}.{extension}"
        for stem in (
            "segmentation_gt_vs_prediction",
            "segmentation_high_agreement_gt_vs_prediction",
            *[f"segmentation_option_{option}" for option in SEGMENTATION_OPTIONS],
            "t2w_real_vs_ensemble",
        )
        for extension in ("png", "pdf", "svg")
    ]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing output: {path}")
        print(f"{path.name}\t{path.stat().st_size}\t{sha256(path)}")


if __name__ == "__main__":
    main()
