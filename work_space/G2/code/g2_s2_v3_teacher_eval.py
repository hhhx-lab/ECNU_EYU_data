#!/usr/bin/env python3
"""Prepare and compare frozen-S2 teacher inference for G1 V3 Stage-5 output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


CHANNELS = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
VALID_LABELS = {0, 1, 2, 3, 4}
REGIONS = {
    "label_1_NETC": {1},
    "label_2_SNFH": {2},
    "label_3_ET": {3},
    "label_4_RC": {4},
    "ET": {3},
    "RC": {4},
    "TC": {1, 3, 4},
    "WT": {1, 2, 3, 4},
}
OFFICIAL_REGIONS = ("ET", "RC", "TC", "WT")
CORE_LABELS = {1, 3, 4}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_source(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def prepare_empty_directory(path: Path, clean: bool) -> None:
    if path.exists():
        if not clean:
            raise FileExistsError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"materialize mode must be symlink or copy, got: {mode}")


def find_case_file(case_dir: Path, case_id: str, modality: str) -> Path:
    candidates = (
        case_dir / f"{case_id}-{modality}.nii.gz",
        case_dir / f"{modality}.nii.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(case_dir.glob(f"*{modality}*.nii.gz"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve {modality} for {case_id}: {matches}")


def same_geometry(reference: nib.spatialimages.SpatialImage, image: nib.spatialimages.SpatialImage) -> bool:
    return (
        reference.shape == image.shape
        and np.allclose(reference.affine, image.affine, atol=1e-4, rtol=0.0)
        and np.allclose(
            reference.header.get_zooms()[:3],
            image.header.get_zooms()[:3],
            atol=1e-5,
            rtol=0.0,
        )
    )


def validate_case_geometry(case_id: str, paths: dict[str, Path]) -> None:
    images = {name: nib.load(str(path)) for name, path in paths.items()}
    reference = images["t1n"]
    for name, image in images.items():
        if not same_geometry(reference, image):
            raise ValueError(f"{case_id}: geometry mismatch for {name}")
        if name != "seg":
            values = image.get_fdata(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{case_id}: {name} contains NaN/Inf")
    segmentation = images["seg"].get_fdata(dtype=np.float32)
    rounded = np.rint(segmentation).astype(np.int16)
    if not np.allclose(segmentation, rounded, atol=1e-6):
        raise ValueError(f"{case_id}: segmentation is not integer-valued")
    invalid = sorted(set(np.unique(rounded)) - VALID_LABELS)
    if invalid:
        raise ValueError(f"{case_id}: invalid segmentation labels {invalid}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_s2_model(
    *,
    checkpoint_path: Path,
    nnunet_results_root: Path,
    expected_dataset_name: str,
    expected_trainer: str,
    configuration: str,
    overwrite: bool,
) -> dict[str, object]:
    import torch

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("S2 checkpoint must contain a dictionary")
    trainer = str(checkpoint.get("trainer_name", ""))
    init_args = checkpoint.get("init_args")
    if not isinstance(init_args, dict):
        raise ValueError("S2 checkpoint is missing init_args")
    plans = init_args.get("plans")
    dataset_json = init_args.get("dataset_json")
    if not isinstance(plans, dict) or not isinstance(dataset_json, dict):
        raise ValueError("S2 checkpoint is missing plans or dataset_json")
    dataset_name = str(plans.get("dataset_name", ""))
    plans_name = str(plans.get("plans_name", ""))
    if dataset_name != expected_dataset_name:
        raise ValueError(
            f"checkpoint dataset {dataset_name} != expected {expected_dataset_name}"
        )
    if trainer != expected_trainer:
        raise ValueError(f"checkpoint trainer {trainer} != expected {expected_trainer}")
    if configuration not in plans.get("configurations", {}):
        raise ValueError(f"checkpoint plans do not contain configuration {configuration}")
    model_root = (
        nnunet_results_root.expanduser().resolve()
        / dataset_name
        / f"{trainer}__{plans_name}__{configuration}"
    )
    if model_root.exists():
        if not overwrite:
            raise FileExistsError(f"S2 model directory already exists: {model_root}")
        shutil.rmtree(model_root)
    fold_root = model_root / "fold_0"
    fold_root.mkdir(parents=True)
    (model_root / "plans.json").write_text(
        json.dumps(plans, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_root / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    installed_checkpoint = fold_root / "checkpoint_final.pth"
    installed_checkpoint.symlink_to(checkpoint_path)
    summary = {
        "model_root": str(model_root),
        "checkpoint_source": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "dataset_name": dataset_name,
        "trainer": trainer,
        "plans_name": plans_name,
        "configuration": configuration,
        "current_epoch": int(checkpoint.get("current_epoch", -1)),
        "checkpoint_materialization": "symlink",
    }
    (model_root / "model_installation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def load_split(split_json: Path) -> dict[str, set[str]]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("split JSON must contain exactly one split object")
    split = payload[0]
    result = {name: set(map(str, split.get(name, []))) for name in ("train", "val", "test")}
    if any(result[left] & result[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("split JSON contains overlapping nnU-Net IDs")
    return result


def prepare_teacher_input(
    *,
    project_root: Path,
    real_root: Path | None,
    mapping_csv: Path,
    split_json: Path,
    stage5_metrics: Path,
    synthetic_root: Path,
    input_root: Path,
    reference_root: Path,
    case_map_path: Path,
    expected_cases: int,
    materialize_mode: str,
    clean: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    real_root = real_root.expanduser().resolve() if real_root is not None else None
    mapping_rows = read_csv(mapping_csv.resolve())
    metrics_rows = read_csv(stage5_metrics.resolve())
    split = load_split(split_json.resolve())
    if len(metrics_rows) != expected_cases:
        raise ValueError(f"Stage-5 case count {len(metrics_rows)} != expected {expected_cases}")
    case_ids = [row.get("subject", "").strip() for row in metrics_rows]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Stage-5 metrics contain blank or duplicate subject IDs")
    mapping = {row.get("source_case_id", "").strip(): row for row in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("mapping contains blank or duplicate source_case_id values")

    input_root = input_root.resolve()
    reference_root = reference_root.resolve()
    prepare_empty_directory(input_root, clean)
    prepare_empty_directory(reference_root, clean)
    case_map_path = case_map_path.resolve()
    case_map_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    try:
        for case_id in case_ids:
            row = mapping.get(case_id)
            if row is None:
                raise ValueError(f"Stage-5 case is missing from real-only mapping: {case_id}")
            if not parse_bool(row.get("eligible_for_realonly", "")):
                raise ValueError(f"Stage-5 case is not real-only eligible: {case_id}")
            nnunet_id = row.get("nnunet_case_id", "").strip()
            if not nnunet_id or nnunet_id not in split["val"]:
                raise ValueError(f"Stage-5 case is not in the fixed validation split: {case_id}")
            case_dir = real_root / case_id if real_root is not None else None
            paths = {}
            for modality in CHANNELS:
                if modality == "t2w":
                    paths[modality] = synthetic_root.resolve() / f"{case_id}-t2w.nii.gz"
                elif case_dir is not None:
                    paths[modality] = find_case_file(case_dir, case_id, modality)
                else:
                    paths[modality] = resolve_source(
                        project_root, row[f"{modality}_source_path"]
                    )
            paths["seg"] = (
                find_case_file(case_dir, case_id, "seg")
                if case_dir is not None
                else resolve_source(project_root, row["seg_source_path"])
            )
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{case_id}: missing teacher input files: {missing}")
            validate_case_geometry(case_id, paths)

            for modality, channel in CHANNELS.items():
                materialize(
                    paths[modality],
                    input_root / f"{nnunet_id}_{channel}.nii.gz",
                    materialize_mode,
                )
            materialize(
                paths["seg"],
                reference_root / f"{case_id}.nii.gz",
                materialize_mode,
            )
            manifest_rows.append(
                {
                    "nnunet_case_id": nnunet_id,
                    "source_case_id": case_id,
                    "split": "val",
                    "t1n_path": str(paths["t1n"]),
                    "t1c_path": str(paths["t1c"]),
                    "t2w_generated_path": str(paths["t2w"]),
                    "t2f_path": str(paths["t2f"]),
                    "reference_seg_path": str(paths["seg"]),
                }
            )
    except Exception:
        shutil.rmtree(input_root, ignore_errors=True)
        shutil.rmtree(reference_root, ignore_errors=True)
        raise

    case_map_path.write_text(
        "".join(f"{row['nnunet_case_id']}\t{row['source_case_id']}\n" for row in manifest_rows),
        encoding="utf-8",
    )
    manifest_path = case_map_path.with_name(case_map_path.stem + "_manifest.csv")
    write_csv(manifest_path, manifest_rows)
    summary = {
        "case_count": len(manifest_rows),
        "nifti_count": len(list(input_root.glob("*.nii.gz"))),
        "reference_count": len(list(reference_root.glob("*.nii.gz"))),
        "channel_order": CHANNELS,
        "split": "val",
        "uses_generated_t2w": True,
        "uses_real_protected_modalities": True,
        "materialize_mode": materialize_mode,
        "input_root": str(input_root),
        "real_root": str(real_root) if real_root is not None else None,
        "reference_root": str(reference_root),
        "case_map": str(case_map_path),
        "manifest": str(manifest_path),
    }
    if summary["case_count"] != expected_cases or summary["nifti_count"] != expected_cases * 4:
        raise RuntimeError(f"teacher input materialization count mismatch: {summary}")
    case_map_path.with_name(case_map_path.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def load_integer_segmentation(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    values = image.get_fdata(dtype=np.float32)
    rounded = np.rint(values).astype(np.int16)
    if not np.allclose(values, rounded, atol=1e-6):
        raise ValueError(f"segmentation is not integer-valued: {path}")
    invalid = sorted(set(np.unique(rounded)) - VALID_LABELS)
    if invalid:
        raise ValueError(f"invalid labels {invalid}: {path}")
    return image, rounded


def binary_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = int(reference.sum()) + int(prediction.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(reference, prediction).sum()) / denominator


def region_mask(segmentation: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    return np.isin(segmentation, tuple(labels))


def connected_components(segmentation: np.ndarray, spacing: tuple[float, float, float]) -> list[dict[str, object]]:
    mask = np.isin(segmentation, tuple(CORE_LABELS))
    components, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    result = []
    for component_id in range(1, count + 1):
        component = components == component_id
        volume = float(component.sum()) * voxel_volume
        result.append({"mask": component, "volume_mm3": volume, "large": volume > 275.0})
    return result


def unmatched_large_count(source: list[dict[str, object]], target: list[dict[str, object]]) -> int:
    target_masks = [row["mask"] for row in target]
    return sum(
        bool(row["large"]) and not any(np.logical_and(row["mask"], mask).any() for mask in target_masks)
        for row in source
    )


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def read_case_map(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"invalid case-map line: {line}")
        pairs.append((parts[0], parts[1]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("case map contains duplicate rows")
    return pairs


def compare_teacher_predictions(
    *,
    baseline_prediction_root: Path,
    generated_prediction_root: Path,
    reference_root: Path,
    case_map_path: Path,
    output_root: Path,
    expected_cases: int,
    max_macro_dice_drop: float,
    max_region_dice_drop: float,
    max_missing_large_fraction: float,
    overwrite: bool,
) -> dict[str, object]:
    pairs = read_case_map(case_map_path.resolve())
    if len(pairs) != expected_cases:
        raise ValueError(f"teacher case count {len(pairs)} != expected {expected_cases}")
    output_root = output_root.resolve()
    prepare_empty_directory(output_root, overwrite)
    case_rows: list[dict[str, object]] = []

    for nnunet_id, case_id in pairs:
        paths = {
            "baseline": baseline_prediction_root.resolve() / f"{case_id}.nii.gz",
            "generated": generated_prediction_root.resolve() / f"{nnunet_id}.nii.gz",
            "reference": reference_root.resolve() / f"{case_id}.nii.gz",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{case_id}: missing teacher comparison files: {missing}")
        reference_image, reference = load_integer_segmentation(paths["reference"])
        baseline_image, baseline = load_integer_segmentation(paths["baseline"])
        generated_image, generated = load_integer_segmentation(paths["generated"])
        for name, image in (("baseline", baseline_image), ("generated", generated_image)):
            if not same_geometry(reference_image, image):
                raise ValueError(f"{case_id}: geometry mismatch for {name} prediction")

        row: dict[str, object] = {"nnunet_case_id": nnunet_id, "source_case_id": case_id}
        for region_name, labels in REGIONS.items():
            ref_mask = region_mask(reference, labels)
            baseline_mask = region_mask(baseline, labels)
            generated_mask = region_mask(generated, labels)
            baseline_dice = binary_dice(ref_mask, baseline_mask)
            generated_dice = binary_dice(ref_mask, generated_mask)
            row[f"baseline_dice_{region_name}"] = baseline_dice
            row[f"generated_dice_{region_name}"] = generated_dice
            row[f"delta_dice_{region_name}"] = generated_dice - baseline_dice
            row[f"prediction_consistency_dice_{region_name}"] = binary_dice(
                baseline_mask, generated_mask
            )

        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
        baseline_lesions = connected_components(baseline, spacing)
        generated_lesions = connected_components(generated, spacing)
        baseline_large = sum(bool(item["large"]) for item in baseline_lesions)
        generated_large = sum(bool(item["large"]) for item in generated_lesions)
        row.update(
            {
                "baseline_lesion_count": len(baseline_lesions),
                "generated_lesion_count": len(generated_lesions),
                "lesion_count_difference": len(generated_lesions) - len(baseline_lesions),
                "baseline_large_lesion_count": baseline_large,
                "generated_large_lesion_count": generated_large,
                "missing_large_lesion_count": unmatched_large_count(
                    baseline_lesions, generated_lesions
                ),
                "extra_large_lesion_count": unmatched_large_count(
                    generated_lesions, baseline_lesions
                ),
            }
        )
        case_rows.append(row)

    write_csv(output_root / "case_metrics.csv", case_rows)
    region_means = {}
    for region_name in REGIONS:
        baseline_mean = finite_mean(row[f"baseline_dice_{region_name}"] for row in case_rows)
        generated_mean = finite_mean(row[f"generated_dice_{region_name}"] for row in case_rows)
        region_means[region_name] = {
            "baseline": baseline_mean,
            "generated": generated_mean,
            "delta": generated_mean - baseline_mean,
            "prediction_consistency": finite_mean(
                row[f"prediction_consistency_dice_{region_name}"] for row in case_rows
            ),
        }
    macro_baseline = finite_mean(region_means[name]["baseline"] for name in OFFICIAL_REGIONS)
    macro_generated = finite_mean(region_means[name]["generated"] for name in OFFICIAL_REGIONS)
    baseline_large_total = sum(int(row["baseline_large_lesion_count"]) for row in case_rows)
    missing_large_total = sum(int(row["missing_large_lesion_count"]) for row in case_rows)
    missing_large_fraction = missing_large_total / baseline_large_total if baseline_large_total else 0.0
    failure_reasons = []
    if macro_generated - macro_baseline < -max_macro_dice_drop:
        failure_reasons.append("macro_region_dice_drop")
    for region_name in OFFICIAL_REGIONS:
        if region_means[region_name]["delta"] < -max_region_dice_drop:
            failure_reasons.append(f"{region_name.lower()}_dice_drop")
    if missing_large_fraction > max_missing_large_fraction:
        failure_reasons.append("missing_large_lesion_fraction")

    summary = {
        "case_count": len(case_rows),
        "teacher_model_role": "frozen_real_only_s2",
        "comparison": "real_t2w_baseline_vs_g1_v3_generated_t2w",
        "region_metrics": region_means,
        "metrics": {
            "macro_region_dice_baseline": macro_baseline,
            "macro_region_dice_generated": macro_generated,
            "macro_region_dice_delta": macro_generated - macro_baseline,
            "baseline_large_lesion_count": baseline_large_total,
            "missing_large_lesion_count": missing_large_total,
            "missing_large_lesion_fraction": missing_large_fraction,
            "extra_large_lesion_count": sum(
                int(row["extra_large_lesion_count"]) for row in case_rows
            ),
        },
        "thresholds": {
            "max_macro_dice_drop": max_macro_dice_drop,
            "max_region_dice_drop": max_region_dice_drop,
            "max_missing_large_fraction": max_missing_large_fraction,
        },
        "failure_reasons": failure_reasons,
        "teacher_technical_gate": "fail" if failure_reasons else "pass",
        "stage6_gate": "hold_for_review",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# G1 V3 Stage-5 冻结 S2 Teacher 对比报告",
        "",
        f"- 病例数：{len(case_rows)}",
        f"- teacher 技术门：`{summary['teacher_technical_gate']}`",
        f"- 阶段 6 门：`{summary['stage6_gate']}`",
        f"- ET/RC/TC/WT macro Dice 变化：{macro_generated - macro_baseline:.6f}",
        f"- 漏掉的大病灶：{missing_large_total}/{baseline_large_total}",
        f"- 失败原因：{';'.join(failure_reasons) if failure_reasons else 'none'}",
        "",
        "该 teacher 结果只检查冻结分割模型对替换 T2W 的敏感性。阶段 6 仍需 G2 paired QC 和人工 montage 复核。",
        "",
    ]
    (output_root / "TEACHER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Build 103-case nnU-Net teacher input")
    prepare.add_argument("--project-root", required=True)
    prepare.add_argument(
        "--real-root",
        help="Optional deployed G1 V3 case root; mapping remains the identity authority",
    )
    prepare.add_argument("--mapping-csv", required=True)
    prepare.add_argument("--split-json", required=True)
    prepare.add_argument("--stage5-metrics", required=True)
    prepare.add_argument("--synthetic-root", required=True)
    prepare.add_argument("--input-root", required=True)
    prepare.add_argument("--reference-root", required=True)
    prepare.add_argument("--case-map", required=True)
    prepare.add_argument("--expected-cases", type=int, default=103)
    prepare.add_argument("--materialize-mode", choices=("symlink", "copy"), default="symlink")
    prepare.add_argument("--clean", action="store_true")

    install = subparsers.add_parser("install-model", help="Install frozen S2 checkpoint metadata")
    install.add_argument("--checkpoint", required=True)
    install.add_argument("--nnunet-results-root", required=True)
    install.add_argument(
        "--expected-dataset-name",
        default="Dataset263_BraTS2026_MET_RealOnly_Current",
    )
    install.add_argument("--expected-trainer", default="nnUNetTrainerBraTS2026RC")
    install.add_argument("--configuration", default="3d_fullres")
    install.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare real-T2W and generated-T2W S2 predictions")
    compare.add_argument("--baseline-prediction-root", required=True)
    compare.add_argument("--generated-prediction-root", required=True)
    compare.add_argument("--reference-root", required=True)
    compare.add_argument("--case-map", required=True)
    compare.add_argument("--output-root", required=True)
    compare.add_argument("--expected-cases", type=int, default=103)
    compare.add_argument("--max-macro-dice-drop", type=float, default=0.02)
    compare.add_argument("--max-region-dice-drop", type=float, default=0.03)
    compare.add_argument("--max-missing-large-fraction", type=float, default=0.05)
    compare.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        summary = prepare_teacher_input(
            project_root=Path(args.project_root),
            real_root=Path(args.real_root) if args.real_root else None,
            mapping_csv=Path(args.mapping_csv),
            split_json=Path(args.split_json),
            stage5_metrics=Path(args.stage5_metrics),
            synthetic_root=Path(args.synthetic_root),
            input_root=Path(args.input_root),
            reference_root=Path(args.reference_root),
            case_map_path=Path(args.case_map),
            expected_cases=args.expected_cases,
            materialize_mode=args.materialize_mode,
            clean=args.clean,
        )
    elif args.command == "install-model":
        summary = install_s2_model(
            checkpoint_path=Path(args.checkpoint),
            nnunet_results_root=Path(args.nnunet_results_root),
            expected_dataset_name=args.expected_dataset_name,
            expected_trainer=args.expected_trainer,
            configuration=args.configuration,
            overwrite=args.overwrite,
        )
    else:
        summary = compare_teacher_predictions(
            baseline_prediction_root=Path(args.baseline_prediction_root),
            generated_prediction_root=Path(args.generated_prediction_root),
            reference_root=Path(args.reference_root),
            case_map_path=Path(args.case_map),
            output_root=Path(args.output_root),
            expected_cases=args.expected_cases,
            max_macro_dice_drop=args.max_macro_dice_drop,
            max_region_dice_drop=args.max_region_dice_drop,
            max_missing_large_fraction=args.max_missing_large_fraction,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
