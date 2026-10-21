#!/usr/bin/env python3
"""Compose G1 Diffusion V2 ROI output into traceable full synthetic cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
GENERATED_PATTERN = re.compile(
    r"^(?P<case_id>BraTS-MET-\d{5}-\d{3})-(?P<modality>t1n|t1c|t2w|t2f)\.nii\.gz$"
)
REQUIRED_CONFIG_FIELDS = (
    "generation_run_id",
    "generator_name",
    "seed",
    "sampling_method",
    "sampling_steps",
    "eta",
    "crop_size",
)


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(value: str | Path, anchor: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (anchor / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def discover_v2_outputs(root: Path) -> dict[str, dict[str, Path]]:
    cases: dict[str, dict[str, Path]] = {}
    for path in sorted(root.rglob("*.nii.gz")):
        match = GENERATED_PATTERN.match(path.name)
        if not match:
            continue
        case_id = match.group("case_id")
        modality = match.group("modality")
        if modality in cases.setdefault(case_id, {}):
            raise ValueError(f"duplicate V2 output for {case_id}/{modality}: {path}")
        cases[case_id][modality] = path
    return cases


def validate_generation_config(config: dict[str, object]) -> None:
    missing = [field for field in REQUIRED_CONFIG_FIELDS if config.get(field) in (None, "")]
    checkpoints = [config.get(f"generator_checkpoint_{mod}") for mod in MODALITIES]
    if not all(checkpoints) and not config.get("diffusion_checkpoint_dir"):
        missing.append("generator_checkpoint_t1n/t1c/t2w/t2f or diffusion_checkpoint_dir")
    if missing:
        raise ValueError("V2 generation metadata is incomplete: " + ", ".join(missing))
    if int(config["sampling_steps"]) <= 0:
        raise ValueError("V2 sampling_steps must be positive")
    if int(config["crop_size"]) <= 0:
        raise ValueError("V2 crop_size must be positive")


def prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"V2 composed output is not empty; pass --overwrite to clean it: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def robust_intensity_map(generated: np.ndarray, source: np.ndarray, support: np.ndarray) -> np.ndarray:
    generated_values = generated[support]
    source_values = source[support]
    source_values = source_values[np.isfinite(source_values)]
    if generated_values.size == 0 or source_values.size == 0:
        raise ValueError("empty generated/source support during intensity mapping")
    low, high = np.percentile(source_values, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("source support has no usable intensity range")
    normalized = (np.clip(generated.astype(np.float32), -1.0, 1.0) + 1.0) / 2.0
    return (low + normalized * (high - low)).astype(np.float32)


def compose_modality(
    generated: np.ndarray,
    source: np.ndarray,
    support: np.ndarray,
    blend_width: float,
) -> np.ndarray:
    mapped = robust_intensity_map(generated, source, support)
    distance = ndimage.distance_transform_edt(support)
    alpha = np.clip(distance / max(blend_width, 1e-6), 0.0, 1.0).astype(np.float32)
    composite = source.astype(np.float32, copy=True)
    composite[support] = (
        mapped[support] * alpha[support]
        + source[support].astype(np.float32) * (1.0 - alpha[support])
    )
    return composite


def save_like(array: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path, dtype=np.float32) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(array.astype(dtype), reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(path))


def source_paths(row: dict[str, str]) -> dict[str, Path]:
    paths = {
        mod: resolve_path(row[f"{mod}_path"])
        for mod in MODALITIES
    }
    paths["seg"] = resolve_path(row["seg_path"])
    return paths


def compose_case(
    case_id: str,
    generated_paths: dict[str, Path],
    source_row: dict[str, str],
    output_run_root: Path,
    blend_width: float,
    support_epsilon: float,
    overwrite: bool,
) -> dict[str, object]:
    missing = sorted(set(MODALITIES) - set(generated_paths))
    if missing:
        raise ValueError(f"V2 output is missing modalities for {case_id}: {missing}")
    if not boolish(source_row.get("allowed_as_v2_source", "")) or source_row.get("split") != "train":
        raise ValueError(f"V2 source is not an authentic master-train case: {case_id}")

    sources = source_paths(source_row)
    missing_sources = [str(path) for path in sources.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"source files missing for {case_id}: {missing_sources}")

    source_images = {mod: nib.load(str(sources[mod])) for mod in MODALITIES}
    generated_images = {mod: nib.load(str(generated_paths[mod])) for mod in MODALITIES}
    source_shape = source_images["t1n"].shape[:3]
    if any(image.shape[:3] != source_shape for image in source_images.values()):
        raise ValueError(f"source geometry mismatch for {case_id}")
    if any(image.shape[:3] != source_shape for image in generated_images.values()):
        raise ValueError(f"V2/source shape mismatch for {case_id}")

    generated_arrays = {
        mod: np.asanyarray(image.dataobj, dtype=np.float32)
        for mod, image in generated_images.items()
    }
    if any(not np.isfinite(array).all() for array in generated_arrays.values()):
        raise ValueError(f"V2 output contains NaN/Inf for {case_id}")
    support = np.logical_or.reduce(
        [np.abs(generated_arrays[mod]) > support_epsilon for mod in MODALITIES]
    )
    if not support.any():
        raise ValueError(f"V2 output support is empty for {case_id}")

    raw_id = f"{case_id}_v2aug_label_0"
    case_dir = output_run_root / raw_id
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"composed case already exists: {case_dir}")
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    for mod in MODALITIES:
        source_array = np.asanyarray(source_images[mod].dataobj, dtype=np.float32)
        composite = compose_modality(
            generated_arrays[mod],
            source_array,
            support,
            blend_width,
        )
        save_like(composite, source_images[mod], case_dir / f"{raw_id}-{mod}.nii.gz")

    seg_image = nib.load(str(sources["seg"]))
    seg_array = np.asanyarray(seg_image.dataobj)
    if seg_array.shape[:3] != source_shape:
        raise ValueError(f"seg/source shape mismatch for {case_id}")
    labels = set(np.unique(seg_array).tolist())
    if not labels.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"illegal source labels for {case_id}: {sorted(labels)}")
    save_like(seg_array, seg_image, case_dir / f"{raw_id}-seg.nii.gz", dtype=np.int16)
    save_like(support.astype(np.uint8), source_images["t1n"], case_dir / f"{raw_id}-generation_support.nii.gz", dtype=np.uint8)

    return {
        "synthetic_raw_id": raw_id,
        "source_case_id": case_id,
        "label_kind": "v2aug",
        "label_index": 0,
        "status": "success",
        "support_voxels": int(support.sum()),
        "raw_case_dir": str(case_dir),
        "error_type": "",
        "error_message": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-output-root", required=True)
    parser.add_argument(
        "--source-manifest",
        default=str(DEFAULT_RESULTS_ROOT / "manifests" / "g1_v2_source_manifest.csv"),
    )
    parser.add_argument("--generation-config", default="")
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--blend-width", type=float, default=3.0)
    parser.add_argument("--support-epsilon", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    v2_root = Path(args.v2_output_root).expanduser().resolve()
    source_manifest = Path(args.source_manifest).expanduser().resolve()
    config_path = Path(args.generation_config).expanduser().resolve() if args.generation_config else v2_root / "generation_config.json"
    output_root = Path(args.output_run_root).expanduser().resolve()
    if not v2_root.is_dir():
        raise SystemExit(f"V2 output root not found: {v2_root}")
    if not source_manifest.is_file():
        raise SystemExit(f"V2 source manifest not found: {source_manifest}")
    if not config_path.is_file():
        raise SystemExit(f"V2 generation_config.json not found: {config_path}")
    if args.blend_width <= 0:
        raise SystemExit("--blend-width must be positive")
    if args.support_epsilon < 0:
        raise SystemExit("--support-epsilon must be non-negative")
    if output_root == v2_root:
        raise SystemExit("--output-run-root must differ from --v2-output-root")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_generation_config(config)

    source_rows = {row["source_case_id"]: row for row in read_csv(source_manifest)}
    discovered = discover_v2_outputs(v2_root)
    if not discovered:
        raise SystemExit(f"no V2 generated modality files found under: {v2_root}")
    prepare_output_root(output_root, args.overwrite)

    rows: list[dict[str, object]] = []
    for case_id, generated_paths in sorted(discovered.items()):
        try:
            if case_id not in source_rows:
                raise ValueError(f"source case absent from G2 V2 manifest: {case_id}")
            row = compose_case(
                case_id,
                generated_paths,
                source_rows[case_id],
                output_root,
                args.blend_width,
                args.support_epsilon,
                args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "synthetic_raw_id": f"{case_id}_v2aug_label_0",
                "source_case_id": case_id,
                "label_kind": "v2aug",
                "label_index": 0,
                "status": "failed",
                "support_voxels": "",
                "raw_case_dir": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        rows.append(row)

    normalized_config = dict(config)
    normalized_config.update({
        "generator_io": "full_generation_composed_v2",
        "generation_mode": "full_generation",
        "source_csv": str(source_manifest),
        "source_csv_version": source_manifest.name,
        "label_channels": 4,
        "rc_policy": "preserve_source_seg",
        "composed_by": "g2_v2_compose_augmentation.py",
        "blend_width": args.blend_width,
        "support_epsilon": args.support_epsilon,
    })
    (output_root / "generation_config.json").write_text(
        json.dumps(normalized_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "generation_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            record["generation_run_id"] = normalized_config["generation_run_id"]
            record["seed"] = normalized_config["seed"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(
        output_root / "synthetic_generation_manifest.csv",
        rows,
        [
            "synthetic_raw_id",
            "source_case_id",
            "label_kind",
            "label_index",
            "status",
            "support_voxels",
            "raw_case_dir",
            "error_type",
            "error_message",
        ],
    )

    failed = sum(row["status"] != "success" for row in rows)
    print(f"discovered_cases={len(discovered)}")
    print(f"composed_cases={len(rows) - failed}")
    print(f"failed_cases={failed}")
    print(f"output_run_root={output_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
