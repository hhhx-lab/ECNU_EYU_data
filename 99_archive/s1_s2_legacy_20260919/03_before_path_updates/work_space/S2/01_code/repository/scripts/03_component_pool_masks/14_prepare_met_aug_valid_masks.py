#!/usr/bin/env python3
"""Create exact preprocessed valid-mask assets for the MET-AUG train loader.

The mask is built from the union of all four original modalities, then subjected
to the same nnU-Net transpose/crop geometry as the preprocessed case.  This is
deliberately not the old single-channel ``t1c != 0`` heuristic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (
    S2_MODALITIES,
    VALID_MASK_MANIFEST_SCHEMA,
    canonical_json_sha256,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-nearest-resample", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"train split is empty or has duplicate IDs: {path}")
    return values


def load_preprocessed_case(
    dataset,
    case_id: str,
) -> tuple[tuple[int, int, int], dict, np.ndarray]:
    """Read one case through the pinned nnU-Net 2.8 four-value API."""
    data, segmentation, previous_stage_segmentation, properties = dataset.load_case(case_id)
    if previous_stage_segmentation is not None:
        raise ValueError(f"{case_id}: MET-AUG does not support cascaded preprocessed data")
    if data.ndim != 4 or data.shape[0] != 4:
        raise ValueError(f"{case_id}: expected preprocessed four-channel array, got {data.shape}")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1 or segmentation.shape[1:] != data.shape[1:]:
        raise ValueError(f"{case_id}: expected aligned one-channel segmentation, got {segmentation.shape}")
    return (
        tuple(int(value) for value in data.shape[1:]),
        dict(properties),
        np.asarray(segmentation[0]),
    )


def _bbox_from_properties(properties: dict, source_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    bbox = properties.get("bbox_used_for_cropping")
    if bbox is None:
        return tuple(slice(0, int(size)) for size in source_shape)  # type: ignore[return-value]
    if len(bbox) != 3:
        raise ValueError(f"bbox_used_for_cropping is malformed: {bbox}")
    result = []
    for bounds, size in zip(bbox, source_shape):
        if len(bounds) != 2:
            raise ValueError(f"bbox dimension is malformed: {bbox}")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop > size or start >= stop:
            raise ValueError(f"bbox is outside source image: {bbox}")
        result.append(slice(start, stop))
    return tuple(result)  # type: ignore[return-value]


def _transpose_from_plans(preprocessed_dir: Path) -> tuple[int, int, int]:
    plans_path = preprocessed_dir.parent / "nnUNetPlans.json"
    if not plans_path.is_file():
        return (0, 1, 2)
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    value = payload.get("transpose_forward", (0, 1, 2))
    result = tuple(int(item) for item in value)
    if sorted(result) != [0, 1, 2]:
        raise ValueError(f"unsupported transpose_forward in {plans_path}: {value}")
    return result


def _resampling_contract_from_plans(
    preprocessed_dir: Path,
) -> tuple[Callable[..., np.ndarray], tuple[float, float, float], str]:
    plans_path = preprocessed_dir.parent / "nnUNetPlans.json"
    if not plans_path.is_file():
        raise FileNotFoundError(f"missing nnU-Net plans: {plans_path}")
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    matching_configurations = [
        name
        for name, configuration in payload.get("configurations", {}).items()
        if configuration.get("data_identifier") == preprocessed_dir.name
    ]
    if len(matching_configurations) != 1:
        raise ValueError(
            "cannot bind valid-mask resampling to one nnU-Net configuration: "
            f"{preprocessed_dir}"
        )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    configuration_name = matching_configurations[0]
    configuration = PlansManager(payload).get_configuration(configuration_name)
    spacing = tuple(float(value) for value in configuration.spacing)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"invalid target spacing in {plans_path}: {spacing}")
    return configuration.resampling_fn_seg, spacing, configuration_name


def align_segmentation(
    segmentation: np.ndarray,
    *,
    properties: dict,
    target_shape: tuple[int, int, int],
    transpose_forward: tuple[int, int, int],
    allow_nearest_resample: bool,
    resampling_fn_seg: Callable[..., np.ndarray] | None = None,
    target_spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    if segmentation.ndim != 3:
        raise ValueError(f"expected a 3D segmentation, got {segmentation.shape}")
    transformed = np.transpose(segmentation, transpose_forward)
    cropped = transformed[
        _bbox_from_properties(properties, tuple(int(value) for value in transformed.shape))
    ]
    if cropped.shape == target_shape:
        return np.asarray(cropped)
    if not allow_nearest_resample:
        raise ValueError(
            "segmentation geometry differs from preprocessed data. Re-run only after "
            "auditing the preprocessing coordinate contract, then explicitly pass "
            "--allow-nearest-resample. "
            f"got={cropped.shape}, expected={target_shape}"
        )
    if resampling_fn_seg is None or target_spacing is None:
        raise ValueError(
            "nnU-Net segmentation resampling contract is required when source and "
            "preprocessed mask geometries differ"
        )
    source_spacing = properties.get("spacing")
    if source_spacing is None or len(source_spacing) != 3:
        raise ValueError("preprocessed properties lack a valid source spacing")
    current_spacing = tuple(float(source_spacing[axis]) for axis in transpose_forward)
    resized = resampling_fn_seg(
        cropped[np.newaxis].astype(np.int16, copy=False),
        target_shape,
        current_spacing,
        target_spacing,
    )
    if resized.ndim != 4 or resized.shape[0] != 1 or tuple(resized.shape[1:]) != target_shape:
        raise ValueError(
            "nnU-Net segmentation resampling produced "
            f"{getattr(resized, 'shape', None)}, expected (1, {target_shape})"
        )
    return np.asarray(resized[0])


def align_mask(
    mask: np.ndarray,
    *,
    properties: dict,
    target_shape: tuple[int, int, int],
    transpose_forward: tuple[int, int, int],
    allow_nearest_resample: bool,
    resampling_fn_seg: Callable[..., np.ndarray] | None = None,
    target_spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    aligned = align_segmentation(
        mask.astype(np.uint8, copy=False),
        properties=properties,
        target_shape=target_shape,
        transpose_forward=transpose_forward,
        allow_nearest_resample=allow_nearest_resample,
        resampling_fn_seg=resampling_fn_seg,
        target_spacing=target_spacing,
    )
    return aligned.astype(bool, copy=False)


def prepare_raw_segmentation_for_nnunet_replay(
    source_segmentation: np.ndarray,
    preprocessing_nonzero_mask: np.ndarray,
) -> np.ndarray:
    if source_segmentation.ndim != 3:
        raise ValueError(f"expected a 3D source segmentation, got {source_segmentation.shape}")
    if preprocessing_nonzero_mask.shape != source_segmentation.shape:
        raise ValueError(
            "preprocessing nonzero mask and source segmentation differ: "
            f"{preprocessing_nonzero_mask.shape} vs {source_segmentation.shape}"
        )
    source = np.asarray(source_segmentation)
    if not np.all(np.isfinite(source)) or not np.array_equal(source, np.rint(source)):
        raise ValueError("source segmentation contains non-finite or non-integer labels")
    source_labels = {int(value) for value in np.unique(source)}
    unexpected = sorted(source_labels - {0, 1, 2, 3, 4})
    if unexpected:
        raise ValueError(f"source segmentation contains unsupported labels: {unexpected}")
    replay = np.rint(source).astype(np.int16, copy=True)
    replay[(replay == 0) & (~preprocessing_nonzero_mask)] = -1
    return replay


def _same_source_geometry(image_properties: dict, label_properties: dict) -> bool:
    image_spacing = image_properties.get("spacing")
    label_spacing = label_properties.get("spacing")
    if image_spacing is None or label_spacing is None:
        return False
    if not np.allclose(image_spacing, label_spacing, atol=1e-5):
        return False
    image_sitk = image_properties.get("sitk_stuff", {})
    label_sitk = label_properties.get("sitk_stuff", {})
    for key in ("spacing", "origin", "direction"):
        if key not in image_sitk or key not in label_sitk:
            return False
        if not np.allclose(image_sitk[key], label_sitk[key], atol=1e-5):
            return False
    return True


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    train_file = Path(args.train_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"valid-mask output is immutable and already exists: {output_dir}")
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir() or not preprocessed_dir.is_dir():
        raise FileNotFoundError("Dataset264 raw or preprocessed input directory is missing")
    train_ids = read_ids(train_file)
    dataset_json_path = dataset_dir / "dataset.json"
    plans_path = preprocessed_dir.parent / "nnUNetPlans.json"
    if not dataset_json_path.is_file() or not plans_path.is_file():
        raise FileNotFoundError("Dataset264 dataset.json or nnUNetPlans.json is missing")
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
    from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    example_image = images_dir / f"{train_ids[0]}_0000.nii.gz"
    reader_class = determine_reader_writer_from_dataset_json(
        dataset_json,
        example_file=str(example_image),
        verbose=False,
    )
    raw_reader = reader_class()
    dataset_class = infer_dataset_class(str(preprocessed_dir))
    preprocessed_dataset = dataset_class(str(preprocessed_dir), train_ids)
    transpose_forward = _transpose_from_plans(preprocessed_dir)
    resampling_fn_seg, target_spacing, resampling_configuration = _resampling_contract_from_plans(
        preprocessed_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir()
    rows: list[dict] = []
    for case_id in train_ids:
        modality_paths = []
        for index, modality in enumerate(S2_MODALITIES):
            path = images_dir / f"{case_id}_{index:04d}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(f"missing {modality}: {path}")
            modality_paths.append(str(path))
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing segmentation: {label_path}")
        source_images, image_properties = raw_reader.read_images(modality_paths)
        source_label, label_properties = raw_reader.read_seg(str(label_path))
        if source_images.ndim != 4 or source_images.shape[0] != 4:
            raise ValueError(f"{case_id}: source reader did not return four modalities")
        if source_label.ndim != 4 or source_label.shape[0] != 1:
            raise ValueError(f"{case_id}: source reader did not return one segmentation")
        if source_images.shape[1:] != source_label.shape[1:]:
            raise ValueError(f"{case_id}: source modality/segmentation shapes differ")
        if not _same_source_geometry(image_properties, label_properties):
            raise ValueError(f"{case_id}: source modality/segmentation geometry differs")
        if not np.all(np.isfinite(source_images)):
            raise ValueError(f"{case_id}: source modalities contain non-finite values")
        raw_valid = np.any(np.isfinite(source_images) & (source_images != 0), axis=0)
        preprocessing_nonzero_mask = create_nonzero_mask(source_images)
        raw_segmentation = prepare_raw_segmentation_for_nnunet_replay(
            source_label[0],
            preprocessing_nonzero_mask,
        )
        target_shape, properties, expected_segmentation = load_preprocessed_case(
            preprocessed_dataset,
            case_id,
        )
        valid_mask = align_mask(
            raw_valid,
            properties=properties,
            target_shape=target_shape,
            transpose_forward=transpose_forward,
            allow_nearest_resample=args.allow_nearest_resample,
            resampling_fn_seg=resampling_fn_seg,
            target_spacing=target_spacing,
        )
        replayed_segmentation = align_segmentation(
            raw_segmentation,
            properties=properties,
            target_shape=target_shape,
            transpose_forward=transpose_forward,
            allow_nearest_resample=args.allow_nearest_resample,
            resampling_fn_seg=resampling_fn_seg,
            target_spacing=target_spacing,
        )
        if not np.array_equal(replayed_segmentation, expected_segmentation):
            raise ValueError(
                f"{case_id}: raw-label nnU-Net replay does not reproduce the preprocessed segmentation"
            )
        foreground_mask = expected_segmentation > 0
        if not np.any(valid_mask):
            raise ValueError(f"{case_id}: valid mask is empty")
        destination = masks_dir / f"{case_id}.npz"
        np.savez_compressed(
            destination,
            valid_mask=valid_mask.astype(np.uint8),
            foreground_mask=foreground_mask.astype(np.uint8),
            shape=np.asarray(target_shape, dtype=np.int32),
        )
        rows.append({
            "case_id": case_id,
            "mask_path": str(Path("masks") / destination.name),
            "sha256": sha256_file(destination),
            "shape": list(target_shape),
            "valid_voxels": int(np.count_nonzero(valid_mask)),
            "foreground_voxels": int(np.count_nonzero(foreground_mask)),
            "source_label_sha256": sha256_file(label_path),
        })
    rows_path = output_dir / "valid_mask_records.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": VALID_MASK_MANIFEST_SCHEMA,
        "dataset_dir": str(dataset_dir),
        "preprocessed_dir": str(preprocessed_dir),
        "builder_code_sha256": sha256_file(Path(__file__)),
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "train_count": len(train_ids),
        "dataset_json_sha256": sha256_file(dataset_json_path),
        "nnunet_plans_sha256": sha256_file(plans_path),
        "records_file": rows_path.name,
        "records_sha256": sha256_file(rows_path),
        "transpose_forward": list(transpose_forward),
        "resampling_configuration": resampling_configuration,
        "target_spacing_mm": list(target_spacing),
        "resampling_backend": "nnunet_configuration_resampling_fn_seg",
        "allow_nearest_resample": bool(args.allow_nearest_resample),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest, exclude=("manifest_sha256",))
    manifest_path = output_dir / "valid_mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_count": len(train_ids),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
