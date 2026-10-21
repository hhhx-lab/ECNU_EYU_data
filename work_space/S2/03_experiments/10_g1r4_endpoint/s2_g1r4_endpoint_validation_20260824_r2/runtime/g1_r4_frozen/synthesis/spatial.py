"""Affine-aware, reversible subject-space transforms for G1 V3 volumes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SpatialTransform:
    native_shape: tuple[int, int, int]
    native_affine: np.ndarray
    target_shape: tuple[int, int, int]
    target_affine: np.ndarray
    target_spacing_mm: float
    foreground_world_min: tuple[float, float, float]
    foreground_world_max: tuple[float, float, float]
    foreground_voxel_count: int
    lesion_voxel_count: int
    margin_mm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "algorithm": "foreground_centered_isotropic_resample",
            "native_shape": list(self.native_shape),
            "native_affine": np.asarray(self.native_affine, dtype=float).tolist(),
            "target_shape": list(self.target_shape),
            "target_affine": np.asarray(self.target_affine, dtype=float).tolist(),
            "target_spacing_mm": self.target_spacing_mm,
            "foreground_world_min": list(self.foreground_world_min),
            "foreground_world_max": list(self.foreground_world_max),
            "foreground_voxel_count": self.foreground_voxel_count,
            "lesion_voxel_count": self.lesion_voxel_count,
            "margin_mm": self.margin_mm,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatialTransform":
        if int(value.get("version", 0)) != 1:
            raise ValueError(f"unsupported spatial transform version: {value.get('version')}")
        return cls(
            native_shape=tuple(int(item) for item in value["native_shape"]),
            native_affine=np.asarray(value["native_affine"], dtype=np.float64),
            target_shape=tuple(int(item) for item in value["target_shape"]),
            target_affine=np.asarray(value["target_affine"], dtype=np.float64),
            target_spacing_mm=float(value["target_spacing_mm"]),
            foreground_world_min=tuple(float(item) for item in value["foreground_world_min"]),
            foreground_world_max=tuple(float(item) for item in value["foreground_world_max"]),
            foreground_voxel_count=int(value["foreground_voxel_count"]),
            lesion_voxel_count=int(value["lesion_voxel_count"]),
            margin_mm=float(value["margin_mm"]),
        )


def _validate_volume(volume: np.ndarray, expected_shape: tuple[int, int, int]) -> None:
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    if tuple(volume.shape) != expected_shape:
        raise ValueError(f"volume shape {volume.shape} != reference shape {expected_shape}")
    if not np.isfinite(volume).all():
        raise ValueError("volume contains NaN or Inf")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def build_foreground_mask(
    normalized_images: Sequence[np.ndarray],
    segmentation: np.ndarray | None = None,
    threshold: float = 0.02,
) -> np.ndarray:
    if not normalized_images:
        raise ValueError("at least one normalized image is required")
    native_shape = tuple(int(item) for item in normalized_images[0].shape)
    for image in normalized_images:
        _validate_volume(np.asarray(image), native_shape)

    mean_image = np.mean(np.stack(normalized_images, axis=0), axis=0)
    foreground = _largest_component(mean_image > threshold)
    if not foreground.any():
        raise ValueError("no foreground was found in the available modalities")

    if segmentation is not None:
        _validate_volume(np.asarray(segmentation), native_shape)
        foreground |= np.asarray(segmentation) > 0
    return foreground


def _world_bounds(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("support mask is empty")
    lower = coordinates.min(axis=0).astype(np.float64) - 0.5
    upper = coordinates.max(axis=0).astype(np.float64) + 0.5
    corners = np.asarray(list(product(*zip(lower, upper))), dtype=np.float64)
    world = nib.affines.apply_affine(affine, corners)
    return world.min(axis=0), world.max(axis=0)


def build_spatial_transform(
    normalized_images: Sequence[np.ndarray],
    native_affine: np.ndarray,
    *,
    segmentation: np.ndarray | None = None,
    target_shape: tuple[int, int, int] = (256, 256, 160),
    base_spacing_mm: float = 1.0,
    margin_mm: float = 5.0,
    foreground_threshold: float = 0.02,
) -> SpatialTransform:
    target_shape = tuple(int(item) for item in target_shape)
    if len(target_shape) != 3 or any(item <= 0 for item in target_shape):
        raise ValueError(f"invalid target shape: {target_shape}")
    if base_spacing_mm <= 0 or margin_mm < 0:
        raise ValueError("spacing must be positive and margin cannot be negative")

    native_affine = np.asarray(native_affine, dtype=np.float64)
    if native_affine.shape != (4, 4) or not np.isfinite(native_affine).all():
        raise ValueError("native affine must be a finite 4x4 matrix")
    native_shape = tuple(int(item) for item in normalized_images[0].shape)
    foreground = build_foreground_mask(
        normalized_images,
        segmentation=segmentation,
        threshold=foreground_threshold,
    )
    world_min, world_max = _world_bounds(foreground, native_affine)
    world_extent = world_max - world_min
    required_spacing = float(
        np.max((world_extent + 2.0 * float(margin_mm)) / np.asarray(target_shape))
    )
    target_spacing = max(float(base_spacing_mm), required_spacing)
    world_center = (world_min + world_max) / 2.0

    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = np.diag([target_spacing] * 3)
    target_center_index = (np.asarray(target_shape, dtype=np.float64) - 1.0) / 2.0
    target_affine[:3, 3] = world_center - target_affine[:3, :3] @ target_center_index

    transform = SpatialTransform(
        native_shape=native_shape,
        native_affine=native_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        target_spacing_mm=target_spacing,
        foreground_world_min=tuple(float(item) for item in world_min),
        foreground_world_max=tuple(float(item) for item in world_max),
        foreground_voxel_count=int(np.count_nonzero(foreground)),
        lesion_voxel_count=int(np.count_nonzero(segmentation)) if segmentation is not None else 0,
        margin_mm=float(margin_mm),
    )
    assert_support_contained(foreground, transform, "foreground")
    if segmentation is not None and np.any(segmentation > 0):
        assert_support_contained(segmentation > 0, transform, "lesion")
    return transform


def assert_support_contained(
    support: np.ndarray,
    transform: SpatialTransform,
    support_name: str,
) -> dict[str, Any]:
    support = np.asarray(support, dtype=bool)
    _validate_volume(support, transform.native_shape)
    world_min, world_max = _world_bounds(support, transform.native_affine)
    corners = np.asarray(list(product(*zip(world_min, world_max))), dtype=np.float64)
    target_indices = nib.affines.apply_affine(
        np.linalg.inv(transform.target_affine), corners
    )
    target_min = target_indices.min(axis=0)
    target_max = target_indices.max(axis=0)
    lower_limit = np.full(3, -0.5, dtype=np.float64)
    upper_limit = np.asarray(transform.target_shape, dtype=np.float64) - 0.5
    if np.any(target_min < lower_limit - 1e-5) or np.any(target_max > upper_limit + 1e-5):
        raise ValueError(
            f"{support_name} support is outside model FOV: "
            f"min={target_min.tolist()} max={target_max.tolist()} "
            f"limits={lower_limit.tolist()}..{upper_limit.tolist()}"
        )
    return {
        "support": support_name,
        "source_voxel_count": int(np.count_nonzero(support)),
        "outside_voxel_count": 0,
        "target_index_min": target_min.tolist(),
        "target_index_max": target_max.tolist(),
    }


def _resample(
    image: np.ndarray,
    source_affine: np.ndarray,
    destination: tuple[tuple[int, int, int], np.ndarray],
    order: int,
) -> np.ndarray:
    image_array = np.asarray(image)
    # NIfTI does not support float16 headers. Model inference may return float16
    # under CUDA autocast, so promote only the temporary interpolation source.
    source_data = (
        image_array.astype(np.float32, copy=False)
        if np.issubdtype(image_array.dtype, np.floating) and image_array.dtype.itemsize < 4
        else image_array
    )
    source = nib.Nifti1Image(source_data, np.asarray(source_affine, dtype=np.float64))
    result = resample_from_to(source, destination, order=order, mode="constant", cval=0.0)
    data = np.asanyarray(result.dataobj)
    if order == 0 and np.issubdtype(image_array.dtype, np.integer):
        return np.rint(data).astype(image_array.dtype, copy=False)
    return np.asarray(data, dtype=np.float32)


def resample_to_model(
    image: np.ndarray,
    transform: SpatialTransform,
    *,
    order: int = 1,
) -> np.ndarray:
    _validate_volume(np.asarray(image), transform.native_shape)
    return _resample(
        image,
        transform.native_affine,
        (transform.target_shape, transform.target_affine),
        order,
    )


def resample_labels_to_model(
    segmentation: np.ndarray,
    transform: SpatialTransform,
) -> np.ndarray:
    segmentation = np.asarray(segmentation)
    _validate_volume(segmentation, transform.native_shape)
    if not np.issubdtype(segmentation.dtype, np.integer):
        rounded = np.rint(segmentation)
        if not np.allclose(segmentation, rounded, atol=1e-6):
            raise ValueError("segmentation contains non-integer labels")
        segmentation = rounded.astype(np.int16)
    assert_support_contained(segmentation > 0, transform, "lesion") if np.any(segmentation > 0) else None

    model_seg = resample_to_model(segmentation, transform, order=0)
    source_coordinates = np.argwhere(segmentation > 0)
    if source_coordinates.size:
        world = nib.affines.apply_affine(transform.native_affine, source_coordinates)
        target = nib.affines.apply_affine(np.linalg.inv(transform.target_affine), world)
        target = np.rint(target).astype(np.int64)
        upper = np.asarray(transform.target_shape, dtype=np.int64) - 1
        if np.any(target < 0) or np.any(target > upper):
            raise ValueError("lesion projection escaped model FOV after containment audit")
        labels = segmentation[tuple(source_coordinates.T)]
        flat = np.ravel_multi_index(tuple(target.T), transform.target_shape)
        projected = model_seg.reshape(-1)
        np.maximum.at(projected, flat, labels)

    source_labels = set(int(value) for value in np.unique(segmentation) if value != 0)
    target_labels = set(int(value) for value in np.unique(model_seg) if value != 0)
    missing = sorted(source_labels - target_labels)
    if missing:
        raise ValueError(f"model-space segmentation lost labels: {missing}")
    return model_seg


def restore_to_native(
    image: np.ndarray,
    transform: SpatialTransform,
    *,
    order: int = 1,
) -> np.ndarray:
    _validate_volume(np.asarray(image), transform.target_shape)
    return _resample(
        image,
        transform.target_affine,
        (transform.native_shape, transform.native_affine),
        order,
    )
