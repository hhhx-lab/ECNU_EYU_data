"""Synchronized 3D patch sampling for seg-guided VAE fine-tuning."""

from itertools import product

import numpy as np


NEIGHBOR_OFFSETS_26 = tuple(
    offset
    for offset in product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
)


def validate_patch_size(patch_size, divisor=4):
    patch_size = tuple(int(value) for value in patch_size)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError(f"Patch size must contain three positive integers, got {patch_size}.")
    if any(value % divisor for value in patch_size):
        raise ValueError(
            f"Every patch dimension must be divisible by {divisor}, got {patch_size}."
        )
    return patch_size


def _label_components_fallback(mask):
    """Dependency-free 26-connected labeling used only when SciPy is unavailable."""
    foreground = {tuple(int(v) for v in coord) for coord in np.argwhere(mask)}
    labels = np.zeros(mask.shape, dtype=np.int32)
    component_id = 0
    shape = mask.shape

    while foreground:
        component_id += 1
        seed = foreground.pop()
        stack = [seed]
        labels[seed] = component_id
        while stack:
            voxel = stack.pop()
            for offset in NEIGHBOR_OFFSETS_26:
                neighbor = tuple(voxel[axis] + offset[axis] for axis in range(3))
                if not all(0 <= neighbor[axis] < shape[axis] for axis in range(3)):
                    continue
                if neighbor in foreground:
                    foreground.remove(neighbor)
                    labels[neighbor] = component_id
                    stack.append(neighbor)
    return labels, component_id


def label_components_26(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")
    try:
        from scipy import ndimage

        return ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    except ImportError:
        return _label_components_fallback(mask)


def choose_tumor_component_center(tumor_mask, rng):
    """Select one 26-connected tumor component uniformly and return its centroid."""
    labels, component_count = label_components_26(tumor_mask)
    if component_count == 0:
        return None, None
    component_id = int(rng.integers(1, component_count + 1))
    coords = np.argwhere(labels == component_id)
    center = np.rint(coords.mean(axis=0)).astype(np.int64)
    return center, component_id


def choose_brain_center(brain_mask, rng):
    coords = np.argwhere(np.asarray(brain_mask) > 0)
    if len(coords) == 0:
        return np.asarray(brain_mask.shape, dtype=np.int64) // 2
    return coords[int(rng.integers(0, len(coords)))].astype(np.int64)


def crop_or_pad_around_center(array, center, patch_size):
    """Crop around a voxel center and zero-pad when the requested box crosses a boundary."""
    array = np.asarray(array)
    patch_size = validate_patch_size(patch_size)
    center = np.asarray(center, dtype=np.int64)
    if array.ndim != 3 or center.shape != (3,):
        raise ValueError(f"Expected a 3D array and 3D center, got {array.shape}, {center}.")

    requested_start = center - np.asarray(patch_size, dtype=np.int64) // 2
    requested_end = requested_start + np.asarray(patch_size, dtype=np.int64)
    source_start = np.maximum(requested_start, 0)
    source_end = np.minimum(requested_end, np.asarray(array.shape, dtype=np.int64))
    destination_start = source_start - requested_start
    destination_end = destination_start + (source_end - source_start)

    result = np.zeros(patch_size, dtype=array.dtype)
    source_slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_end))
    destination_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(destination_start, destination_end)
    )
    result[destination_slices] = array[source_slices]
    return result


def sample_synchronized_patch(
    images,
    seg,
    patch_size,
    tumor_probability,
    rng,
    brain_threshold=0.02,
):
    """Sample one center, then apply exactly the same crop to all modalities and masks."""
    patch_size = validate_patch_size(patch_size)
    if len(images) != 4:
        raise ValueError(f"Expected four modalities, got {len(images)}.")
    if not 0.0 <= tumor_probability <= 1.0:
        raise ValueError("tumor_probability must be in [0, 1].")
    if any(np.asarray(image).shape != np.asarray(seg).shape for image in images):
        raise ValueError("All modalities and segmentation must have the same shape.")

    mean_image = np.mean(images, axis=0)
    brain_mask = (mean_image > brain_threshold).astype(np.float32)
    tumor_mask = (np.asarray(seg) > 0).astype(np.float32)
    healthy_mask = np.clip(brain_mask - tumor_mask, 0.0, 1.0)

    use_tumor = bool(tumor_mask.any()) and float(rng.random()) < tumor_probability
    component_id = None
    if use_tumor:
        center, component_id = choose_tumor_component_center(tumor_mask, rng)
        mode = "tumor"
    else:
        center = choose_brain_center(brain_mask, rng)
        mode = "brain"

    return {
        "images": [crop_or_pad_around_center(image, center, patch_size) for image in images],
        "seg": crop_or_pad_around_center(seg, center, patch_size),
        "brain_mask": crop_or_pad_around_center(brain_mask, center, patch_size),
        "tumor_mask": crop_or_pad_around_center(tumor_mask, center, patch_size),
        "healthy_mask": crop_or_pad_around_center(healthy_mask, center, patch_size),
        "mode": mode,
        "component_id": component_id,
        "center": tuple(int(value) for value in center),
    }
