"""nnU-Net-style cropping/normalization followed by exact physical-grid resampling.

physical_spacing_v2 uses requested spacing for BOTH interpolation and the output affine.
Unlike shape-based nnU-Net/skimage resize, rounding the output shape does not stretch
the voxel grid. Arrays are consequently not claimed to be bit-identical to S2.
Legacy shape-based helpers remain available for comparisons, but preprocess_case uses
resample_physical regardless of the available backend. Axis order is the input order.
"""

from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, map_coordinates

_sk_resize = None
GRID_VERSION = "physical_spacing_v2"

ANISO_THRESHOLD = 3

try:  # 1. nnU-Net v2 itself
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero as _nn_crop_to_nonzero
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape as _nn_resample_to_shape,
        compute_new_shape as _nn_compute_new_shape,
    )
    BACKEND = "nnunetv2"
except Exception:  # noqa: BLE001
    _nn_crop_to_nonzero = _nn_resample_to_shape = _nn_compute_new_shape = None
    try:  # 2. scikit-image (what nnU-Net uses internally)
        from skimage.transform import resize as _sk_resize
        BACKEND = "skimage"
    except Exception:  # noqa: BLE001
        _sk_resize = None
        BACKEND = "scipy"


# ---------------------------------------------------------------------------
# resize primitives
# ---------------------------------------------------------------------------

def _resize(image, new_shape, order, mode="edge", clip=True, anti_aliasing=False):
    """skimage.transform.resize(image, new_shape, order, mode='edge', anti_aliasing=False)."""
    if _sk_resize is not None:
        return _sk_resize(image, new_shape, order, mode=mode, clip=clip, anti_aliasing=anti_aliasing)
    image = np.asarray(image, dtype=float)
    factors = [n / o for n, o in zip(new_shape, image.shape)]
    out = ndimage.zoom(image, factors, order=order, mode="nearest", grid_mode=True)
    if clip and order > 0:
        np.clip(out, image.min(), image.max(), out=out)
    return out


def _resize_segmentation(segmentation, new_shape, order=3):
    """batchgenerators.augmentations.utils.resize_segmentation."""
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    if order == 0:
        return _resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True).astype(tpe)
    reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
    for c in unique_labels:
        reshaped_multihot = _resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True)
        reshaped[reshaped_multihot >= 0.5] = c
    return reshaped


# ---------------------------------------------------------------------------
# nnU-Net v2.1 default_resampling (verbatim logic)
# ---------------------------------------------------------------------------

def get_do_separate_z(spacing, anisotropy_threshold=ANISO_THRESHOLD):
    return (np.max(spacing) / np.min(spacing)) > anisotropy_threshold


def get_lowres_axis(new_spacing):
    return np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]


def compute_new_shape(old_shape, old_spacing, new_spacing):
    if _nn_compute_new_shape is not None:
        return np.asarray(_nn_compute_new_shape(old_shape, old_spacing, new_spacing))
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def _resample_data_or_seg(data, new_shape, is_seg=False, axis=None, order=3, do_separate_z=False, order_z=0):
    assert data.ndim == 4, "data must be (c, x, y, z)"
    resize_fn = _resize_segmentation if is_seg else _resize
    kwargs = OrderedDict() if is_seg else {"mode": "edge", "anti_aliasing": False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if not np.any(shape != new_shape):
        return data
    data = data.astype(float)
    if do_separate_z:
        assert len(axis) == 1, "only one anisotropic axis supported"
        axis = axis[0]
        if axis == 0:
            new_shape_2d = new_shape[1:]
        elif axis == 1:
            new_shape_2d = new_shape[[0, 2]]
        else:
            new_shape_2d = new_shape[:-1]
        out = []
        for c in range(data.shape[0]):
            slices = []
            for slice_id in range(shape[axis]):
                if axis == 0:
                    slices.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                elif axis == 1:
                    slices.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                else:
                    slices.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs))
            reshaped_data = np.stack(slices, axis)
            if shape[axis] != new_shape[axis]:
                rows, cols, dim = new_shape
                orig_rows, orig_cols, orig_dim = reshaped_data.shape
                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = float(orig_rows) / rows * (map_rows + 0.5) - 0.5
                map_cols = float(orig_cols) / cols * (map_cols + 0.5) - 0.5
                map_dims = float(orig_dim) / dim * (map_dims + 0.5) - 0.5
                coord_map = np.array([map_rows, map_cols, map_dims])
                if not is_seg or order_z == 0:
                    out.append(map_coordinates(reshaped_data, coord_map, order=order_z, mode="nearest")[None])
                else:
                    reshaped = np.zeros(new_shape, dtype=dtype_data)
                    for cl in np.sort(np.unique(reshaped_data.ravel())):
                        multihot = np.round(map_coordinates((reshaped_data == cl).astype(float), coord_map,
                                                            order=order_z, mode="nearest"))
                        reshaped[multihot > 0.5] = cl
                    out.append(reshaped[None])
            else:
                out.append(reshaped_data[None])
        result = np.vstack(out)
    else:
        result = np.vstack([resize_fn(data[c], new_shape, order, **kwargs)[None] for c in range(data.shape[0])])
    return result.astype(dtype_data)


def resample_data_or_seg_to_shape(data, new_shape, current_spacing, new_spacing, is_seg=False,
                                  order=3, order_z=0, force_separate_z=None,
                                  separate_z_anisotropy_threshold=ANISO_THRESHOLD):
    if _nn_resample_to_shape is not None:
        return _nn_resample_to_shape(data, new_shape, current_spacing, new_spacing, is_seg=is_seg,
                                     order=order, order_z=order_z, force_separate_z=force_separate_z,
                                     separate_z_anisotropy_threshold=separate_z_anisotropy_threshold)
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = get_lowres_axis(current_spacing) if force_separate_z else None
    elif get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(current_spacing)
    elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(new_spacing)
    else:
        do_separate_z, axis = False, None
    if axis is not None and len(axis) in (2, 3):
        do_separate_z = False
    return _resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)


# ---------------------------------------------------------------------------
# cropping + normalization
# ---------------------------------------------------------------------------

def crop_to_nonzero(data, seg, nonzero_label=-1):
    """nnU-Net crop_to_nonzero: crop to the bbox of the (hole-filled) union nonzero mask.

    Returns (data, seg, bbox) where seg is -1 outside the nonzero mask and bbox is
    [[lo, hi), ...] per spatial axis in the input array.
    """
    if _nn_crop_to_nonzero is not None:
        return _nn_crop_to_nonzero(data, seg, nonzero_label)
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        nonzero_mask |= data[c] != 0
    nonzero_mask = binary_fill_holes(nonzero_mask)
    coords = np.argwhere(nonzero_mask)
    lo, hi = coords.min(0), coords.max(0) + 1
    bbox = [[int(a), int(b)] for a, b in zip(lo, hi)]
    slicer = tuple(slice(a, b) for a, b in bbox)
    data = data[(slice(None), *slicer)]
    seg = seg[(slice(None), *slicer)]
    nonzero_mask = nonzero_mask[slicer][None]
    seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    return data, seg, bbox


def zscore_masked(data, seg):
    """nnU-Net ZScoreNormalization with use_mask_for_norm=True, applied per channel."""
    out = np.empty(data.shape, dtype=np.float32)
    mask = seg[0] >= 0
    for c in range(data.shape[0]):
        image = data[c].astype(np.float32)
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
        out[c] = image
    return out


def physical_grid(shape, spacing, target_spacing):
    """An exact-spacing grid sharing the input's low voxel-cell boundary.

    Output index j samples input at (j + 0.5) * target/current - 0.5.
    Shape rounding changes the far edge, never the requested voxel spacing.
    """
    spacing, target = np.asarray(spacing, float), np.asarray(target_spacing, float)
    if spacing.shape != (3,) or target.shape != (3,) or not np.isfinite([spacing, target]).all():
        raise ValueError("Spacing must contain three finite values")
    if np.any(spacing <= 0) or np.any(target <= 0):
        raise ValueError("Spacing must be positive")
    factor = target / spacing
    new_shape = np.maximum(1, np.rint(np.asarray(shape) / factor).astype(int))
    return new_shape, factor, (factor - 1) / 2


def resample_physical(data, new_shape, factor, offset, spacing, target_spacing, is_seg=False):
    """Use the same physical coordinates as the saved affine, including anisotropic scans."""
    axis = None
    for candidate in (spacing, target_spacing):
        if get_do_separate_z(candidate):
            axes = get_lowres_axis(candidate)
            if len(axes) == 1:
                axis = int(axes[0])
            break

    def interpolate(volume, order):
        if axis is None:
            return ndimage.affine_transform(volume, np.diag(factor), offset,
                                            output_shape=tuple(new_shape), order=order, mode="nearest")
        # nnU-Net-style low-resolution-axis nearest interpolation, on an exact physical grid.
        f1, o1 = factor.copy(), offset.copy()
        intermediate_shape = np.asarray(new_shape).copy()
        f1[axis], o1[axis] = 1, 0
        intermediate_shape[axis] = volume.shape[axis]
        intermediate = ndimage.affine_transform(volume, np.diag(f1), o1,
                                                output_shape=tuple(intermediate_shape), order=order,
                                                mode="nearest")
        f2, o2 = np.ones(3), np.zeros(3)
        f2[axis], o2[axis] = factor[axis], offset[axis]
        return ndimage.affine_transform(intermediate, np.diag(f2), o2,
                                       output_shape=tuple(new_shape), order=0, mode="nearest")

    channels = []
    for volume in data:
        if is_seg:
            result = interpolate(volume, 0)
            # Label-wise interpolation avoids inventing fractional/new class IDs.
            for label in np.unique(volume):
                probability = interpolate((volume == label).astype(np.float32), 1)
                result[probability >= 0.5] = label
        else:
            result = interpolate(volume, 3)
            np.clip(result, volume.min(), volume.max(), out=result)
        channels.append(result)
    return np.asarray(channels, dtype=data.dtype)


def preprocess_case(data, seg, spacing, target_spacing=(1.0, 1.0, 1.0)):
    """Full nnU-Net v2 3d_fullres preprocessing for one case.

    Args:
        data:    (C, X, Y, Z) float32 image channels
        seg:     (1, X, Y, Z) integer label map
        spacing: voxel spacing (mm) in the same axis order as data
    Returns:
        data_out (C, X', Y', Z') float32, seg_out (1, X', Y', Z') int8 (-1 = outside brain), info dict
    """
    data = np.asarray(data, dtype=np.float32)
    seg = np.asarray(seg)
    if data.ndim != 4 or seg.shape != (1, *data.shape[1:]):
        raise ValueError("Expected data (C,X,Y,Z) and seg (1,X,Y,Z)")
    if not np.isfinite(data).all() or not np.isfinite(seg).all():
        raise ValueError("Non-finite input")
    if not np.equal(seg, np.rint(seg)).all() or not np.isin(seg, [0, 1, 2, 3, 4]).all():
        raise ValueError("Invalid segmentation labels")
    physical_grid(data.shape[1:], spacing, target_spacing)
    if not np.any(data):
        raise ValueError("Empty brain mask")
    seg = seg.astype(np.int16, copy=True)
    shape_before = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg)
    shape_cropped = data.shape[1:]
    new_shape, factor, offset = physical_grid(shape_cropped, spacing, target_spacing)
    data = zscore_masked(data, seg)                       # normalization BEFORE resampling (as nnU-Net)
    data = resample_physical(data, new_shape, factor, offset, spacing, target_spacing)
    seg = resample_physical(seg, new_shape, factor, offset, spacing, target_spacing, is_seg=True)
    info = {
        "backend": BACKEND,
        "grid_version": GRID_VERSION,
        "shape_before_cropping": [int(v) for v in shape_before],
        "bbox_used_for_cropping": [[int(a), int(b)] for a, b in bbox],
        "shape_after_cropping": [int(v) for v in shape_cropped],
        "shape_after_resampling": [int(v) for v in new_shape],
        "original_spacing": [float(v) for v in spacing],
        "target_spacing": [float(v) for v in target_spacing],
    }
    return data.astype(np.float32), seg.astype(np.int8), info
