"""nnU-Net v2 compatible preprocessing (crop -> masked z-score -> resample).

Reproduces ``DefaultPreprocessor.run_case_npy`` of nnU-Net v2 for a 3d_fullres
configuration with ``ZScoreNormalization`` (use_mask_for_norm=True) and the
default resampling (data: order 3, seg: order 1, order_z 0, force_separate_z None),
so that the diffusion model is trained on exactly the representation that the
S2 on-the-fly augmentation feeds it at inference time.

Backend selection:
    1. ``nnunetv2`` installed       -> its own crop / resampling functions (bit-identical to S2)
    2. ``scikit-image`` installed   -> a verbatim copy of the nnU-Net v2.1 resampling code
    3. neither                      -> scipy re-implementation of skimage.transform.resize
                                       (grid_mode zoom + clip); numerically near-identical

All arrays use the axis order in which they are passed in. nnU-Net's resampling is
equivariant to axis permutation, so working in nibabel (x, y, z) order gives the
transpose of nnU-Net's SimpleITK (z, y, x) arrays.
"""

from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, map_coordinates

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
    seg = np.asarray(seg).astype(np.int16)
    shape_before = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg)
    shape_cropped = data.shape[1:]
    new_shape = compute_new_shape(shape_cropped, spacing, target_spacing)
    data = zscore_masked(data, seg)                       # normalization BEFORE resampling (as nnU-Net)
    data = resample_data_or_seg_to_shape(data, new_shape, spacing, target_spacing, is_seg=False,
                                         order=3, order_z=0, force_separate_z=None)
    seg = resample_data_or_seg_to_shape(seg, new_shape, spacing, target_spacing, is_seg=True,
                                        order=1, order_z=0, force_separate_z=None)
    info = {
        "backend": BACKEND,
        "shape_before_cropping": [int(v) for v in shape_before],
        "bbox_used_for_cropping": [[int(a), int(b)] for a, b in bbox],
        "shape_after_cropping": [int(v) for v in shape_cropped],
        "shape_after_resampling": [int(v) for v in new_shape],
        "original_spacing": [float(v) for v in spacing],
        "target_spacing": [float(v) for v in target_spacing],
    }
    return data.astype(np.float32), seg.astype(np.int8), info
