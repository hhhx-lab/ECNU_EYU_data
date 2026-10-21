import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.transforms import Randomizable
from torch import clone as clone
import numpy as np
from scipy import ndimage


def zscore_then_rescale(arr, target_min=-1.0, target_max=1.0):
    """[DEPRECATED] z-score then minmax rescale to [-1,1].

    Kept for backward compatibility with old checkpoints. New training should use
    ``brain_zscore_normalize`` (matching S2 nnUNet preprocessing) instead.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return arr
    z = (arr - mean) / std
    z_min, z_max = np.min(z), np.max(z)
    if z_max == z_min:
        return np.full_like(arr, (target_min + target_max) / 2)
    return (z - z_min) / (z_max - z_min) * (target_max - target_min) + target_min


def brain_zscore_normalize(arr, brain_mask=None):
    """Z-score normalize using only brain-tissue voxels (matching S2 nnUNet).

    S2 nnUNet computes mean/std from ``seg >= 0`` (non-padding brain voxels)
    and applies:  data[mask] = (data[mask] - mean) / max(std, 1e-8)

    When *brain_mask* is None, all voxels in *arr* are treated as brain
    (the crop is centred on a lesion and fully inside the brain).

    Returns array with brain tissue ~N(0,1) in the same type (e.g. MetaTensor
    or np.ndarray) as the input. No [-1,1] rescaling.
    """
    is_torch = isinstance(arr, torch.Tensor)
    if is_torch:
        if brain_mask is None:
            mask = slice(None)
        elif isinstance(brain_mask, torch.Tensor):
            mask = brain_mask.to(arr.device)
        else:
            mask = torch.as_tensor(brain_mask, device=arr.device, dtype=torch.bool)
        vox = arr[mask]
        if vox.numel() == 0:
            return arr
        mean = vox.mean()
        std = vox.std()
        if std < 1e-8:
            std = 1.0
        result = torch.zeros_like(arr)
        result[mask] = (vox - mean) / std
        return result

    # numpy path
    arr_np = np.asarray(arr, dtype=np.float32)
    if brain_mask is None:
        brain_mask = np.ones_like(arr_np, dtype=bool)
    vox = arr_np[brain_mask]
    if vox.size == 0:
        return arr
    mean = vox.mean()
    std = vox.std()
    if std < 1e-8:
        std = 1.0
    result_np = np.zeros_like(arr_np, dtype=np.float32)
    result_np[brain_mask] = (vox - mean) / std
    return result_np


MAX_CROP_SHIFT = 8  # upper bound of the adaptive random offset in GaussianNoiseTumour
_CC26 = np.ones((3, 3, 3), dtype=np.int16)


def effective_n_voxels(tumour_binary, centre):
    """Voxel count of the 26-connected component at `centre`; if `centre` is background, of the
    component whose centroid is nearest to `centre` (first one on ties); 0 if there is no tumour.

    [1mm] Same result as the original per-component ndimage.center_of_mass loop, but labels only the
    tumour bounding box and gets all centroids from one bincount pass (several times faster).
    """
    if not tumour_binary.any():
        return 0
    lo, hi = [], []
    for ax in range(3):
        nz = np.flatnonzero(tumour_binary.any(axis=tuple(a for a in range(3) if a != ax)))
        lo.append(int(nz[0]))
        hi.append(int(nz[-1]) + 1)
    sub = tumour_binary[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    labeled, n_cc = ndimage.label(sub, structure=_CC26)
    c_local = tuple(c - l for c, l in zip(centre, lo))
    if all(0 <= c < s for c, s in zip(c_local, sub.shape)) and labeled[c_local] > 0:
        return int((labeled == labeled[c_local]).sum())
    flat = labeled.ravel()
    counts = np.bincount(flat, minlength=n_cc + 1)
    grids = np.indices(sub.shape)
    d2 = np.zeros(n_cc)
    for ax in range(3):
        cent = np.bincount(flat, weights=grids[ax].ravel() + lo[ax], minlength=n_cc + 1)[1:] / counts[1:]
        d2 += (cent - centre[ax]) ** 2
    return int(counts[int(np.argmin(d2)) + 1])


class CropLesionROId(MapTransform):
    """[1mm] Deterministic pre-crop so that CacheDataset stores only a small ROI per CSV row.

    Keeps the region that every possible GaussianNoiseTumour window of this row can touch:
    the un-shifted crop window (same formula as GaussianNoiseTumour) extended by
    MAX_CROP_SHIFT voxels on each side and clamped to the volume. Full volumes are dropped,
    so the cache holds ~ (target_size + 2*8)^3 voxels per row instead of the whole brain.
    Bbox / centre fields stay in whole-volume coordinates; the ROI origin is stored in
    ``roi_offset`` and the volume shape in ``vol_shape``.
    """

    def __init__(self, keys: KeysCollection, target_size: int = 64):
        super().__init__(keys)
        self.scan_key, self.label_key = keys
        self.target_size = target_size

    def __call__(self, data):
        d = dict(data)
        scan = d[self.scan_key]
        _, max_x, max_y, max_z = scan.shape
        ts = self.target_size
        lo, hi = [], []
        for ax, dim in zip("xyz", (max_x, max_y, max_z)):
            e_min, e_max = d[f"{ax}_extreme_min"], d[f"{ax}_extreme_max"]
            pad = (ts - (e_max - e_min)) / 2
            c = -0.5 if pad < 0 else 0.5
            base, top = e_min - int(pad), e_max + int(pad + c)
            lo.append(max(0, base - MAX_CROP_SHIFT))
            hi.append(min(dim, top + MAX_CROP_SHIFT))
        sl = (slice(None), slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        to_np = lambda a: a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        # [1mm] Guard: the row's bbox must contain its tumour voxels. Fails for CSVs written with the
        # original (z, y, x) axis convention or built from a different DataSet -> rebuild the CSV.
        bb = [int(d[f"{ax}_extreme_{m}"]) for ax in "xyz" for m in ("min", "max")]
        n_in = int((to_np(d[self.label_key])[0, bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] > 0).sum())
        n_row = int(d["n_voxels"])
        if n_in < n_row:
            raise RuntimeError(
                f"CSV row (label {d.get('label_meta_dict', {}).get('filename_or_obj', '?')}, bbox {bb}): "
                f"only {n_in}/{n_row} tumour voxels inside its bbox. The CSV "
                f"does not match the label volume (old axis convention or different DataSet) - rebuild it with "
                f"src/train/csv_creator.py from this package.")
        d[self.scan_key] = torch.from_numpy(np.ascontiguousarray(to_np(scan)[sl], dtype=np.float32))
        d[self.label_key] = torch.from_numpy(np.ascontiguousarray(to_np(d[self.label_key])[sl]).round().astype(np.uint8))
        d["roi_offset"] = (int(lo[0]), int(lo[1]), int(lo[2]))
        d["vol_shape"] = (int(max_x), int(max_y), int(max_z))
        return d


class GaussianNoiseTumour(MapTransform, Randomizable):
    """Crop/pad a lesion-centred window, normalise it and add tumour-region noise.

    [1mm] Changes w.r.t. the original GliGAN transform (all optional / backward compatible):
      * Randomizable: the adaptive offset and the noise are re-drawn at every iteration
        (MONAI's CacheDataset no longer freezes them, it caches up to this transform).
      * ROI mode: if the sample was pre-cropped by CropLesionROId, the window is computed in
        whole-volume coordinates exactly as before and then read from the cached ROI.
      * label_transform: in ROI mode the single-channel label is converted to the multi-channel
        label after cropping (element-wise, hence identical to converting the full volume first).
      * normalization="nnunet": the scan is already nnU-Net normalised (whole-case masked
        z-score), so the crop is NOT re-normalised; padding/background value is 0 as for "zscore".
    """

    def __init__(self, keys: KeysCollection, normalization="minmax", target_size: int = 64,
                 label_transform=None):
        super().__init__(keys)
        self.keys = keys
        self.normalization = normalization
        self.target_size = target_size
        self.label_transform = label_transform
        self.zero_background = normalization in ("zscore", "nnunet")

    def __call__(self, data):
        d = dict(data)
        scan_key = self.keys
        scan_data = d[scan_key]
        roi_mode = "roi_offset" in d
        if roi_mode:
            max_x, max_y, max_z = d["vol_shape"]
            off_x, off_y, off_z = d["roi_offset"]
        else:
            _, max_x, max_y, max_z = scan_data.shape
            off_x = off_y = off_z = 0
        scan_crop = clone(scan_data)
        label = d["label"]
        label_crop = clone(label)

        x_extreme_dif = d["x_extreme_max"] - d["x_extreme_min"]
        y_extreme_dif = d["y_extreme_max"] - d["y_extreme_min"]
        z_extreme_dif = d["z_extreme_max"] - d["z_extreme_min"]

        ts = self.target_size

        # ---- Adaptive random offset (break positional overfitting) ----
        def _margin(ext_min, ext_max, _ts):
            size = ext_max - ext_min
            pad_each = (_ts - size) // 2
            return min(8, max(0, pad_each - 2))

        x_shift = _margin(d["x_extreme_min"], d["x_extreme_max"], ts)
        y_shift = _margin(d["y_extreme_min"], d["y_extreme_max"], ts)
        z_shift = _margin(d["z_extreme_min"], d["z_extreme_max"], ts)

        if x_shift > 0:
            ox = self.R.randint(-x_shift, x_shift + 1)
            ox = max(-d["x_extreme_min"], min(ox, max_x - d["x_extreme_max"]))
            d["x_extreme_min"] += ox
            d["x_extreme_max"] += ox
            d["center_x"] += ox
        if y_shift > 0:
            oy = self.R.randint(-y_shift, y_shift + 1)
            oy = max(-d["y_extreme_min"], min(oy, max_y - d["y_extreme_max"]))
            d["y_extreme_min"] += oy
            d["y_extreme_max"] += oy
            d["center_y"] += oy
        if z_shift > 0:
            oz = self.R.randint(-z_shift, z_shift + 1)
            oz = max(-d["z_extreme_min"], min(oz, max_z - d["z_extreme_max"]))
            d["z_extreme_min"] += oz
            d["z_extreme_max"] += oz
            d["center_z"] += oz
        # -----------------------------------------------------------------

        x_pad = (ts - x_extreme_dif) / 2
        y_pad = (ts - y_extreme_dif) / 2
        z_pad = (ts - z_extreme_dif) / 2

        if x_pad < 0:
            C_x = -0.5
        else:
            C_x = 0.5

        if y_pad < 0:
            C_y = -0.5
        else:
            C_y = 0.5

        if z_pad < 0:
            C_z = -0.5
        else:
            C_z = 0.5

        x_base = d["x_extreme_min"] - int(x_pad)
        x_top = d["x_extreme_max"] + int(x_pad + C_x)
        y_base = d["y_extreme_min"] - int(y_pad)
        y_top = d["y_extreme_max"] + int(y_pad + C_y)
        z_base = d["z_extreme_min"] - int(z_pad)
        z_top = d["z_extreme_max"] + int(z_pad + C_z)

        # Verifying the need for padding
        x_base_pad = 0
        y_base_pad = 0
        z_base_pad = 0
        x_top_pad = 0
        y_top_pad = 0
        z_top_pad = 0

        if x_base < 0:
            x_base_pad = -x_base
            x_base = 0

        if y_base < 0:
            y_base_pad = -y_base
            y_base = 0

        if z_base < 0:
            z_base_pad = -z_base
            z_base = 0

        if x_top > max_x:
            x_top_pad = x_top - max_x
            x_top = max_x

        if y_top > max_y:
            y_top_pad = y_top - max_y
            y_top = max_y

        if z_top > max_z:
            z_top_pad = z_top - max_z
            z_top = max_z
        ##################################
        # Crop the label (ROI mode: whole-volume window shifted into ROI coordinates)
        label_crop = label_crop[:, x_base - off_x: x_top - off_x,
                                y_base - off_y: y_top - off_y, z_base - off_z: z_top - off_z]
        if roi_mode and self.label_transform is not None:
            label_crop = self.label_transform(label_crop)  # single-channel -> multi-channel (TC/WT/ET/RC)

        # Crop and Normalise the scan
        scan_crop = scan_crop[:, x_base - off_x: x_top - off_x,
                              y_base - off_y: y_top - off_y, z_base - off_z: z_top - off_z]

        # Guard against empty crop (e.g. when scan dimensions don't match CSV bbox)
        if scan_crop.numel() == 0:
            scan_shape = scan_data.shape
            raise RuntimeError(
                f"Empty crop produced for lesion. "
                f"scan_shape={scan_shape}, "
                f"x_range=[{x_base},{x_top}], y_range=[{y_base},{y_top}], z_range=[{z_base},{z_top}], "
                f"original_bbox: x=[{d['x_extreme_min']},{d['x_extreme_max']}], "
                f"y=[{d['y_extreme_min']},{d['y_extreme_max']}], "
                f"z=[{d['z_extreme_min']},{d['z_extreme_max']}], "
                f"target_size={self.target_size}, "
                f"x_pad={x_pad:.1f}, y_pad={y_pad:.1f}, z_pad={z_pad:.1f}"
            )

        # Resize if crop exceeds target_size
        scan_crop, label_crop = self._resize_crop_if_needed(scan_crop, label_crop)

        if self.normalization == "zscore":
            scan_crop = brain_zscore_normalize(scan_crop)
        elif self.normalization == "nnunet":
            pass  # [1mm] already nnU-Net normalised (whole-case masked z-score): keep values as-is
        else:
            scan_crop = self.rescale_array(arr=scan_crop, minv=-1, maxv=1)
        d["scan_crop"] = scan_crop

        # Scan and label with padding to target_size
        # zscore / nnunet mode: pad with 0 (matching nnUNet constant_values=0)
        # minmax mode: pad with -1 (background convention)
        pad_val = 0 if self.zero_background else -1
        scan_crop_pad = clone(scan_crop)
        scan_crop_pad = np.pad(scan_crop_pad,
                               pad_width=((0, 0), (x_base_pad, x_top_pad),
                                          (y_base_pad, y_top_pad),
                                          (z_base_pad, z_top_pad)),
                               mode='constant', constant_values=(pad_val, pad_val))
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad,
                                pad_width=((0, 0), (x_base_pad, x_top_pad),
                                           (y_base_pad, y_top_pad),
                                           (z_base_pad, z_top_pad)),
                                mode='constant', constant_values=(0, 0))

        scan_noisy = self.add_gaussian_noise_tumour(scan=scan_crop_pad, label=label_crop_pad,
                                                     is_zscore=self.zero_background)
        if self.zero_background:
            # No rescaling — data stays in z-score space (~N(0,1) in brain)
            pass
        else:
            scan_noisy = self.rescale_array_numpy(arr=scan_noisy, minv=-1, maxv=1)

        if roi_mode:
            # [1mm] the cached ROI stays in the cache; do not pass it (or its bookkeeping) to collate
            for k in (scan_key, "label", "roi_offset", "vol_shape", f"{scan_key}_meta_dict", "label_meta_dict"):
                d.pop(k, None)
        else:
            d[scan_key] = scan_data
        d[f"{scan_key}_crop"] = scan_crop
        d[f"{scan_key}_crop_pad"] = scan_crop_pad
        d[f"{scan_key}_noisy"] = scan_noisy
        d["label_crop"] = label_crop
        d["label_crop_pad"] = label_crop_pad

        # ---- Compute effective_n_voxels from actual crop window ----
        # Uses max CC size within the 64^3 crop (not CSV n_voxels),
        # so loss weighting reflects what the model actually sees.
        # Use CC at crop centre (32,32,32) as the target lesion.
        # Falls back to nearest CC if centre falls on background.
        # This prevents large debris fragments from suppressing the
        # loss weight of small central lesions.
        tumour_binary = np.any(label_crop_pad, axis=0)
        d["effective_n_voxels"] = effective_n_voxels(tumour_binary, (ts // 2, ts // 2, ts // 2))

        return d

    def _resize_crop_if_needed(self, scan_crop, label_crop):
        """If the cropped region exceeds target_size in any dimension, zoom both
        scan and label down proportionally so the largest dimension fits target_size.
        Returns (scan_crop, label_crop) — may be resized or unchanged."""
        max_dim = max(scan_crop.shape[1:])
        if max_dim <= self.target_size:
            return scan_crop, label_crop
        scale = self.target_size / max_dim
        from scipy.ndimage import zoom as ndimage_zoom
        # Scan
        new_shape = np.maximum(np.round(np.array(scan_crop.shape[1:]) * scale), 1).astype(int)
        zoomed_s = np.zeros((scan_crop.shape[0],) + tuple(new_shape), dtype=np.float32)
        factors = tuple(new_shape.astype(float) / np.array(scan_crop.shape[1:]))
        for c in range(scan_crop.shape[0]):
            arr = scan_crop[c].cpu().numpy() if hasattr(scan_crop, 'cpu') else np.asarray(scan_crop[c])
            zoomed_s[c] = ndimage_zoom(arr.astype(np.float32), factors, order=1)
        scan_crop = torch.from_numpy(zoomed_s)
        # Label
        new_shape_l = np.maximum(np.round(np.array(label_crop.shape[1:]) * scale), 1).astype(int)
        zoomed_l = np.zeros((label_crop.shape[0],) + tuple(new_shape_l), dtype=np.float32)
        factors_l = tuple(new_shape_l.astype(float) / np.array(label_crop.shape[1:]))
        for c in range(label_crop.shape[0]):
            arr = label_crop[c].cpu().numpy() if hasattr(label_crop, 'cpu') else np.asarray(label_crop[c])
            zoomed_l[c] = ndimage_zoom(arr.astype(np.float32), factors_l, order=1)
        label_crop = torch.from_numpy(zoomed_l)
        return scan_crop, label_crop

    def rescale_array(self, arr, minv, maxv):
        if arr.numel() == 0:
            return arr
        mina = torch.min(arr)
        maxa = torch.max(arr)
        if mina == maxa:
            return arr * minv
        norm = (arr - mina) / (maxa - mina)
        return (norm * (maxv - minv)) + minv

    def rescale_array_numpy(self, arr, minv, maxv):
        mina = np.min(arr)
        maxa = np.max(arr)
        if mina == maxa:
            return arr * minv
        norm = (arr - mina) / (maxa - mina)
        return (norm * (maxv - minv)) + minv

    def add_gaussian_noise_tumour(self, scan, label, is_zscore=False):
        scan_noisy = np.copy(scan)
        ts = self.target_size
        noise = np.full((1, ts, ts, ts), 1000., dtype=scan_noisy.dtype)
        tumour_mask = np.any(label, axis=0)  # (ts, ts, ts) bool
        n_tumour = int(tumour_mask.sum())
        if n_tumour > 0:
            noise[0, tumour_mask] = self.R.randn(n_tumour).astype(scan_noisy.dtype)
        # zscore: background/padding is 0; minmax: background/padding is -1
        bg_val = 0 if is_zscore else -1
        np.copyto(scan_noisy, noise, where=np.logical_and(noise < 100, scan_noisy != bg_val))
        return scan_noisy
