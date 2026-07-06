import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone
import numpy as np
from scipy import ndimage


def zscore_then_rescale(arr, target_min=-1.0, target_max=1.0):
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return arr
    z = (arr - mean) / std
    z_min, z_max = np.min(z), np.max(z)
    if z_max == z_min:
        return np.full_like(arr, (target_min + target_max) / 2)
    return (z - z_min) / (z_max - z_min) * (target_max - target_min) + target_min


class GaussianNoiseTumour(MapTransform):
    def __init__(self, keys: KeysCollection, normalization="minmax", target_size: int = 64):
        super().__init__(keys)
        self.keys = keys
        self.normalization = normalization
        self.target_size = target_size

    def __call__(self, data):
        d = dict(data)
        scan_key = self.keys
        scan_data = d[scan_key]
        _, max_x, max_y, max_z = scan_data.shape
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
            ox = np.random.randint(-x_shift, x_shift + 1)
            ox = max(-d["x_extreme_min"], min(ox, max_x - d["x_extreme_max"]))
            d["x_extreme_min"] += ox
            d["x_extreme_max"] += ox
            d["center_x"] += ox
        if y_shift > 0:
            oy = np.random.randint(-y_shift, y_shift + 1)
            oy = max(-d["y_extreme_min"], min(oy, max_y - d["y_extreme_max"]))
            d["y_extreme_min"] += oy
            d["y_extreme_max"] += oy
            d["center_y"] += oy
        if z_shift > 0:
            oz = np.random.randint(-z_shift, z_shift + 1)
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
        # Crop the label
        label_crop = label_crop[:, x_base: x_top, y_base: y_top, z_base: z_top]

        # Crop and Normalise the scan
        scan_crop = scan_crop[:, x_base: x_top, y_base: y_top, z_base: z_top]

        # Resize if crop exceeds target_size
        scan_crop, label_crop = self._resize_crop_if_needed(scan_crop, label_crop)

        if self.normalization == "zscore":
            scan_crop = zscore_then_rescale(scan_crop, target_min=-1.0, target_max=1.0)
        else:
            scan_crop = self.rescale_array(arr=scan_crop, minv=-1, maxv=1)
        d["scan_crop"] = scan_crop

        # Scan and label with padding to target_size
        scan_crop_pad = clone(scan_crop)
        scan_crop_pad = np.pad(scan_crop_pad,
                               pad_width=((0, 0), (x_base_pad, x_top_pad),
                                          (y_base_pad, y_top_pad),
                                          (z_base_pad, z_top_pad)),
                               mode='constant', constant_values=(-1, -1))
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad,
                                pad_width=((0, 0), (x_base_pad, x_top_pad),
                                           (y_base_pad, y_top_pad),
                                           (z_base_pad, z_top_pad)),
                                mode='constant', constant_values=(0, 0))

        scan_noisy = self.add_gaussian_noise_tumour(scan=scan_crop_pad, label=label_crop_pad)
        if self.normalization == "zscore":
            scan_noisy = zscore_then_rescale(scan_noisy, target_min=-1.0, target_max=1.0)
        else:
            scan_noisy = self.rescale_array_numpy(arr=scan_noisy, minv=-1, maxv=1)

        d[scan_key] = scan_data
        d[f"{scan_key}_crop"] = scan_crop
        d[f"{scan_key}_crop_pad"] = scan_crop_pad
        d[f"{scan_key}_noisy"] = scan_noisy
        d["label_crop"] = label_crop
        d["label_crop_pad"] = label_crop_pad

        # ---- Compute effective_n_voxels from actual crop window ----
        # Uses max CC size within the 64^3 crop (not CSV n_voxels),
        # so loss weighting reflects what the model actually sees.
        tumour_binary = np.any(label_crop_pad, axis=0)
        if tumour_binary.any():
            structure = np.ones((3, 3, 3), dtype=np.int16)
            labeled, n_cc = ndimage.label(tumour_binary, structure=structure)
            # Use CC at crop centre (32,32,32) as the target lesion.
            # Falls back to nearest CC if centre falls on background.
            # This prevents large debris fragments from suppressing the
            # loss weight of small central lesions.
            centre = (ts // 2, ts // 2, ts // 2)
            cc_at_centre = labeled[centre]
            if cc_at_centre > 0:
                d["effective_n_voxels"] = int((labeled == cc_at_centre).sum())
            else:
                best_cc = None
                best_dist = float('inf')
                for cid in range(1, n_cc + 1):
                    centroid = ndimage.center_of_mass(tumour_binary, labeled, cid)
                    dist = np.sqrt(sum((c - o) ** 2 for c, o in zip(centroid, centre)))
                    if dist < best_dist:
                        best_dist = dist
                        best_cc = cid
                d["effective_n_voxels"] = int((labeled == best_cc).sum()) if best_cc else 0
        else:
            d["effective_n_voxels"] = 0

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

    def add_gaussian_noise_tumour(self, scan, label):
        scan_noisy = np.copy(scan)
        ts = self.target_size
        noise = np.full((1, ts, ts, ts), 1000., dtype=scan_noisy.dtype)
        tumour_mask = np.any(label, axis=0)  # (ts, ts, ts) bool
        n_tumour = int(tumour_mask.sum())
        if n_tumour > 0:
            noise[0, tumour_mask] = np.random.randn(n_tumour).astype(scan_noisy.dtype)
        np.copyto(scan_noisy, noise, where=np.logical_and(noise < 100, scan_noisy != -1))
        return scan_noisy
