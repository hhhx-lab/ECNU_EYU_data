from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
)


MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def nonzero_zscore_normalize(image, eps=1e-8):
    """Per-modality Z-score inside the finite nonzero brain mask.

    Must be used identically for train / val / inference.
    Background (zeros / non-finite) stays zero.
    """
    image = np.asarray(image, dtype=np.float32).copy()
    for channel in range(image.shape[0]):
        volume = image[channel]
        finite = np.isfinite(volume)
        mask = finite & (volume != 0)
        volume[~finite] = 0.0
        if not np.any(mask):
            continue
        values = volume[mask]
        mean = float(values.mean())
        std = float(values.std())
        volume[mask] = (values - mean) / max(std, eps)
        volume[~mask] = 0.0
    return image


def _pad_to_patch(array, patch_size):
    spatial_shape = array.shape[-3:]
    pads = []
    for current, required in zip(spatial_shape, patch_size):
        total = max(0, required - current)
        before = total // 2
        pads.append((before, total - before))
    if not any(before or after for before, after in pads):
        return array
    return np.pad(array, ((0, 0), *pads), mode="constant")


def _crop_around_center(array, center, patch_size):
    slices = [slice(None)]
    for coordinate, current, required in zip(center, array.shape[-3:], patch_size):
        start = int(coordinate) - required // 2
        start = min(max(start, 0), current - required)
        slices.append(slice(start, start + required))
    return array[tuple(slices)]


def lesion_balanced_crop(sample, patch_size, lesion_probability=0.8):
    """70-80% crop around a uniformly chosen connected lesion; rest random brain.

    Uniform lesion sampling prevents large tumors from dominating random
    positive-voxel sampling, which is important for small BraTS-MET metastases.
    """
    patch_size = tuple(int(value) for value in patch_size)
    arrays = {
        key: _pad_to_patch(np.asarray(sample[key]), patch_size)
        for key in ("image", "tumor", "rc")
    }
    lesion_mask = (arrays["tumor"][0] > 0) | (arrays["rc"][0] > 0)
    use_lesion = bool(np.any(lesion_mask)) and np.random.random() < lesion_probability

    if use_lesion:
        components, count = ndimage.label(
            lesion_mask,
            structure=ndimage.generate_binary_structure(3, 3),
        )
        component_id = int(np.random.randint(1, count + 1))
        coordinates = np.argwhere(components == component_id)
        center = coordinates[int(np.random.randint(0, len(coordinates)))]
    else:
        brain_mask = np.any(arrays["image"] != 0, axis=0)
        coordinates = np.argwhere(brain_mask)
        if len(coordinates):
            center = coordinates[int(np.random.randint(0, len(coordinates)))]
        else:
            center = np.asarray(arrays["image"].shape[-3:]) // 2

    for key, array in arrays.items():
        sample[key] = _crop_around_center(array, center, patch_size)
    return sample


def build_train_augmentation(config=None):
    """Light geometric / intensity aug; avoid strong elastic warps on small mets."""
    config = config or {}
    if not config.get("enabled", True):
        return EnsureTyped(keys=["image", "tumor", "rc"])

    rotation = float(config.get("rotation_radians", 0.10))
    translation = float(config.get("translation_voxels", 5.0))
    scale = float(config.get("scale_range", 0.08))
    return Compose(
        [
            EnsureTyped(keys=["image", "tumor", "rc"]),
            RandFlipd(
                keys=["image", "tumor", "rc"],
                prob=float(config.get("flip_probability", 0.5)),
                spatial_axis=0,
            ),
            RandAffined(
                keys=["image", "tumor", "rc"],
                prob=float(config.get("affine_probability", 0.2)),
                rotate_range=(rotation, rotation, rotation),
                translate_range=(translation, translation, translation),
                scale_range=(scale, scale, scale),
                mode=("bilinear", "nearest", "nearest"),
                padding_mode="zeros",
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=float(config.get("contrast_probability", 0.15)),
                gamma=tuple(config.get("gamma_range", (0.9, 1.1))),
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=float(config.get("noise_probability", 0.15)),
                mean=0.0,
                std=float(config.get("noise_std", 0.01)),
            ),
            RandBiasFieldd(
                keys=["image"],
                prob=float(config.get("bias_field_probability", 0.1)),
                coeff_range=tuple(config.get("bias_field_coeff_range", (0.0, 0.05))),
            ),
        ]
    )


class BraTSMultiTaskDataset(torch.utils.data.Dataset):
    """Train: lesion-balanced patch crop + light aug.
    Val/test: full volume + identical nonzero Z-score (for SWI evaluation).
    """

    def __init__(
        self,
        case_list,
        data_root,
        patch_size=(96, 96, 96),
        train=True,
        lesion_probability=0.8,
        augmentation=None,
        normalize=True,
    ):
        self.case_list = [str(case) for case in case_list if str(case).strip()]
        self.data_root = Path(data_root)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.train = bool(train)
        self.lesion_probability = float(lesion_probability)
        self.normalize = bool(normalize)
        self.transform = build_train_augmentation(augmentation) if self.train else None

        self.case_dirs = {
            path.name: path
            for path in self.data_root.rglob("BraTS-MET-*")
            if path.is_dir()
        }
        missing = sorted(set(self.case_list) - set(self.case_dirs))
        if missing:
            raise FileNotFoundError(
                f"Cannot find {len(missing)} split cases under {self.data_root}: "
                f"{missing[:10]}"
            )

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        case_dir = self.case_dirs[case]

        modalities = []
        reference_image = None
        for modality in MODALITIES:
            nifti = nib.load(case_dir / f"{case}-{modality}.nii.gz")
            if reference_image is None:
                reference_image = nifti
            modalities.append(nifti.get_fdata(dtype=np.float32))

        image = np.stack(modalities)
        if self.normalize:
            image = nonzero_zscore_normalize(image)

        tumor = nib.load(case_dir / "tumor_label.nii.gz").get_fdata(dtype=np.float32)
        rc = nib.load(case_dir / "rc_label.nii.gz").get_fdata(dtype=np.float32)
        if image.shape[-3:] != tumor.shape or tumor.shape != rc.shape:
            raise ValueError(
                f"Geometry shape mismatch for {case}: image={image.shape[-3:]}, "
                f"tumor={tumor.shape}, rc={rc.shape}"
            )

        sample = {
            "image": image.astype(np.float32, copy=False),
            "tumor": tumor[None].astype(np.int64, copy=False),
            "rc": rc[None].astype(np.int64, copy=False),
        }

        if self.train:
            sample = lesion_balanced_crop(
                sample,
                self.patch_size,
                lesion_probability=self.lesion_probability,
            )
            sample = self.transform(sample)
        else:
            # Full volume for sliding-window validation / inference.
            sample = {key: torch.as_tensor(value) for key, value in sample.items()}

        sample["image"] = sample["image"].float()
        sample["tumor"] = sample["tumor"].long()
        sample["rc"] = sample["rc"].long()
        sample["case"] = case
        sample["spacing"] = torch.tensor(
            reference_image.header.get_zooms()[:3],
            dtype=torch.float32,
        )
        return sample
