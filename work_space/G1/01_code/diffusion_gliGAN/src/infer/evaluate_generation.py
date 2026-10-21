"""
Evaluate generated MRI quality against real scans.

Two evaluation modes:
  - patch (default): per-lesion crop evaluation, compares generated vs real in crop_size^3
  - whole_brain: multi-lesion whole-brain generation with Gaussian blending,
    compares generated vs real in tumour regions (masked metrics).

Usage:
    cd Segmentation_Tasks/GliGAN/src/infer
    # Patch-level evaluation (per-lesion crops)
    python evaluate_generation.py \
        --diffusion_ckpt_dir ../../Checkpoint/brats_2024 \
        --csv_path ../../Checkpoint/brats2024/brats2024.csv \
        --dataset BRATS_2024 \
        --output_dir ./eval_results \
        --crop_size 64 \
        --device cuda
    # Whole-brain evaluation (multi-lesion blending)
    python evaluate_generation.py \
        --diffusion_ckpt_dir ../../Checkpoint/brats_2024 \
        --csv_path ../../Checkpoint/brats2024/brats2024.csv \
        --dataset BRATS_2024 \
        --output_dir ./eval_results \
        --crop_size 64 --evaluation_mode whole_brain \
        --device cuda
"""

import os
import sys
import argparse
import csv
import glob
import hashlib
import json
import random
import re

import numpy as np
import torch
import nibabel as nib
from scipy import ndimage

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusion_inference_utils import (
    make_diffusion_coefficients,
    sample_tumour_diffusion_full,
)
from src.networks.DiffusionNetwork import get_diffusion_network
from src.utils.gaussian_noise_tumour import brain_zscore_normalize

# Import mlutli-lesion generation functions from generate_from_label
from generate_from_label import (
    connected_component_analysis,
    merge_nearby_lesions,
    extract_single_crop,
    make_gaussian_weight_3d,
    label_to_multichannel,
    _tile_generate_lesion,
)

# Import model.py for add_noise_schedule_args
import importlib.util
def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_diffusion_utils_local = _import_from_path(
    "diffusion_utils_local",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model.py"),
)


ALL_MODALITIES = ["t1c", "t1n", "t2w", "t2f"]


def _derive_case_seed(base_seed, case_id, modality):
    payload = f"{int(base_seed)}:{case_id}:{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _set_random_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path):
    match = re.search(r"diffusion_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _find_checkpoint(ckpt_dir, modality, checkpoint_step=None):
    weights_dir = os.path.join(ckpt_dir, modality, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weight directory not found: {weights_dir}")
    if checkpoint_step is not None:
        checkpoint_path = os.path.join(
            weights_dir, f"diffusion_{int(checkpoint_step)}.pt"
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Diffusion checkpoint step {checkpoint_step} not found: "
                f"{checkpoint_path}"
            )
        return checkpoint_path
    candidates = glob.glob(os.path.join(weights_dir, "diffusion_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No diffusion checkpoint found in: {weights_dir}")
    return max(candidates, key=_checkpoint_step)


# ===========================================================================
# Metric functions (numpy, 3D)
# ===========================================================================

def compute_mse(a, b):
    return np.mean((a - b) ** 2)


def compute_mae(a, b):
    return np.mean(np.abs(a - b))


def compute_psnr(a, b, max_val=2.0):
    """PSNR for [-1,1] range: max_val = 2.0."""
    mse = compute_mse(a, b)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim_3d(a, b, max_val=2.0, win_size=7):
    """
    3D SSIM: compute per axial slice, return mean.
    For volumes smaller than win_size in any dimension, fall back to global SSIM.
    """
    k1, k2 = 0.01, 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Gaussian window
    x = np.arange(win_size) - win_size // 2
    gauss = np.exp(-(x ** 2) / (2.0 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    gauss_2d = np.outer(gauss, gauss)

    # If volume is large enough in z, do slice-wise; otherwise global
    if a.shape[0] >= win_size:
        ssim_vals = []
        for z in range(a.shape[0]):
            val = _ssim_2d(a[z], b[z], max_val, c1, c2, gauss_2d, win_size)
            if not np.isnan(val):
                ssim_vals.append(val)
        return np.mean(ssim_vals) if ssim_vals else 0.0
    else:
        return _ssim_3d_global(a, b, max_val, c1, c2)


def _ssim_2d(slice_a, slice_b, max_val, c1, c2, gauss_2d, win_size):
    """Single-slice SSIM with Gaussian window."""
    mu_a = _conv2d_valid(slice_a, gauss_2d, win_size)
    mu_b = _conv2d_valid(slice_b, gauss_2d, win_size)
    if mu_a is None:
        return float("nan")

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = _conv2d_valid(slice_a ** 2, gauss_2d, win_size) - mu_a_sq
    sigma_b_sq = _conv2d_valid(slice_b ** 2, gauss_2d, win_size) - mu_b_sq
    sigma_ab = _conv2d_valid(slice_a * slice_b, gauss_2d, win_size) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return float(np.mean(ssim_map))


def _conv2d_valid(img, kernel, win_size):
    """2D valid convolution via scipy (optimized C implementation)."""
    from scipy.signal import convolve2d
    h, w = img.shape
    kh, kw = kernel.shape
    if h < kh or w < kw:
        return None
    return convolve2d(img, kernel, mode='valid')


def _ssim_3d_global(vol_a, vol_b, max_val, c1, c2):
    """Global SSIM for volumes too small for sliding window."""
    mu_a = np.mean(vol_a)
    mu_b = np.mean(vol_b)
    sigma_a_sq = np.var(vol_a)
    sigma_b_sq = np.var(vol_b)
    sigma_ab = np.mean((vol_a - mu_a) * (vol_b - mu_b))
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(numerator / (denominator + 1e-8))


def compute_per_lesion_ssim(real_whole, gen_whole, cc_list, max_val=2.0):
    """Per-lesion weighted SSIM: compute SSIM on each lesion's bbox window,
    then average weighted by lesion voxel count.

    For lesions smaller than the SSIM sliding window, the global SSIM fallback
    (already built into compute_ssim_3d) handles them transparently.

    Falls back to full-volume SSIM if cc_list is empty or all lesions fail.
    """
    if not cc_list:
        return compute_ssim_3d(real_whole, gen_whole, max_val)

    ssim_vals = []
    weights = []
    for cc in cc_list:
        x_min, x_max, y_min, y_max, z_min, z_max = cc['bbox']
        x_min = max(0, x_min); x_max = min(real_whole.shape[2], x_max)
        y_min = max(0, y_min); y_max = min(real_whole.shape[1], y_max)
        z_min = max(0, z_min); z_max = min(real_whole.shape[0], z_max)

        if x_max <= x_min or y_max <= y_min or z_max <= z_min:
            continue

        real_crop = real_whole[z_min:z_max, y_min:y_max, x_min:x_max]
        gen_crop = gen_whole[z_min:z_max, y_min:y_max, x_min:x_max]
        n_vox = cc['n_voxels']
        if n_vox < 1:
            continue

        ssim_val = compute_ssim_3d(real_crop, gen_crop, max_val)
        if not np.isnan(ssim_val):
            ssim_vals.append(ssim_val)
            weights.append(float(n_vox))

    if not ssim_vals:
        return compute_ssim_3d(real_whole, gen_whole, max_val)

    return float(np.average(ssim_vals, weights=weights))


def _extract_reference_content(scan_whole, coords, content_shape, crop_size):
    """Build a training-aligned z-score reference for one generated crop."""
    z0, z1, y0, y1, x0, x1 = coords
    window = np.asarray(scan_whole[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    window_shape = tuple(int(value) for value in window.shape)
    content_shape = tuple(int(value) for value in content_shape)
    if window_shape != content_shape:
        factors = tuple(
            target / source for target, source in zip(content_shape, window_shape)
        )
        window = ndimage.zoom(window, factors, order=1)

    padding = []
    for dimension in content_shape:
        total = crop_size - dimension
        if total < 0:
            raise ValueError(
                f"Reference content dimension {dimension} exceeds crop_size={crop_size}"
            )
        before = total // 2
        padding.append((before, total - before))
    padded = np.pad(window, padding, mode="constant", constant_values=0)
    normalized = brain_zscore_normalize(padded)
    slices = tuple(
        slice(before, before + dimension)
        for (before, _), dimension in zip(padding, content_shape)
    )
    content = normalized[slices]
    if window_shape != content_shape:
        factors = tuple(
            target / source for target, source in zip(window_shape, content_shape)
        )
        content = ndimage.zoom(content, factors, order=1)
    if content.shape != window_shape:
        raise ValueError(
            f"Reference content shape mismatch: actual={content.shape} expected={window_shape}"
        )
    return np.asarray(content, dtype=np.float32)


def _tile_reference_content(scan_whole, coords, crop_size):
    """Blend per-tile z-score references using the generation tile geometry."""
    z0, z1, y0, y1, x0, x1 = coords
    window_shape = (z1 - z0, y1 - y0, x1 - x0)
    accum = np.zeros(window_shape, dtype=np.float32)
    weight = np.zeros(window_shape, dtype=np.float32)
    stride = max(crop_size // 2, 1)
    gaussian = make_gaussian_weight_3d(
        (crop_size, crop_size, crop_size), sigma=crop_size / 3.0
    )

    starts = [list(range(0, dimension, stride)) for dimension in window_shape]
    for local_z in starts[0]:
        for local_y in starts[1]:
            for local_x in starts[2]:
                end_z = min(local_z + crop_size, window_shape[0])
                end_y = min(local_y + crop_size, window_shape[1])
                end_x = min(local_x + crop_size, window_shape[2])
                tile_shape = (
                    end_z - local_z,
                    end_y - local_y,
                    end_x - local_x,
                )
                global_slices = (
                    slice(z0 + local_z, z0 + end_z),
                    slice(y0 + local_y, y0 + end_y),
                    slice(x0 + local_x, x0 + end_x),
                )
                tile = np.asarray(scan_whole[global_slices], dtype=np.float32)
                padding = []
                for dimension in tile_shape:
                    total = crop_size - dimension
                    before = total // 2
                    padding.append((before, total - before))
                padded = np.pad(tile, padding, mode="constant", constant_values=0)
                normalized = brain_zscore_normalize(padded)
                valid_slices = tuple(
                    slice(before, before + dimension)
                    for (before, _), dimension in zip(padding, tile_shape)
                )
                normalized_valid = normalized[valid_slices]
                gaussian_valid = gaussian[valid_slices]
                local_slices = (
                    slice(local_z, end_z),
                    slice(local_y, end_y),
                    slice(local_x, end_x),
                )
                accum[local_slices] += normalized_valid * gaussian_valid
                weight[local_slices] += gaussian_valid

    reference = np.zeros_like(accum)
    valid = weight > 1e-8
    reference[valid] = accum[valid] / weight[valid]
    return reference, weight


def _compute_masked_ssim_3d(reference, generated, mask, max_val):
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("SSIM support mask is empty")
    lower = np.maximum(coordinates.min(axis=0) - 4, 0)
    upper = np.minimum(coordinates.max(axis=0) + 5, np.asarray(mask.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    local_mask = mask[slices].astype(np.float32)
    return compute_ssim_3d(
        reference[slices] * local_mask,
        generated[slices] * local_mask,
        max_val=max_val,
    )


def _save_array_like(array, reference_image, output_path, dtype):
    values = np.asarray(array, dtype=dtype)
    if values.shape != reference_image.shape:
        raise ValueError(
            f"Output shape {values.shape} does not match reference {reference_image.shape}"
        )
    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = reference_image.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(values, reference_image.affine, header=header),
        output_path,
    )


# ===========================================================================
# Preprocessing: replicate training crop+pad+normalize
# ===========================================================================

def preprocess_scan(scan_path, label_path, csv_row, dataset_type, crop_size=64,
                    normalization="minmax"):
    """Apply the SAME preprocessing as GaussianNoiseTumour training transform.

    1. Load scan and label
    2. Crop to tumour bbox (with margin to target crop_size)
    3. If crop exceeds crop_size, zoom down proportionally (matching training)
    4. Pad to exactly crop_size^3
    5. Normalize scan to [-1, 1]

    Returns:
        scan_crop_pad: (1, crop_size, crop_size, crop_size) float32
        label_crop_pad: (C, crop_size, crop_size, crop_size) float32
    """
    scan = nib.load(scan_path).get_fdata().astype(np.float32)
    label_data = nib.load(label_path).get_fdata().astype(np.int16)

    scan = scan[np.newaxis, ...]
    label_data = label_data[np.newaxis, ...]

    _, max_x, max_y, max_z = scan.shape

    x_min = int(csv_row["x_extreme_min"])
    x_max = int(csv_row["x_extreme_max"])
    y_min = int(csv_row["y_extreme_min"])
    y_max = int(csv_row["y_extreme_max"])
    z_min = int(csv_row["z_extreme_min"])
    z_max = int(csv_row["z_extreme_max"])

    x_ext = x_max - x_min
    y_ext = y_max - y_min
    z_ext = z_max - z_min

    x_pad = (crop_size - x_ext) / 2
    y_pad = (crop_size - y_ext) / 2
    z_pad = (crop_size - z_ext) / 2

    C_x = -0.5 if x_pad < 0 else 0.5
    C_y = -0.5 if y_pad < 0 else 0.5
    C_z = -0.5 if z_pad < 0 else 0.5

    x_base = int(x_min - int(x_pad))
    x_top = int(x_max + int(x_pad + C_x))
    y_base = int(y_min - int(y_pad))
    y_top = int(y_max + int(y_pad + C_y))
    z_base = int(z_min - int(z_pad))
    z_top = int(z_max + int(z_pad + C_z))

    x_base_pad = 0 if x_base >= 0 else -x_base
    y_base_pad = 0 if y_base >= 0 else -y_base
    z_base_pad = 0 if z_base >= 0 else -z_base
    x_top_pad = 0 if x_top <= max_x else x_top - max_x
    y_top_pad = 0 if y_top <= max_y else y_top - max_y
    z_top_pad = 0 if z_top <= max_z else z_top - max_z

    x_base = max(0, x_base)
    y_base = max(0, y_base)
    z_base = max(0, z_base)
    x_top = min(max_x, x_top)
    y_top = min(max_y, y_top)
    z_top = min(max_z, z_top)

    scan_crop = scan[:, x_base:x_top, y_base:y_top, z_base:z_top]
    label_crop = label_data[:, x_base:x_top, y_base:y_top, z_base:z_top]

    # Resize if crop exceeds crop_size (matching training _resize_crop_if_needed)
    max_dim = max(scan_crop.shape[1:])
    if max_dim > crop_size:
        scale = crop_size / max_dim
        from scipy.ndimage import zoom as ndimage_zoom
        new_shape = np.maximum(np.round(np.array(scan_crop.shape[1:]) * scale), 1).astype(int)
        zoomed_scan = np.zeros((1,) + tuple(new_shape), dtype=np.float32)
        factors = tuple(new_shape.astype(float) / np.array(scan_crop.shape[1:]))
        zoomed_scan[0] = ndimage_zoom(scan_crop[0].astype(np.float32), factors, order=1)
        scan_crop = zoomed_scan
        new_shape_l = np.maximum(np.round(np.array(label_crop.shape[1:]) * scale), 1).astype(int)
        zoomed_label = np.zeros((1,) + tuple(new_shape_l), dtype=np.float32)
        factors_l = tuple(new_shape_l.astype(float) / np.array(label_crop.shape[1:]))
        zoomed_label[0] = ndimage_zoom(label_crop[0].astype(np.float32), factors_l, order=1)
        label_crop = zoomed_label
        # Recompute padding for resized crop
        x_top_pad = crop_size - zoomed_scan.shape[1] - x_base_pad
        y_top_pad = crop_size - zoomed_scan.shape[2] - y_base_pad
        z_top_pad = crop_size - zoomed_scan.shape[3] - z_base_pad

    # Normalize scan
    if normalization == "zscore":
        # Pure z-score (matching S2 nnUNet preprocessing). No [-1,1] rescaling.
        scan_crop = brain_zscore_normalize(scan_crop)
        pad_val = 0
    else:
        mina, maxa = np.min(scan_crop), np.max(scan_crop)
        if maxa > mina:
            scan_crop = (scan_crop - mina) / (maxa - mina) * 2.0 - 1.0
        pad_val = -1

    # Pad to crop_size^3
    scan_crop_pad = np.pad(
        scan_crop,
        pad_width=((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad),
                    (z_base_pad, z_top_pad)),
        mode="constant", constant_values=(pad_val, pad_val),
    )
    label_crop_pad = np.pad(
        label_crop,
        pad_width=((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad),
                    (z_base_pad, z_top_pad)),
        mode="constant", constant_values=(0, 0),
    )

    label_crop_pad = _label_to_multichannel(label_crop_pad, dataset_type)

    return scan_crop_pad.astype(np.float32), label_crop_pad.astype(np.float32)


def _label_to_multichannel(label, dataset_type):
    """Convert single-channel integer label to multi-channel. Mirrors training."""
    label_1ch = label[0].astype(np.int16)
    if dataset_type == "BRATS_2024":
        n_channels = 4
        mc = np.zeros((n_channels,) + label_1ch.shape, dtype=np.float32)
        mc[0] = ((label_1ch == 1) | (label_1ch == 3)).astype(np.float32)  # TC
        mc[1] = ((label_1ch == 1) | (label_1ch == 2) | (label_1ch == 3)).astype(np.float32)  # WT
        mc[2] = (label_1ch == 3).astype(np.float32)  # ET
        mc[3] = (label_1ch == 4).astype(np.float32)  # RC
    else:
        n_channels = 3
        mc = np.zeros((n_channels,) + label_1ch.shape, dtype=np.float32)
        mc[0] = ((label_1ch == 1) | (label_1ch == 3)).astype(np.float32)  # TC
        mc[1] = ((label_1ch == 1) | (label_1ch == 2) | (label_1ch == 3)).astype(np.float32)  # WT
        mc[2] = (label_1ch == 3).astype(np.float32)  # ET
    return mc


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated MRI quality (SSIM/PSNR/MSE/MAE) vs real scans"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with scan paths and bbox info")
    parser.add_argument("--diffusion_ckpt_dir", type=str, default="",
                        help="Root dir: {dir}/{modality}/weights/diffusion_*.pt "
                             "(not needed for --self_test)")
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Load exactly diffusion_<step>.pt for every selected modality. "
             "If omitted, the highest available step is used.",
    )
    parser.add_argument("--dataset", type=str, default="BRATS_2024",
                        choices=["BRATS_2023", "BRATS_2024", "BRATS_GOAT_2024"])
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Where to save metrics JSON and per-case details")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampling_steps", type=int, default=18,
                        help="DDIM sampling steps (0 = full n_steps)")
    parser.add_argument("--sampling_method", type=str, default="edm_heun",
                        choices=["ddpm", "ddim", "edm_heun", "lognsr_ode"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--generator_type", type=str, default="Unet_NnU")
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore"])
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "t1c", "t1n", "t2w", "t2f"])
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Limit number of test cases (0=all)")
    parser.add_argument("--seed", type=int, default=20260720,
                        help="Base seed; each patient/modality gets a stable derived seed")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="CSV split to evaluate (default: val)")
    parser.add_argument("--self_test", action="store_true",
                        help="Self-comparison mode: compare real scan with itself "
                             "(no diffusion model needed, verifies preprocessing + metrics)")
    parser.add_argument("--crop_size", default=64, type=int,
                        help="Crop/pad target size (must match training)")
    parser.add_argument("--use_compile", action="store_true",
                        help="Enable torch.compile for the model (PyTorch >= 2.0)")
    parser.add_argument("--large_lesion_mode", default="resize", type=str,
                        choices=["resize", "skip", "tile"],
                        help="How to handle lesions > crop_size (whole_brain mode only): "
                             "resize / skip / tile")
    parser.add_argument("--evaluation_mode", default="patch", type=str,
                        choices=["patch", "whole_brain"],
                        help="patch: per-lesion crop eval; whole_brain: multi-lesion full-brain eval")
    parser.add_argument(
        "--save_support_volumes",
        action="store_true",
        help="Save generated/reference z-score support NIfTI files for G2 QC.",
    )
    _diffusion_utils_local.add_noise_schedule_args(parser)
    parser.add_argument("--cfg_weight", default=1.0, type=float,
                        help="CFG weight: 1.0=normal, >1=stronger conditioning (2.0-3.0 typical)")
    args = parser.parse_args()
    if args.save_support_volumes and args.evaluation_mode != "whole_brain":
        parser.error("--save_support_volumes requires --evaluation_mode whole_brain")

    os.makedirs(args.output_dir, exist_ok=True)
    _set_random_seed(args.seed)

    if not args.self_test and not args.diffusion_ckpt_dir:
        parser.error("--diffusion_ckpt_dir is required (unless --self_test)")

    # ---- Read checkpoint metadata for model architecture ----
    ckpt_metadata = None
    if not args.self_test:
        for mod in (args.modality.split(",") if args.modality != "all"
                    else ["t1c", "t1n", "t2w", "t2f"]):
            try:
                ckpt_path = _find_checkpoint(
                    args.diffusion_ckpt_dir,
                    mod.strip(),
                    checkpoint_step=args.checkpoint_step,
                )
            except FileNotFoundError:
                continue
            ckpt_metadata = torch.load(ckpt_path, map_location="cpu")
            print(f"[Metadata] architecture params from {ckpt_path}")
            break

    # Proxy args for get_diffusion_network
    class _Args:
        pass
    args_proxy = _Args()
    args_proxy.generator_type = (
        ckpt_metadata.get("generator_type", args.generator_type)
        if ckpt_metadata is not None else args.generator_type)
    args_proxy.feature_size = (
        ckpt_metadata.get("feature_size", args.feature_size)
        if ckpt_metadata is not None else args.feature_size)
    args_proxy.use_checkpoint = False
    args_proxy.out_channels = (
        ckpt_metadata.get("out_channels", 1) if ckpt_metadata is not None else 1)
    args_proxy.crop_size = (
        ckpt_metadata.get("crop_size", args.crop_size)
        if ckpt_metadata is not None else args.crop_size)
    args_proxy.network_channels = (
        ckpt_metadata.get("network_channels") if ckpt_metadata is not None else None)
    args_proxy.network_strides = (
        ckpt_metadata.get("network_strides") if ckpt_metadata is not None else None)
    if args_proxy.crop_size != args.crop_size:
        raise ValueError(
            f"--crop_size={args.crop_size} does not match checkpoint crop_size={args_proxy.crop_size}")
    if (ckpt_metadata is not None and ckpt_metadata.get("normalization")
            and ckpt_metadata["normalization"] != args.normalization):
        raise ValueError(
            f"--normalization={args.normalization} does not match checkpoint "
            f"normalization={ckpt_metadata['normalization']}")

    # Architecture params: read from checkpoint, fall back to CLI / guessing
    if ckpt_metadata is not None:
        args_proxy.noise_embedding_mode = ckpt_metadata.get("noise_embedding_mode", "continuous")
        args_proxy.time_ch_count = ckpt_metadata.get("time_ch_count", 8)
        args_proxy.n_steps = ckpt_metadata.get("n_steps", args.n_steps)
        args_proxy.p_uncond = ckpt_metadata.get("p_uncond", 0.0)
        # Infer data in_channels from the first conv layer's input channel dim.
        # get_diffusion_network internally adds time_ch_count on top, so
        # args_proxy.in_channels must be data channels only (scan + label),
        # NOT include time_ch_count.
        ckpt_in_channels = ckpt_metadata.get("in_channels")
        sd = ckpt_metadata.get("state_dict", ckpt_metadata)
        ckpt_in = None
        if ckpt_in_channels is None:
            for k, v in sd.items():
                if hasattr(v, "shape") and len(v.shape) >= 2 and any(
                    p in k for p in ("model.0.conv.weight", "encoder1", "encoder_stages.0",
                                     "conv1", "input_layer")):
                    ckpt_in = v.shape[1]
                    break
        if ckpt_in_channels is not None:
            args_proxy.in_channels = int(ckpt_in_channels)
            print(f"[Metadata] in_channels={args_proxy.in_channels} (checkpoint metadata)")
        elif ckpt_in is not None:
            # ckpt_in = backbone_in_ch = data_in_channels + time_ch_count
            args_proxy.in_channels = ckpt_in - args_proxy.time_ch_count
            print(f"[Metadata] in_channels={args_proxy.in_channels} (from checkpoint, backbone_in={ckpt_in})")
        else:
            label_channels = 4 if args.dataset == "BRATS_2024" else 3
            args_proxy.in_channels = 1 + label_channels
            print(f"[Metadata] in_channels={args_proxy.in_channels} (guessed from dataset)")
    else:
        args_proxy.time_ch_count = 8
        args_proxy.n_steps = args.n_steps
        label_channels = 4 if args.dataset == "BRATS_2024" else 3
        args_proxy.in_channels = 1 + label_channels
        if args.sampling_method in ("edm_heun", "lognsr_ode"):
            args_proxy.noise_embedding_mode = "continuous"
        else:
            args_proxy.noise_embedding_mode = "discrete"
        print(f"[Metadata] no checkpoint found, in_channels={args_proxy.in_channels} (guessed)")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load CSV
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError(f"CSV has no split column: {args.csv_path}")
        df = df[df["split"] == args.split].copy()
    if args.max_cases > 0:
        df = df.head(args.max_cases)
    if len(df) == 0:
        raise ValueError(
            f"No CSV rows to evaluate after split={args.split}, max_cases={args.max_cases}")
    print(f"[CSV] {len(df)} case(s), split={args.split}, from {args.csv_path}")

    # Filter modalities
    if args.modality == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [args.modality]

    ckpt_paths = {}
    if not args.self_test:
        for mod in modalities:
            try:
                ckpt_paths[mod] = _find_checkpoint(
                    args.diffusion_ckpt_dir,
                    mod,
                    checkpoint_step=args.checkpoint_step,
                )
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
        if not ckpt_paths:
            raise RuntimeError(
                f"No diffusion checkpoints found under {args.diffusion_ckpt_dir}")

    # Accumulate results
    all_results = {}
    generation_manifest_rows = []
    per_modality_sums = {m: {"mse": [], "mae": [], "psnr": [],
                               "ssim": [],           # patch mode SSIM
                               "ssim_whole": [],     # whole_brain: full-volume SSIM (dominated by background)
                               "ssim_support": [],   # whole_brain: generated-support bbox SSIM
                               "ssim_lesion": [],}   # whole_brain: per-lesion weighted SSIM (tumour-region quality)
                         for m in modalities}

    if not args.self_test:
        # Backward compat: --beta_schedule overrides default --noise_schedule
        if hasattr(args, "beta_schedule") and args.beta_schedule != "cosine":
            args.noise_schedule = args.beta_schedule

        schedule_meta = ckpt_metadata or {}
        schedule_override = schedule_meta.get("schedule_config") or {}
        n_steps_actual = schedule_meta.get("n_steps", args.n_steps)
        noise_schedule = schedule_meta.get("noise_schedule", args.noise_schedule)
        if hasattr(args, "beta_schedule") and args.beta_schedule != "cosine":
            noise_schedule = args.beta_schedule

        schedule_cfg = make_diffusion_coefficients(
            n_steps=n_steps_actual, device=device,
            noise_schedule=noise_schedule,
            sigma_data=schedule_override.get("sigma_data", args.sigma_data),
            sigma_max=schedule_override.get("sigma_max", args.sigma_max),
            sigma_min=schedule_override.get("sigma_min", args.sigma_min),
            rho=schedule_override.get("rho", args.rho),
            gamma_max=schedule_override.get("gamma_max", args.gamma_max),
            gamma_min=schedule_override.get("gamma_min", args.gamma_min),
            snr_shift=schedule_override.get("snr_shift", args.snr_shift),
        )
        print(f"[Schedule] {noise_schedule}, n_steps={n_steps_actual}, "
              f"sampling_method={args.sampling_method}")
        # Legacy unpacking
        betas = schedule_cfg.betas
        alphas_bar_sqrt = schedule_cfg.alphas_bar_sqrt
        one_minus_alphas_bar_sqrt = schedule_cfg.one_minus_alphas_bar_sqrt
        alphas_bar = schedule_cfg.alphas_bar

    gli_gan_root= os.path.join(os.path.dirname(__file__), "..", "..")
    spatial_size = (args.crop_size, args.crop_size, args.crop_size)
    n_steps_for_sampling = n_steps_actual if not args.self_test else args.n_steps
    sampling_steps = args.sampling_steps if args.sampling_steps > 0 else n_steps_for_sampling

    # ---------- PATCH mode: per-lesion crop evaluation ----------
    if args.evaluation_mode == "patch":
        for mod in modalities:
            print(f"\n{'='*60}")
            print(f"Evaluating modality: {mod}  [patch mode, crop_size={args.crop_size}]")

            if args.self_test:
                model = None
            else:
                ckpt_path = ckpt_paths.get(mod)
                if ckpt_path is None:
                    print(f"  [SKIP] No checkpoint for {mod}")
                    continue
                model = get_diffusion_network(args_proxy)
                ckpt = torch.load(ckpt_path, map_location=torch.device(device))
                state_dict = ckpt.get("state_dict", ckpt)
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                if args.use_compile:
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        print(f"  [{mod}] torch.compile enabled")
                    else:
                        print(f"  [{mod}] Warning: torch.compile not available")
                print(f"  Model: {ckpt_path}")

            case_results = {}
            for idx, row in df.iterrows():
                case_id = str(row.get("lesion_id", f"case_{idx}"))
                scan_path = row[f"scan_{mod}"]
                label_path = row["label"]

                scan_full = os.path.join(gli_gan_root, scan_path)
                label_full = os.path.join(gli_gan_root, label_path)
                if not os.path.isfile(scan_full):
                    print(f"  [WARN] Missing scan: {scan_full}")
                    continue
                if not os.path.isfile(label_full):
                    print(f"  [WARN] Missing label: {label_full}")
                    continue

                scan_real, label_mc = preprocess_scan(
                    scan_full, label_full, row, args.dataset,
                    crop_size=args.crop_size, normalization=args.normalization)
                real_np = scan_real.squeeze(0)

                if args.self_test:
                    gen_np = real_np
                else:
                    case_seed = _derive_case_seed(args.seed, case_id, mod)
                    _set_random_seed(case_seed)
                    label_tensor = torch.from_numpy(label_mc).float().unsqueeze(0).to(device)
                    generated = sample_tumour_diffusion_full(
                        model=model,
                        label_cond=label_tensor,
                        spatial_size=spatial_size,
                        n_steps=n_steps_for_sampling,
                        betas=betas,
                        alphas_bar_sqrt=alphas_bar_sqrt,
                        one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                        device=device,
                        method=args.sampling_method,
                        sampling_steps=sampling_steps,
                        eta=args.eta,
                        alphas_bar=alphas_bar,
                        noise_schedule_cfg=schedule_cfg if args.sampling_method in ("edm_heun", "lognsr_ode") else None,
                        cfg_weight=args.cfg_weight,
                    )
                    gen_np = generated.squeeze(0).squeeze(0).cpu().numpy()

                data_range = 6.0 if args.normalization == "zscore" else 2.0
                mse = compute_mse(real_np, gen_np)
                mae = compute_mae(real_np, gen_np)
                psnr = compute_psnr(real_np, gen_np, max_val=data_range)
                ssim = compute_ssim_3d(real_np, gen_np, max_val=data_range)

                per_modality_sums[mod]["mse"].append(mse)
                per_modality_sums[mod]["mae"].append(mae)
                per_modality_sums[mod]["psnr"].append(psnr if not np.isinf(psnr) else 100.0)
                per_modality_sums[mod]["ssim"].append(ssim)

                case_results[case_id] = {"mse": round(mse, 6), "mae": round(mae, 6),
                                          "psnr": round(psnr, 3), "ssim": round(ssim, 4)}
                print(f"  [{case_id}] MSE={mse:.4f} MAE={mae:.4f} PSNR={psnr:.2f}dB SSIM={ssim:.4f}")

            all_results[mod] = case_results

    # ---------- WHOLE_BRAIN mode: multi-lesion blending evaluation ----------
    else:
        print(f"\n[Mode] whole_brain — grouping by patient_id, "
              f"crop_size={args.crop_size}, large_lesion={args.large_lesion_mode}")

        # Group CSV rows by patient
        patient_groups = df.groupby("patient_id")

        # Load models for all modalities
        models = {}
        for mod in modalities:
            if args.self_test:
                models[mod] = None
                continue
            ckpt_path = ckpt_paths.get(mod)
            if ckpt_path is None:
                print(f"  [SKIP] No checkpoint for {mod}")
                continue
            model = get_diffusion_network(args_proxy)
            ckpt = torch.load(ckpt_path, map_location=torch.device(device))
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            if args.use_compile:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model)
                    print(f"  [{mod}] torch.compile enabled")
                else:
                    print(f"  [{mod}] Warning: torch.compile not available")
            models[mod] = model
            print(f"  [{mod}] loaded: {ckpt_path}")

        if not models:
            raise RuntimeError("No diffusion models loaded. Check --diffusion_ckpt_dir")
        primary_modality = next(iter(models))

        sample_kwargs_full = {}
        if not args.self_test:
            sample_kwargs_full = dict(
                n_steps=n_steps_for_sampling, betas=betas, alphas_bar_sqrt=alphas_bar_sqrt,
                one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, device=device,
                method=args.sampling_method, sampling_steps=sampling_steps,
                eta=args.eta, alphas_bar=alphas_bar,
                noise_schedule_cfg=schedule_cfg if args.sampling_method in ("edm_heun", "lognsr_ode") else None,
                cfg_weight=args.cfg_weight,
            )

        # Evaluate each patient
        for patient_id, patient_rows in patient_groups:
            print(f"\n  Patient: {patient_id} ({len(patient_rows)} lesion(s) in CSV)")

            label_path = patient_rows.iloc[0]["label"]
            label_full = os.path.join(gli_gan_root, label_path)
            if not os.path.isfile(label_full):
                print(f"    [WARN] Missing label: {label_full}")
                continue

            label_image = nib.load(label_full)
            label_data = label_image.get_fdata().astype(np.int16)
            original_shape = label_data.shape
            mask_binary = (label_data != 0)
            if not np.any(mask_binary):
                print(f"    [WARN] No tumour in label, skipping")
                continue

            label_mc_full, n_label_ch = label_to_multichannel(label_data, args.dataset)

            # CC analysis + merge (same pipeline as generate_from_label.py)
            cc_list = connected_component_analysis(mask_binary)
            cc_list = merge_nearby_lesions(cc_list, merge_dist=16, crop_size=args.crop_size)
            print(f"    {len(cc_list)} lesion(s) after merging")

            gauss_weight = make_gaussian_weight_3d(
                (args.crop_size, args.crop_size, args.crop_size),
                sigma=args.crop_size / 3.0)

            # Generate + blend for EACH modality
            for mod in modalities:
                if mod not in models:
                    continue
                model = models[mod]
                case_seed = _derive_case_seed(args.seed, patient_id, mod)
                _set_random_seed(case_seed)

                scan_path = patient_rows.iloc[0][f"scan_{mod}"]
                scan_full = os.path.join(gli_gan_root, scan_path)
                if not os.path.isfile(scan_full):
                    print(f"    [WARN] Missing scan for {mod}: {scan_full}")
                    continue
                scan_image = nib.load(scan_full)
                scan_whole = scan_image.get_fdata().astype(np.float32)
                if scan_whole.shape != original_shape:
                    raise ValueError(
                        f"{patient_id}/{mod}: scan shape {scan_whole.shape} "
                        f"does not match label shape {original_shape}"
                    )

                accum = np.zeros(original_shape, dtype=np.float32)
                weight_blend = np.zeros(original_shape, dtype=np.float32)
                reference_accum = np.zeros(original_shape, dtype=np.float32)
                reference_weight = np.zeros(original_shape, dtype=np.float32)

                for lesion_idx, cc in enumerate(cc_list):
                    label_cube, coords, content_shape = extract_single_crop(
                        label_mc_full, cc['bbox'], original_shape, crop_size=args.crop_size)
                    z0, z1, y0, y1, x0, x1 = coords
                    window_shape = (z1 - z0, y1 - y0, x1 - x0)
                    was_resized = (content_shape != window_shape)
                    is_oversized = any(d > args.crop_size for d in window_shape)

                    # Skip oversized lesions in skip mode
                    if args.large_lesion_mode == "skip" and is_oversized:
                        continue

                    if args.large_lesion_mode == "tile" and is_oversized and not args.self_test:
                        # TILE mode
                        gen_content, gw_content = _tile_generate_lesion(
                            label_mc=label_mc_full, coords=coords,
                            crop_size=args.crop_size, model=model,
                            spatial_size=spatial_size, device=device,
                            sample_kwargs=sample_kwargs_full,
                        )
                        reference_content, reference_tile_weight = _tile_reference_content(
                            scan_whole, coords, args.crop_size
                        )
                        if not np.allclose(
                            gw_content, reference_tile_weight, atol=1e-6, rtol=0.0
                        ):
                            raise ValueError(
                                f"{patient_id}/{mod}: generated/reference tile weights differ"
                            )
                        accum[z0:z1, y0:y1, x0:x1] += gen_content * gw_content
                        weight_blend[z0:z1, y0:y1, x0:x1] += gw_content
                        reference_accum[z0:z1, y0:y1, x0:x1] += (
                            reference_content * reference_tile_weight
                        )
                        reference_weight[z0:z1, y0:y1, x0:x1] += reference_tile_weight
                    else:
                        # RESIZE or small lesion
                        if args.self_test:
                            gen_cube = label_cube[:1] if label_cube.shape[0] > 0 else label_cube
                            gen_cube = np.broadcast_to(gen_cube, (1,) + spatial_size).copy()
                        else:
                            label_tensor = torch.from_numpy(label_cube).float().unsqueeze(0).to(device)
                            generated = sample_tumour_diffusion_full(
                                model=model, label_cond=label_tensor,
                                spatial_size=spatial_size, **sample_kwargs_full)
                            gen_cube = generated.squeeze(0).cpu().numpy()

                        gen_np = gen_cube.squeeze(0)

                        # Extract valid content
                        cz, cy, cx = content_shape
                        z_pb = (args.crop_size - cz) // 2
                        y_pb = (args.crop_size - cy) // 2
                        x_pb = (args.crop_size - cx) // 2

                        gen_valid = gen_np[z_pb:z_pb + cz, y_pb:y_pb + cy, x_pb:x_pb + cx]
                        gw_valid = gauss_weight[z_pb:z_pb + cz, y_pb:y_pb + cy, x_pb:x_pb + cx]
                        reference_valid = _extract_reference_content(
                            scan_whole, coords, content_shape, args.crop_size
                        )

                        if was_resized:
                            wz, wy, wx = window_shape
                            from scipy.ndimage import zoom as ndimage_zoom
                            gen_valid = ndimage_zoom(gen_valid,
                                                      (wz / cz, wy / cy, wx / cx), order=1)
                            gw_valid = ndimage_zoom(gw_valid,
                                                     (wz / cz, wy / cy, wx / cx), order=1)

                        accum[z0:z1, y0:y1, x0:x1] += gen_valid * gw_valid
                        weight_blend[z0:z1, y0:y1, x0:x1] += gw_valid
                        reference_accum[z0:z1, y0:y1, x0:x1] += (
                            reference_valid * gw_valid
                        )
                        reference_weight[z0:z1, y0:y1, x0:x1] += gw_valid

                # Normalize blended whole-brain result
                gen_whole = np.zeros_like(accum)
                valid_mask = weight_blend > 1e-8
                gen_whole[valid_mask] = accum[valid_mask] / weight_blend[valid_mask]

                reference_mask = reference_weight > 1e-8
                if not np.array_equal(valid_mask, reference_mask):
                    raise ValueError(
                        f"{patient_id}/{mod}: generated/reference support masks differ"
                    )
                real_whole = np.zeros_like(reference_accum)
                real_whole[reference_mask] = (
                    reference_accum[reference_mask] / reference_weight[reference_mask]
                )
                if not np.isfinite(gen_whole[valid_mask]).all():
                    raise ValueError(f"{patient_id}/{mod}: non-finite generated support")
                if not np.isfinite(real_whole[reference_mask]).all():
                    raise ValueError(f"{patient_id}/{mod}: non-finite reference support")

                # Masked metrics (tumour region only)
                tumour_mask = mask_binary
                tumour_outside_support = int(np.count_nonzero(tumour_mask & ~valid_mask))
                if tumour_outside_support:
                    raise ValueError(
                        f"{patient_id}/{mod}: {tumour_outside_support} tumour voxels "
                        "fall outside generation support"
                    )
                real_tumour = real_whole[tumour_mask]
                gen_tumour = gen_whole[tumour_mask]

                if len(real_tumour) == 0:
                    print(f"    [{mod}] No tumour voxels, skipping")
                    continue

                data_range = 6.0 if args.normalization == "zscore" else 2.0
                mse = compute_mse(real_tumour, gen_tumour)
                mae = compute_mae(real_tumour, gen_tumour)
                psnr = compute_psnr(real_tumour, gen_tumour, max_val=data_range)
                # Full-volume SSIM remains background-sensitive and is auxiliary only.
                ssim_whole = compute_ssim_3d(real_whole, gen_whole, max_val=data_range)
                ssim_support = _compute_masked_ssim_3d(
                    real_whole, gen_whole, valid_mask, max_val=data_range
                )
                ssim_lesion = compute_per_lesion_ssim(real_whole, gen_whole, cc_list, max_val=data_range)

                per_modality_sums[mod]["mse"].append(mse)
                per_modality_sums[mod]["mae"].append(mae)
                per_modality_sums[mod]["psnr"].append(psnr if not np.isinf(psnr) else 100.0)
                per_modality_sums[mod]["ssim_whole"].append(ssim_whole)
                per_modality_sums[mod]["ssim_support"].append(ssim_support)
                per_modality_sums[mod]["ssim_lesion"].append(ssim_lesion)

                case_id = str(patient_id)
                all_results.setdefault(mod, {})[case_id] = {
                    "mse": round(mse, 6), "mae": round(mae, 6),
                    "psnr": round(psnr, 3),
                    "ssim_whole": round(ssim_whole, 4),
                    "ssim_support": round(ssim_support, 4),
                    "ssim_lesion": round(ssim_lesion, 4),
                    "support_voxels": int(valid_mask.sum()),
                    "tumour_outside_support": tumour_outside_support,
                    "case_seed": case_seed}
                if args.save_support_volumes:
                    source_case_id = f"BraTS-MET-{patient_id}"
                    generated_path = os.path.join(
                        args.output_dir,
                        "generated_zscore",
                        f"{source_case_id}-{mod}.nii.gz",
                    )
                    reference_path = os.path.join(
                        args.output_dir,
                        "reference_zscore",
                        f"{source_case_id}-{mod}.nii.gz",
                    )
                    support_path = os.path.join(
                        args.output_dir,
                        "support",
                        f"{source_case_id}-support.nii.gz",
                    )
                    output_label_path = os.path.join(
                        args.output_dir,
                        "labels",
                        f"{source_case_id}-seg.nii.gz",
                    )
                    _save_array_like(
                        gen_whole, scan_image, generated_path, np.float32
                    )
                    _save_array_like(
                        real_whole, scan_image, reference_path, np.float32
                    )
                    if mod == primary_modality:
                        _save_array_like(
                            valid_mask.astype(np.uint8),
                            scan_image,
                            support_path,
                            np.uint8,
                        )
                        _save_array_like(
                            label_data.astype(np.uint8),
                            label_image,
                            output_label_path,
                            np.uint8,
                        )
                    generation_manifest_rows.append(
                        {
                            "source_case_id": source_case_id,
                            "patient_id": str(patient_id),
                            "modality": mod,
                            "case_seed": case_seed,
                            "checkpoint_path": ckpt_paths.get(mod, "self_test"),
                            "checkpoint_step": (
                                _checkpoint_step(ckpt_paths[mod])
                                if mod in ckpt_paths
                                else "self_test"
                            ),
                            "generated_zscore_path": os.path.abspath(generated_path),
                            "reference_zscore_path": os.path.abspath(reference_path),
                            "support_path": os.path.abspath(support_path),
                            "label_path": os.path.abspath(output_label_path),
                            "support_voxels": int(valid_mask.sum()),
                            "tumour_voxels": int(tumour_mask.sum()),
                            "tumour_outside_support": tumour_outside_support,
                            "normalization": "per_crop_or_tile_brain_zscore",
                        }
                    )
                print(f"    [{mod}] masked-MSE={mse:.4f} MAE={mae:.4f} "
                      f"PSNR={psnr:.2f}dB ssim_whole={ssim_whole:.4f} "
                      f"ssim_support={ssim_support:.4f} "
                      f"ssim_lesion={ssim_lesion:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = {}
    for mod in modalities:
        metrics = per_modality_sums[mod]
        if not metrics["mse"]:
            continue
        # Compute mean/std only for non-empty lists
        avg = {}
        std = {}
        for k, v in metrics.items():
            if v:
                avg[k] = round(float(np.mean(v)), 4)
                std[f"{k}_std"] = round(float(np.std(v)), 4)
        summary[mod] = {**avg, **std}

        # Base line: MSE, MAE, PSNR
        parts = [f"MSE={avg['mse']:.4f}±{std['mse_std']:.4f}",
                 f"MAE={avg['mae']:.4f}±{std['mae_std']:.4f}",
                 f"PSNR={avg['psnr']:.2f}±{std['psnr_std']:.2f}dB"]

        # SSIM: mode-dependent labels
        if avg.get("ssim") is not None:
            parts.append(f"SSIM={avg['ssim']:.4f}±{std['ssim_std']:.4f}")
        if avg.get("ssim_whole") is not None:
            parts.append(f"ssim_whole={avg['ssim_whole']:.4f}±{std['ssim_whole_std']:.4f} (full-brain)")
        if avg.get("ssim_support") is not None:
            parts.append(
                f"ssim_support={avg['ssim_support']:.4f}±"
                f"{std['ssim_support_std']:.4f} (generated-support bbox)"
            )
        if avg.get("ssim_lesion") is not None:
            parts.append(f"ssim_lesion={avg['ssim_lesion']:.4f}±{std['ssim_lesion_std']:.4f} (per-lesion weighted)")

        print(f"  {mod}: " + "  ".join(parts))

    # Save JSON
    def _to_python(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    generation_manifest_path = None
    if args.save_support_volumes:
        if not generation_manifest_rows:
            raise RuntimeError("No support volumes were generated")
        generation_manifest_path = os.path.join(
            args.output_dir, "generation_manifest.csv"
        )
        with open(generation_manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(generation_manifest_rows[0])
            )
            writer.writeheader()
            writer.writerows(generation_manifest_rows)

    result_path = os.path.join(args.output_dir, "metrics.json")
    checkpoint_metadata = {}
    for modality, checkpoint_path in ckpt_paths.items():
        checkpoint_metadata[modality] = {
            "path": os.path.abspath(checkpoint_path),
            "step": _checkpoint_step(checkpoint_path),
            "bytes": os.path.getsize(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        }
    run_metadata = {
        "csv_path": os.path.abspath(args.csv_path),
        "csv_sha256": _sha256_file(args.csv_path),
        "dataset": args.dataset,
        "split": args.split,
        "evaluation_mode": args.evaluation_mode,
        "modalities": modalities,
        "checkpoint_step": args.checkpoint_step,
        "checkpoints": checkpoint_metadata,
        "normalization": args.normalization,
        "reference_normalization": "per_crop_or_tile_brain_zscore",
        "noise_schedule": args.noise_schedule,
        "sampling_method": args.sampling_method,
        "sampling_steps": sampling_steps,
        "seed": args.seed,
        "large_lesion_mode": args.large_lesion_mode,
        "crop_size": args.crop_size,
        "max_cases": args.max_cases,
        "save_support_volumes": args.save_support_volumes,
        "generation_manifest": (
            os.path.abspath(generation_manifest_path)
            if generation_manifest_path is not None
            else None
        ),
        "generation_manifest_rows": len(generation_manifest_rows),
    }
    with open(result_path, "w") as f:
        json.dump(
            {"metadata": run_metadata, "per_case": all_results, "summary": summary},
            f,
            indent=2,
            default=_to_python,
        )
    print(f"\nMetrics saved to: {result_path}")


if __name__ == "__main__":
    main()
