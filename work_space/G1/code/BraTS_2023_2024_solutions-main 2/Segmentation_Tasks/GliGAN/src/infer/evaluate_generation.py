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
import glob
import json
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


def _checkpoint_step(path):
    match = re.search(r"diffusion_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _find_checkpoint(ckpt_dir, modality):
    weights_dir = os.path.join(ckpt_dir, modality, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weight directory not found: {weights_dir}")
    ckpt_files = glob.glob(os.path.join(weights_dir, "diffusion_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No diffusion checkpoint found in: {weights_dir}")
    return max(ckpt_files, key=_checkpoint_step)


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
        mean = np.mean(scan_crop)
        std = np.std(scan_crop)
        if std > 0:
            scan_crop = (scan_crop - mean) / std
        z_min_v, z_max_v = np.min(scan_crop), np.max(scan_crop)
        if z_max_v > z_min_v:
            scan_crop = (scan_crop - z_min_v) / (z_max_v - z_min_v) * 2.0 - 1.0
    else:
        mina, maxa = np.min(scan_crop), np.max(scan_crop)
        if maxa > mina:
            scan_crop = (scan_crop - mina) / (maxa - mina) * 2.0 - 1.0

    # Pad to crop_size^3
    scan_crop_pad = np.pad(
        scan_crop,
        pad_width=((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad),
                    (z_base_pad, z_top_pad)),
        mode="constant", constant_values=(-1, -1),
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
    parser.add_argument("--dataset", type=str, default="BRATS_2024",
                        choices=["BRATS_2023", "BRATS_2024", "BRATS_GOAT_2024"])
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Where to save metrics JSON and per-case details")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampling_steps", type=int, default=50,
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
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="CSV split to evaluate. Default: val. Use all for full CSV.")
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
    _diffusion_utils_local.add_noise_schedule_args(parser)
    parser.add_argument("--cfg_weight", default=1.0, type=float,
                        help="CFG weight: 1.0=normal, >1=stronger conditioning (2.0-3.0 typical)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.self_test and not args.diffusion_ckpt_dir:
        parser.error("--diffusion_ckpt_dir is required (unless --self_test)")

    # ---- Read checkpoint metadata for model architecture ----
    ckpt_metadata = None
    if not args.self_test:
        for mod in (args.modality.split(",") if args.modality != "all"
                    else ["t1c", "t1n", "t2w", "t2f"]):
            try:
                ckpt_path = _find_checkpoint(args.diffusion_ckpt_dir, mod.strip())
            except FileNotFoundError:
                continue
            if ckpt_path:
                ckpt_metadata = torch.load(ckpt_path, map_location="cpu")
                print(f"[Metadata] architecture params from {ckpt_path}")
                break

    # Proxy args for get_diffusion_network
    class _Args:
        pass
    args_proxy = _Args()
    args_proxy.generator_type = args.generator_type
    args_proxy.feature_size = args.feature_size
    args_proxy.use_checkpoint = False
    args_proxy.out_channels = 1
    args_proxy.crop_size = args.crop_size

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
        sd = ckpt_metadata.get("state_dict", ckpt_metadata)
        ckpt_in = None
        for k, v in sd.items():
            if hasattr(v, "shape") and len(v.shape) >= 2:
                # Match first conv of MONAI UNet backbone (e.g. backbone.model.0.conv.weight),
                # SwinUNETR encoder1, PlainConvUNet encoder_stages, or LabelDenoiser3D
                if any(p in k for p in ("model.0.conv.weight", "encoder1", "encoder_stages.0",
                                         "conv1", "input_layer")):
                    ckpt_in = v.shape[1]
                    break
        if ckpt_in is not None:
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
    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split].copy()
    elif args.split != "all" and "split" not in df.columns:
        print(f"[WARN] CSV has no 'split' column; evaluating all rows instead of split={args.split}")
    if args.max_cases > 0:
        df = df.head(args.max_cases)
    if len(df) == 0:
        raise ValueError(f"No CSV rows to evaluate after split={args.split}, max_cases={args.max_cases}")
    print(f"[CSV] {len(df)} test case(s) from {args.csv_path}")

    # Filter modalities
    if args.modality == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [args.modality]

    ckpt_paths = {}
    if not args.self_test:
        for mod in modalities:
            try:
                ckpt_paths[mod] = _find_checkpoint(args.diffusion_ckpt_dir, mod)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
        if not ckpt_paths:
            raise RuntimeError(
                f"No diffusion checkpoints found under {args.diffusion_ckpt_dir}. "
                "Expected {ckpt_dir}/{modality}/weights/diffusion_*.pt")

    # Accumulate results
    all_results = {}
    per_modality_sums = {m: {"mse": [], "mae": [], "psnr": [],
                               "ssim": [],           # patch mode SSIM
                               "ssim_whole": [],     # whole_brain: full-volume SSIM (dominated by background)
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
                model.load_state_dict(ckpt.get("state_dict", ckpt))
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

                mse = compute_mse(real_np, gen_np)
                mae = compute_mae(real_np, gen_np)
                psnr = compute_psnr(real_np, gen_np)
                ssim = compute_ssim_3d(real_np, gen_np)

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
            model.load_state_dict(ckpt.get("state_dict", ckpt))
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
            raise RuntimeError("No diffusion models loaded. Check --diffusion_ckpt_dir.")

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

            label_data = nib.load(label_full).get_fdata().astype(np.int16)
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

                accum = np.zeros(original_shape, dtype=np.float32)
                weight_blend = np.zeros(original_shape, dtype=np.float32)

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
                        accum[z0:z1, y0:y1, x0:x1] += gen_content * gw_content
                        weight_blend[z0:z1, y0:y1, x0:x1] += gw_content
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

                        if was_resized:
                            wz, wy, wx = window_shape
                            from scipy.ndimage import zoom as ndimage_zoom
                            gen_valid = ndimage_zoom(gen_valid,
                                                      (wz / cz, wy / cy, wx / cx), order=1)
                            gw_valid = ndimage_zoom(gw_valid,
                                                     (wz / cz, wy / cy, wx / cx), order=1)

                        accum[z0:z1, y0:y1, x0:x1] += gen_valid * gw_valid
                        weight_blend[z0:z1, y0:y1, x0:x1] += gw_valid

                # Normalize blended whole-brain result
                gen_whole = np.zeros_like(accum)
                valid_mask = weight_blend > 1e-8
                gen_whole[valid_mask] = accum[valid_mask] / weight_blend[valid_mask]

                # Load & normalize real whole-brain scan
                scan_path = patient_rows.iloc[0][f"scan_{mod}"]
                scan_full = os.path.join(gli_gan_root, scan_path)
                if not os.path.isfile(scan_full):
                    print(f"    [WARN] Missing scan for {mod}: {scan_full}")
                    continue
                real_whole = nib.load(scan_full).get_fdata().astype(np.float32)

                if args.normalization == "zscore":
                    mean = np.mean(real_whole)
                    std = np.std(real_whole)
                    if std > 0:
                        real_whole = (real_whole - mean) / std
                    rmin, rmax = np.min(real_whole), np.max(real_whole)
                    if rmax > rmin:
                        real_whole = (real_whole - rmin) / (rmax - rmin) * 2.0 - 1.0
                else:
                    rmin, rmax = np.min(real_whole), np.max(real_whole)
                    if rmax > rmin:
                        real_whole = (real_whole - rmin) / (rmax - rmin) * 2.0 - 1.0

                # Masked metrics (tumour region only)
                tumour_mask = mask_binary
                real_tumour = real_whole[tumour_mask]
                gen_tumour = gen_whole[tumour_mask]

                if len(real_tumour) == 0:
                    print(f"    [{mod}] No tumour voxels, skipping")
                    continue

                mse = compute_mse(real_tumour, gen_tumour)
                mae = compute_mae(real_tumour, gen_tumour)
                psnr = compute_psnr(real_tumour, gen_tumour)
                # Two SSIM variants, served different purposes:
                #   ssim_whole : whole-brain volume — dominated by background, signals pipeline health
                #   ssim_lesion: per-lesion bbox, voxel-count weighted — signals tumour quality
                ssim_whole = compute_ssim_3d(real_whole, gen_whole)
                ssim_lesion = compute_per_lesion_ssim(real_whole, gen_whole, cc_list)

                per_modality_sums[mod]["mse"].append(mse)
                per_modality_sums[mod]["mae"].append(mae)
                per_modality_sums[mod]["psnr"].append(psnr if not np.isinf(psnr) else 100.0)
                per_modality_sums[mod]["ssim_whole"].append(ssim_whole)
                per_modality_sums[mod]["ssim_lesion"].append(ssim_lesion)

                case_id = str(patient_id)
                all_results.setdefault(mod, {})[case_id] = {
                    "mse": round(mse, 6), "mae": round(mae, 6),
                    "psnr": round(psnr, 3),
                    "ssim_whole": round(ssim_whole, 4),
                    "ssim_lesion": round(ssim_lesion, 4)}
                print(f"    [{mod}] masked-MSE={mse:.4f} MAE={mae:.4f} "
                      f"PSNR={psnr:.2f}dB ssim_whole={ssim_whole:.4f} "
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

    result_path = os.path.join(args.output_dir, "metrics.json")
    with open(result_path, "w") as f:
        json.dump({"per_case": all_results, "summary": summary}, f, indent=2, default=_to_python)
    print(f"\nMetrics saved to: {result_path}")


if __name__ == "__main__":
    main()
