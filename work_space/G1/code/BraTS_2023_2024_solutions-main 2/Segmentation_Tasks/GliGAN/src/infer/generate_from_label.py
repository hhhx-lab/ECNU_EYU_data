"""
Pipeline 2: Generate full 4-modality brain MRI from a tumour label.

Supports multi-lesion inference: connected-component analysis,
per-lesion crop generation, and whole-brain Gaussian blending.

Usage:
    cd Segmentation_Tasks/GliGAN/src/infer
    python generate_from_label.py \
        --label_path /path/to/tumour_label.nii.gz \
        --diffusion_ckpt_dir ../../Checkpoint/brats_2024 \
        --dataset BRATS_2023 \
        --output_dir ./generated_scans \
        --sampling_steps 50 \
        --crop_size 64 \
        --device cuda
"""

import os
import sys
import argparse
import glob
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
    load_diffusion_model,
)

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


# ---------------------------------------------------------------------------
# Connected-component analysis (mirrors csv_creator.py logic)
# ---------------------------------------------------------------------------

def connected_component_analysis(mask_binary):
    """Return list of {centroid, bbox, n_voxels} per connected component."""
    structure = np.ones((3, 3, 3), dtype=np.int16)
    labeled, n_cc = ndimage.label(mask_binary, structure=structure)
    components = []
    for cc_id in range(1, n_cc + 1):
        cc_mask = (labeled == cc_id)
        coords = np.argwhere(cc_mask)
        centroid = coords.mean(axis=0)  # (z, y, x)
        z_min, z_max = coords[:, 0].min(), coords[:, 0].max() + 1
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max() + 1
        x_min, x_max = coords[:, 2].min(), coords[:, 2].max() + 1
        components.append({
            'centroid': (round(centroid[2]), round(centroid[1]), round(centroid[0])),
            'bbox': (x_min, x_max, y_min, y_max, z_min, z_max),
            'n_voxels': int(cc_mask.sum()),
        })
    return components


def merge_nearby_lesions(components, merge_dist=16, crop_size=64):
    """Merge lesions whose centroid distance < merge_dist voxels.
    Does NOT merge if the merged bbox would exceed crop_size in any dimension."""
    if len(components) <= 1:
        return components
    merged = []
    used = [False] * len(components)
    for i, ci in enumerate(components):
        if used[i]:
            continue
        group = [ci]
        used[i] = True
        for j, cj in enumerate(components):
            if used[j]:
                continue
            dx = ci['centroid'][0] - cj['centroid'][0]
            dy = ci['centroid'][1] - cj['centroid'][1]
            dz = ci['centroid'][2] - cj['centroid'][2]
            if np.sqrt(dx*dx + dy*dy + dz*dz) < merge_dist:
                x_mins_tmp = [g['bbox'][0] for g in group] + [cj['bbox'][0]]
                x_maxs_tmp = [g['bbox'][1] for g in group] + [cj['bbox'][1]]
                y_mins_tmp = [g['bbox'][2] for g in group] + [cj['bbox'][2]]
                y_maxs_tmp = [g['bbox'][3] for g in group] + [cj['bbox'][3]]
                z_mins_tmp = [g['bbox'][4] for g in group] + [cj['bbox'][4]]
                z_maxs_tmp = [g['bbox'][5] for g in group] + [cj['bbox'][5]]
                merged_x = max(x_maxs_tmp) - min(x_mins_tmp)
                merged_y = max(y_maxs_tmp) - min(y_mins_tmp)
                merged_z = max(z_maxs_tmp) - min(z_mins_tmp)
                if merged_x <= crop_size and merged_y <= crop_size and merged_z <= crop_size:
                    group.append(cj)
                    used[j] = True
        if len(group) == 1:
            merged.append(ci)
        else:
            x_mins = [g['bbox'][0] for g in group]
            x_maxs = [g['bbox'][1] for g in group]
            y_mins = [g['bbox'][2] for g in group]
            y_maxs = [g['bbox'][3] for g in group]
            z_mins = [g['bbox'][4] for g in group]
            z_maxs = [g['bbox'][5] for g in group]
            n_voxels = sum(g['n_voxels'] for g in group)
            cx = round(np.mean([g['centroid'][0] for g in group]))
            cy = round(np.mean([g['centroid'][1] for g in group]))
            cz = round(np.mean([g['centroid'][2] for g in group]))
            merged.append({
                'centroid': (cx, cy, cz),
                'bbox': (min(x_mins), max(x_maxs), min(y_mins), max(y_maxs),
                         min(z_mins), max(z_maxs)),
                'n_voxels': n_voxels,
            })
    return merged


# ---------------------------------------------------------------------------
# Label → multi-channel conversion (per dataset type)
# ---------------------------------------------------------------------------

def label_to_multichannel(label_data, dataset_type):
    """Convert integer label to multi-channel float32 array."""
    if dataset_type in ("BRATS_2024",):
        n_channels = 4
        label_mc = np.zeros((n_channels,) + label_data.shape, dtype=np.float32)
        label_mc[0] = ((label_data == 1) | (label_data == 3)).astype(np.float32)  # TC
        label_mc[1] = ((label_data == 1) | (label_data == 2) | (label_data == 3)).astype(np.float32)  # WT
        label_mc[2] = (label_data == 3).astype(np.float32)  # ET
        label_mc[3] = (label_data == 4).astype(np.float32)  # RC
    elif dataset_type == "BRATS_2024_MENINGIOMA":
        n_channels = 1
        label_mc = np.zeros((1,) + label_data.shape, dtype=np.float32)
        label_mc[0] = (label_data > 0).astype(np.float32)
    else:  # BRATS_2023 / BRATS_GOAT_2024
        n_channels = 3
        label_mc = np.zeros((n_channels,) + label_data.shape, dtype=np.float32)
        label_mc[0] = ((label_data == 1) | (label_data == 3)).astype(np.float32)  # TC
        label_mc[1] = ((label_data == 1) | (label_data == 2) | (label_data == 3)).astype(np.float32)  # WT
        label_mc[2] = (label_data == 3).astype(np.float32)  # ET
    return label_mc, n_channels


# ---------------------------------------------------------------------------
# Per-lesion crop extraction
# ---------------------------------------------------------------------------

def extract_single_crop(label_mc, bbox, original_shape, crop_size=64):
    """
    Extract a crop around the lesion bbox, with margin and symmetric padding.

    Args:
        label_mc: (C, Z, Y, X) multi-channel label (full brain)
        bbox: (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coords
        original_shape: (Z, Y, X) of the full brain
        crop_size: target cubic size

    Returns:
        label_cube: (C, crop_size, crop_size, crop_size) float32
        coords: (z0, z1, y0, y1, x0, x1) — global coords of the crop window in brain space
        content_shape: (cz, cy, cx) — shape of the label content after optional resize,
                       before padding. Used to map generated output back to window size.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    n_channels = label_mc.shape[0]
    Z, Y, X = original_shape

    # Add margin (10% of bbox, min 4 voxels)
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    mx = max(int(dx * 0.1), 4)
    my = max(int(dy * 0.1), 4)
    mz = max(int(dz * 0.1), 4)

    # Compute crop window
    x0 = max(0, x_min - mx)
    x1 = min(X, x_max + mx)
    y0 = max(0, y_min - my)
    y1 = min(Y, y_max + my)
    z0 = max(0, z_min - mz)
    z1 = min(Z, z_max + mz)

    crop = label_mc[:, z0:z1, y0:y1, x0:x1]  # (C, cz, cy, cx)

    # Resize if any dimension > crop_size - 4
    max_dim = max(crop.shape[1:])
    if max_dim > crop_size - 4:
        scale = (crop_size - 4) / max_dim
        from scipy.ndimage import zoom as ndimage_zoom
        new_dims = np.maximum(np.round(np.array(crop.shape[1:]) * scale), 1).astype(int)
        resized = np.zeros((n_channels,) + tuple(new_dims), dtype=np.float32)
        for c in range(n_channels):
            resized[c] = ndimage_zoom(crop[c].astype(np.float32),
                                       tuple(new_dims / np.array(crop.shape[1:])),
                                       order=1)
        crop = resized

    # Symmetric padding to crop_size³
    cz, cy, cx = crop.shape[1:]
    z_pad_total = crop_size - cz
    y_pad_total = crop_size - cy
    x_pad_total = crop_size - cx

    z_pad_before = z_pad_total // 2
    z_pad_after = z_pad_total - z_pad_before
    y_pad_before = y_pad_total // 2
    y_pad_after = y_pad_total - y_pad_before
    x_pad_before = x_pad_total // 2
    x_pad_after = x_pad_total - x_pad_before

    result = np.zeros((n_channels, crop_size, crop_size, crop_size), dtype=np.float32)
    result[:,
           z_pad_before:z_pad_before + cz,
           y_pad_before:y_pad_before + cy,
           x_pad_before:x_pad_before + cx] = crop

    coords = (z0, z1, y0, y1, x0, x1)
    content_shape = (crop.shape[1], crop.shape[2], crop.shape[3])  # after optional resize
    return result, coords, content_shape


def make_gaussian_weight_3d(shape, sigma=None):
    """
    Create a 3D Gaussian weight volume centered in the array.

    Args:
        shape: (D, H, W) tuple
        sigma: standard deviation (default: max(shape)/3)

    Returns:
        weight: ndarray of shape with values in [0, 1]
    """
    if sigma is None:
        sigma = max(shape) / 3.0
    D, H, W = shape
    zd = np.arange(D) - (D - 1) / 2.0
    yd = np.arange(H) - (H - 1) / 2.0
    xd = np.arange(W) - (W - 1) / 2.0
    zgrid, ygrid, xgrid = np.meshgrid(zd, yd, xd, indexing='ij')
    dist_sq = zgrid**2 + ygrid**2 + xgrid**2
    return np.exp(-dist_sq / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Tile generation for large lesions
# ---------------------------------------------------------------------------

def _tile_generate_lesion(label_mc, coords, crop_size, model, spatial_size, device,
                          sample_kwargs):
    """Generate a large lesion by tiling crop_size^3 windows with overlap.

    Slides a crop_size^3 window through the lesion region, generates each tile
    independently via the diffusion model, and blends them with Gaussian weights.

    Args:
        label_mc: (C, Z_full, Y_full, X_full) multi-channel label
        coords: (z0, z1, y0, y1, x0, x1) global voxel coords of the lesion window
        crop_size: cubic tile / crop size
        model: diffusion model (one modality)
        spatial_size: (crop_size, crop_size, crop_size)
        device: torch device
        sample_kwargs: dict of kwargs forwarded to sample_tumour_diffusion_full

    Returns:
        gen_content: (window_z, window_y, window_x) generated float32 content
        gw_content: (window_z, window_y, window_x) blending weights
    """
    z0, z1, y0, y1, x0, x1 = coords
    window_z = z1 - z0
    window_y = y1 - y0
    window_x = x1 - x0

    accum = np.zeros((window_z, window_y, window_x), dtype=np.float32)
    weight = np.zeros((window_z, window_y, window_x), dtype=np.float32)

    stride = max(crop_size // 2, 1)
    gauss_full = make_gaussian_weight_3d((crop_size, crop_size, crop_size),
                                          sigma=crop_size / 3.0)

    z_starts = list(range(0, window_z, stride))
    y_starts = list(range(0, window_y, stride))
    x_starts = list(range(0, window_x, stride))
    total_tiles = len(z_starts) * len(y_starts) * len(x_starts)
    tile_idx = 0

    for iz in z_starts:
        for iy in y_starts:
            for ix in x_starts:
                tile_idx += 1
                # Tile bounds in lesion-window-local coords
                tz0, tz1 = iz, min(iz + crop_size, window_z)
                ty0, ty1 = iy, min(iy + crop_size, window_y)
                tx0, tx1 = ix, min(ix + crop_size, window_x)
                tz, ty, tx = tz1 - tz0, ty1 - ty0, tx1 - tx0

                # Extract label tile from full-brain label at native resolution
                gz0, gz1 = z0 + tz0, z0 + tz1
                gy0, gy1 = y0 + ty0, y0 + ty1
                gx0, gx1 = x0 + tx0, x0 + tx1
                label_tile = label_mc[:, gz0:gz1, gy0:gy1, gx0:gx1]

                # Pad to crop_size^3
                zp = (crop_size - tz) // 2
                zp2 = crop_size - tz - zp
                yp = (crop_size - ty) // 2
                yp2 = crop_size - ty - yp
                xp = (crop_size - tx) // 2
                xp2 = crop_size - tx - xp

                label_cube = np.pad(label_tile,
                                    ((0, 0), (zp, zp2), (yp, yp2), (xp, xp2)),
                                    mode='constant', constant_values=0)

                label_tensor = torch.from_numpy(label_cube).float().unsqueeze(0).to(device)

                generated = sample_tumour_diffusion_full(
                    model=model, label_cond=label_tensor,
                    spatial_size=spatial_size, **sample_kwargs)

                gen_np = generated.squeeze(0).squeeze(0).cpu().numpy()

                gen_valid = gen_np[zp:zp + tz, yp:yp + ty, xp:xp + tx]
                gw_valid = gauss_full[zp:zp + tz, yp:yp + ty, xp:xp + tx]

                accum[tz0:tz1, ty0:ty1, tx0:tx1] += gen_valid * gw_valid
                weight[tz0:tz1, ty0:ty1, tx0:tx1] += gw_valid

                print(f"    [tile {tile_idx}/{total_tiles}] "
                      f"pos=({tz0}:{tz1},{ty0}:{ty1},{tx0}:{tx1}) "
                      f"content=({tz},{ty},{tx})")

    return accum, weight


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_checkpoint(ckpt_dir, modality):
    """Find the latest checkpoint for a given modality."""
    weights_dir = os.path.join(ckpt_dir, modality, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weight directory not found: {weights_dir}")
    ckpt_files = glob.glob(os.path.join(weights_dir, "diffusion_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No diffusion checkpoint found in: {weights_dir}")
    return max(ckpt_files, key=_checkpoint_step)


def _checkpoint_step(path):
    match = re.search(r"diffusion_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 2: Generate full 4-modality brain MRI from a tumour label"
    )
    parser.add_argument("--label_path", type=str, required=True,
                        help="Path to tumour label nii.gz")
    parser.add_argument("--diffusion_ckpt_dir", type=str, required=True,
                        help="Root dir of diffusion checkpoints ({dir}/{modality}/weights/diffusion_*.pt)")
    parser.add_argument("--output_dir", type=str, default="./generated_scans",
                        help="Output directory for generated nii.gz files")
    parser.add_argument("--dataset", type=str, default="BRATS_2023",
                        help="Dataset type: BRATS_2023 / BRATS_2024 / BRATS_GOAT_2024 / BRATS_2024_MENINGIOMA")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="DDPM/DDIM accelerated sampling steps (0 = use full n_steps)")
    parser.add_argument("--sampling_method", type=str, default="edm_heun",
                        choices=["ddpm", "ddim", "edm_heun", "lognsr_ode"],
                        help="Sampling: 'edm_heun', 'lognsr_ode', 'ddpm', 'ddim'")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM / logsnr stochasticity (0=deterministic, 1≈DDPM)")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Total diffusion steps")
    parser.add_argument("--beta_schedule", type=str, default="cosine",
                        help="Beta schedule type (legacy; use --noise_schedule for EDM/logsnr)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda / cpu")
    parser.add_argument("--generator_type", type=str, default="Unet_NnU",
                        help="Backbone: Unet_NnU / PlainConvUNet / SwinUNETR / AttentionUnet / Unet")
    parser.add_argument("--feature_size", type=int, default=48,
                        help="Feature size for SwinUNETR")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "t1c", "t1n", "t2w", "t2f"],
                        help="Run a single modality or 'all' (default: all)")
    parser.add_argument("--crop_size", default=64, type=int,
                        help="Crop/pad target size (64=default, 96=glioma)")
    parser.add_argument("--use_compile", action="store_true",
                        help="Enable torch.compile for the model (PyTorch >= 2.0)")
    parser.add_argument("--merge_dist", default=16, type=int,
                        help="Merge lesions closer than this distance (voxels) into one crop")
    parser.add_argument("--large_lesion_mode", default="resize", type=str,
                        choices=["resize", "skip", "tile"],
                        help="How to handle lesions > crop_size: resize (zoom in/out), "
                             "skip (ignore lesion), tile (sliding window with overlap)")
    _diffusion_utils_local.add_noise_schedule_args(parser)
    parser.add_argument("--cfg_weight", default=1.0, type=float,
                        help="CFG weight: 1.0=normal, >1=stronger conditioning (2.0-3.0 typical)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build proxy args for get_diffusion_network ----
    class _Args:
        pass
    args_proxy = _Args()
    args_proxy.generator_type = args.generator_type
    args_proxy.feature_size = args.feature_size
    args_proxy.use_checkpoint = False
    args_proxy.out_channels = 1
    args_proxy.crop_size = args.crop_size

    if args.dataset == "BRATS_2024":
        label_channels = 4
    elif args.dataset == "BRATS_2024_MENINGIOMA":
        label_channels = 1
    else:
        label_channels = 3
    args_proxy.in_channels = 1 + label_channels
    # noise_embedding_mode / time_ch_count are read from checkpoint by
    # load_diffusion_model(); the guess below is a fallback for old checkpoints.
    if args.sampling_method in ("edm_heun", "lognsr_ode"):
        args_proxy.noise_embedding_mode = "continuous"
    else:
        args_proxy.noise_embedding_mode = "discrete"

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] using: {device}")

    # ---- Load label and do CC analysis ----
    print(f"\n[Step 1/5] Loading label: {args.label_path}")
    label_data = nib.load(args.label_path).get_fdata().astype(np.int16)
    original_shape = label_data.shape  # (Z, Y, X)
    print(f"  original shape: {original_shape}")

    # Convert to multi-channel (for crop extraction)
    label_mc, n_label_ch = label_to_multichannel(label_data, args.dataset)
    print(f"  multi-channel label: {label_mc.shape}")

    # Binary mask for CC analysis
    mask_binary = (label_data != 0)
    if not np.any(mask_binary):
        raise ValueError("Label contains no tumour regions (all zeros).")

    cc_list = connected_component_analysis(mask_binary)
    print(f"  found {len(cc_list)} connected component(s)")
    cc_list = merge_nearby_lesions(cc_list, merge_dist=args.merge_dist, crop_size=args.crop_size)
    print(f"  after merging (dist < {args.merge_dist} vox): {len(cc_list)} lesion group(s)")

    print(f"  {len(cc_list)} lesion(s) will be processed")

    if len(cc_list) == 0:
        raise RuntimeError("No lesions found.")

    # ---- Extract per-lesion crops ----
    print(f"\n[Step 2/5] Extracting per-lesion crops (crop_size={args.crop_size})...")
    lesion_crops = []
    for i, cc in enumerate(cc_list):
        label_cube, coords, content_shape = extract_single_crop(
            label_mc, cc['bbox'], original_shape, crop_size=args.crop_size)
        window_shape = (coords[1] - coords[0], coords[3] - coords[2], coords[5] - coords[4])
        was_resized = (content_shape != window_shape)
        lesion_crops.append({
            'label_cube': label_cube,
            'coords': coords,
            'content_shape': content_shape,
            'was_resized': was_resized,
            'centroid': cc['centroid'],
            'n_voxels': cc['n_voxels'],
        })
        resize_note = " (resized)" if was_resized else ""
        print(f"  lesion {i}: bbox={cc['bbox']}, window={window_shape}, "
              f"content={content_shape}{resize_note}, n_voxels={cc['n_voxels']}")

    # ---- Load models first (need metadata for noise schedule) ----
    print(f"\n[Step 3/5] Loading diffusion models...")

    if args.modality == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [args.modality]

    models = {}
    ckpt_metadata = None
    for mod in modalities:
        weights_dir = os.path.join(args.diffusion_ckpt_dir, mod, "weights")
        if not os.path.isdir(weights_dir):
            print(f"  [SKIP] weights dir not found: {weights_dir}")
            continue
        ckpt_path = _find_checkpoint(args.diffusion_ckpt_dir, mod)
        print(f"  [{mod}] loading: {ckpt_path}")
        models[mod], meta = load_diffusion_model(ckpt_path, args_proxy, device,
                                                 use_compile=args.use_compile)
        if ckpt_metadata is None:
            ckpt_metadata = meta

    if not models:
        raise RuntimeError("No diffusion models loaded. Check --diffusion_ckpt_dir.")

    # ---- Build diffusion coefficients from checkpoint metadata ----
    noise_schedule = ckpt_metadata["noise_schedule"]
    schedule_override = ckpt_metadata["schedule_config"] or {}
    n_steps_actual = ckpt_metadata["n_steps"]
    if hasattr(args, "beta_schedule") and args.beta_schedule != "cosine":
        noise_schedule = args.beta_schedule

    print(f"\n[Step 4/5] Building noise schedule from checkpoint: "
          f"schedule={noise_schedule}, n_steps={n_steps_actual}")
    print(f"  sampling: method={args.sampling_method}")

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
    betas = schedule_cfg.betas
    alphas_bar_sqrt = schedule_cfg.alphas_bar_sqrt
    one_minus_alphas_bar_sqrt = schedule_cfg.one_minus_alphas_bar_sqrt
    alphas_bar = schedule_cfg.alphas_bar

    sampling_steps = args.sampling_steps if args.sampling_steps > 0 else n_steps_actual
    print(f"  sampling steps: {sampling_steps}, eta={args.eta}")

    spatial_size = (args.crop_size, args.crop_size, args.crop_size)

    # ---- Per-lesion generation + Gaussian blending ----
    # Accumulators for whole-brain stitching
    accum = {}   # mod → np.zeros(original_shape, float32)
    weight = {}  # mod → np.zeros(original_shape, float32)

    for mod in modalities:
        if mod not in models:
            continue
        accum[mod] = np.zeros(original_shape, dtype=np.float32)
        weight[mod] = np.zeros(original_shape, dtype=np.float32)

    # Pre-compute Gaussian weight volume for a single crop
    gauss_weight = make_gaussian_weight_3d(
        (args.crop_size, args.crop_size, args.crop_size),
        sigma=args.crop_size / 3.0)

    for lesion_idx, lesion in enumerate(lesion_crops):
        coords = lesion['coords']
        z0, z1, y0, y1, x0, x1 = coords
        window_shape = (z1 - z0, y1 - y0, x1 - x0)
        is_oversized = any(d > args.crop_size for d in window_shape)

        # --- SKIP mode: ignore oversized lesions ---
        if args.large_lesion_mode == "skip" and is_oversized:
            print(f"\n  Lesion {lesion_idx + 1}/{len(lesion_crops)}: "
                  f"SKIPPED (window {window_shape} > crop_size {args.crop_size})")
            continue

        print(f"\n  Lesion {lesion_idx + 1}/{len(lesion_crops)}: "
              f"coords=({z0}:{z1},{y0}:{y1},{x0}:{x1}), "
              f"window={window_shape}, n_voxels={lesion['n_voxels']}"
              f"{' [tile]' if args.large_lesion_mode == 'tile' and is_oversized else ''}"
              f"{' [resize]' if args.large_lesion_mode == 'resize' and is_oversized else ''}")

        for mod in modalities:
            if mod not in models:
                continue
            model = models[mod]

            if args.large_lesion_mode == "tile" and is_oversized:
                # --- TILE mode: sliding window generation with Gaussian blending ---
                gen_content, gw_content = _tile_generate_lesion(
                    label_mc=label_mc,
                    coords=coords,
                    crop_size=args.crop_size,
                    model=model,
                    spatial_size=spatial_size,
                    device=device,
                    sample_kwargs=dict(
                        n_steps=n_steps_actual,
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
                    ),
                )
                accum[mod][z0:z1, y0:y1, x0:x1] += gen_content * gw_content
                weight[mod][z0:z1, y0:y1, x0:x1] += gw_content

            else:
                # --- RESIZE mode (or small lesion): single-crop generation ---
                label_cube = lesion['label_cube']
                label_tensor = torch.from_numpy(label_cube).float().unsqueeze(0).to(device)

                generated = sample_tumour_diffusion_full(
                    model=model,
                    label_cond=label_tensor,
                    spatial_size=spatial_size,
                    n_steps=n_steps_actual,
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

                # Extract the valid (non-padding) content from the crop_size^3 output
                cz, cy, cx = lesion['content_shape']
                z_pad_before = (args.crop_size - cz) // 2
                y_pad_before = (args.crop_size - cy) // 2
                x_pad_before = (args.crop_size - cx) // 2

                gen_valid = gen_np[z_pad_before:z_pad_before + cz,
                                   y_pad_before:y_pad_before + cy,
                                   x_pad_before:x_pad_before + cx]
                gw_valid = gauss_weight[z_pad_before:z_pad_before + cz,
                                         y_pad_before:y_pad_before + cy,
                                         x_pad_before:x_pad_before + cx]

                # If the crop was resized (lesion > crop_size), zoom back up to
                # the original window size before placing in the whole brain
                if lesion['was_resized']:
                    window_z = z1 - z0
                    window_y = y1 - y0
                    window_x = x1 - x0
                    from scipy.ndimage import zoom as ndimage_zoom
                    gen_valid = ndimage_zoom(gen_valid,
                                              (window_z / cz, window_y / cy, window_x / cx),
                                              order=1)
                    gw_valid = ndimage_zoom(gw_valid,
                                             (window_z / cz, window_y / cy, window_x / cx),
                                             order=1)

                accum[mod][z0:z1, y0:y1, x0:x1] += gen_valid * gw_valid
                weight[mod][z0:z1, y0:y1, x0:x1] += gw_valid

    # ---- Normalize and save ----
    print(f"\n[Step 5/5] Normalizing blending weights and saving...")

    basename = os.path.basename(args.label_path)
    for ext in [".nii.gz", ".nii"]:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    for suffix in ["_seg", "-seg", "_label", "-label"]:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break

    for mod in modalities:
        if mod not in accum:
            continue

        # Normalize: divide by weight where weight > 0
        final = np.zeros(original_shape, dtype=np.float32)
        mask = weight[mod] > 1e-8
        final[mask] = accum[mod][mask] / weight[mod][mask]

        output_path = os.path.join(args.output_dir, f"{basename}-{mod}.nii.gz")
        img = nib.Nifti1Image(final, np.eye(4))
        nib.save(img, output_path)
        print(f"  saved: {output_path}")

    print(f"\nDone! Generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
