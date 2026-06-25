"""
Pipeline 2: Generate full 4-modality brain MRI from a tumour label.

Independent of on_the_fly_augmentation.py. Takes a label nii.gz as input,
runs full diffusion generation (no inpainting) for each of the 4 modalities,
and saves the results as nii.gz files.

Usage:
    cd Segmentation_Tasks/GliGAN/src/infer
    python generate_from_label.py \\
        --label_path /path/to/tumour_label.nii.gz \\
        --diffusion_ckpt_dir ../../Checkpoint/brats_2024 \\
        --dataset BRATS_2023 \\
        --output_dir ./generated_scans \\
        --sampling_steps 50 \\
        --device cuda
"""

import os
import sys
import argparse
import glob

import numpy as np
import torch
import nibabel as nib

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusion_inference_utils import (
    make_diffusion_coefficients,
    sample_tumour_diffusion_full,
)
from src.networks.DiffusionNetwork import get_diffusion_network

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


# All 4 modalities
ALL_MODALITIES = ["t1c", "t1n", "t2w", "t2f"]


def load_label_and_prepare(label_path, dataset_type, target_size=96):
    """
    Load a label nii.gz, convert to multi-channel, crop to tumour bounding box,
    and pad to target_size³.

    Returns:
        label_cube:  (C, target_size, target_size, target_size) float32
        meta:        dict with crop/pad info for potential inverse mapping
    """
    label_data = nib.load(label_path).get_fdata().astype(np.int16)
    print(f"[Label] original shape: {label_data.shape}")

    # ---- Convert to multi-channel ----
    if dataset_type == "BRATS_2024":
        n_channels = 4
        label_mc = np.zeros((n_channels,) + label_data.shape, dtype=np.float32)
        label_mc[0] = ((label_data == 1) | (label_data == 3)).astype(np.float32)  # TC
        label_mc[1] = ((label_data == 1) | (label_data == 2) | (label_data == 3)).astype(np.float32)  # WT
        label_mc[2] = (label_data == 3).astype(np.float32)  # ET
        label_mc[3] = (label_data == 4).astype(np.float32)  # RC
    else:  # BRATS_2023 / BRATS_GOAT_2024
        n_channels = 3
        label_mc = np.zeros((n_channels,) + label_data.shape, dtype=np.float32)
        label_mc[0] = ((label_data == 1) | (label_data == 3)).astype(np.float32)  # TC
        label_mc[1] = ((label_data == 1) | (label_data == 2) | (label_data == 3)).astype(np.float32)  # WT
        label_mc[2] = (label_data == 3).astype(np.float32)  # ET

    # ---- Crop to tumour bounding box ----
    non_zero = (label_data != 0)
    if not np.any(non_zero):
        raise ValueError("Label contains no tumour regions (all zeros).")

    coords = np.where(non_zero)
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1

    # Add margin (10% of bbox size, min 4 voxels)
    dz, dy, dx = z_max - z_min, y_max - y_min, x_max - x_min
    margin_z = max(int(dz * 0.1), 4)
    margin_y = max(int(dy * 0.1), 4)
    margin_x = max(int(dx * 0.1), 4)

    z0 = max(0, z_min - margin_z)
    z1 = min(label_data.shape[0], z_max + margin_z)
    y0 = max(0, y_min - margin_y)
    y1 = min(label_data.shape[1], y_max + margin_y)
    x0 = max(0, x_min - margin_x)
    x1 = min(label_data.shape[2], x_max + margin_x)

    crop_mc = label_mc[:, z0:z1, y0:y1, x0:x1]  # (C, crop_z, crop_y, crop_x)
    print(f"[Label] tumour bbox: ({z0}:{z1}, {y0}:{y1}, {x0}:{x1}), crop shape: {crop_mc.shape}")

    # ---- Resize if any dimension > target_size ----
    max_dim = max(crop_mc.shape[1:])
    if max_dim > target_size - 4:
        scale = (target_size - 4) / max_dim
        from scipy.ndimage import zoom as ndimage_zoom
        new_dims = np.maximum(np.round(np.array(crop_mc.shape[1:]) * scale), 1).astype(int)
        resized = np.zeros((n_channels,) + tuple(new_dims), dtype=np.float32)
        for c in range(n_channels):
            resized[c] = ndimage_zoom(crop_mc[c].astype(np.float32),
                                       tuple(new_dims / np.array(crop_mc.shape[1:])),
                                       order=1)
        crop_mc = resized
        print(f"[Label] resized to: {crop_mc.shape}")

    # ---- Pad to target_size³, centred ----
    result = np.zeros((n_channels, target_size, target_size, target_size), dtype=np.float32)
    z_start = (target_size - crop_mc.shape[1]) // 2
    y_start = (target_size - crop_mc.shape[2]) // 2
    x_start = (target_size - crop_mc.shape[3]) // 2
    result[:,
           z_start:z_start + crop_mc.shape[1],
           y_start:y_start + crop_mc.shape[2],
           x_start:x_start + crop_mc.shape[3]] = crop_mc
    print(f"[Label] padded to: {result.shape}")

    meta = {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "original_shape": label_data.shape, "n_channels": n_channels}

    return result, meta


def load_diffusion_model(ckpt_dir, modality, args_proxy, device):
    """Load a single diffusion model checkpoint for the given modality."""
    weights_dir = os.path.join(ckpt_dir, modality, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weight directory not found: {weights_dir}")

    ckpt_files = sorted(glob.glob(os.path.join(weights_dir, "diffusion_*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No diffusion checkpoint found in: {weights_dir}")

    ckpt_path = ckpt_files[-1]  # latest
    print(f"  [{modality}] loading: {ckpt_path}")

    model = get_diffusion_network(args_proxy)
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


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
                        help="Dataset type: BRATS_2023 / BRATS_2024 / BRATS_GOAT_2024")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="DDPM/DDIM accelerated sampling steps (0 = use full n_steps)")
    parser.add_argument("--sampling_method", type=str, default="ddpm",
                        choices=["ddpm", "ddim", "edm_heun", "lognsr_ode"],
                        help="Sampling: 'ddpm', 'ddim', 'edm_heun', 'lognsr_ode'")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM / logsnr stochasticity (0=deterministic, 1≈DDPM)")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Total diffusion steps")
    parser.add_argument("--beta_schedule", type=str, default="cosine",
                        help="Beta schedule type (legacy; use --noise_schedule for EDM/logsnr)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda / cpu")
    parser.add_argument("--generator_type", type=str, default="SwinUNETR",
                        help="Backbone: SwinUNETR / AttentionUnet / Unet / Unet_NnU / PlainConvUNet")
    parser.add_argument("--feature_size", type=int, default=48,
                        help="Feature size for SwinUNETR")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "t1c", "t1n", "t2w", "t2f"],
                        help="Run a single modality or 'all' (default: all)")
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
    # Determine label channels from dataset type
    if args.dataset == "BRATS_2024":
        label_channels = 4
    else:
        label_channels = 3
    args_proxy.in_channels = 1 + label_channels  # scan(1) + label(C)
    # Set noise embedding mode based on sampling method
    if args.sampling_method in ("edm_heun", "lognsr_ode"):
        args_proxy.noise_embedding_mode = "continuous"
    else:
        args_proxy.noise_embedding_mode = "discrete"

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] using: {device}")

    # ---- Load and prepare label ----
    print(f"\n[Step 1/4] Loading label: {args.label_path}")
    label_cube, meta = load_label_and_prepare(args.label_path, args.dataset, target_size=96)
    label_tensor = torch.from_numpy(label_cube).float().unsqueeze(0).to(device)
    print(f"  label_tensor shape: {label_tensor.shape}")

    # Backward compat: --beta_schedule overrides default --noise_schedule
    if hasattr(args, "beta_schedule") and args.beta_schedule != "cosine":
        args.noise_schedule = args.beta_schedule

    # ---- Build diffusion coefficients / schedule config ----
    print(f"\n[Step 2/4] Building noise schedule (schedule={args.noise_schedule}, method={args.sampling_method})")
    schedule_cfg = make_diffusion_coefficients(
        n_steps=args.n_steps, device=device,
        noise_schedule=args.noise_schedule,
        sigma_data=args.sigma_data, sigma_max=args.sigma_max,
        sigma_min=args.sigma_min, rho=args.rho,
        gamma_max=args.gamma_max, gamma_min=args.gamma_min,
        snr_shift=args.snr_shift,
    )
    # Legacy unpacking for backward compat
    betas = schedule_cfg.betas
    alphas_bar_sqrt = schedule_cfg.alphas_bar_sqrt
    one_minus_alphas_bar_sqrt = schedule_cfg.one_minus_alphas_bar_sqrt
    alphas_bar = schedule_cfg.alphas_bar

    sampling_steps = args.sampling_steps if args.sampling_steps > 0 else args.n_steps
    print(f"  sampling: method={args.sampling_method}, steps={sampling_steps}, eta={args.eta}")

    # ---- Generate 4 modalities ----
    basename = os.path.basename(args.label_path)
    # Strip .nii.gz or .nii
    for ext in [".nii.gz", ".nii"]:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    # Strip common suffixes
    for suffix in ["_seg", "-seg", "_label", "-label"]:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break

    # Filter modalities
    if args.modality == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [args.modality]

    print(f"\n[Step 3/4] Generating {len(modalities)} modality(ies)...")

    for mod in modalities:
        print(f"\n  --- {mod} ---")

        weights_dir = os.path.join(args.diffusion_ckpt_dir, mod, "weights")
        if not os.path.isdir(weights_dir):
            print(f"  [SKIP] weights dir not found: {weights_dir}")
            continue

        model = load_diffusion_model(args.diffusion_ckpt_dir, mod, args_proxy, device)

        generated = sample_tumour_diffusion_full(
            model=model,
            label_cond=label_tensor,
            spatial_size=(96, 96, 96),
            n_steps=args.n_steps,
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

        generated_np = generated.squeeze(0).squeeze(0).cpu().numpy()  # (96, 96, 96)

        output_path = os.path.join(args.output_dir, f"{basename}-{mod}.nii.gz")
        img = nib.Nifti1Image(generated_np.astype(np.float32), np.eye(4))
        nib.save(img, output_path)
        print(f"  saved: {output_path}")

    print(f"\n[Step 4/4] Done! Generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
