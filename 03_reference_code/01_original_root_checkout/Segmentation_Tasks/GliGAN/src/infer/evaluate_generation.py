"""
Evaluate generated MRI quality against real scans.

For each test case, applies the SAME crop+pad+normalize preprocessing as training
to the real scan, then compares the 96³ generated output against the 96³ preprocessed
real scan using SSIM, PSNR, MSE, and MAE.

Usage:
    cd Segmentation_Tasks/GliGAN/src/infer
    python evaluate_generation.py \
        --diffusion_ckpt_dir ../../Checkpoint/brats_2024 \
        --csv_path ../../Checkpoint/brats2024/brats2024.csv \
        --dataset BRATS_2024 \
        --output_dir ./eval_results \
        --device cuda
"""

import os
import sys
import argparse
import glob
import json

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


ALL_MODALITIES = ["t1c", "t1n", "t2w", "t2f"]


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
    """2D valid convolution."""
    h, w = img.shape
    kh, kw = kernel.shape
    if h < kh or w < kw:
        return None
    oh, ow = h - kh + 1, w - kw + 1
    result = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            result[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return result


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


# ===========================================================================
# Preprocessing: replicate training crop+pad+normalize
# ===========================================================================

def preprocess_scan(scan_path, label_path, csv_row, dataset_type, normalization="minmax"):
    """
    Apply the SAME preprocessing as GaussianNoiseTumour training transform:
    1. Load scan and label
    2. Crop to tumour bbox (with padding to make ~96)
    3. Pad to exactly 96³
    4. Normalize scan to [-1, 1]

    Returns:
        scan_crop_pad: (1, 96, 96, 96) float32 — normalized to [-1, 1]
        label_crop_pad: (C, 96, 96, 96) float32 — multi-channel label
    """
    scan = nib.load(scan_path).get_fdata().astype(np.float32)
    label_data = nib.load(label_path).get_fdata().astype(np.int16)

    # Add channel dim: (1, D, H, W)
    scan = scan[np.newaxis, ...]
    label_data = label_data[np.newaxis, ...]

    _, max_x, max_y, max_z = scan.shape

    # Bounding box from CSV
    x_min, x_max = int(csv_row["x_extreme_min"]), int(csv_row["x_extreme_max"])
    y_min, y_max = int(csv_row["y_extreme_min"]), int(csv_row["y_extreme_max"])
    z_min, z_max = int(csv_row["z_extreme_min"]), int(csv_row["z_extreme_max"])

    x_ext = x_max - x_min
    y_ext = y_max - y_min
    z_ext = z_max - z_min

    x_pad = (96 - x_ext) / 2
    y_pad = (96 - y_ext) / 2
    z_pad = (96 - z_ext) / 2

    C_x = -0.5 if x_pad < 0 else 0.5
    C_y = -0.5 if y_pad < 0 else 0.5
    C_z = -0.5 if z_pad < 0 else 0.5

    x_base = int(x_min - int(x_pad))
    x_top = int(x_max + int(x_pad + C_x))
    y_base = int(y_min - int(y_pad))
    y_top = int(y_max + int(y_pad + C_y))
    z_base = int(z_min - int(z_pad))
    z_top = int(z_max + int(z_pad + C_z))

    # Compute edge padding
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

    # Crop
    scan_crop = scan[:, x_base:x_top, y_base:y_top, z_base:z_top]
    label_crop = label_data[:, x_base:x_top, y_base:y_top, z_base:z_top]

    # Normalize scan
    if normalization == "zscore":
        # z-score then rescale to [-1, 1]
        mean = np.mean(scan_crop)
        std = np.std(scan_crop)
        if std > 0:
            scan_crop = (scan_crop - mean) / std
        z_min, z_max = np.min(scan_crop), np.max(scan_crop)
        if z_max > z_min:
            scan_crop = (scan_crop - z_min) / (z_max - z_min) * 2.0 - 1.0
    else:
        # minmax to [-1, 1]
        mina, maxa = np.min(scan_crop), np.max(scan_crop)
        if maxa > mina:
            scan_crop = (scan_crop - mina) / (maxa - mina) * 2.0 - 1.0

    # Pad to 96³
    scan_crop_pad = np.pad(
        scan_crop,
        pad_width=((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad)),
        mode="constant", constant_values=(-1, -1),
    )
    label_crop_pad = np.pad(
        label_crop,
        pad_width=((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad)),
        mode="constant", constant_values=(0, 0),
    )

    # Convert label to multi-channel
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
    parser.add_argument("--sampling_method", type=str, default="ddpm",
                        choices=["ddpm", "ddim", "edm_heun", "lognsr_ode"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--generator_type", type=str, default="SwinUNETR")
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore"])
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "t1c", "t1n", "t2w", "t2f"])
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Limit number of test cases (0=all)")
    parser.add_argument("--self_test", action="store_true",
                        help="Self-comparison mode: compare real scan with itself "
                             "(no diffusion model needed, verifies preprocessing + metrics)")
    _diffusion_utils_local.add_noise_schedule_args(parser)
    parser.add_argument("--cfg_weight", default=1.0, type=float,
                        help="CFG weight: 1.0=normal, >1=stronger conditioning (2.0-3.0 typical)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.self_test and not args.diffusion_ckpt_dir:
        parser.error("--diffusion_ckpt_dir is required (unless --self_test)")

    # Proxy args for get_diffusion_network
    class _Args:
        pass
    args_proxy = _Args()
    args_proxy.generator_type = args.generator_type
    args_proxy.feature_size = args.feature_size
    args_proxy.use_checkpoint = False
    args_proxy.out_channels = 1
    label_channels = 4 if args.dataset == "BRATS_2024" else 3
    args_proxy.in_channels = 1 + label_channels
    # Set noise embedding mode based on sampling method
    if args.sampling_method in ("edm_heun", "lognsr_ode"):
        args_proxy.noise_embedding_mode = "continuous"
    else:
        args_proxy.noise_embedding_mode = "discrete"

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load CSV
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    if args.max_cases > 0:
        df = df.head(args.max_cases)
    print(f"[CSV] {len(df)} test case(s) from {args.csv_path}")

    # Filter modalities
    if args.modality == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [args.modality]

    # Accumulate results
    all_results = {}
    per_modality_sums = {m: {"mse": [], "mae": [], "psnr": [], "ssim": []}
                         for m in modalities}

    if not args.self_test:
        # Backward compat: --beta_schedule overrides default --noise_schedule
        if hasattr(args, "beta_schedule") and args.beta_schedule != "cosine":
            args.noise_schedule = args.beta_schedule

        schedule_cfg = make_diffusion_coefficients(
            n_steps=args.n_steps, device=device,
            noise_schedule=args.noise_schedule,
            sigma_data=args.sigma_data, sigma_max=args.sigma_max,
            sigma_min=args.sigma_min, rho=args.rho,
            gamma_max=args.gamma_max, gamma_min=args.gamma_min,
            snr_shift=args.snr_shift,
        )
        # Legacy unpacking
        betas = schedule_cfg.betas
        alphas_bar_sqrt = schedule_cfg.alphas_bar_sqrt
        one_minus_alphas_bar_sqrt = schedule_cfg.one_minus_alphas_bar_sqrt
        alphas_bar = schedule_cfg.alphas_bar
        sampling_steps = args.sampling_steps if args.sampling_steps > 0 else args.n_steps

    for mod in modalities:
        print(f"\n{'='*60}")
        print(f"Evaluating modality: {mod}")

        if args.self_test:
            model = None
        else:
            weights_dir = os.path.join(args.diffusion_ckpt_dir, mod, "weights")
            ckpt_files = sorted(glob.glob(os.path.join(weights_dir, "diffusion_*.pt")))
            if not ckpt_files:
                print(f"  [SKIP] No checkpoint in: {weights_dir}")
                continue

            model = get_diffusion_network(args_proxy)
            ckpt = torch.load(ckpt_files[-1], map_location=torch.device(device))
            model.load_state_dict(ckpt.get("state_dict", ckpt))
            model.to(device)
            model.eval()
            print(f"  Model: {ckpt_files[-1]}")

        case_results = {}
        for idx, row in df.iterrows():
            case_id = row.get("id", f"case_{idx}")
            scan_path = row[f"scan_{mod}"]
            label_path = row["label"]

            # Check files exist (handle relative paths: CSV is relative to GliGAN/)
            gli_gan_root = os.path.join(os.path.dirname(__file__), "..", "..")
            scan_full = os.path.join(gli_gan_root, scan_path)
            label_full = os.path.join(gli_gan_root, label_path)
            if not os.path.isfile(scan_full):
                print(f"  [WARN] Missing scan: {scan_full}")
                continue
            if not os.path.isfile(label_full):
                print(f"  [WARN] Missing label: {label_full}")
                continue

            # Preprocess real scan
            scan_real, label_mc = preprocess_scan(
                scan_full, label_full, row, args.dataset, args.normalization)
            # scan_real: (1, 96, 96, 96), label_mc: (C, 96, 96, 96)
            real_np = scan_real.squeeze(0)  # (96, 96, 96)

            if args.self_test:
                # Self-comparison: use real scan as "generated"
                gen_np = real_np
            else:
                # Generate from label
                label_tensor = torch.from_numpy(label_mc).float().unsqueeze(0).to(device)
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
                gen_np = generated.squeeze(0).squeeze(0).cpu().numpy()  # (96, 96, 96)

            # Compute metrics
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

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = {}
    for mod in modalities:
        metrics = per_modality_sums[mod]
        if not metrics["mse"]:
            continue
        avg = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
        std = {f"{k}_std": round(float(np.std(v)), 4) for k, v in metrics.items()}
        summary[mod] = {**avg, **std}
        print(f"  {mod}: MSE={avg['mse']:.4f}±{std['mse_std']:.4f}  "
              f"MAE={avg['mae']:.4f}±{std['mae_std']:.4f}  "
              f"PSNR={avg['psnr']:.2f}±{std['psnr_std']:.2f}dB  "
              f"SSIM={avg['ssim']:.4f}±{std['ssim_std']:.4f}")

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
