#!/usr/bin/env python
"""
VAE Reconstruction Quality Comparison

Compares frozen (pretrained) vs fine-tuned VAE reconstruction quality
on the validation set. Reports per-subject SSIM and MSE for whole volume,
tumor regions, and healthy brain regions.

Usage:
    # Baseline only (before fine-tuning)
    python validate_vae_recon.py \
        --data_csv data/data_csv.csv \
        --data_dir data/input \
        --vae_weights weights/vae/autoencoder_epoch273.pt \
        --output_dir training/vae_finetuned \
        --mode baseline

    # Compare baseline vs fine-tuned
    python validate_vae_recon.py \
        --data_csv data/data_csv.csv \
        --data_dir data/input \
        --vae_weights weights/vae/autoencoder_epoch273.pt \
        --finetuned_weights training/vae_finetuned/best_model.pt \
        --output_dir training/vae_finetuned \
        --mode compare
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.losses import SSIMLoss
from PIL import Image, ImageDraw
from tqdm import tqdm

import configs
import synthesis.utils as utils

# ---------------------------------------------------------------------------
#  Data loading helpers (imported from finetune_vae.py)
# ---------------------------------------------------------------------------

def load_split_ids(csv_path, split="val"):
    subjects = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "val") == split:
                subjects.append(row)
    return subjects


def load_and_preprocess_image(path):
    img, _ = utils.load_nifti(path)
    img = utils.robust_normalize(img)
    img, _ = utils.resize_center_crop_pad(img, configs.SHAPE_PREPROCESS_IMG)
    return img


def load_and_preprocess_seg(path):
    seg, _ = utils.load_nifti(path)
    seg, _ = utils.resize_center_crop_pad(seg, configs.SHAPE_PREPROCESS_IMG)
    return seg.astype(np.int16)


def build_masks(images_list, seg, threshold=0.02):
    mean_img = np.mean(images_list, axis=0)
    brain_mask = (mean_img > threshold).astype(np.float32)
    tumor_mask = (seg > 0).astype(np.float32)
    healthy_mask = np.clip(brain_mask - tumor_mask, 0.0, 1.0)
    return brain_mask, tumor_mask, healthy_mask


# ---------------------------------------------------------------------------
#  Metric computation
# ---------------------------------------------------------------------------

def compute_mse(pred, target, mask=None):
    """Compute MSE. If mask is provided, compute on masked region only."""
    if mask is not None and mask.sum() > 0:
        diff = (pred[mask > 0] - target[mask > 0]) ** 2
        return float(diff.mean())
    return float(np.mean((pred - target) ** 2))


def compute_ssim(pred, target, mask=None, data_range=1.0, device="cpu"):
    """Compute 3D SSIM, using a tight padded ROI when a mask is provided."""
    if mask is not None and mask.sum() > 0:
        coords = np.argwhere(mask > 0)
        lower = np.maximum(coords.min(axis=0) - 8, 0)
        upper = np.minimum(coords.max(axis=0) + 9, np.asarray(mask.shape))
        roi = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        pred_s = pred[roi]
        target_s = target[roi]
    else:
        pred_s = pred
        target_s = target

    t1 = torch.from_numpy(np.asarray(pred_s, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(np.asarray(target_s, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    ssim_fn = SSIMLoss(spatial_dims=3, data_range=data_range).to(device)
    return float(1.0 - ssim_fn(t1, t2).item())


@torch.no_grad()
def reconstruct_subject(vae, images, device):
    """Encode-decode all 4 modalities for one subject. Returns list of recon npys."""
    recons = []
    kls = []
    for img_np in images:
        img_t = utils.prepare_image(img_np, vae)
        mu, sigma = vae.encode(img_t)
        sigma = sigma.clamp_min(torch.finfo(sigma.dtype).eps)
        kl = 0.5 * torch.mean(
            mu.square() + sigma.square() - 1.0 - 2.0 * torch.log(sigma)
        )
        recon_t = vae.decode(mu)
        recons.append(recon_t.squeeze().cpu().numpy())
        kls.append(float(kl.item()))
    return recons, kls


def to_uint8_slice(volume, z_index, normalize=False):
    image = np.rot90(volume[:, :, z_index])
    if normalize:
        image = image - image.min()
        denom = float(image.max())
        image = image / denom if denom > 0 else np.zeros_like(image)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def save_comparison_samples(
    val_subjects, deltas, vae_frozen, vae_ft, data_dir, output_dir, count, device
):
    """Save T2W axial montages for the worst mean tumor-SSIM deltas."""
    if count <= 0 or vae_ft is None or not deltas:
        return

    by_subject = {}
    for row in deltas:
        by_subject.setdefault(row["subject"], []).append(row["delta_tumor_SSIM"])
    worst_ids = [
        subject_id
        for subject_id, _ in sorted(
            by_subject.items(), key=lambda item: float(np.mean(item[1]))
        )[:count]
    ]
    subjects_by_id = {row["id"]: row for row in val_subjects}
    sample_dir = os.path.join(output_dir, "comparison_samples")
    os.makedirs(sample_dir, exist_ok=True)

    for subject_id in worst_ids:
        subject = subjects_by_id[subject_id]
        t2w_path = os.path.join(
            data_dir, subject_id, os.path.basename(subject["t2w"])
        )
        seg_path = os.path.join(
            data_dir, subject_id, os.path.basename(subject["seg"])
        )
        target = load_and_preprocess_image(t2w_path)
        seg = load_and_preprocess_seg(seg_path)
        frozen = reconstruct_subject(vae_frozen, [target], device)[0][0]
        finetuned = reconstruct_subject(vae_ft, [target], device)[0][0]
        tumor_per_slice = (seg > 0).sum(axis=(0, 1))
        z_index = int(np.argmax(tumor_per_slice)) if tumor_per_slice.max() > 0 else target.shape[2] // 2

        panels = [
            ("Target T2W", to_uint8_slice(target, z_index)),
            ("Pretrained", to_uint8_slice(frozen, z_index)),
            ("Fine-tuned", to_uint8_slice(finetuned, z_index)),
            ("Absolute error", to_uint8_slice(np.abs(finetuned - target), z_index, normalize=True)),
        ]
        panel_images = []
        for label, array in panels:
            panel = Image.new("L", (array.shape[1], array.shape[0] + 24), color=0)
            panel.paste(Image.fromarray(array, mode="L"), (0, 24))
            ImageDraw.Draw(panel).text((6, 6), label, fill=255)
            panel_images.append(panel)
        montage = Image.new(
            "L",
            (sum(panel.width for panel in panel_images), max(panel.height for panel in panel_images)),
            color=0,
        )
        x_offset = 0
        for panel in panel_images:
            montage.paste(panel, (x_offset, 0))
            x_offset += panel.width
        montage.save(os.path.join(sample_dir, f"{subject_id}_t2w.png"))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VAE Reconstruction Quality Comparison")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--vae_weights", type=str,
                   default=os.path.join(configs.PATH_WEIGHTS, "vae", "autoencoder_epoch273.pt"))
    p.add_argument("--finetuned_weights", type=str, default=None,
                   help="Path to fine-tuned VAE checkpoint")
    p.add_argument("--output_dir", type=str, default="training/vae_finetuned")
    p.add_argument("--mode", type=str, default="compare",
                   choices=["baseline", "compare"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_samples", type=int, default=5,
                   help="Save middle-slice PNG for worst-N subjects (0=disabled)")
    p.add_argument("--min_delta_tumor_ssim", type=float, default=0.03)
    p.add_argument("--min_delta_whole_ssim", type=float, default=-0.005)
    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Optional deterministic val-subject limit for smoke tests",
    )
    return p.parse_args()


def load_vae(weights_path, device):
    parser = ConfigParser(configs.NETWORKS_CONFIG)
    parser.parse(True)
    vae = parser.get_parsed_content("autoencoder_def").to(device)
    chk = torch.load(weights_path, weights_only=True, map_location=device)
    vae.load_state_dict(chk)
    vae.eval()
    return vae


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # load val subjects
    val_subjects = load_split_ids(args.data_csv, "val")
    if args.max_subjects is not None:
        if args.max_subjects <= 0:
            raise ValueError("--max_subjects must be positive when provided.")
        val_subjects = val_subjects[:args.max_subjects]
    print(f"Val subjects: {len(val_subjects)}")
    if not val_subjects:
        raise ValueError("No rows with split='val' were found in data_csv.csv.")

    # load frozen VAE
    print(f"Loading frozen VAE: {args.vae_weights}")
    vae_frozen = load_vae(args.vae_weights, device)

    # load fine-tuned VAE if comparing
    vae_ft = None
    finetuned_path = None
    if args.mode == "compare" and args.finetuned_weights:
        finetuned_path = args.finetuned_weights
        print(f"Loading fine-tuned VAE: {finetuned_path}")
        vae_ft = load_vae(finetuned_path, device)
    elif args.mode == "compare":
        # default: look for best_model.pt
        best_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_path):
            finetuned_path = best_path
            print(f"Loading fine-tuned VAE: {finetuned_path}")
            vae_ft = load_vae(finetuned_path, device)
        else:
            raise FileNotFoundError(
                f"Comparison requested, but fine-tuned weights were not found: {best_path}"
            )

    # metrics
    metric_names = ["whole_SSIM", "whole_MSE",
                    "tumor_SSIM", "tumor_MSE",
                    "healthy_SSIM", "healthy_MSE", "KL"]
    frozen_rows = []
    ft_rows = []
    all_deltas = []
    failures = []

    for subj in tqdm(val_subjects, desc="Evaluating"):
        try:
            # load all modalities
            images = []
            for mod in configs.MODALITY_LIST:
                fname = os.path.basename(subj[mod])
                fpath = os.path.join(args.data_dir, subj["id"], fname)
                images.append(load_and_preprocess_image(fpath))

            seg_path = os.path.join(args.data_dir, subj["id"],
                                    os.path.basename(subj["seg"]))
            seg = load_and_preprocess_seg(seg_path)
            _, tumor_mask, healthy_mask = build_masks(images, seg)

            # --- frozen VAE ---
            recons_frozen, kls_frozen = reconstruct_subject(vae_frozen, images, device)

            frozen_mod_metrics = []
            for i, mod in enumerate(configs.MODALITY_LIST):
                row = {"subject": subj["id"], "modality": mod}
                row["whole_SSIM"] = compute_ssim(recons_frozen[i], images[i], device=device)
                row["whole_MSE"] = compute_mse(recons_frozen[i], images[i])
                row["tumor_SSIM"] = compute_ssim(recons_frozen[i], images[i], tumor_mask, device=device)
                row["tumor_MSE"] = compute_mse(recons_frozen[i], images[i], tumor_mask)
                row["healthy_SSIM"] = compute_ssim(recons_frozen[i], images[i], healthy_mask, device=device)
                row["healthy_MSE"] = compute_mse(recons_frozen[i], images[i], healthy_mask)
                row["KL"] = kls_frozen[i]
                frozen_mod_metrics.append(row)

            frozen_rows.extend(frozen_mod_metrics)

            # --- fine-tuned VAE (if comparing) ---
            if vae_ft is not None:
                recons_ft, kls_ft = reconstruct_subject(vae_ft, images, device)

                ft_mod_metrics = []
                for i, mod in enumerate(configs.MODALITY_LIST):
                    row = {"subject": subj["id"], "modality": mod}
                    row["whole_SSIM"] = compute_ssim(recons_ft[i], images[i], device=device)
                    row["whole_MSE"] = compute_mse(recons_ft[i], images[i])
                    row["tumor_SSIM"] = compute_ssim(recons_ft[i], images[i], tumor_mask, device=device)
                    row["tumor_MSE"] = compute_mse(recons_ft[i], images[i], tumor_mask)
                    row["healthy_SSIM"] = compute_ssim(recons_ft[i], images[i], healthy_mask, device=device)
                    row["healthy_MSE"] = compute_mse(recons_ft[i], images[i], healthy_mask)
                    row["KL"] = kls_ft[i]
                    ft_mod_metrics.append(row)

                    # compute delta
                    delta = {
                        "subject": subj["id"],
                        "modality": mod,
                        "delta_whole_SSIM": row["whole_SSIM"] - frozen_mod_metrics[i]["whole_SSIM"],
                        "delta_tumor_SSIM": row["tumor_SSIM"] - frozen_mod_metrics[i]["tumor_SSIM"],
                        "delta_tumor_MSE": row["tumor_MSE"] - frozen_mod_metrics[i]["tumor_MSE"],
                    }
                    all_deltas.append(delta)

                ft_rows.extend(ft_mod_metrics)
        except Exception as e:
            print(f"\n  [WARN] {subj['id']}: {e}")
            failures.append({"subject": subj["id"], "error": repr(e)})
            continue

    # --- write CSVs ---
    def write_csv(rows, fname):
        if not rows:
            return
        with open(os.path.join(args.output_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    write_csv(frozen_rows, "baseline_metrics.csv")
    if ft_rows:
        write_csv(ft_rows, "finetuned_metrics.csv")

    if all_deltas:
        write_csv(all_deltas, "delta_metrics.csv")
    if failures:
        write_csv(failures, "evaluation_failures.csv")
        raise RuntimeError(
            f"VAE reconstruction evaluation failed for {len(failures)} validation subjects."
        )

    # --- summary ---
    def summarize(rows, label):
        if not rows:
            return
        print(f"\n{'=' * 50}")
        print(f"  {label}  (n={len(rows)})")
        print(f"{'=' * 50}")
        for m in ["whole_SSIM", "whole_MSE", "tumor_SSIM", "tumor_MSE",
                  "healthy_SSIM", "healthy_MSE"]:
            vals = [r[m] for r in rows]
            print(f"  {m:>14s}: {np.mean(vals):.4f}  ± {np.std(vals):.4f}  "
                  f"[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    summarize(frozen_rows, "FROZEN VAE")
    if ft_rows:
        summarize(ft_rows, "FINE-TUNED VAE")

    if all_deltas:
        print(f"\n{'=' * 50}")
        print(f"  DELTA (fine-tuned - frozen)")
        print(f"{'=' * 50}")
        for m in ["delta_whole_SSIM", "delta_tumor_SSIM", "delta_tumor_MSE"]:
            vals = [d[m] for d in all_deltas]
            n_pos = sum(1 for v in vals if v > 0)
            print(f"  {m:>18s}: mean={np.mean(vals):+.4f}  "
                  f"median={np.median(vals):+.4f}  "
                  f"positive={n_pos}/{len(vals)}")

        # top/bottom 5 by tumor_SSIM improvement
        deltas_sorted = sorted(all_deltas,
                               key=lambda d: d["delta_tumor_SSIM"])
        print(f"\n  Bottom 5 (worst degradation):")
        for d in deltas_sorted[:5]:
            print(f"    {d['subject']} {d['modality']}: "
                  f"delta_tumor_SSIM={d['delta_tumor_SSIM']:+.4f}")
        print(f"  Top 5 (best improvement):")
        for d in deltas_sorted[-5:][::-1]:
            print(f"    {d['subject']} {d['modality']}: "
                  f"delta_tumor_SSIM={d['delta_tumor_SSIM']:+.4f}")

        mean_delta_tumor_ssim = float(np.mean(
            [d["delta_tumor_SSIM"] for d in all_deltas]
        ))
        mean_delta_whole_ssim = float(np.mean(
            [d["delta_whole_SSIM"] for d in all_deltas]
        ))
        accepted = (
            mean_delta_tumor_ssim >= args.min_delta_tumor_ssim
            and mean_delta_whole_ssim >= args.min_delta_whole_ssim
        )
        selected_weights = finetuned_path if accepted else args.vae_weights
        selection = {
            "accepted": accepted,
            "decision": "use_finetuned" if accepted else "keep_pretrained",
            "selected_weights": os.path.abspath(selected_weights),
            "pretrained_weights": os.path.abspath(args.vae_weights),
            "finetuned_weights": os.path.abspath(finetuned_path),
            "validation_subjects": len(val_subjects),
            "validation_rows": len(all_deltas),
            "criteria": {
                "min_mean_delta_tumor_ssim": args.min_delta_tumor_ssim,
                "min_mean_delta_whole_ssim": args.min_delta_whole_ssim,
            },
            "observed": {
                "mean_delta_tumor_ssim": mean_delta_tumor_ssim,
                "mean_delta_whole_ssim": mean_delta_whole_ssim,
                "mean_delta_tumor_mse": float(np.mean(
                    [d["delta_tumor_MSE"] for d in all_deltas]
                )),
            },
        }
        with open(os.path.join(args.output_dir, "vae_selection.json"), "w") as f:
            json.dump(selection, f, indent=2)
        print(f"\nVAE selection: {selection['decision']}")
        print(f"Selected weights: {selection['selected_weights']}")
        save_comparison_samples(
            val_subjects,
            all_deltas,
            vae_frozen,
            vae_ft,
            args.data_dir,
            args.output_dir,
            args.save_samples,
            device,
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
