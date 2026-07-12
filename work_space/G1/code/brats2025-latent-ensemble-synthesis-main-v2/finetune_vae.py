#!/usr/bin/env python
"""
VAE Fine-tuning for BraTS Metastasis — tumor-centered patch training

Fine-tunes MAISI AutoencoderKlMaisi on BraTS-MET data with seg-guided loss
to improve small lesion preservation in the latent space.

Usage:
    python finetune_vae.py \
        --data_csv data/data_csv.csv \
        --data_dir data/input \
        --vae_weights weights/vae/autoencoder_epoch273.pt \
        --output_dir training/vae_finetuned \
        --epochs 3 \
        --patch-size 128 128 96 \
        --tumor-patch-probability 0.8 \
        --device cuda:0

Architecture: AutoencoderKlMaisi (MONAI MAISI)
    Training input:  (1, 128, 128, 96) per modality
    Training latent: (4, 32, 32, 24)
    Training output: (1, 128, 128, 96) reconstructed

Seg labels (BraTS MET):
    0 = background / unannotated healthy brain
    1 = NETC (non-enhancing tumor core)
    2 = SNFH (surrounding non-enhancing FLAIR hyperintensity)
    3 = ET  (enhancing tumor)
    4 = RC  (resection cavity)

Loss:
    loss = mse_whole + 0.1 * mse_healthy + lambda_tumor * mse_tumor + 0.01 * kl
    where lambda_tumor = clamp(n_healthy / n_tumor, 3, 30)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from monai.bundle import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

import configs
import synthesis.utils as utils
from vae_patch_sampling import sample_synchronized_patch, validate_patch_size


def parse_args():
    p = argparse.ArgumentParser(description="VAE tumor-centered patch fine-tuning")
    p.add_argument("--data_csv", type=str, required=True,
                   help="Path to data_csv.csv with split column")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to data/input/ containing subject folders")
    p.add_argument("--vae_weights", type=str,
                   default=os.path.join(configs.PATH_WEIGHTS, "vae", "autoencoder_epoch273.pt"),
                   help="Path to pretrained VAE weights")
    p.add_argument("--output_dir", type=str, default="training/vae_finetuned",
                   help="Directory for checkpoints and logs")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Number of subjects accumulated before each optimizer step")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr_encoder", type=float, default=2e-6)
    p.add_argument("--lr_decoder", type=float, default=1e-6)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(128, 128, 96),
        metavar=("X", "Y", "Z"),
        help="Synchronized 3D training patch size; each dimension must be divisible by 4",
    )
    p.add_argument(
        "--tumor-patch-probability",
        type=float,
        default=0.8,
        help="Probability of uniformly selecting a tumor component as patch center",
    )
    p.add_argument(
        "--quick-val-subjects",
        type=int,
        default=20,
        help="Fixed seeded val subset evaluated after each epoch",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many validations without tumor-MSE improvement; 0 disables",
    )
    p.add_argument("--lambda_tumor_min", type=float, default=3.0)
    p.add_argument("--lambda_tumor_max", type=float, default=30.0)
    p.add_argument("--kl_weight", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--amp-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="CUDA autocast precision; auto prefers bfloat16 when supported",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MONAI activation checkpointing; disabled by default for patch training",
    )
    p.add_argument("--skip_initial_validation", action="store_true",
                   help="Skip the duplicate initial MSE validation when a baseline job already ran")
    p.add_argument("--max_train_subjects", type=int, default=None,
                   help="Optional deterministic subject limit for server smoke tests")
    p.add_argument("--max_val_subjects", type=int, default=None,
                   help="Optional deterministic validation limit for server smoke tests")
    p.add_argument("--dry_run", action="store_true",
                   help="Print config and exit without training")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_split_ids(csv_path, split="train"):
    """Return list of subject dicts for a given split from data_csv.csv."""
    subjects = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "train") == split:
                subjects.append(row)
    return subjects


def load_and_preprocess_image(path):
    """Load NIfTI, robust_normalize, resize_center_crop_pad to (256,256,160)."""
    img, _ = utils.load_nifti(path)
    img = utils.robust_normalize(img)
    img, _ = utils.resize_center_crop_pad(img, configs.SHAPE_PREPROCESS_IMG)
    return img


def load_and_preprocess_seg(path):
    """Load seg NIfTI, resize_center_crop_pad to (256,256,160).

    Uses nearest-neighbour behaviour: resize_center_crop_pad only crops/pads,
    no interpolation, so integer labels are preserved.
    """
    seg, _ = utils.load_nifti(path)
    seg, _ = utils.resize_center_crop_pad(seg, configs.SHAPE_PREPROCESS_IMG)
    return seg.astype(np.int16)


# ---------------------------------------------------------------------------
#  Mask construction
# ---------------------------------------------------------------------------

def build_masks(images_list, seg, threshold=0.02):
    """Build brain / tumor / healthy masks.

    Args:
        images_list: list of 4 np.ndarrays (256,256,160), already preprocessed.
        seg: np.ndarray (256,256,160), integer BraTS labels.
        threshold: intensity threshold for brain mask.

    Returns:
        brain_mask, tumor_mask, healthy_mask  (all float32)
    """
    # brain_mask from mean image intensity
    mean_img = np.mean(images_list, axis=0)
    brain_mask = (mean_img > threshold).astype(np.float32)

    # tumor_mask: all non-zero labels (1=NETC, 2=SNFH, 3=ET, 4=RC)
    tumor_mask = (seg > 0).astype(np.float32)

    # healthy brain = brain minus tumor (floor at 0)
    healthy_mask = np.clip(brain_mask - tumor_mask, 0.0, 1.0)

    return brain_mask, tumor_mask, healthy_mask


# ---------------------------------------------------------------------------
#  VAE helpers
# ---------------------------------------------------------------------------

def gaussian_kl(mu, sigma):
    """KL(q(z|x) || N(0, I)) for MONAI's (mu, sigma) encoder output."""
    sigma = sigma.clamp_min(torch.finfo(sigma.dtype).eps)
    variance = sigma.square()
    log_variance = 2.0 * torch.log(sigma)
    return 0.5 * torch.mean(mu.square() + variance - 1.0 - log_variance)


def checkpoint_blocks(module, tensor):
    """Checkpoint each MAISI block so backward does not retain a full-volume graph."""
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        return checkpoint(module, tensor, use_reentrant=False)
    for block in blocks:
        tensor = checkpoint(block, tensor, use_reentrant=False)
    return tensor


def vae_encode(vae, img_tensor, sample=True, checkpoint_activations=False):
    """Encode image using MONAI's (mu, sigma) return contract."""
    if checkpoint_activations:
        encoded = checkpoint_blocks(vae.encoder, img_tensor)
        mu = vae.quant_conv_mu(encoded)
        log_variance = vae.quant_conv_log_sigma(encoded).clamp(-30.0, 20.0)
        sigma = torch.exp(log_variance / 2)
    else:
        mu, sigma = vae.encode(img_tensor)
    z = mu + torch.randn_like(mu) * sigma if sample else mu
    kl = gaussian_kl(mu, sigma)
    return z, kl


def vae_decode(vae, z, checkpoint_activations=False):
    """Decode latent to image space."""
    if checkpoint_activations:
        z = vae.post_quant_conv(z)
        return checkpoint_blocks(vae.decoder, z)
    return vae.decode(z)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(vae, val_subjects, data_dir, device):
    """Compute reconstruction metrics on validation set."""
    vae.eval()
    metrics = {"whole_mse": [], "tumor_mse": [], "healthy_mse": [], "kl": []}
    failures = []

    for subj in tqdm(val_subjects, desc="Validating", leave=False):
        try:
            images = []
            for mod in configs.MODALITY_LIST:
                fname = os.path.basename(subj[mod])
                fpath = os.path.join(data_dir, subj["id"], fname)
                images.append(load_and_preprocess_image(fpath))

            seg_path = os.path.join(data_dir, subj["id"],
                                    os.path.basename(subj["seg"]))
            seg = load_and_preprocess_seg(seg_path)

            _, tumor_mask, healthy_mask = build_masks(images, seg)

            for i, mod in enumerate(configs.MODALITY_LIST):
                img_np = images[i]
                img_t = utils.prepare_image(img_np, vae)
                z, kl = vae_encode(vae, img_t, sample=False)
                recon_t = vae_decode(vae, z)
                recon_np = recon_t.squeeze().cpu().numpy()

                mse_whole = float(np.mean((recon_np - img_np) ** 2))
                nt = max(int(tumor_mask.sum()), 1)
                nh = max(int(healthy_mask.sum()), 1)
                mse_tumor = float(
                    ((recon_np - img_np) ** 2 * tumor_mask).sum() / nt
                )
                mse_healthy = float(
                    ((recon_np - img_np) ** 2 * healthy_mask).sum() / nh
                )

                metrics["whole_mse"].append(mse_whole)
                metrics["tumor_mse"].append(mse_tumor)
                metrics["healthy_mse"].append(mse_healthy)
                metrics["kl"].append(float(kl.item()))
        except Exception as e:
            print(f"  [WARN] validation: {subj['id']}: {e}")
            failures.append({"subject": subj["id"], "error": repr(e)})
            continue

    if failures:
        raise RuntimeError(
            f"VAE validation failed for {len(failures)}/{len(val_subjects)} subjects: "
            f"{failures[:3]}"
        )
    if any(not values for values in metrics.values()):
        raise RuntimeError("VAE validation produced no complete metric set.")
    vae.train()
    return {k: np.mean(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ----- setup -----
    os.makedirs(args.output_dir, exist_ok=True)
    args.patch_size = validate_patch_size(args.patch_size)
    if not 0.0 <= args.tumor_patch_probability <= 1.0:
        raise ValueError("--tumor-patch-probability must be in [0, 1].")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch_size must be positive.")
    if args.val_interval <= 0 or args.save_interval <= 0:
        raise ValueError("--val_interval and --save_interval must be positive.")
    if args.quick_val_subjects <= 0:
        raise ValueError("--quick-val-subjects must be positive.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    patch_rng = np.random.default_rng(args.seed)

    # ----- load data -----
    train_subjects = load_split_ids(args.data_csv, "train")
    val_subjects = load_split_ids(args.data_csv, "val")
    if args.max_train_subjects is not None:
        train_subjects = train_subjects[:args.max_train_subjects]
    if args.max_val_subjects is not None:
        val_subjects = val_subjects[:args.max_val_subjects]
    quick_val_rng = np.random.default_rng(args.seed + 10_000)
    quick_val_count = min(args.quick_val_subjects, len(val_subjects))
    quick_val_indices = quick_val_rng.choice(
        len(val_subjects), size=quick_val_count, replace=False
    )
    quick_val_subjects = [val_subjects[int(index)] for index in quick_val_indices]
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)} (quick fixed subset: {len(quick_val_subjects)})")
    print(f"Train samples per epoch (subjects x 4 modalities): "
          f"{len(train_subjects) * 4}")
    print(
        f"Training patch: {args.patch_size}; tumor/brain target probability: "
        f"{args.tumor_patch_probability:.0%}/{1.0 - args.tumor_patch_probability:.0%}"
    )

    if not train_subjects:
        raise ValueError("No rows with split='train' were found in data_csv.csv.")
    if not val_subjects:
        raise ValueError("No rows with split='val' were found in data_csv.csv.")

    quick_val_path = os.path.join(args.output_dir, "quick_val_subjects.json")
    with open(quick_val_path, "w") as f:
        json.dump(
            {
                "seed": args.seed + 10_000,
                "count": len(quick_val_subjects),
                "subject_ids": [subject["id"] for subject in quick_val_subjects],
            },
            f,
            indent=2,
        )

    if args.dry_run:
        print("\n[Dry run] Configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.amp_dtype == "auto":
        amp_name = (
            "bfloat16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        amp_name = args.amp_dtype
    if amp_name == "bfloat16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 AMP was requested, but this CUDA device does not support it.")
    amp_enabled = device.type == "cuda" and amp_name != "float32"
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    use_scaler = amp_enabled and amp_name == "float16"
    print(f"AMP dtype: {amp_name}; GradScaler: {'enabled' if use_scaler else 'disabled'}")

    # ----- load VAE -----
    print(f"\nLoading VAE from: {args.vae_weights}")
    parser = ConfigParser(configs.NETWORKS_CONFIG)
    parser.parse(True)
    vae = parser.get_parsed_content("autoencoder_def").to(device)
    chk = torch.load(args.vae_weights, weights_only=True, map_location=device)
    vae.load_state_dict(chk)
    # AutoencoderKL checkpoints the whole encoder/decoder at once. Full BraTS
    # volumes still exceed 40 GB during backward, so training checkpoints each
    # MAISI block explicitly below instead.
    vae.use_checkpoint = False
    print(f"VAE loaded ({sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params)")
    print(
        "Per-block gradient checkpointing: "
        f"{'enabled' if args.gradient_checkpointing else 'disabled'}"
    )

    # ----- optimizer (differential LRs) -----
    encoder_params = []
    decoder_params = []
    for name, param in vae.named_parameters():
        if "encoder" in name or "quant_conv_mu" in name or "quant_conv_log_sigma" in name:
            encoder_params.append(param)
        elif "decoder" in name:
            decoder_params.append(param)
        else:
            decoder_params.append(param)  # default to decoder group

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": decoder_params, "lr": args.lr_decoder},
    ], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    print(f"Optimizer: Encoder LR={args.lr_encoder}, Decoder LR={args.lr_decoder}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # ----- tensorboard -----
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # ----- compute baseline metrics -----
    baseline = {}
    if not args.skip_initial_validation:
        print("\nComputing baseline (frozen VAE) validation metrics ...")
        baseline = validate(vae, quick_val_subjects, args.data_dir, device)
        print(f"Baseline: whole_mse={baseline['whole_mse']:.6f}  "
              f"tumor_mse={baseline['tumor_mse']:.6f}  "
              f"healthy_mse={baseline['healthy_mse']:.6f}")
        with open(os.path.join(args.output_dir, "baseline_mse_metrics.json"), "w") as f:
            json.dump(baseline, f, indent=2)
    else:
        print("\nInitial MSE validation skipped; standalone baseline results are retained.")

    # ----- training -----
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    best_tumor_mse = float("inf")
    best_epoch = 0
    global_step = 0
    optimizer_steps = 0
    history = []
    completed_epochs = 0
    epochs_without_improvement = 0
    early_stop_reason = None
    total_patch_counts = {"tumor": 0, "brain": 0}
    training_started_at = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    vae.train()

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.monotonic()
        # shuffle subjects each epoch
        perm = np.random.permutation(len(train_subjects))
        epoch_losses = {"total": [], "mse_whole": [], "mse_tumor": [],
                        "mse_healthy": [], "kl": []}
        epoch_optimizer_steps = 0
        epoch_patch_counts = {"tumor": 0, "brain": 0}

        pbar = tqdm(range(0, len(train_subjects), args.batch_size),
                    desc=f"Epoch {epoch}/{args.epochs}")
        for batch_start in pbar:
            batch_subjs = [train_subjects[perm[i]]
                           for i in range(batch_start,
                                          min(batch_start + args.batch_size,
                                              len(train_subjects)))]
            if not batch_subjs:
                continue

            batch_refs = []

            # ---- load batch data ----
            for subj in batch_subjs:
                try:
                    images = []
                    for mod in configs.MODALITY_LIST:
                        fname = os.path.basename(subj[mod])
                        fpath = os.path.join(args.data_dir, subj["id"], fname)
                        images.append(load_and_preprocess_image(fpath))

                    seg_path = os.path.join(args.data_dir, subj["id"],
                                            os.path.basename(subj["seg"]))
                    seg = load_and_preprocess_seg(seg_path)
                    patch = sample_synchronized_patch(
                        images=images,
                        seg=seg,
                        patch_size=args.patch_size,
                        tumor_probability=args.tumor_patch_probability,
                        rng=patch_rng,
                    )
                    batch_refs.append(patch)
                    epoch_patch_counts[patch["mode"]] += 1
                    total_patch_counts[patch["mode"]] += 1
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load or sample training subject {subj['id']}. "
                        "All selected train subjects are required."
                    ) from e

            if not batch_refs:
                continue

            # ---- forward + backward per modality ----
            optimizer.zero_grad()
            total_loss = 0.0
            gradient_items = len(batch_refs) * len(configs.MODALITY_LIST)

            for ref in batch_refs:
                for i, mod in enumerate(configs.MODALITY_LIST):
                    img_np = ref["images"][i]
                    tmask = ref["tumor_mask"]
                    hmask = ref["healthy_mask"]

                    img_t = utils.prepare_image(img_np, vae)

                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        z, kl = vae_encode(
                            vae,
                            img_t,
                            sample=True,
                            checkpoint_activations=args.gradient_checkpointing,
                        )
                        recon_t = vae_decode(
                            vae,
                            z,
                            checkpoint_activations=args.gradient_checkpointing,
                        )

                        target_t = torch.from_numpy(img_np).to(
                            device=device, dtype=recon_t.dtype
                        ).unsqueeze(0).unsqueeze(0)
                        diff = (recon_t - target_t).square()

                        # whole mse
                        loss_whole = diff.mean()

                        # tumor mse
                        tmask_t = torch.from_numpy(tmask).to(
                            device=device, dtype=diff.dtype
                        ).unsqueeze(0).unsqueeze(0)
                        nt = max(int(tmask.sum()), 1)
                        loss_tumor = (diff * tmask_t).sum() / nt

                        # healthy mse
                        hmask_t = torch.from_numpy(hmask).to(
                            device=device, dtype=diff.dtype
                        ).unsqueeze(0).unsqueeze(0)
                        nh = max(int(hmask.sum()), 1)
                        loss_healthy = (diff * hmask_t).sum() / nh

                        # lambda_tumor (adaptive)
                        lambda_tumor = nh / nt
                        lambda_tumor = max(args.lambda_tumor_min,
                                           min(args.lambda_tumor_max, lambda_tumor))

                        # total loss
                        loss = (loss_whole
                                + 0.1 * loss_healthy
                                + lambda_tumor * loss_tumor
                                + args.kl_weight * kl)

                    normalized_loss = loss / gradient_items
                    if use_scaler:
                        scaler.scale(normalized_loss).backward()
                    else:
                        normalized_loss.backward()

                    total_loss += loss.item()

                    # track
                    epoch_losses["total"].append(loss.item())
                    epoch_losses["mse_whole"].append(loss_whole.item())
                    epoch_losses["mse_tumor"].append(loss_tumor.item())
                    epoch_losses["mse_healthy"].append(loss_healthy.item())
                    epoch_losses["kl"].append(kl.item())

                    global_step += 1

            # ---- optimizer step ----
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    vae.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise RuntimeError(
                        "GradScaler skipped an optimizer step because gradients overflowed."
                    )
            else:
                torch.nn.utils.clip_grad_norm_(
                    vae.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
            optimizer_steps += 1
            epoch_optimizer_steps += 1

            avg_in_batch = total_loss / (len(batch_refs) * 4)
            pbar.set_postfix({"loss": f"{avg_in_batch:.4f}"})

        if epoch_optimizer_steps == 0:
            raise RuntimeError(f"Epoch {epoch} completed without an optimizer step.")
        scheduler.step()

        # ---- epoch summary ----
        avg_total = np.mean(epoch_losses["total"])
        avg_whole = np.mean(epoch_losses["mse_whole"])
        avg_tumor = np.mean(epoch_losses["mse_tumor"])
        avg_healthy = np.mean(epoch_losses["mse_healthy"])
        avg_kl = np.mean(epoch_losses["kl"])

        completed_epochs = epoch
        epoch_seconds = time.monotonic() - epoch_started_at
        print(f"\nEpoch {epoch} summary — "
              f"total={avg_total:.6f}  whole={avg_whole:.6f}  "
              f"tumor={avg_tumor:.6f}  healthy={avg_healthy:.6f}  "
              f"kl={avg_kl:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"patches(tumor/brain)={epoch_patch_counts['tumor']}/"
              f"{epoch_patch_counts['brain']}  seconds={epoch_seconds:.1f}")

        writer.add_scalar("train/total", avg_total, epoch)
        writer.add_scalar("train/mse_whole", avg_whole, epoch)
        writer.add_scalar("train/mse_tumor", avg_tumor, epoch)
        writer.add_scalar("train/mse_healthy", avg_healthy, epoch)
        writer.add_scalar("train/kl", avg_kl, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ---- validation ----
        val_metrics = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_metrics = validate(vae, quick_val_subjects, args.data_dir, device)
            print(f"Val — whole_mse={val_metrics['whole_mse']:.6f}  "
                  f"tumor_mse={val_metrics['tumor_mse']:.6f}  "
                  f"healthy_mse={val_metrics['healthy_mse']:.6f}  "
                  f"kl={val_metrics['kl']:.6f}")
            writer.add_scalar("val/whole_mse", val_metrics["whole_mse"], epoch)
            writer.add_scalar("val/tumor_mse", val_metrics["tumor_mse"], epoch)
            writer.add_scalar("val/healthy_mse", val_metrics["healthy_mse"], epoch)

            # save best
            if val_metrics["tumor_mse"] < best_tumor_mse:
                best_tumor_mse = val_metrics["tumor_mse"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(vae.state_dict(), best_path)
                print(f"  -> Best model saved (tumor_mse={best_tumor_mse:.6f})")
            else:
                epochs_without_improvement += 1
                print(
                    "  -> No tumor-MSE improvement "
                    f"({epochs_without_improvement}/{args.early_stopping_patience or 'disabled'})"
                )

        history_row = {
            "epoch": epoch,
            "train_total": float(avg_total),
            "train_whole_mse": float(avg_whole),
            "train_tumor_mse": float(avg_tumor),
            "train_healthy_mse": float(avg_healthy),
            "train_kl": float(avg_kl),
            "lr_encoder": float(scheduler.get_last_lr()[0]),
            "lr_decoder": float(scheduler.get_last_lr()[1]),
            "val_whole_mse": "" if val_metrics is None else float(val_metrics["whole_mse"]),
            "val_tumor_mse": "" if val_metrics is None else float(val_metrics["tumor_mse"]),
            "val_healthy_mse": "" if val_metrics is None else float(val_metrics["healthy_mse"]),
            "val_kl": "" if val_metrics is None else float(val_metrics["kl"]),
            "tumor_patches": epoch_patch_counts["tumor"],
            "brain_patches": epoch_patch_counts["brain"],
            "epoch_seconds": float(epoch_seconds),
        }
        history.append(history_row)
        history_path = os.path.join(args.output_dir, "training_history.csv")
        with open(history_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=history_row.keys())
            writer_csv.writeheader()
            writer_csv.writerows(history)

        # ---- checkpoint ----
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_tumor_mse": best_tumor_mse,
                "baseline_metrics": baseline,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        if (
            val_metrics is not None
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            early_stop_reason = (
                f"No quick-val tumor-MSE improvement for "
                f"{epochs_without_improvement} consecutive validations."
            )
            print(f"Early stopping after epoch {epoch}: {early_stop_reason}")
            break

    # ----- done -----
    writer.close()
    training_seconds = time.monotonic() - training_started_at
    peak_cuda_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else None
    )
    peak_cuda_reserved_gib = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        if device.type == "cuda"
        else None
    )
    print(f"\nFinished. Best epoch: {best_epoch} "
          f"(validation tumor_mse: {best_tumor_mse:.6f})")
    print(f"Optimizer steps: {optimizer_steps}")
    print(f"Training seconds (including quick val): {training_seconds:.1f}")
    if peak_cuda_memory_gib is not None:
        print(
            f"Peak CUDA memory: allocated={peak_cuda_memory_gib:.2f} GiB, "
            f"reserved={peak_cuda_reserved_gib:.2f} GiB"
        )
    print(
        f"Sampled patches: tumor={total_patch_counts['tumor']}, "
        f"brain={total_patch_counts['brain']}"
    )
    print(f"Outputs in: {args.output_dir}")

    # Save final config for reproducibility
    config_out = os.path.join(args.output_dir, "finetune_config.json")
    with open(config_out, "w") as f:
        json.dump({**vars(args),
                   "best_epoch": best_epoch,
                   "completed_epochs": completed_epochs,
                   "optimizer_steps": optimizer_steps,
                   "training_seconds": training_seconds,
                   "peak_cuda_memory_gib": peak_cuda_memory_gib,
                   "peak_cuda_reserved_gib": peak_cuda_reserved_gib,
                   "quick_val_subject_ids": [
                       subject["id"] for subject in quick_val_subjects
                   ],
                   "total_patch_counts": total_patch_counts,
                   "early_stop_reason": early_stop_reason,
                   "baseline_metrics": baseline,
                   "best_tumor_mse": best_tumor_mse,
                   "timestamp": datetime.now().isoformat()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
