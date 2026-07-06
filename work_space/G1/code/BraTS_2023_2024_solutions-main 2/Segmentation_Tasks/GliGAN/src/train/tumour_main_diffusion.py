import os
import argparse
import torch
import pickle
from time import time
import warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless backend, safe with DataLoader workers
import matplotlib.pyplot as plt


import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
# Use the provided diffusion model framework
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.utils.data_utils import get_loader
from src.networks.DiffusionNetwork import get_diffusion_network

# Import diffusion utilities from the provided model.py at repo root
import importlib.util
def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_diffusion_utils = _import_from_path("diffusion_utils",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model.py"))


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def load_ckp(args, model, optimizer):
    model_pth = f"../../Checkpoint/{args.logdir}"
    print(f"Loading model from {model_pth}")
    ckpt = torch.load(os.path.join(os.path.join(model_pth, args.modality, "weights"),
                                   f"diffusion_{args.resume_iter}.pt"))
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    model.global_step = ckpt["global_step"]
    model.epoch = ckpt["epoch"]
    print(f"Pre-trained weights loaded. Resuming from epoch {ckpt['epoch']}, step {ckpt['global_step']}")
    return model, optimizer, ckpt["epoch"], ckpt["global_step"]


def get_nets(args):
    if args.generator_type == "SwinUNETR":
        print("Using TimeConditionedSwinUNETR (Diffusion)")
    elif args.generator_type == "AttentionUnet":
        print("Using TimeConditionedAttentionUnet (Diffusion)")
    elif args.generator_type == "Unet":
        print("Using TimeConditionedUNet (Diffusion)")
    elif args.generator_type == "Unet_NnU":
        print("Using TimeConditionedUNet_NnU (Diffusion + nnU-Net planner)")
    elif args.generator_type == "PlainConvUNet":
        print("Using TimeConditionedPlainConvUNet (Diffusion + PlainConvUNet backbone)")
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")

    model = get_diffusion_network(args, n_steps=args.n_steps)
    model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.optim_lr,
                                   weight_decay=args.reg_weight, betas=(0.5, 0.999))
    return model, optimizer


def create_dirs(args, HOME_DIR):
    for subdir in ["", f"/{args.modality}", f"/{args.modality}/weights",
                   f"/{args.modality}/loss_lists", f"/{args.modality}/checkpoint_scans"]:
        p = HOME_DIR + subdir
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Directory {p} created")
    print("## ALL dirs set ##")


def save_sample(args, image, reality, iter_num, path, label=False):
    if label:
        try:
            image = image.float()
            new_image = torch.empty_like(image[0])
            TC = image[0]
            WT = image[1]
            ET = image[2]
            RC = image[3] if image.shape[0] > 3 else torch.zeros_like(image[0])
            NETC = TC - ET
            SNFH = WT - ET - NETC
            new_image[NETC > 0] = 1
            new_image[SNFH > 0] = 2
            new_image[image[2] > 0] = 3
            if image.shape[0] > 3:
                new_image[image[3] > 0] = 4
            image = new_image
        except IndexError:
            image = torch.sum(image, axis=0).float()

    feat = np.squeeze(image.data.cpu().numpy())
    feat = nib.Nifti1Image(feat, affine=np.eye(4))
    nib.save(feat, f"{path}/{args.modality}/checkpoint_scans/{iter_num}_{reality}.nii.gz")


def save_losses(args, loss_names, losses_lists, HOME_DIR, epoch=None):
    HOME_DIR = os.path.join(HOME_DIR, args.modality, "loss_lists")
    for index, loss in enumerate(loss_names):
        fpath = os.path.join(HOME_DIR, f"{loss}.txt")
        if os.path.exists(fpath) and os.stat(fpath).st_size != 0:
            with open(fpath, "rb") as fp:
                b = pickle.load(fp)
        else:
            b = list()
        val = losses_lists[index][-1]
        b.append(val)
        with open(fpath, "wb") as fp:
            pickle.dump(b, fp)
        # Human-readable log (overwrite each session, keep aligned with pickle)
        logpath = os.path.join(HOME_DIR, f"{loss}.log")
        epoch_str = f"{epoch}" if epoch is not None else str(len(b))
        with open(logpath, "w" if epoch == 1 else "a") as fp:
            fp.write(f"{epoch_str}\t{val:.6f}\n")


def draw_curve(list_iter, dic_loss, losses, colour, file_name, HOME_DIR):
    plt.figure()
    for idx, loss in enumerate(losses):
        plt.plot(list_iter, dic_loss[f'{loss}'], f'{colour[idx]}', label=f'{loss}')
    plt.legend()
    plt.savefig(os.path.join(HOME_DIR, f'{file_name}.jpg'))
    plt.close()


def train(args, global_step, train_loader, model, optimizer, scaler, schedule_cfg, HOME_DIR):
    model.train()

    loss_list = []

    x_crop_pad, y_crop_pad = None, None

    for step, batch in enumerate(train_loader):
        t1 = time()

        scan_key = f"scan_{args.modality}"  # e.g. "scan_t1c" / "scan_t1n" / "scan_t2w" / "scan_t2f"
        x_crop_pad = batch[f"{scan_key}_crop_pad"].cuda()  # clean scan
        y_crop_pad = batch["label_crop_pad"].cuda()  # label (condition)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # ---- Compute per-sample loss weights ----
            # per_crop_factor = 1 + small_lesion_weight × clamp(threshold / n_voxels, 0, clamp_max)
            # patient_scale    = f(patient_n_crops)
            # final weight     = per_crop_factor / patient_scale
            # Use crop-window max CC size if available, fall back to CSV n_voxels
            n_voxels = batch.get("effective_n_voxels")
            if n_voxels is None:
                n_voxels = batch.get("n_voxels")
            if args.small_lesion_weight > 0 and n_voxels is not None:
                n_voxels_t = n_voxels.float().to(x_crop_pad.device)
                ratio = args.small_lesion_threshold / (n_voxels_t + 1e-6)
                per_crop_factor = 1.0 + args.small_lesion_weight * torch.clamp(
                    ratio, 0.0, args.small_lesion_clamp)
            else:
                per_crop_factor = torch.ones(x_crop_pad.shape[0], device=x_crop_pad.device)

            # Patient-level balancing
            patient_n_crops = batch.get("patient_n_crops")
            if args.patient_balance_mode != "none" and patient_n_crops is not None:
                patient_n_crops_t = patient_n_crops.float().to(x_crop_pad.device)
                if args.patient_balance_mode == "divide":
                    patient_scale = patient_n_crops_t
                else:  # sqrt
                    patient_scale = torch.sqrt(patient_n_crops_t)
                sample_weights = per_crop_factor / patient_scale
            else:
                sample_weights = per_crop_factor

            # ---- Loss computation ----
            if schedule_cfg.name in ("edm",):
                loss = _diffusion_utils.edm_loss_fn(
                    model=model, x_0=x_crop_pad, cond=y_crop_pad,
                    schedule_cfg=schedule_cfg, device=x_crop_pad.device,
                    p_uncond=args.p_uncond, sample_weights=sample_weights)
            elif schedule_cfg.name in ("lognsr",):
                loss = _diffusion_utils.lognsr_loss_fn(
                    model=model, x_0=x_crop_pad, cond=y_crop_pad,
                    schedule_cfg=schedule_cfg, device=x_crop_pad.device,
                    p_uncond=args.p_uncond, sample_weights=sample_weights)
            else:
                loss = _diffusion_utils.diffusion_loss_fn(
                    model=model,
                    batch_y=x_crop_pad,
                    batch_x=y_crop_pad,
                    alphas_bar_sqrt=schedule_cfg.alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt=schedule_cfg.one_minus_alphas_bar_sqrt,
                    n_steps=schedule_cfg.n_steps,
                    device=x_crop_pad.device,
                    p_uncond=args.p_uncond,
                    sample_weights=sample_weights,
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_list.append(loss.item())

        print("Step:{}/{}, Loss:{:.6f}, Time:{:.4f}"
              .format(global_step, args.num_steps, loss.item(), time() - t1))

        if global_step >= args.num_steps:
            break
        global_step += 1

    return global_step, model, optimizer, loss_list, x_crop_pad, y_crop_pad


def __main__():
    parser = argparse.ArgumentParser(description="Tumour Diffusion Model Training")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save the experiment")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers")
    parser.add_argument("--in_channels", default=None, type=int,
                        help="Number of input channels (auto: BRATS_2023=4, BRATS_2024=5). "
                             "If set, overrides auto-detection.")
    parser.add_argument("--out_channels", default=1, type=int, help="Number of output channels")
    parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--use_compile", action="store_true",
                        help="Enable torch.compile for the model (PyTorch >= 2.0, best on A100+)")
    parser.add_argument("--optim_lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="Regularization weight")
    parser.add_argument("--num_steps", default=100000, type=int, help="Number of training iterations")
    parser.add_argument("--n_steps", default=1000, type=int, help="Number of diffusion steps")
    parser.add_argument("--beta_schedule", default=None, type=str, help="[DEPRECATED] Use --noise_schedule instead")
    parser.add_argument("--resume_iter", default=None, type=str, help="Iteration number to resume")
    parser.add_argument("--noise_type", default="gaussian_tumour", type=str, help="Type of noise")
    parser.add_argument("--generator_type", default="Unet_NnU", type=str, help="Backbone type")
    parser.add_argument("--time_ch_count", default=8, type=int,
                        help="Number of time embedding channels to concatenate to input (default: 8)")
    parser.add_argument("--modality", default="t1c", type=str, help="Modality to train")
    parser.add_argument("--csv_path", default="", type=str, help="Path to CSV")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"],
                        help="CSV split to load for training (default: train)")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore"],
                        help="Scan normalization method (default: minmax)")
    parser.add_argument("--crop_size", default=64, type=int,
                        help="Crop/pad target size (64=default, 96=glioma, 48=tiny lesions)")
    parser.add_argument("--small_lesion_weight", default=3.0, type=float,
                        help="Small lesion loss weight factor. 0=off. "
                             "Loss is multiplied by 1 + weight * clamp(threshold / n_voxels, 0, clamp).")
    parser.add_argument("--small_lesion_threshold", default=27.0, type=float,
                        help="Voxel threshold: lesions smaller than this get full weight boost")
    parser.add_argument("--small_lesion_clamp", default=1.0, type=float,
                        help="Upper clamp for threshold/n_voxels ratio. 1=conservative, 2-3=amplify tiny")
    parser.add_argument("--patient_balance_mode", default="none", type=str,
                        choices=["none", "divide", "sqrt"],
                        help="Patient-level balance: none (off), divide (loss/n_crops), "
                             "sqrt (loss/sqrt(n_crops))")
    _diffusion_utils.add_noise_schedule_args(parser)
    parser.add_argument("--p_uncond", default=0.0, type=float,
                        help="CFG: condition dropout probability during training (0=off)")
    args = parser.parse_args()

    HOME_DIR = f"../../Checkpoint/{args.logdir}"
    create_dirs(args, HOME_DIR=HOME_DIR)

    # Normalize dataset name
    dl = args.dataset.lower()
    if "2024" in dl and "goat" in dl and "brats" in dl:
        args.dataset = "BRATS_GOAT_2024"
    elif "brats" in dl and "2024" in dl and "goat" not in dl and "meningioma" not in dl:
        args.dataset = "BRATS_2024"
    elif "brats" in dl and "2023" in dl and "goat" not in dl and "meningioma" not in dl:
        args.dataset = "BRATS_2023"
    elif "brats" in dl and "meningioma" in dl:
        args.dataset = "BRATS_2024_MENINGIOMA"
    else:
        raise ValueError("Unknown dataset")

    # Auto-detect in_channels from dataset if not specified
    if args.in_channels is None:
        if args.dataset == "BRATS_2024":
            args.in_channels = 5  # 1 scan + 4 label channels
        elif args.dataset == "BRATS_2024_MENINGIOMA":
            args.in_channels = 2  # 1 scan + 1 label channel
        else:
            args.in_channels = 4  # 1 scan + 3 label channels (BRATS_2023 / GOAT)
        print(f"[Auto] in_channels={args.in_channels} (detected from {args.dataset})")

    global_step = 0
    epoch = 0

    # Backward compat: --beta_schedule (if explicitly set) overrides --noise_schedule
    if getattr(args, "beta_schedule", None) is not None:
        args.noise_schedule = args.beta_schedule

    # Build noise schedule (legacy beta / EDM / logsnr) BEFORE get_nets()
    # because get_nets() reads args.noise_embedding_mode
    schedule_cfg = _diffusion_utils.make_noise_schedule(
        schedule=args.noise_schedule, n_steps=args.n_steps,
        beta_start=args.beta_start, beta_end=args.beta_end,
        sigma_data=args.sigma_data, sigma_max=args.sigma_max,
        sigma_min=args.sigma_min, rho=args.rho,
        gamma_max=args.gamma_max, gamma_min=args.gamma_min,
        snr_shift=args.snr_shift,
    )
    args.noise_embedding_mode = "continuous" if schedule_cfg.name in ("edm", "lognsr") else "discrete"
    print(f"[Schedule] {schedule_cfg.name}, embedding_mode={args.noise_embedding_mode}")

    model, optimizer = get_nets(args)

    # AMP: gradient scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # torch.compile: kernel fusion optimization (PyTorch >= 2.0)
    if args.use_compile:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("[Optimization] torch.compile enabled for model")
        else:
            print("[Warning] torch.compile not available, skipping (requires PyTorch >= 2.0)")

    if args.resume_iter is not None:
        global_step = int(args.resume_iter)
        model, optimizer, epoch, global_step = load_ckp(args, model, optimizer)
    # ---- Ensure CSV has patient_n_crops column for patient-level balancing ----
    import pandas as pd
    if args.csv_path == "":
        csv_dir = f"../../Checkpoint/{args.logdir}"
        csv_file = None
        for fn in os.listdir(csv_dir):
            if fn.endswith("csv"):
                csv_file = os.path.join(csv_dir, fn)
                break
    else:
        csv_file = args.csv_path
    if csv_file and os.path.isfile(csv_file):
        df_csv = pd.read_csv(csv_file)
        if "patient_id" not in df_csv.columns:
            raise ValueError(f"CSV missing required column 'patient_id': {csv_file}")
        patient_counts = df_csv.groupby("patient_id").size()
        expected_counts = df_csv["patient_id"].map(patient_counts).astype(int)
        needs_update = (
            "patient_n_crops" not in df_csv.columns
            or not df_csv["patient_n_crops"].astype(int).equals(expected_counts)
        )
        if needs_update:
            patient_counts = df_csv.groupby("patient_id").size()
            df_csv["patient_n_crops"] = expected_counts
            df_csv.to_csv(csv_file, index=False)
            print(f"[CSV] Rebuilt patient_n_crops column ({len(patient_counts)} patients)")
    # ---------------------------------------------------------------------------

    train_loader = get_loader(args=args)

    dic_loss = {'loss': []}
    list_iter = []

    while global_step < args.num_steps:
        epoch += 1
        global_step, model, optimizer, loss_list, x_crop_pad, y_crop_pad = train(
            args=args, global_step=global_step, train_loader=train_loader,
            model=model, optimizer=optimizer, scaler=scaler,
            schedule_cfg=schedule_cfg, HOME_DIR=HOME_DIR)

        # Save sample visualizations
        save_sample(args=args, image=x_crop_pad[0], reality="x_crop_pad",
                    iter_num=epoch, path=HOME_DIR)
        save_sample(args=args, image=y_crop_pad[0], reality="y_crop_pad",
                    iter_num=epoch, path=HOME_DIR, label=True)

        dic_loss['loss'].append(np.mean(loss_list))
        losses_lists = [dic_loss['loss']]
        loss_names = ["loss_diffusion"]
        save_losses(args=args, loss_names=loss_names, losses_lists=losses_lists, HOME_DIR=HOME_DIR, epoch=epoch)
        list_iter.append(epoch)
        draw_curve(list_iter=list_iter, dic_loss=dic_loss,
                   losses=['loss'], colour=['b-'],
                   file_name=f"{args.modality}_diffusion_train_loss", HOME_DIR=HOME_DIR)

        # Save checkpoint
        if (epoch % 10 == 0) or (global_step >= args.num_steps):
            if global_step >= args.num_steps:
                print(f"LAST SAVE. global_step: {global_step}")
            checkpoint = {
                "global_step": global_step,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_steps": args.n_steps,
                "noise_schedule": schedule_cfg.name,
                "noise_embedding_mode": args.noise_embedding_mode,
                "time_ch_count": args.time_ch_count,
                "schedule_config": {
                    k: v for k, v in schedule_cfg.__dict__.items()
                    if v is not None and k not in ("betas", "alphas_bar_sqrt",
                        "one_minus_alphas_bar_sqrt", "alphas_bar")
                },
                "p_uncond": args.p_uncond,
            }
            save_ckp(checkpoint, f"{HOME_DIR}/{args.modality}/weights/diffusion_{global_step}.pt")
            print(f"Saved in: {HOME_DIR}/{args.modality}/weights/diffusion_{global_step}.pt")


if __name__ == "__main__":
    __main__()
