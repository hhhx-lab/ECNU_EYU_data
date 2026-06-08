import os
import argparse
import time
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import torch
from torch import nn, optim

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Resized, ToTensord, ResizeWithPadOrCropd,
)
from monai.data import CSVDataset, CacheDataset, DataLoader
from monai.data.utils import pad_list_data_collate

import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.networks.DiffusionNetwork import LabelDenoiser3D
from src.utils.crop_label import CropLabel

import importlib.util
def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_diffusion_utils = _import_from_path("diffusion_utils",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model.py"))


def create_train_loader(args, CSV_PATH, BATCH_SIZE, WORKERS, LABEL_TRANSFORM):
    col_names = ['scan_t1ce', 'label', 'center_x', 'center_y', 'center_z',
                 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max',
                 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
    col_types = {'center_x': {'type': int}, 'center_y': {'type': int},
                 'center_z': {'type': int}, 'x_extreme_min': {'type': int},
                 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int},
                 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int},
                 'z_extreme_max': {'type': int}, 'x_size': {'type': int},
                 'y_size': {'type': int}, 'z_size': {'type': int}}

    train_transforms = Compose([
        LoadImaged(keys=['label']),
        EnsureChannelFirstd(keys=["label"]),
        EnsureTyped(keys=["label"]),
        LABEL_TRANSFORM(keys="label"),
        CropLabel(keys=["label"]),
        Resized(keys=["label_crop_pad"], spatial_size=(64, 64, 64)),
        ToTensord(keys=['label_crop_pad'], dtype="float32")
    ])

    train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types)
    warnings.warn("The data loader will load all labels to memory.")

    train_ds = CacheDataset(
        data=train_CSVdataset,
        transform=train_transforms,
        cache_rate=1,
        copy_cache=False,
        progress=True,
        num_workers=WORKERS,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=WORKERS,
                              drop_last=True, shuffle=True, collate_fn=pad_list_data_collate)
    print(f"Number of elements in Train loader: {len(train_loader)}")
    return train_loader


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def create_dirs(HOME_DIR):
    for subdir in ["", "/label", "/label/weights", "/label/loss_lists", "/label/checkpoint_scans"]:
        p = HOME_DIR + subdir
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Directory {p} created")
    print("## ALL dirs set ##")


def save_sample(args, image, reality, iter_num, path, sum=False):
    if sum:
        image = torch.sum(image, axis=0).float()
    feat = np.squeeze(image.data.cpu().numpy())
    feat = nib.Nifti1Image(feat, affine=np.eye(4))
    nib.save(feat, f"{path}/label/checkpoint_scans/{iter_num}_{reality}.nii.gz")


def save_losses(loss_names, losses_lists, HOME_DIR):
    loss_dir = os.path.join(HOME_DIR, "label", "loss_lists")
    for index, loss in enumerate(loss_names):
        b = list()
        fpath = os.path.join(loss_dir, f"{loss}.txt")
        if not os.path.exists(fpath):
            Path(fpath).touch()
        if os.stat(fpath).st_size != 0:
            with open(fpath, "rb") as fp:
                b = pickle.load(fp)
                b.append(losses_lists[index][-1])
        with open(fpath, "wb") as fp:
            pickle.dump(b, fp)


def draw_curve(flag, loss_list, losses, colour, file_name, HOME_DIR):
    for idx, loss in enumerate(losses):
        plt.plot(range(len(loss_list)), loss_list, f'{colour[idx]}', label=f'{loss}')
    if flag:
        plt.ylabel('Loss')
        plt.xlabel('Iter(*1000)')
        plt.legend()
    plt.savefig(os.path.join(HOME_DIR, f'{file_name}.jpg'))
    plt.clf()
    plt.close()


def train_epoch(model, optimizer, train_loader, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in train_loader:
        real_labels = batch['label_crop_pad'].to(device)  # [B, C, 64, 64, 64]

        optimizer.zero_grad()

        # Use diffusion_loss_fn: it takes model, batch_y (target), batch_x (condition=None here)
        # But diffusion_loss_fn expects condition. For unconditional label generation, we pass None.
        # We need to adapt: create a variant without condition.
        batch_size = real_labels.shape[0]
        t = torch.randint(0, n_steps, size=(batch_size,), device=device)
        t_input = t.unsqueeze(-1)
        a = alphas_bar_sqrt[t].to(device)
        # Reshape a for broadcasting: [B, 1, 1, 1, 1]
        a = a.view(-1, 1, 1, 1, 1)
        aml = one_minus_alphas_bar_sqrt[t].to(device).view(-1, 1, 1, 1, 1)
        e = torch.randn_like(real_labels).to(device)
        y_t = real_labels * a + aml * e
        output = model(y_t, t_input.squeeze(-1))
        loss = (e - output).square().mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def __main__():
    parser = argparse.ArgumentParser(description="Label Diffusion Model Training")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers")
    parser.add_argument("--in_channels", default=3, type=int, help="Input channels (label channels)")
    parser.add_argument("--out_channels", default=3, type=int, help="Output channels")
    parser.add_argument("--total_iter", default=200000, type=int, help="Training iterations")
    parser.add_argument("--n_steps", default=1000, type=int, help="Diffusion steps")
    parser.add_argument("--beta_schedule", default="cosine", type=str, help="Beta schedule")
    parser.add_argument("--resume_iter", default=None, type=str, help="Iteration to resume")
    parser.add_argument("--csv_path", default="", type=str, help="CSV path")
    parser.add_argument("--dataset", default="", type=str, help="Dataset name")
    parser.add_argument("--optim_lr", default=2e-4, type=float, help="Learning rate")
    args = parser.parse_args()

    HOME_DIR = f"../../Checkpoint/{args.logdir}"
    create_dirs(HOME_DIR=HOME_DIR)

    # Dataset setup
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

    if args.csv_path == "":
        for file_name in os.listdir(f"../../Checkpoint/{args.logdir}"):
            if file_name.endswith("csv"):
                CSV_PATH = os.path.join(f"../../Checkpoint/{args.logdir}", file_name)
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    # Label transform
    if args.dataset == "BRATS_2023" or args.dataset == "BRATS_GOAT_2024":
        from src.utils.convert_to_multi_channel_based_on_brats_classes import \
            ConvertToMultiChannelBasedOnBratsGliomaClasses2023d as LABEL_TRANSFORM
        if int(args.in_channels) != 3:
            print("WARNING: in_channels should be 3 for this dataset")
    elif args.dataset == "BRATS_2024":
        from src.utils.convert_to_multi_channel_based_on_brats_classes import \
            ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels) != 4:
            print("WARNING: in_channels should be 4 for this dataset")
    elif args.dataset == "BRATS_2024_MENINGIOMA":
        from src.utils.convert_to_multi_channel_based_on_brats_classes import \
            ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels) != 1:
            print("WARNING: in_channels should be 1 for this dataset")

    # Create data loader
    train_loader = create_train_loader(
        args=args, CSV_PATH=CSV_PATH, BATCH_SIZE=args.batch_size,
        WORKERS=args.num_workers, LABEL_TRANSFORM=LABEL_TRANSFORM)

    # Build model
    model = LabelDenoiser3D(in_channels=args.out_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.optim_lr)

    start = 0
    if args.resume_iter is not None:
        start = int(args.resume_iter)
        ckpt_path = f"{HOME_DIR}/label/weights/diffusion_label_{start}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start = ckpt["iteration"]
            print(f"Resumed from iteration {start}")

    # Build beta schedule
    betas = _diffusion_utils.make_beta_schedule(
        schedule=args.beta_schedule, num_timesteps=args.n_steps)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - alphas_bar)

    loss_history = []
    flag = True

    TOTAL_ITER = args.total_iter
    for iteration in range(start, TOTAL_ITER + 1):
        avg_loss = train_epoch(
            model=model, optimizer=optimizer, train_loader=train_loader,
            alphas_bar_sqrt=alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
            n_steps=args.n_steps, device=device)

        loss_history.append(avg_loss)
        print(f"[{iteration}/{TOTAL_ITER}] Loss: {avg_loss:<8.6f}")

        # Save checkpoint every 1000 iterations
        if iteration % 1000 == 0:
            checkpoint = {
                "iteration": iteration,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_steps": args.n_steps,
                "beta_schedule": args.beta_schedule,
            }
            save_ckp(checkpoint, f"{HOME_DIR}/label/weights/diffusion_label_{iteration}.pt")
            print(f"Saved checkpoint at iteration {iteration}")

            loss_names = ["loss_diffusion_label"]
            losses_lists = [loss_history]
            save_losses(loss_names, losses_lists, HOME_DIR)
            draw_curve(flag=flag, loss_list=loss_history, losses=['label_diffusion'],
                       colour=['b-'], file_name="label_diffusion_loss", HOME_DIR=HOME_DIR)
            flag = False

    print("Finished training label diffusion model")


if __name__ == "__main__":
    __main__()
