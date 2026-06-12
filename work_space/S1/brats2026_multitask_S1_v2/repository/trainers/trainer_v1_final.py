import os
import sys
import yaml
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter

ROOT = "/root/autodl-tmp/brats2026/repository"

sys.path.append(f"{ROOT}/datasets")
sys.path.append(f"{ROOT}/models")
sys.path.append(f"{ROOT}/losses")

from brats_multitask_dataset import BraTSMultiTaskDataset
from multitask_unet import MultiTaskUNet
from multitask_loss import MultiTaskLoss
from dice_metric import dice_score


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default="/root/autodl-tmp/brats2026/repository/configs/multitask_v1.yaml"
)

args = parser.parse_args()

CONFIG = args.config


with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)


seed = cfg["seed"]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = "cuda"


with open(cfg["data"]["train_split"]) as f:
    train_cases = [
        x.strip()
        for x in f
    ]

with open(cfg["data"]["val_split"]) as f:
    val_cases = [
        x.strip()
        for x in f
    ]


train_ds = BraTSMultiTaskDataset(
    train_cases,
    cfg["data"]["data_root"],
    patch_size=tuple(
        cfg["train"]["patch_size"]
    ),
    train=True
)

val_ds = BraTSMultiTaskDataset(
    val_cases,
    cfg["data"]["data_root"],
    patch_size=tuple(
        cfg["train"]["patch_size"]
    ),
    train=False
)


train_loader = DataLoader(
    train_ds,
    batch_size=cfg["train"]["batch_size"],
    shuffle=True,
    num_workers=cfg["train"]["num_workers"]
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2
)


print("train =", len(train_ds))
print("val   =", len(val_ds))


model = MultiTaskUNet().to(device)

criterion = MultiTaskLoss().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["train"]["lr"]
)

scaler = GradScaler()

writer = SummaryWriter(
    cfg["logging"]["tensorboard_dir"]
)

print("trainer initialized")

import torch.nn.functional as F

CHECKPOINT_DIR = cfg["checkpoint"]["save_dir"]

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

best_loss = 1e9

start_epoch = 0

resume_path = cfg["train"]["resume"]

if resume_path != "":

    print("loading checkpoint:", resume_path)

    ckpt = torch.load(
        resume_path,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    start_epoch = ckpt["epoch"] + 1

    print(
        "resume from epoch",
        start_epoch
    )




def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    for step, batch in enumerate(train_loader):

        image = batch["image"].to(device)

        tumor = batch["tumor"].to(device)

        rc = batch["rc"].to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(image)

            loss_dict = criterion(
                outputs["tumor"],
                outputs["rc"],
                tumor,
                rc
            )

            loss = loss_dict["loss"]

            if torch.isnan(loss):

                print("\n===== NAN DETECTED =====")

                print(batch["case"])

                print(
                    "tumor unique =",
                    torch.unique(tumor)
                )

                print(
                    "rc unique =",
                    torch.unique(rc)
                )

                continue

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        if step % 5 == 0:

            print(
                f"epoch {epoch} "
                f"step {step} "
                f"loss {loss.item():.4f}"
            )

    running_loss /= len(train_loader)

    writer.add_scalar(
        "train/loss",
        running_loss,
        epoch
    )

    return running_loss


@torch.no_grad()
def validate(epoch):

    model.eval()

    val_loss = 0.0

    tumor_dice_total = 0.0

    rc_dice_total = 0.0

    for batch in val_loader:

        image = batch["image"].to(device)

        tumor = batch["tumor"].to(device)

        rc = batch["rc"].to(device)

        outputs = model(image)

        loss_dict = criterion(
            outputs["tumor"],
            outputs["rc"],
            tumor,
            rc
        )

        val_loss += loss_dict["loss"].item()

        tumor_pred = torch.argmax(
            outputs["tumor"],
            dim=1,
            keepdim=True
        )

        rc_pred = torch.argmax(
            outputs["rc"],
            dim=1,
            keepdim=True
        )

        tumor_dice_total += dice_score(
            (tumor_pred > 0),
            (tumor > 0)
        ).item()

        rc_dice_total += dice_score(
            rc_pred,
            rc
        ).item()


    val_loss /= len(val_loader)

    tumor_dice = (
        tumor_dice_total /
        len(val_loader)
    )

    rc_dice = (
        rc_dice_total /
        len(val_loader)
    )

    writer.add_scalar(
        "val/loss",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "val/tumor_dice",
        tumor_dice,
        epoch
    )

    writer.add_scalar(
        "val/rc_dice",
        rc_dice,
        epoch
    )

    print(
        f"epoch {epoch} "
        f"val_loss {val_loss:.4f} "
        f"tumor_dice {tumor_dice:.4f} "
        f"rc_dice {rc_dice:.4f}"
    )

    return val_loss


for epoch in range(
    start_epoch,
    cfg["train"]["epochs"]
):

    train_loss = train_one_epoch(epoch)

    val_loss = validate(epoch)

    latest_path = os.path.join(
        CHECKPOINT_DIR,
        "latest.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        latest_path
    )


    if val_loss < best_loss:

        best_loss = val_loss

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            },
            best_path
        )

        print(
            f"new best model "
            f"{val_loss:.4f}"
        )

writer.close()

print("training complete")
