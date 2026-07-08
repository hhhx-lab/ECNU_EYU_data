import os
import sys
import yaml
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "datasets"))
sys.path.append(str(ROOT / "models"))
sys.path.append(str(ROOT / "losses"))

from brats_multitask_dataset import BraTSMultiTaskDataset
from multitask_unet import MultiTaskUNet
from multitask_loss import MultiTaskLoss
from dice_metric import dice_score


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default=str(ROOT / "configs" / "multitask_v1.yaml")
)

args = parser.parse_args()

def resolve_config_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def resolve_repo_path(path):
    if path is None or path == "":
        return ""
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


CONFIG = resolve_config_path(args.config)


with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["data_root"] = os.environ.get(
    "BRATS_TRAIN_ROOT",
    cfg["data"]["data_root"]
)
split_dir = os.environ.get("BRATS_SPLIT_DIR")
if split_dir:
    cfg["data"]["train_split"] = str(
        Path(split_dir) / Path(cfg["data"]["train_split"]).name
    )
    cfg["data"]["val_split"] = str(
        Path(split_dir) / Path(cfg["data"]["val_split"]).name
    )
cfg["data"]["train_split"] = os.environ.get(
    "BRATS_TRAIN_SPLIT",
    cfg["data"]["train_split"]
)
cfg["data"]["val_split"] = os.environ.get(
    "BRATS_VAL_SPLIT",
    cfg["data"]["val_split"]
)
cfg["data"]["data_root"] = resolve_repo_path(cfg["data"]["data_root"])
cfg["data"]["train_split"] = resolve_repo_path(cfg["data"]["train_split"])
cfg["data"]["val_split"] = resolve_repo_path(cfg["data"]["val_split"])
cfg["train"]["resume"] = resolve_repo_path(
    os.environ.get("S1_RESUME", cfg["train"].get("resume", ""))
)
cfg["checkpoint"]["save_dir"] = resolve_repo_path(
    os.environ.get("S1_CHECKPOINT_DIR", cfg["checkpoint"]["save_dir"])
)
cfg["logging"]["tensorboard_dir"] = resolve_repo_path(
    os.environ.get("S1_TENSORBOARD_DIR", cfg["logging"]["tensorboard_dir"])
)
if "inference" in cfg and "output_dir" in cfg["inference"]:
    cfg["inference"]["output_dir"] = resolve_repo_path(
        os.environ.get("S1_OUTPUT_DIR", cfg["inference"]["output_dir"])
    )


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
