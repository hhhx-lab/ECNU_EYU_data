import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai.inferers import sliding_window_inference

import sys
sys.path.append(
    "/root/autodl-tmp/brats2026/repository"
)

from models.multitask_unet import MultiTaskUNet


def load_case(case_dir):

    case_dir = Path(case_dir)

    case = case_dir.name

    mods = []

    for mod in [
        "t1c",
        "t1n",
        "t2f",
        "t2w"
    ]:

        arr = nib.load(
            case_dir /
            f"{case}-{mod}.nii.gz"
        ).get_fdata()

        mods.append(arr)

    image = np.stack(mods)

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return image


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--case_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    device = "cuda"

    model = MultiTaskUNet().to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    image = load_case(
        args.case_dir
    ).to(device)

    with torch.no_grad():

        tumor_logits = sliding_window_inference(
            image,
            roi_size=(96,96,96),
            sw_batch_size=1,
            predictor=lambda x:
                model(x)["tumor"]
        )

        rc_logits = sliding_window_inference(
            image,
            roi_size=(96,96,96),
            sw_batch_size=1,
            predictor=lambda x:
                model(x)["rc"]
        )

    tumor = (
        tumor_logits
        .argmax(1)
        .cpu()
        .numpy()[0]
    )

    rc = (
        rc_logits
        .argmax(1)
        .cpu()
        .numpy()[0]
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    nib.save(
        nib.Nifti1Image(
            tumor.astype(np.uint8),
            np.eye(4)
        ),
        output_dir /
        "tumor_pred.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            rc.astype(np.uint8),
            np.eye(4)
        ),
        output_dir /
        "rc_pred.nii.gz"
    )

    print("prediction saved")


if __name__ == "__main__":
    main()
