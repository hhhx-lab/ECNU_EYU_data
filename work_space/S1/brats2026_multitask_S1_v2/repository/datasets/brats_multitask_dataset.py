from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CenterSpatialCropd,
    SpatialPadd,
)


class BraTSMultiTaskDataset:

    def __init__(
        self,
        case_list,
        data_root,
        patch_size=(96,96,96),
        train=True
    ):

        self.case_list = case_list

        self.data_root = Path(data_root)

        self.case_dirs = {}

        for p in self.data_root.rglob("BraTS-MET-*"):

            if p.is_dir():

                self.case_dirs[p.name] = p

        self.train = train

        if train:

            self.transform = Compose(
                [
                    SpatialPadd(
                        keys=[
                            "image",
                            "tumor",
                            "rc"
                        ],
                        spatial_size=patch_size
                    ),

                    RandCropByPosNegLabeld(
                        keys=[
                            "image",
                            "tumor",
                            "rc"
                        ],
                        label_key="tumor",
                        spatial_size=patch_size,
                        pos=1,
                        neg=1,
                        num_samples=1
                    )
                ]
            )

        else:

            self.transform = Compose(
                [
                    SpatialPadd(
                        keys=[
                            "image",
                            "tumor",
                            "rc"
                        ],
                        spatial_size=patch_size
                    ),

                    CenterSpatialCropd(
                        keys=[
                            "image",
                            "tumor",
                            "rc"
                        ],
                        roi_size=patch_size
                    )
                ]
            )

    def __len__(self):

        return len(self.case_list)

    def __getitem__(self, idx):

        case = self.case_list[idx]

        case_dir = self.case_dirs[case]

        image = []

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

            image.append(arr)

        image = np.stack(image)

        tumor = nib.load(
            case_dir /
            "tumor_label.nii.gz"
        ).get_fdata()

        rc = nib.load(
            case_dir /
            "rc_label.nii.gz"
        ).get_fdata()

        sample = {

            "image":
                image.astype(np.float32),

            "tumor":
                tumor[None].astype(np.int64),

            "rc":
                rc[None].astype(np.int64),
        }

        sample = self.transform(sample)

        if isinstance(sample, list):
            sample = sample[0]

        sample["image"] = sample["image"].float()

        sample["tumor"] = sample["tumor"].long()

        sample["rc"] = sample["rc"].long()

        sample["case"] = case

        return sample
