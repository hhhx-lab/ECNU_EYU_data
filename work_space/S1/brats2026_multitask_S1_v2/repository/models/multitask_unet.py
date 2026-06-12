import torch
import torch.nn as nn

from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=64,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.tumor_head = nn.Conv3d(
            64,
            4,
            kernel_size=1,
        )

        self.rc_head = nn.Conv3d(
            64,
            2,
            kernel_size=1,
        )

    def forward(self, x):

        feat = self.backbone(x)

        tumor_logits = self.tumor_head(feat)

        rc_logits = self.rc_head(feat)

        return {
            "tumor": tumor_logits,
            "rc": rc_logits,
        }
