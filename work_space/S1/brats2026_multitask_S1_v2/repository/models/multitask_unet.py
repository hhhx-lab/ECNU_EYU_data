import torch
import torch.nn as nn

from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feature_channels=64,
        tumor_classes=4,
        rc_classes=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ):
        super().__init__()

        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=num_res_units,
        )

        self.tumor_head = nn.Conv3d(
            feature_channels,
            tumor_classes,
            kernel_size=1,
        )

        self.rc_head = nn.Conv3d(
            feature_channels,
            rc_classes,
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


def model_kwargs_from_config(config):
    config = config or {}
    return {
        "in_channels": int(config.get("in_channels", 4)),
        "feature_channels": int(config.get("feature_channels", 64)),
        "tumor_classes": int(config.get("tumor_classes", 4)),
        "rc_classes": int(config.get("rc_classes", 2)),
        "channels": tuple(config.get("channels", (32, 64, 128, 256, 512))),
        "strides": tuple(config.get("strides", (2, 2, 2, 2))),
        "num_res_units": int(config.get("num_res_units", 2)),
    }
