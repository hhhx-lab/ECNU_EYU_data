import torch
import torch.nn as nn

from monai.losses import DiceCELoss


class MultiTaskLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.tumor_loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True
        )

        self.rc_loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True
        )

        # V5 uncertainty weighting

        self.log_sigma_tumor = nn.Parameter(
            torch.zeros(1)
        )

        self.log_sigma_rc = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        tumor_logits,
        rc_logits,
        tumor_target,
        rc_target
    ):

        lt = self.tumor_loss(
            tumor_logits,
            tumor_target
        )

        lr = self.rc_loss(
            rc_logits,
            rc_target
        )

        loss = (
            torch.exp(-self.log_sigma_tumor) * lt
            + self.log_sigma_tumor
            + torch.exp(-self.log_sigma_rc) * lr
            + self.log_sigma_rc
        )

        return {
            "loss": loss,
            "tumor_loss": lt.detach(),
            "rc_loss": lr.detach(),
            "sigma_tumor":
                torch.exp(
                    self.log_sigma_tumor
                ).detach(),
            "sigma_rc":
                torch.exp(
                    self.log_sigma_rc
                ).detach()
        }
