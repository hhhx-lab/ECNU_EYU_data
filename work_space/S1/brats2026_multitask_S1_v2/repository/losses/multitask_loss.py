import torch
import torch.nn as nn

from monai.losses import DiceCELoss


class MultiTaskLoss(nn.Module):
    """DiceCE multi-task loss with optional uncertainty weighting.

    Uncertainty weighting can silently down-weight the harder RC branch.
    We therefore:
      1. expose effective weights every step for logging / TensorBoard
      2. clamp log-sigma so RC cannot be driven arbitrarily close to zero weight
    """

    def __init__(
        self,
        use_uncertainty: bool = True,
        fixed_tumor_weight: float = 1.0,
        fixed_rc_weight: float = 1.0,
        max_log_sigma: float = 2.0,
        min_log_sigma: float = -2.0,
    ):
        super().__init__()

        self.tumor_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.rc_loss = DiceCELoss(to_onehot_y=True, softmax=True)

        self.use_uncertainty = bool(use_uncertainty)
        self.fixed_tumor_weight = float(fixed_tumor_weight)
        self.fixed_rc_weight = float(fixed_rc_weight)
        self.max_log_sigma = float(max_log_sigma)
        self.min_log_sigma = float(min_log_sigma)

        # Homoscedastic uncertainty parameters (Kendall et al.)
        self.log_sigma_tumor = nn.Parameter(torch.zeros(1))
        self.log_sigma_rc = nn.Parameter(torch.zeros(1))

    def _clamp_log_sigma(self, log_sigma: torch.Tensor) -> torch.Tensor:
        return torch.clamp(log_sigma, min=self.min_log_sigma, max=self.max_log_sigma)

    def forward(
        self,
        tumor_logits,
        rc_logits,
        tumor_target,
        rc_target,
    ):
        lt = self.tumor_loss(tumor_logits, tumor_target)
        lr = self.rc_loss(rc_logits, rc_target)

        if self.use_uncertainty:
            log_sigma_t = self._clamp_log_sigma(self.log_sigma_tumor)
            log_sigma_r = self._clamp_log_sigma(self.log_sigma_rc)
            # precision = exp(-log_sigma); larger sigma => smaller task weight
            w_t = torch.exp(-log_sigma_t)
            w_r = torch.exp(-log_sigma_r)
            loss = w_t * lt + log_sigma_t + w_r * lr + log_sigma_r
            sigma_t = torch.exp(log_sigma_t)
            sigma_r = torch.exp(log_sigma_r)
        else:
            w_t = tumor_logits.new_tensor(self.fixed_tumor_weight)
            w_r = rc_logits.new_tensor(self.fixed_rc_weight)
            loss = w_t * lt + w_r * lr
            sigma_t = 1.0 / w_t.clamp_min(1e-8)
            sigma_r = 1.0 / w_r.clamp_min(1e-8)

        return {
            "loss": loss,
            "tumor_loss": lt.detach(),
            "rc_loss": lr.detach(),
            "weight_tumor": w_t.detach() if torch.is_tensor(w_t) else w_t,
            "weight_rc": w_r.detach() if torch.is_tensor(w_r) else w_r,
            "sigma_tumor": sigma_t.detach() if torch.is_tensor(sigma_t) else sigma_t,
            "sigma_rc": sigma_r.detach() if torch.is_tensor(sigma_r) else sigma_r,
            "log_sigma_tumor": (
                self._clamp_log_sigma(self.log_sigma_tumor).detach()
                if self.use_uncertainty
                else tumor_logits.new_zeros(1)
            ),
            "log_sigma_rc": (
                self._clamp_log_sigma(self.log_sigma_rc).detach()
                if self.use_uncertainty
                else rc_logits.new_zeros(1)
            ),
        }
