"""Joint multi-head sliding-window inference for MultiTaskUNet.

Runs the shared backbone once per window and returns both tumor and RC logits.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from monai.inferers import sliding_window_inference


def multitask_predictor(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Concatenate tumor + RC logits along channel dim for MONAI SWI."""

    def _predict(window: torch.Tensor) -> torch.Tensor:
        outputs = model(window)
        return torch.cat([outputs["tumor"], outputs["rc"]], dim=1)

    return _predict


def split_multitask_logits(
    logits: torch.Tensor,
    tumor_classes: int = 4,
    rc_classes: int = 2,
) -> Dict[str, torch.Tensor]:
    if logits.shape[1] != tumor_classes + rc_classes:
        raise ValueError(
            f"Expected {tumor_classes + rc_classes} channels, got {logits.shape[1]}"
        )
    return {
        "tumor": logits[:, :tumor_classes],
        "rc": logits[:, tumor_classes : tumor_classes + rc_classes],
    }


@torch.no_grad()
def sliding_window_multitask(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    mode: str = "gaussian",
    tumor_classes: int = 4,
    rc_classes: int = 2,
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Full-volume joint inference for tumor and RC heads.

    Args:
        model: MultiTaskUNet-like module returning {"tumor", "rc"}.
        image: Tensor shaped (B, C, H, W, D) or (C, H, W, D).
        roi_size: Sliding-window ROI size.
        sw_batch_size: Windows processed per SWI step.
        overlap: Sliding-window overlap ratio.
        mode: Blending mode for overlapping windows.
        tumor_classes / rc_classes: Head channel counts used to split logits.
        device: Optional device override for the input batch.
        amp_dtype: If set, run SWI under autocast with this dtype.
    """
    model.eval()
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if device is not None:
        image = image.to(device)

    predictor = multitask_predictor(model)

    def _run() -> torch.Tensor:
        return sliding_window_inference(
            inputs=image,
            roi_size=tuple(int(v) for v in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=predictor,
            overlap=float(overlap),
            mode=mode,
        )

    if amp_dtype is not None and image.is_cuda:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = _run()
    else:
        logits = _run()

    return split_multitask_logits(
        logits,
        tumor_classes=tumor_classes,
        rc_classes=rc_classes,
    )


def logits_to_label_maps(
    outputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert joint SWI logits to integer tumor and RC maps (B, H, W, D)."""
    tumor = outputs["tumor"].argmax(dim=1)
    rc = outputs["rc"].argmax(dim=1)
    return tumor, rc
