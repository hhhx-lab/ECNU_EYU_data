"""Reusable architecture and loss components for S2 small-lesion variants."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


A1_N_STAGES = 5
_ENCODER_STAGE_FIELDS = (
    "features_per_stage",
    "kernel_sizes",
    "strides",
    "n_conv_per_stage",
)


def truncate_architecture_kwargs_for_a1(architecture_kwargs: dict) -> dict:
    """Return the five-stage A-1 architecture without mutating the nnU-Net plan."""
    result = deepcopy(architecture_kwargs)
    original_stages = int(result.get("n_stages", 0))
    if original_stages < A1_N_STAGES:
        raise ValueError(
            f"n_stages must be at least {A1_N_STAGES}, got {original_stages}"
        )

    for field in _ENCODER_STAGE_FIELDS:
        values = result.get(field)
        if not isinstance(values, (list, tuple)) or len(values) < A1_N_STAGES:
            actual_length = len(values) if isinstance(values, (list, tuple)) else 0
            raise ValueError(
                f"{field} must contain at least {A1_N_STAGES} entries, "
                f"got {actual_length}"
            )
        result[field] = deepcopy(list(values[:A1_N_STAGES]))

    decoder_field = "n_conv_per_stage_decoder"
    decoder_values = result.get(decoder_field)
    decoder_stages = A1_N_STAGES - 1
    if not isinstance(decoder_values, (list, tuple)) or len(decoder_values) < decoder_stages:
        actual_length = (
            len(decoder_values) if isinstance(decoder_values, (list, tuple)) else 0
        )
        raise ValueError(
            f"{decoder_field} must contain at least {decoder_stages} entries, "
            f"got {actual_length}"
        )
    result[decoder_field] = deepcopy(list(decoder_values[:decoder_stages]))
    result["n_stages"] = A1_N_STAGES
    return result


def a1_deep_supervision_scales(
    pool_op_kernel_sizes: Sequence[Sequence[int]],
) -> list[list[float]]:
    """Compute scales for the four decoder outputs of a five-stage A-1 UNet."""
    if len(pool_op_kernel_sizes) < A1_N_STAGES:
        raise ValueError(
            f"pool_op_kernel_sizes must contain at least {A1_N_STAGES} entries"
        )
    dimensions = len(pool_op_kernel_sizes[0])
    cumulative = [1] * dimensions
    scales: list[list[float]] = []
    for stride in pool_op_kernel_sizes[:A1_N_STAGES]:
        if len(stride) != dimensions:
            raise ValueError("pool_op_kernel_sizes contains inconsistent dimensions")
        cumulative = [value * int(step) for value, step in zip(cumulative, stride)]
        scales.append([1.0 / value for value in cumulative])
    return scales[:-1]


class FocalCrossEntropyLoss(nn.CrossEntropyLoss):
    """Multiclass focal CE with nnU-Net target and class-weight semantics."""

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        gamma: float = 2.0,
    ) -> None:
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.gamma = float(gamma)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            if target.shape[1] != 1:
                raise ValueError("target channel dimension must be one")
            target = target[:, 0]
        target = target.long()
        valid = target != self.ignore_index
        safe_target = torch.where(valid, target, 0)

        ce = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_probabilities = F.log_softmax(input, dim=1)
        log_pt = torch.gather(
            log_probabilities, 1, safe_target.unsqueeze(1)
        ).squeeze(1)
        focal = (1.0 - log_pt.exp()).pow(self.gamma) * ce
        focal = torch.where(valid, focal, 0.0)

        if self.reduction == "none":
            return focal
        if self.reduction == "sum":
            return focal.sum()
        if not torch.any(valid):
            return focal.sum()
        if self.weight is None:
            return focal.sum() / valid.sum()
        denominator = self.weight[safe_target][valid].sum()
        return focal.sum() / denominator.clamp_min(torch.finfo(focal.dtype).eps)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_network(network: nn.Module) -> nn.Module:
    module = network.module if hasattr(network, "module") else network
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def load_matching_pretrained_weights(
    network: nn.Module,
    checkpoint_path: str | Path,
    *,
    audit_path: str | Path | None = None,
) -> dict:
    """Load only shape-compatible non-segmentation weights and record provenance."""
    source_path = Path(checkpoint_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("network_weights")
    if not isinstance(source_state, dict):
        raise ValueError(f"checkpoint has no network_weights mapping: {source_path}")

    module = _unwrap_network(network)
    target_state = module.state_dict()
    loaded: dict[str, Tensor] = {}
    skipped_shape: list[str] = []
    skipped_segmentation: list[str] = []
    skipped_missing_target: list[str] = []

    for key, source_tensor in source_state.items():
        if ".seg_layers." in key or key.startswith("seg_layers."):
            skipped_segmentation.append(key)
        elif key not in target_state:
            skipped_missing_target.append(key)
        elif target_state[key].shape != source_tensor.shape:
            skipped_shape.append(key)
        else:
            loaded[key] = source_tensor.to(
                device=target_state[key].device,
                dtype=target_state[key].dtype,
            )

    if not loaded:
        raise RuntimeError(
            f"checkpoint contains no compatible parameters for the target network: {source_path}"
        )
    target_state.update(loaded)
    module.load_state_dict(target_state)

    target_non_segmentation = [
        key
        for key in target_state
        if ".seg_layers." not in key and not key.startswith("seg_layers.")
    ]
    audit = {
        "source_checkpoint": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_parameter_keys": len(source_state),
        "target_parameter_keys": len(target_state),
        "loaded_parameter_fraction": len(loaded) / len(target_non_segmentation),
        "loaded_keys": sorted(loaded),
        "skipped_shape_keys": sorted(skipped_shape),
        "skipped_segmentation_keys": sorted(skipped_segmentation),
        "skipped_missing_target_keys": sorted(skipped_missing_target),
        "target_missing_source_keys": sorted(set(target_state) - set(source_state)),
    }
    if audit_path is not None:
        output_path = Path(audit_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    return audit
