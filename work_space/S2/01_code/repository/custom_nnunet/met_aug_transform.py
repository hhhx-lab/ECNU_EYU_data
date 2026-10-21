"""Batchgeneratorsv2 transform that applies the atomic MET-AUG Route A engine."""

from __future__ import annotations

from typing import Any

import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

try:
    import torch
except ImportError:  # pragma: no cover - runtime always includes Torch
    torch = None

try:
    from .met_aug_core import EventContext, MetAugContractError, MetAugEngine
except ImportError:
    from met_aug_core import EventContext, MetAugContractError, MetAugEngine


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise MetAugContractError("MET-AUG transform must run in the CPU main dataloader")
        return value.detach().numpy()
    raise MetAugContractError(f"unsupported MET-AUG batch type: {type(value)!r}")


def _restore_type(value: np.ndarray, template: Any) -> Any:
    if isinstance(template, np.ndarray):
        return value.astype(template.dtype, copy=False)
    if torch is not None and isinstance(template, torch.Tensor):
        return torch.from_numpy(value).to(dtype=template.dtype)
    raise MetAugContractError(f"unsupported MET-AUG batch type: {type(template)!r}")


class MetAugRouteATransform(BasicTransform):
    """Apply one Route A transaction before nnU-Net spatial/intensity transforms.

    nnU-Net 2.8 invokes ``BasicTransform`` once per cropped patch, before batch
    collation. The matching ``MetAugDataLoader`` therefore injects the target
    identity and crop sidecars into this single-sample dictionary. All sidecars
    are removed before the remaining batchgenerators transforms run.
    """

    def __init__(self, engine: MetAugEngine) -> None:
        super().__init__()
        self.engine = engine
        self.epoch = 0
        self.rank = 0
        self.worker = 0
        self._patch_counter = 0

    def set_epoch(self, epoch: int, *, rank: int = 0, worker: int = 0) -> None:
        self.epoch = int(epoch)
        self.rank = int(rank)
        self.worker = int(worker)
        self._patch_counter = 0

    def apply(self, data_dict: dict, **params) -> dict:
        del params
        image_key = "image" if "image" in data_dict else "data"
        segmentation_key = "segmentation" if "segmentation" in data_dict else "seg"
        if image_key not in data_dict or segmentation_key not in data_dict:
            raise MetAugContractError("MET-AUG batch lacks image/data or seg/segmentation")
        for key in (
            "met_aug_valid_mask",
            "met_aug_patch_origin",
            "met_aug_full_shape",
            "met_aug_case_id",
        ):
            if key not in data_dict:
                raise MetAugContractError(f"MET-AUG batch lacks required metadata key: {key}")
        image_template = data_dict[image_key]
        segmentation_template = data_dict[segmentation_key]
        images = _to_numpy(image_template)
        segmentations = _to_numpy(segmentation_template)
        valid_masks = _to_numpy(data_dict["met_aug_valid_mask"])
        patch_origins = _to_numpy(data_dict["met_aug_patch_origin"])
        full_shapes = _to_numpy(data_dict["met_aug_full_shape"])
        case_id = str(data_dict["met_aug_case_id"])
        if images.ndim != 4 or images.shape[0] != 4:
            raise MetAugContractError(f"MET-AUG expects one four-channel patch, got {images.shape}")
        if segmentations.ndim != 4 or segmentations.shape[0] != 1:
            raise MetAugContractError(f"MET-AUG expects one one-channel segmentation patch, got {segmentations.shape}")
        if (
            valid_masks.shape != tuple(images.shape[1:])
            or patch_origins.shape != (3,)
            or full_shapes.shape != (3,)
        ):
            raise MetAugContractError("MET-AUG sidecars cannot be aligned with the current patch")
        context = EventContext(
            epoch=self.epoch,
            rank=self.rank,
            worker=self.worker,
            case_id=case_id,
            patch_index=self._patch_counter,
            patch_origin=tuple(int(value) for value in patch_origins),
            full_shape=tuple(int(value) for value in full_shapes),
        )
        self._patch_counter += 1
        image, segmentation, _result = self.engine.apply(
            image=images,
            segmentation=segmentations,
            valid_mask=valid_masks.astype(bool, copy=False),
            context=context,
        )
        data_dict[image_key] = _restore_type(image, image_template)
        data_dict[segmentation_key] = _restore_type(segmentation, segmentation_template)
        data_dict.pop("met_aug_valid_mask", None)
        data_dict.pop("met_aug_patch_origin", None)
        data_dict.pop("met_aug_full_shape", None)
        data_dict.pop("met_aug_case_id", None)
        return data_dict
