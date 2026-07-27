"""nnU-Net 2.8 data-loader adapter that carries exact valid masks into MET-AUG.

The base nnU-Net loader knows the random crop bbox but discards it from the
batch.  Route A needs that bbox to slice a precomputed four-modality brain mask
in the identical preprocessed coordinate system.  Recording it here avoids the
unsafe ``single modality != 0`` shortcut used by the legacy bridge.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from threadpoolctl import threadpool_limits

try:
    from .met_aug_core import (
        VALID_MASK_MANIFEST_SCHEMA,
        MetAugContractError,
        canonical_json_sha256,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        VALID_MASK_MANIFEST_SCHEMA,
        MetAugContractError,
        canonical_json_sha256,
        sha256_file,
    )


class PreprocessedValidMaskStore:
    """Verified LRU reader for the immutable valid-mask sidecars."""

    def __init__(self, manifest_path: str | Path, *, cache_size: int = 24):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
            raise MetAugContractError("unsupported MET-AUG valid-mask manifest schema")
        for key in (
            "builder_code_sha256",
            "dataset_json_sha256",
            "nnunet_plans_sha256",
            "train_file_sha256",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise MetAugContractError(f"valid-mask manifest does not bind {key}")
        expected = str(payload.get("manifest_sha256", ""))
        actual = canonical_json_sha256(payload, exclude=("manifest_sha256",))
        if not expected or expected != actual:
            raise MetAugContractError("valid-mask manifest SHA256 mismatch")
        records_path = self.manifest_path.parent / str(payload.get("records_file", ""))
        if not records_path.is_file() or sha256_file(records_path) != payload.get("records_sha256"):
            raise MetAugContractError("valid-mask records are missing or drifted")
        self.records: dict[str, dict[str, Any]] = {}
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", ""))
            if not case_id or case_id in self.records:
                raise MetAugContractError("valid-mask records have empty or duplicate case IDs")
            self.records[case_id] = row
        if int(payload.get("train_count", -1)) != len(self.records):
            raise MetAugContractError("valid-mask train count differs from record count")
        self.root = self.manifest_path.parent
        self.identity_sha256 = actual
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def load(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(case_id)
        if cached is not None:
            self._cache.move_to_end(case_id)
            return cached[0].copy(), cached[1].copy()
        row = self.records.get(case_id)
        if row is None:
            raise MetAugContractError(f"case is absent from valid-mask assets: {case_id}")
        path = (self.root / str(row["mask_path"])).resolve()
        if self.root not in path.parents or not path.is_file():
            raise MetAugContractError(f"valid-mask payload is unavailable: {path}")
        if sha256_file(path) != row.get("sha256"):
            raise MetAugContractError(f"valid-mask payload SHA256 drifted: {case_id}")
        with np.load(path, allow_pickle=False) as payload:
            valid = payload["valid_mask"].astype(bool, copy=True)
            foreground = payload["foreground_mask"].astype(bool, copy=True)
        expected_shape = tuple(int(value) for value in row["shape"])
        if valid.shape != expected_shape or foreground.shape != expected_shape:
            raise MetAugContractError(f"valid-mask payload shape drifted: {case_id}")
        self._cache[case_id] = (valid, foreground)
        self._cache.move_to_end(case_id)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return valid.copy(), foreground.copy()


def _crop_with_padding(
    mask: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    full_shape = np.asarray(mask.shape, dtype=int)
    lower_array = np.asarray(lower, dtype=int)
    upper_array = np.asarray(upper, dtype=int)
    if np.any(upper_array <= lower_array):
        raise MetAugContractError(f"invalid nnU-Net crop bbox: lower={lower}, upper={upper}")
    clipped_lower = np.maximum(lower_array, 0)
    clipped_upper = np.minimum(upper_array, full_shape)
    if np.any(clipped_upper <= clipped_lower):
        cropped = np.zeros((0, 0, 0), dtype=bool)
    else:
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(clipped_lower, clipped_upper))
        cropped = mask[slices]
    padding = tuple(
        (int(clipped_lower[index] - lower_array[index]), int(upper_array[index] - clipped_upper[index]))
        for index in range(3)
    )
    padded = np.pad(cropped, padding, mode="constant", constant_values=False)
    if padded.shape != expected_shape:
        raise MetAugContractError(
            f"valid-mask crop shape mismatch: got {padded.shape}, expected {expected_shape}"
        )
    return padded


class MetAugDataLoader(nnUNetDataLoader):
    """Route A loader that supplies sidecars before nnU-Net invokes transforms."""

    def __init__(self, *args, valid_mask_store: PreprocessedValidMaskStore, **kwargs):
        super().__init__(*args, **kwargs)
        if self.patch_size_was_2d:
            raise MetAugContractError("MET-AUG Route A supports only 3D nnU-Net patches")
        self.valid_mask_store = valid_mask_store

    def generate_train_batch(self):
        """Inject target metadata before the first per-sample transform.

        ``nnUNetDataLoader.generate_train_batch`` calls transforms inside its
        per-case loop and only collates afterward. Calling ``super`` and adding
        sidecars to its return value therefore arrives too late for Route A.
        This is the upstream 2.8 implementation with only the explicit
        sidecar payload added at that pre-transform point.
        """
        if self.transforms is None:
            raise MetAugContractError("MET-AUG Route A requires a pre-spatial transform")
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for batch_index, case_id in enumerate(selected_keys):
                    force_foreground = self.get_do_oversample(batch_index)
                    data, segmentation, previous_stage_segmentation, properties = self._data.load_case(case_id)
                    if previous_stage_segmentation is not None:
                        raise MetAugContractError("MET-AUG Route A does not support cascaded nnU-Net data")
                    if data.ndim != 4 or data.shape[0] != 4:
                        raise MetAugContractError(
                            f"MET-AUG Route A requires four-channel source data, got {data.shape}"
                        )
                    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
                        raise MetAugContractError(
                            "MET-AUG Route A requires a one-channel source segmentation"
                        )
                    lower, upper = self.get_bbox(
                        data.shape[1:], force_foreground, properties["class_locations"]
                    )
                    lower = tuple(int(value) for value in lower)
                    upper = tuple(int(value) for value in upper)
                    bbox = [[start, stop] for start, stop in zip(lower, upper)]
                    data_patch = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    segmentation_patch = torch.from_numpy(
                        crop_and_pad_nd(segmentation, bbox, -1, cast_cropped_to=np.int16)
                    ).to(torch.int16)
                    valid_mask, _foreground_mask = self.valid_mask_store.load(str(case_id))
                    valid_patch = _crop_with_padding(
                        valid_mask,
                        lower,
                        upper,
                        tuple(int(value) for value in data_patch.shape[1:]),
                    )
                    transformed = self.transforms(
                        image=data_patch,
                        segmentation=segmentation_patch,
                        met_aug_valid_mask=valid_patch,
                        met_aug_patch_origin=np.asarray(lower, dtype=np.int32),
                        met_aug_full_shape=np.asarray(valid_mask.shape, dtype=np.int32),
                        met_aug_case_id=str(case_id),
                    )
                    leaked_sidecars = {
                        "met_aug_valid_mask",
                        "met_aug_patch_origin",
                        "met_aug_full_shape",
                        "met_aug_case_id",
                    } & set(transformed)
                    if leaked_sidecars:
                        raise MetAugContractError(
                            f"MET-AUG sidecars leaked into downstream transforms: {sorted(leaked_sidecars)}"
                        )
                    data_sample = transformed["image"]
                    segmentation_sample = transformed["segmentation"]
                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[batch_index] = data_sample
                    if isinstance(segmentation_sample, list):
                        if seg_all is None:
                            seg_all = [
                                torch.empty((self.batch_size, *sample.shape), dtype=sample.dtype)
                                for sample in segmentation_sample
                            ]
                        for output_index, sample in enumerate(segmentation_sample):
                            seg_all[output_index][batch_index] = sample
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *segmentation_sample.shape),
                                dtype=segmentation_sample.dtype,
                            )
                        seg_all[batch_index] = segmentation_sample
        return {"data": data_all, "target": seg_all, "keys": selected_keys}
