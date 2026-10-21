"""Single-process nnU-Net transform backed by the frozen G1 diffusion models."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

try:
    from .online_diffusion_contract import g1_to_s2_layout, s2_to_g1_layout
except ImportError:
    from online_diffusion_contract import g1_to_s2_layout, s2_to_g1_layout


class OnlineDiffusionTransform(BasicTransform):
    def __init__(
        self,
        g1_code_dir: Path,
        checkpoint_dir: Path,
        label_pool_paths: list[str],
        checkpoint_steps: dict[str, int],
        sampling_steps: int,
        augment_probability: float,
        second_tumour_probability: float,
        max_tumours: int,
        device: str,
        report_every: int = 25,
    ) -> None:
        super().__init__()
        infer_dir = g1_code_dir / "src" / "infer"
        for path in (g1_code_dir, infer_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from on_the_fly_augmentation import OnTheFlyTumourAugmenter

        self.augmenter = OnTheFlyTumourAugmenter(
            diffusion_ckpt_dir=str(checkpoint_dir),
            label_pool_paths=label_pool_paths,
            dataset_type="BRATS_2024",
            sampling_steps=sampling_steps,
            sampling_method="edm_heun",
            device=device,
            generator_type="Unet_NnU",
            normalization="zscore",
            crop_size=64,
            checkpoint_steps=checkpoint_steps,
            augment_probability=augment_probability,
            second_tumour_probability=second_tumour_probability,
            max_tumours=max_tumours,
        )
        self.report_every = report_every
        self.calls = 0
        self.modified = 0
        self.total_seconds = 0.0

    def apply(self, data_dict: dict, **params) -> dict:
        image = data_dict.get("image")
        segmentation = data_dict.get("segmentation")
        if image is None or segmentation is None:
            raise ValueError("Online diffusion requires image and segmentation")
        if image.device.type != "cpu" or segmentation.device.type != "cpu":
            raise ValueError("Online diffusion transform must run in the main CPU dataloader")

        image_np = image.detach().numpy().astype(np.float32, copy=True)
        seg_np = segmentation.detach().numpy().astype(np.int16, copy=True)
        g1_image, g1_seg = s2_to_g1_layout(image_np, seg_np)

        started = time.perf_counter()
        with torch.no_grad():
            g1_image, g1_seg, was_modified = self.augmenter.augment_sample(
                g1_image, g1_seg, np.random)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        self.calls += 1
        self.modified += int(was_modified)
        self.total_seconds += elapsed
        if was_modified:
            s2_image, s2_seg = g1_to_s2_layout(g1_image, g1_seg)
            data_dict["image"] = torch.from_numpy(s2_image).to(dtype=image.dtype)
            data_dict["segmentation"] = torch.from_numpy(s2_seg).to(
                dtype=segmentation.dtype)
        if self.calls % self.report_every == 0:
            print(
                "ONLINE_DIFFUSION_STATS "
                f"calls={self.calls} modified={self.modified} "
                f"mean_seconds={self.total_seconds / self.calls:.3f}",
                flush=True,
            )
        return data_dict
