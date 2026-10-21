"""Pure layout conversion between S2 nnU-Net patches and G1 Diffusion V3."""

from __future__ import annotations

import numpy as np


# S2 channels: t1n,t1c,t2w,t2f. G1 channels: t1c,t1n,t2w,t2f.
CHANNEL_SWAP = (1, 0, 2, 3)


def s2_to_g1_layout(
    image: np.ndarray, segmentation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert C,Z,Y,X S2 tensors to C,X,Y,Z G1 tensors."""
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"Expected S2 image shape (4,Z,Y,X), got {image.shape}")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise ValueError(f"Expected S2 seg shape (1,Z,Y,X), got {segmentation.shape}")
    if image.shape[1:] != segmentation.shape[1:]:
        raise ValueError(
            f"S2 image/seg shapes differ: {image.shape} vs {segmentation.shape}")
    g1_image = image[list(CHANNEL_SWAP)].transpose(0, 3, 2, 1).copy()
    g1_seg = segmentation.transpose(0, 3, 2, 1).copy()
    return g1_image, g1_seg


def g1_to_s2_layout(
    image: np.ndarray, segmentation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert C,X,Y,Z G1 tensors back to C,Z,Y,X S2 tensors."""
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"Expected G1 image shape (4,X,Y,Z), got {image.shape}")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise ValueError(f"Expected G1 seg shape (1,X,Y,Z), got {segmentation.shape}")
    if image.shape[1:] != segmentation.shape[1:]:
        raise ValueError(
            f"G1 image/seg shapes differ: {image.shape} vs {segmentation.shape}")
    s2_image = image.transpose(0, 3, 2, 1)[list(CHANNEL_SWAP)].copy()
    s2_seg = segmentation.transpose(0, 3, 2, 1).copy()
    return s2_image, s2_seg
