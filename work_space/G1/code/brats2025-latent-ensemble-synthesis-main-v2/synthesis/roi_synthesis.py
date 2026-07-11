"""
Per-Lesion ROI Synthesis Pipeline (Inference-Only)

Detects individual lesions from the BraTS seg, merges overlapping ROIs,
runs EncDec + BBDM synthesis on each ROI (pad-to-cube, no zoom), and
feather-blends the enhanced ROIs back into the full-image ensemble result.

This is a parallel pipeline to the existing full-image synthesis — both
can coexist and the user selects which to use at inference time.

Usage (from evaluate.py or main.py with --per_lesion flag):
    from synthesis.roi_synthesis import run_per_lesion_synthesis
    final_img = run_per_lesion_synthesis(s_data, synthesis_type, vae, device)
"""

import os
import sys

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

import configs
import synthesis.utils as utils
import synthesis.pipeline as pipeline


# ---------------------------------------------------------------------------
#  Lesion detection and ROI management
# ---------------------------------------------------------------------------

def detect_lesions(seg, min_volume=8):
    """Find individual lesions via connected components.

    Args:
        seg: (256,256,160) integer BraTS seg.
        min_volume: minimum voxel count for a lesion to be considered.

    Returns:
        list of dicts: [{"label": int, "volume": int,
                         "bbox": (x1,x2,y1,y2,z1,z2)}, ...]
    """
    tumor_mask = (seg > 0).astype(np.int32)
    labeled, num_features = ndimage.label(tumor_mask)

    lesions = []
    for label_id in range(1, num_features + 1):
        vol = int(np.sum(labeled == label_id))
        if vol < min_volume:
            continue
        coords = np.argwhere(labeled == label_id)
        x1, x2 = coords[:, 0].min(), coords[:, 0].max()
        y1, y2 = coords[:, 1].min(), coords[:, 1].max()
        z1, z2 = coords[:, 2].min(), coords[:, 2].max()
        lesions.append({
            "label": label_id,
            "volume": vol,
            "bbox": (x1, x2, y1, y2, z1, z2),
        })
    return lesions


def merge_overlapping_bboxes(bboxes, margin=32, max_dim=256):
    """Merge bboxes that overlap after adding margin.

    Args:
        bboxes: list of (x1,x2,y1,y2,z1,z2) tuples.
        margin: voxels to expand each bbox before overlap check.
        max_dim: maximum image dimension for clipping.

    Returns:
        list of merged (x1,x2,y1,y2,z1,z2) tuples.
    """
    if not bboxes:
        return []

    # Expand each bbox with margin
    expanded = []
    for bbox in bboxes:
        x1, x2, y1, y2, z1, z2 = bbox
        expanded.append([
            max(0, x1 - margin), min(max_dim - 1, x2 + margin),
            max(0, y1 - margin), min(max_dim - 1, y2 + margin),
            max(0, z1 - margin), min(max_dim - 1, z2 + margin),
        ])

    # Greedy merge: sort by x1, merge overlapping
    expanded.sort(key=lambda b: b[0])
    merged = [expanded[0]]

    for bbox in expanded[1:]:
        last = merged[-1]
        # Check 3D overlap
        overlap_x = last[0] <= bbox[1] and last[1] >= bbox[0]
        overlap_y = last[2] <= bbox[3] and last[3] >= bbox[2]
        overlap_z = last[4] <= bbox[5] and last[5] >= bbox[4]
        if overlap_x and overlap_y and overlap_z:
            merged[-1] = [
                min(last[0], bbox[0]), max(last[1], bbox[1]),
                min(last[2], bbox[2]), max(last[3], bbox[3]),
                min(last[4], bbox[4]), max(last[5], bbox[5]),
            ]
        else:
            merged.append(bbox)

    return [tuple(m) for m in merged]


# ---------------------------------------------------------------------------
#  ROI extraction, padding, and blending
# ---------------------------------------------------------------------------

def extract_roi(img, bbox, margin=0):
    """Extract a region of interest from a 3D image.

    Args:
        img: 3D np.ndarray.
        bbox: (x1,x2,y1,y2,z1,z2).
        margin: additional voxels around the bbox.

    Returns:
        (roi, src_slice_tuple) — the cropped array and the slice used.
    """
    x1, x2, y1, y2, z1, z2 = bbox
    x1 = max(0, x1 - margin)
    x2 = min(img.shape[0], x2 + margin + 1)
    y1 = max(0, y1 - margin)
    y2 = min(img.shape[1], y2 + margin + 1)
    z1 = max(0, z1 - margin)
    z2 = min(img.shape[2], z2 + margin + 1)

    src_slice = (slice(x1, x2), slice(y1, y2), slice(z1, z2))
    return img[src_slice].copy(), src_slice


def pad_to_cube(roi, min_size=64, pad_value=0.0):
    """Zero-pad a 3D array to a cube >= min_size.

    Args:
        roi: 3D np.ndarray.
        min_size: minimum edge length.
        pad_value: fill value for padding.

    Returns:
        (padded_roi, pad_slices) where pad_slices is (x_slice, y_slice, z_slice)
        that maps the original ROI into the padded cube.
    """
    dx, dy, dz = roi.shape
    target = max(dx, dy, dz, min_size)

    # Compute pad amounts: center the ROI
    pad_x_before = (target - dx) // 2
    pad_x_after = target - dx - pad_x_before
    pad_y_before = (target - dy) // 2
    pad_y_after = target - dy - pad_y_before
    pad_z_before = (target - dz) // 2
    pad_z_after = target - dz - pad_z_before

    padded = np.full((target, target, target), pad_value, dtype=roi.dtype)
    x_slice = slice(pad_x_before, pad_x_before + dx)
    y_slice = slice(pad_y_before, pad_y_before + dy)
    z_slice = slice(pad_z_before, pad_z_before + dz)
    padded[x_slice, y_slice, z_slice] = roi

    return padded, (x_slice, y_slice, z_slice)


def create_feather_mask(shape, blend_width):
    """Create a 3D feathering mask that ramps from 1 (center) to 0 (edge).

    Args:
        shape: (dx, dy, dz) of the ROI.
        blend_width: voxels of feathering at each edge.

    Returns:
        3D np.float32 mask in [0, 1].
    """
    dx, dy, dz = shape
    mask = np.ones(shape, dtype=np.float32)

    for dim, size in enumerate([dx, dy, dz]):
        ramp = np.ones(size, dtype=np.float32)
        if size > 2 * blend_width:
            ramp[:blend_width] = np.linspace(0, 1, blend_width)
            ramp[-blend_width:] = np.linspace(1, 0, blend_width)

        # Broadcast to 3D
        shape_1d = [1, 1, 1]
        shape_1d[dim] = size
        mask *= ramp.reshape(shape_1d)

    return mask


def feather_blend(canvas, patch, dst_bbox, blend_width=16):
    """Blend a patch into a canvas using feathering.

    Args:
        canvas: (dx, dy, dz) full-image canvas.
        patch: (px, py, pz) the synthesized ROI.
        dst_bbox: (x1,x2,y1,y2,z1,z2) destination in canvas coordinates.
        blend_width: voxels of feathering.

    Returns:
        canvas with patch blended in (modified in-place).
    """
    x1, x2, y1, y2, z1, z2 = dst_bbox
    px, py, pz = patch.shape
    cx, cy, cz = x2 - x1, y2 - y1, z2 - z1

    # safety check: patch should match bbox size
    if (px, py, pz) != (cx, cy, cz):
        # crop patch or adjust
        px = min(px, cx)
        py = min(py, cy)
        pz = min(pz, cz)
        patch = patch[:px, :py, :pz]

    dst_slice = (slice(x1, x1 + px), slice(y1, y1 + py), slice(z1, z1 + pz))

    feather = create_feather_mask(patch.shape, blend_width)
    canvas[dst_slice] = (canvas[dst_slice] * (1 - feather) +
                         patch * feather)
    return canvas


# ---------------------------------------------------------------------------
#  ROI synthesis core
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_roi(roi_images, unet_encdec, unet_bbdm, conditions_model,
                   noise_scheduler, vae, device, roi_desc=""):
    """Run EncDec + BBDM synthesis on a single padded ROI.

    Args:
        roi_images: list of 3 np.ndarrays [t1n, t1c, t2f] for this ROI.
        unet_encdec: EncDec UNet.
        unet_bbdm: BBDM UNet.
        conditions_model: BBDM condition tokens model.
        noise_scheduler: BBDM noise scheduler.
        vae: VAE autoencoder.
        device: torch device.
        roi_desc: description string for progress bar.

    Returns:
        synthesized T2W np.ndarray for this ROI.
    """
    # 1. VAE encode each modality
    latens_list = []
    for img in roi_images:
        latent = pipeline.encode_image(img, vae)
        latens_list.append(latent)

    n_latent_channels = 4

    # 2. EncDec synthesis on ROI latents
    to_modality_one_hot = torch.tensor(
        utils.create_modality_one_hot(configs.MISSING_MODALITY)
    ).float().to(device)

    from_modality_latents = np.concatenate(latens_list, axis=0)
    from_modality_latents = torch.tensor(from_modality_latents).half().to(device).unsqueeze(0)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        syn_lat_encdec = unet_encdec(
            x=from_modality_latents,
            modality_tensor=to_modality_one_hot,
        )
        syn_lat_encdec = syn_lat_encdec.detach().cpu().squeeze(0).numpy()

    # 3. BBDM synthesis on ROI latents
    conditioning = torch.tensor([[[1.], [1.]]], device=device)
    conditioning_emb = conditions_model(conditioning)

    to_modality_index = configs.MODALITY_LIST.index(configs.MISSING_MODALITY)
    latens_list_bbdm = utils.preprare_bbdm_latens(latens_list, to_modality_index)
    from_lat_bbdm = np.concatenate(latens_list_bbdm, axis=0)
    from_lat_bbdm = torch.tensor(from_lat_bbdm).half().to(device).unsqueeze(0)

    utils.set_seed(42)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        latents_denoised = from_lat_bbdm
        for i in range(noise_scheduler.sample_step):
            t = noise_scheduler.steps[i]
            t_tensor = torch.zeros((from_lat_bbdm.shape[0],), device=device)
            t_tensor.fill_(t)
            noise_pred = unet_bbdm(
                x=latents_denoised,
                timesteps=t_tensor,
                context=conditioning_emb,
            )
            latents_denoised, _ = noise_scheduler.backward_diffusion(
                i, latents_denoised, from_lat_bbdm, noise_pred
            )
        del noise_pred
        latents_denoised = latents_denoised.detach().cpu().squeeze(0).numpy()
        syn_lat_bbdm = latents_denoised[
            n_latent_channels * to_modality_index:
            n_latent_channels * (to_modality_index + 1)
        ]

    # 4. Decode and ensemble
    syn_img_encdec = pipeline.decode_latents(syn_lat_encdec, vae)
    syn_img_bbdm = pipeline.decode_latents(syn_lat_bbdm, vae)
    syn_img = utils.combine_images([syn_img_encdec, syn_img_bbdm],
                                   combination_type='mean')

    return syn_img


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_per_lesion_synthesis(s_data, base_syn_img, vae, unet_encdec,
                             unet_bbdm, conditions_model, noise_scheduler,
                             seg, device, verbose=False):
    """Per-lesion ROI synthesis overlay on top of full-image ensemble.

    Args:
        s_data: subject data dict (from pipeline.prepare_s_data or similar).
        base_syn_img: full-image ensemble synthesis result (256,256,160).
        vae: VAE model.
        unet_encdec: EncDec model.
        unet_bbdm: BBDM model.
        conditions_model: BBDM condition tokens.
        noise_scheduler: BBDM noise scheduler.
        seg: (256,256,160) preprocessed BraTS seg.
        device: torch device.
        verbose: print progress.

    Returns:
        final_syn_img: (256,256,160) with ROI-enhanced regions.
    """
    # 1. Detect lesions and compute ROIs
    lesions = detect_lesions(seg, min_volume=8)

    if not lesions:
        if verbose:
            print("  No lesions detected, returning base synthesis.")
        return base_syn_img

    bboxes = [l["bbox"] for l in lesions]
    merged_bboxes = merge_overlapping_bboxes(bboxes, margin=32)

    if verbose:
        print(f"  Detected {len(lesions)} lesions -> {len(merged_bboxes)} "
              f"ROIs after merge.")

    # 2. Load preprocessed images for ROI extraction
    # Get images (already loaded in s_data)
    # We need the preprocessed images, not just latents
    # Reconstruct from latents or re-load
    imgs_pp_list = s_data.get("imgs_pp_list", None)
    if imgs_pp_list is None:
        # If not available, skip per-lesion enhancement
        if verbose:
            print("  [WARN] Preprocessed images not in s_data, "
                  "skipping per-lesion.")
        return base_syn_img

    # 3. Process each merged ROI
    final_img = base_syn_img.copy()
    margin = 32
    crop_margin = 32  # extra context beyond the merged bbox

    for roi_idx, bbox in enumerate(tqdm(merged_bboxes,
                                        desc="  ROI synthesis",
                                        disable=not verbose or len(merged_bboxes) <= 1)):
        # Expand bbox with crop margin
        x1, x2, y1, y2, z1, z2 = bbox
        expanded_bbox = (
            x1 - crop_margin, x2 + crop_margin,
            y1 - crop_margin, y2 + crop_margin,
            z1 - crop_margin, z2 + crop_margin,
        )

        # Extract ROI from each modality
        roi_mods = []
        roi_slices = None
        for mod_img in imgs_pp_list:
            roi, slc = extract_roi(mod_img, expanded_bbox, margin=0)
            if roi_slices is None:
                roi_slices = slc
            # Pad to cube
            roi_padded, pad_slices = pad_to_cube(roi, min_size=64)
            roi_mods.append(roi_padded)

        # Synthesize
        syn_roi = synthesize_roi(
            roi_mods, unet_encdec, unet_bbdm, conditions_model,
            noise_scheduler, vae, device,
            roi_desc=f"ROI {roi_idx + 1}/{len(merged_bboxes)}"
        )

        # Unpad: extract original region from padded synthesis
        syn_roi_unpadded = syn_roi[pad_slices[0], pad_slices[1], pad_slices[2]]

        # Compute dst bbox in full-image coords
        dst_bbox = (
            roi_slices[0].start, roi_slices[0].start + syn_roi_unpadded.shape[0],
            roi_slices[1].start, roi_slices[1].start + syn_roi_unpadded.shape[1],
            roi_slices[2].start, roi_slices[2].start + syn_roi_unpadded.shape[2],
        )

        # Feather blend
        feather_blend(final_img, syn_roi_unpadded, dst_bbox, blend_width=16)

    return final_img


def prepare_roi_models(device):
    """Load all models needed for ROI synthesis once (shared across subjects).

    Returns:
        dict with vae, unet_encdec, unet_bbdm, conditions_model, noise_scheduler.
    """
    vae = pipeline.instantiate_vae_model(device)
    unet_encdec = pipeline.instantiate_encdec_model(device)
    unet_bbdm, conditions_model, noise_scheduler = \
        pipeline.instantiate_bbdm_model(device)
    return {
        "vae": vae,
        "unet_encdec": unet_encdec,
        "unet_bbdm": unet_bbdm,
        "conditions_model": conditions_model,
        "noise_scheduler": noise_scheduler,
    }
