#!/usr/bin/env python3
"""Run lightweight EncDec/BBDM forward-contract checks on a CUDA device."""

from __future__ import annotations

import argparse

import torch

import configs
from models.bbdm.bb_scheduler import Scheduler
from models.bbdm.condition_tokens import ConditionTokens
from models.bbdm.unet import DiffusionModelUNetMaisi as BBDMUNet
from models.encdec.unet import DiffusionModelUNetMaisi as EncDecUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--spatial-shape", type=int, nargs=3, default=(16, 16, 16))
    return parser.parse_args()


def build_encdec(device: torch.device) -> EncDecUNet:
    cfg = configs.NETWORKS_CONFIG["encdec"]["unet"]
    return EncDecUNet(
        spatial_dims=cfg["spatial_dims"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        num_res_blocks=cfg["num_res_blocks"],
        num_channels=cfg["num_channels"],
        attention_levels=cfg["attention_levels"],
        num_head_channels=cfg["num_head_channels"],
        with_conditioning=cfg["with_conditioning"],
        use_flash_attention=cfg["use_flash_attention"],
    ).to(device).eval()


def build_bbdm(device: torch.device):
    cfg = configs.NETWORKS_CONFIG["bbdm"]
    unet_cfg = cfg["unet"]
    unet = BBDMUNet(
        spatial_dims=unet_cfg["spatial_dims"],
        in_channels=unet_cfg["in_channels"],
        out_channels=unet_cfg["out_channels"],
        num_res_blocks=unet_cfg["num_res_blocks"],
        num_channels=unet_cfg["num_channels"],
        attention_levels=unet_cfg["attention_levels"],
        num_head_channels=unet_cfg["num_head_channels"],
        with_conditioning=unet_cfg["with_conditioning"],
        use_flash_attention=unet_cfg["use_flash_attention"],
        include_from_modality=unet_cfg["include_from_modality"],
        include_to_modality=unet_cfg["include_to_modality"],
        cross_attention_dim=unet_cfg["cross_attention_dim"],
        transformer_num_layers=unet_cfg["transformer_num_layers"],
        upcast_attention=unet_cfg["upcast_attention"],
    ).to(device).eval()
    condition_cfg = cfg["conditions_model"]
    conditions = ConditionTokens(
        num_conditions=condition_cfg["num_conditions"],
        embed_dim=condition_cfg["embed_dim"],
        hidden_dim=condition_cfg["hidden_dim"],
        use_self_attention=condition_cfg["use_self_attention"],
        n_heads=condition_cfg["n_heads"],
        n_layers=condition_cfg["n_layers"],
    ).to(device).eval()
    scheduler_cfg = cfg["bb_scheduler"]
    scheduler = Scheduler(**scheduler_cfg)
    return unet, conditions, scheduler


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 to test batch alignment.")
    if any(size < 16 or size % 8 for size in args.spatial_shape):
        raise ValueError("Every spatial dimension must be >=16 and divisible by 8.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke was requested but no CUDA device is visible.")

    device = torch.device(args.device)
    batch = args.batch_size
    spatial = tuple(args.spatial_shape)
    print(f"Device: {device}; batch={batch}; spatial={spatial}; BBDM s={configs.BBDM_S}")

    encdec = build_encdec(device)
    encdec_input = torch.randn((batch, 12, *spatial), device=device)
    modality = torch.zeros((batch, 4), device=device)
    modality[:, configs.MODALITY_LIST.index(configs.MISSING_MODALITY)] = 1
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        encdec_output = encdec(x=encdec_input, modality_tensor=modality)
    expected_encdec = (batch, 4, *spatial)
    if tuple(encdec_output.shape) != expected_encdec or not torch.isfinite(encdec_output).all():
        raise RuntimeError(
            f"EncDec contract failed: got {tuple(encdec_output.shape)}, expected {expected_encdec}"
        )
    print(f"EncDec forward passed: {tuple(encdec_output.shape)}")
    del encdec, encdec_input, encdec_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bbdm, condition_model, scheduler = build_bbdm(device)
    clean = torch.randn((batch, 16, *spatial), device=device)
    endpoint = torch.randn_like(clean)
    noise = torch.randn_like(clean)
    timesteps = torch.arange(batch, device=device, dtype=torch.long)
    noisy, objective = scheduler.forward_diffusion(timesteps, clean, endpoint, noise)
    context = condition_model(torch.ones((batch, 2, 1), device=device))
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = bbdm(x=noisy, timesteps=timesteps, context=context)
    expected_bbdm = (batch, 16, *spatial)
    if tuple(prediction.shape) != expected_bbdm or not torch.isfinite(prediction).all():
        raise RuntimeError(
            f"BBDM contract failed: got {tuple(prediction.shape)}, expected {expected_bbdm}"
        )
    if tuple(objective.shape) != expected_bbdm or tuple(context.shape[:2]) != (batch, 2):
        raise RuntimeError("BBDM scheduler or condition-token batch contract failed.")
    print(f"BBDM forward passed: {tuple(prediction.shape)}; context={tuple(context.shape)}")
    print("G1 V3 model smoke passed.")


if __name__ == "__main__":
    main()
