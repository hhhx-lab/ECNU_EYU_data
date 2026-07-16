"""S1 multi-task inference with joint (single-pass) sliding-window backbone.

Tumor and RC heads share one backbone forward per window, cutting ~half of the
redundant SWI compute compared to running two separate predictors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "metrics"))

from brats_multitask_dataset import nonzero_zscore_normalize  # noqa: E402
from brats_validation_metrics import compose_label_map  # noqa: E402
from multitask_unet import MultiTaskUNet, model_kwargs_from_config  # noqa: E402
from sliding_window_multitask import (  # noqa: E402
    logits_to_label_maps,
    sliding_window_multitask,
)

MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def load_case(case_dir: Path, normalize: bool = True):
    case_dir = Path(case_dir)
    case = case_dir.name
    mods = []
    reference = None
    for mod in MODALITIES:
        nifti = nib.load(case_dir / f"{case}-{mod}.nii.gz")
        if reference is None:
            reference = nifti
        mods.append(nifti.get_fdata(dtype=np.float32))

    image = np.stack(mods)
    if normalize:
        image = nonzero_zscore_normalize(image)

    tensor = torch.from_numpy(image.astype(np.float32, copy=False)).unsqueeze(0)
    return tensor, reference, case


def resolve_amp_dtype(name: str | None):
    if not name or name.lower() in ("none", "off", "false"):
        return None
    name = name.lower()
    if name in ("auto",):
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float16 if torch.cuda.is_available() else None
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown amp dtype: {name}")


def main():
    parser = argparse.ArgumentParser(description="S1 joint multi-task SWI inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML to read model/roi settings from",
    )
    parser.add_argument("--roi_size", nargs=3, type=int, default=None)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--amp_dtype",
        default="auto",
        help="auto | bf16 | fp16 | none",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable nonzero brain Z-score (not recommended)",
    )
    parser.add_argument(
        "--save_fused",
        action="store_true",
        help="Also write fused BraTS-style seg (RC=4 over tumor labels)",
    )
    args = parser.parse_args()

    model_cfg = {}
    roi_size = (96, 96, 96)
    tumor_classes = 4
    rc_classes = 2
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        val_cfg = cfg.get("validation", {})
        train_cfg = cfg.get("train", {})
        roi_size = tuple(
            val_cfg.get("roi_size", train_cfg.get("patch_size", list(roi_size)))
        )
        tumor_classes = int(model_cfg.get("tumor_classes", 4))
        rc_classes = int(model_cfg.get("rc_classes", 2))
    if args.roi_size is not None:
        roi_size = tuple(args.roi_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskUNet(**model_kwargs_from_config(model_cfg)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    image, reference, case = load_case(
        args.case_dir, normalize=not args.no_normalize
    )
    image = image.to(device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    if device.type != "cuda":
        amp_dtype = None

    with torch.no_grad():
        outputs = sliding_window_multitask(
            model=model,
            image=image,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            tumor_classes=tumor_classes,
            rc_classes=rc_classes,
            amp_dtype=amp_dtype,
        )
        tumor_t, rc_t = logits_to_label_maps(outputs)

    tumor = tumor_t[0].cpu().numpy().astype(np.uint8)
    rc = rc_t[0].cpu().numpy().astype(np.uint8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = reference.affine
    header = reference.header.copy()

    nib.save(
        nib.Nifti1Image(tumor, affine, header),
        output_dir / "tumor_pred.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(rc, affine, header),
        output_dir / "rc_pred.nii.gz",
    )

    if args.save_fused:
        fused = compose_label_map(tumor, rc)
        nib.save(
            nib.Nifti1Image(fused.astype(np.uint8), affine, header),
            output_dir / f"{case}.nii.gz",
        )

    print(f"prediction saved for {case} -> {output_dir}")
    print(f"joint SWI roi={roi_size} amp={amp_dtype} normalize={not args.no_normalize}")


if __name__ == "__main__":
    main()
