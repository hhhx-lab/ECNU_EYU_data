#!/usr/bin/env python3
"""Run frozen S2 inference without importing training-only augmentation code."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import nnunetv2.inference.predict_from_raw_data as predict_module

from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import (
    nnUNetTrainerBraTS2026RC,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--export-workers", type=int, default=4)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    model_root = Path(args.model_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen S2 inference")

    # nnU-Net's standard finder imports the complete training trainer first.
    # This checkpoint uses the standard plans-defined network builder, so frozen
    # inference can provide that exact builder without training-only transforms.
    predict_module.recursive_find_trainer_class_by_name = (
        lambda trainer_name: nnUNetTrainerBraTS2026RC
        if trainer_name == "nnUNetTrainerBraTS2026RC"
        else (_ for _ in ()).throw(ValueError(f"unexpected trainer: {trainer_name}"))
    )
    predictor = predict_module.nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_root),
        use_folds=(args.fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    print(f"Frozen S2 checkpoint loaded with trainer={predictor.trainer_name}")
    predictor.predict_from_files(
        str(input_root),
        str(output_root),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=args.preprocess_workers,
        num_processes_segmentation_export=args.export_workers,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
