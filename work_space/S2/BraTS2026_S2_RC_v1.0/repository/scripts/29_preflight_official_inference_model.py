#!/usr/bin/env python3
"""Load the frozen official-inference model without reading official cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import nnunetv2.inference.predict_from_raw_data as predict_module

from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import (
    nnUNetTrainerBraTS2026RC,
)


EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAINER = "nnUNetTrainerBraTS2026RCFocalCompletionFineTune"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    checkpoint = model_root / "fold_0/checkpoint_final.pth"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Checkpoint SHA256 drifted")

    predict_module.recursive_find_trainer_class_by_name = (
        lambda trainer_name: nnUNetTrainerBraTS2026RC
        if trainer_name == EXPECTED_TRAINER
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
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_root),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    if predictor.trainer_name != EXPECTED_TRAINER:
        raise SystemExit(f"Loaded trainer drifted: {predictor.trainer_name}")
    parameter_count = sum(parameter.numel() for parameter in predictor.network.parameters())
    if parameter_count <= 0:
        raise SystemExit("Loaded model has no parameters")
    print(
        json.dumps(
            {
                "status": "pass",
                "model_root": str(model_root),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "trainer": predictor.trainer_name,
                "parameter_count": parameter_count,
                "visible_cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
