"""Deadline-aware S2 fine-tuning with frozen G1 Diffusion V3 augmentation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

try:
    from .nnUNetTrainerBraTS2026RC import nnUNetTrainerBraTS2026RC
    from .online_diffusion_transform import OnlineDiffusionTransform
except ImportError:
    from nnUNetTrainerBraTS2026RC import nnUNetTrainerBraTS2026RC
    from online_diffusion_transform import OnlineDiffusionTransform


def required_path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class nnUNetTrainerBraTS2026RCOnlineDiffusion(nnUNetTrainerBraTS2026RC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("S2_ONLINE_EPOCHS", "200"))
        self.initial_lr = float(os.environ.get("S2_ONLINE_INITIAL_LR", "0.001"))
        self.save_every = int(os.environ.get("S2_ONLINE_SAVE_EVERY", "25"))
        self.requires_single_threaded_augmentation = True
        self._online_diffusion_transform = None

    def _initialize_online_diffusion(self) -> None:
        if self._online_diffusion_transform is not None:
            return
        if self.is_ddp:
            raise RuntimeError("Online diffusion trainer currently requires one S2 GPU")

        g1_code_dir = required_path("G1_DIFFUSION_CODE_DIR", directory=True)
        checkpoint_dir = required_path("G1_DIFFUSION_CHECKPOINT_DIR", directory=True)
        selection_path = required_path("G1_DIFFUSION_CHECKPOINT_SELECTION")
        label_pool_path = required_path("G1_DIFFUSION_LABEL_POOL")
        gate_path = required_path("G2_DIFFUSION_QC_GATE")

        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_steps = selection.get("checkpoint_steps", selection)
        expected_modalities = {"t1c", "t1n", "t2w", "t2f"}
        if set(checkpoint_steps) != expected_modalities:
            raise ValueError(
                f"Checkpoint selection must contain {sorted(expected_modalities)}")
        checkpoint_steps = {key: int(value) for key, value in checkpoint_steps.items()}

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("decision") != "approve":
            raise RuntimeError(f"G2 diffusion gate is not approved: {gate_path}")
        if gate.get("checkpoint_selection_sha256") != sha256(selection_path):
            raise RuntimeError("G2 gate does not match the selected checkpoint file")
        if gate.get("normalization") != "zscore":
            raise RuntimeError("G2 gate must approve zscore normalization")
        if gate.get("sampling_method") != "edm_heun":
            raise RuntimeError("G2 gate must approve edm_heun sampling")

        label_paths = [
            line.strip() for line in label_pool_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(label_paths) != 823:
            raise ValueError(f"Expected 823 train labels, found {len(label_paths)}")
        missing = [path for path in label_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Label pool contains missing files: {missing[:5]}")

        sampling_steps = int(gate.get("sampling_steps", 18))
        self._online_diffusion_transform = OnlineDiffusionTransform(
            g1_code_dir=g1_code_dir,
            checkpoint_dir=checkpoint_dir,
            label_pool_paths=label_paths,
            checkpoint_steps=checkpoint_steps,
            sampling_steps=sampling_steps,
            augment_probability=float(os.environ.get("S2_ONLINE_AUGMENT_PROB", "0.6")),
            second_tumour_probability=float(
                os.environ.get("S2_ONLINE_SECOND_TUMOUR_PROB", "0.4")),
            max_tumours=int(os.environ.get("S2_ONLINE_MAX_TUMOURS", "2")),
            device=str(self.device),
        )
        self.print_to_log_file(
            "G1 online diffusion initialized: "
            f"steps={checkpoint_steps}, sampling_steps={sampling_steps}"
        )

    def get_pre_spatial_training_transforms(self):
        if self._online_diffusion_transform is None:
            raise RuntimeError("Online diffusion was not initialized before dataloader creation")
        return [self._online_diffusion_transform]

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        self._initialize_online_diffusion()
        super().on_train_start()
