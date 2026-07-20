"""Completion-only fine-tuning from the fixed real-only S2 checkpoint."""

from __future__ import annotations

import os

try:
    from .nnUNetTrainerBraTS2026RC import nnUNetTrainerBraTS2026RC
except ImportError:
    from nnUNetTrainerBraTS2026RC import nnUNetTrainerBraTS2026RC


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class nnUNetTrainerBraTS2026RCCompletionFineTune(nnUNetTrainerBraTS2026RC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = positive_int("S2_COMPLETION_EPOCHS", 200)
        self.initial_lr = positive_float("S2_COMPLETION_INITIAL_LR", 0.001)
        self.save_every = positive_int("S2_COMPLETION_SAVE_EVERY", 25)
