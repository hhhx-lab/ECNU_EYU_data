"""Dataset264 completion fine-tuning with focal cross-entropy."""

try:
    from .nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from .small_lesion_trainer_mixins import FocalLossMixin
except ImportError:
    from nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from small_lesion_trainer_mixins import FocalLossMixin


class nnUNetTrainerBraTS2026RCFocalCompletionFineTune(
    FocalLossMixin,
    nnUNetTrainerBraTS2026RCCompletionFineTune,
):
    pass
