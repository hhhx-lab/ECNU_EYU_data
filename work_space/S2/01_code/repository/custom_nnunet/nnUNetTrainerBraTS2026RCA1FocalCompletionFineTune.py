"""Dataset264 completion fine-tuning with A-1 and focal CE."""

try:
    from .nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from .small_lesion_trainer_mixins import A1ArchitectureMixin, FocalLossMixin
except ImportError:
    from nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from small_lesion_trainer_mixins import A1ArchitectureMixin, FocalLossMixin


class nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune(
    A1ArchitectureMixin,
    FocalLossMixin,
    nnUNetTrainerBraTS2026RCCompletionFineTune,
):
    pass
