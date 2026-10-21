"""Dataset264 completion fine-tuning with the five-stage A-1 architecture."""

try:
    from .nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from .small_lesion_trainer_mixins import A1ArchitectureMixin
except ImportError:
    from nnUNetTrainerBraTS2026RCCompletionFineTune import (
        nnUNetTrainerBraTS2026RCCompletionFineTune,
    )
    from small_lesion_trainer_mixins import A1ArchitectureMixin


class nnUNetTrainerBraTS2026RCA1CompletionFineTune(
    A1ArchitectureMixin,
    nnUNetTrainerBraTS2026RCCompletionFineTune,
):
    pass
