from importlib.resources import files
from pathlib import Path

from brats_evaluation.evaluation import evaluate_single_exam
from brats_evaluation.metrics_parser import parse_seg_results, parse_mets_results


def config_path(name: str) -> Path:
    """Return the absolute path to a bundled Panoptica config.

    Valid names (case-sensitive): "mets", "gli", "ped", "MenRT", "MenPre", "GoAT".
    Example: ``config_path("mets")`` → ``.../brats_evaluation/configs/config_mets.yaml``.
    """
    path = Path(str(files("brats_evaluation").joinpath("configs", f"config_{name}.yaml")))
    if not path.exists():
        raise FileNotFoundError(
            f"No bundled config named {name!r}. "
            f"Expected file at {path}."
        )
    return path


__all__ = [
    "evaluate_single_exam",
    "parse_seg_results",
    "parse_mets_results",
    "config_path",
]
