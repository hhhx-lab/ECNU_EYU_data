"""
BraTS 2026 Evaluation Module (METs Challenge)

Provides the official Panoptica-based BraTS evaluation pipeline,
hardcoded for the METs (brain metastasis) challenge using config_mets.yaml.

Usage:
    from evaluation.run_evaluation import run_brats_evaluation
    results = run_brats_evaluation(ref_path="./gt/", pred_path="./pred/")
"""

from evaluation.brats_evaluation import (
    evaluate_single_exam,
    parse_seg_results,
    parse_mets_results,
)

__all__ = [
    "evaluate_single_exam",
    "parse_seg_results",
    "parse_mets_results",
    "run_brats_evaluation",
]