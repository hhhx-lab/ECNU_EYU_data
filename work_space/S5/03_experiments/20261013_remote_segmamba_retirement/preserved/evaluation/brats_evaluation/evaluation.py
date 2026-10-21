import argparse
import os
import json
import re
from pathlib import Path

from panoptica import Panoptica_Evaluator


def evaluate_single_exam(
    prediction_filepath: str,
    reference_filepath: str,
    subject_identifier: str,
    evaluator: Panoptica_Evaluator,
) -> dict:
    """
    Evaluates a single subject's segmentation using the Panoptica framework.

    Args:
        prediction_filepath: Absolute path to the prediction NIfTI file.
        reference_filepath: Absolute path to the ground truth (reference) NIfTI file.
        subject_identifier: A unique identifier for the subject (e.g., filename).
        evaluator: The initialized Panoptica evaluator.

    Returns:
        dict: Evaluation results for the subject.
    """
    try:
        group_to_result = evaluator.evaluate(prediction_filepath, reference_filepath)
        results_dict = {k: r.to_dict() for k, r in group_to_result.items()}
        results_dict["subject_name"] = subject_identifier
        return results_dict
    except FileNotFoundError as e:
        print(f"Error: File not found for subject {subject_identifier}. Details: {e}")
        return {"subject_name": subject_identifier, "error": str(e)}
    except Exception as e:
        print(f"Error evaluating subject {subject_identifier}. Details: {e}")
        return {"subject_name": subject_identifier, "error": str(e)}