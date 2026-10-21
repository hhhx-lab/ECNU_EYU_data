import argparse
import os
import json
import re
from importlib.resources import files
from pathlib import Path

from panoptica import Panoptica_Evaluator

BUNDLED_CONFIGS = ["mets", "gli", "ped", "MenRT", "MenPre", "GoAT"]

def evaluate_single_exam(
    prediction_filepath: str,
    reference_filepath: str,
    subject_identifier: str,
    evaluator: Panoptica_Evaluator,
) -> dict:
    """
    Evaluates a single subject's segmentation using the Panoptica framework.

    Args:
        prediction_filepath (str): Absolute path to the prediction NIfTI file.
        reference_filepath (str): Absolute path to the ground truth (reference) NIfTI file.
        subject_identifier (str): A unique identifier for the subject (e.g., filename).
        evaluator (Panoptica_Evaluator): The initialized Panoptica evaluator.

    Returns:
        dict: A dictionary containing the evaluation results for the subject.
              Includes metrics for each defined region and the subject_identifier.
              Returns an empty dictionary if evaluation fails.
    """
    try:
        # Perform the evaluation
        group_to_result = evaluator.evaluate(
            prediction_filepath,
            reference_filepath,
        )

        # Convert results to a dictionary and add subject identifier
        results_dict = {k: r.to_dict(True) for k, r in group_to_result.items()}
        results_dict["subject_name"] = subject_identifier
        
        return results_dict

    except FileNotFoundError as e:
        print(f"Error: File not found for subject {subject_identifier}. Details: {e}")
        return {"subject_name": subject_identifier, "error": str(e)}
    except Exception as e:
        print(f"Error evaluating subject {subject_identifier}. Details: {e}")
        return {"subject_name": subject_identifier, "error": str(e)}

def main():
    """
    Main function to parse arguments, iterate through subjects, and perform evaluation.
    """
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(
        description="Evaluate medical image segmentations using Panoptica."
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        required=True,
        help="Path to the directory containing reference (ground truth) NIfTI files.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to the directory containing prediction NIfTI files.",
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=str,
        choices=BUNDLED_CONFIGS,
        help="Name of a bundled Panoptica config (e.g., mets, gli, ped).",
    )
    config_group.add_argument(
        "--config_path",
        type=str,
        help="Path to a custom Panoptica configuration YAML file.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="./panoptica_evaluation_summary.json",
        help="Output path for the JSON file summarizing all evaluation metrics.",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=None,
        help="Number of subjects to process. If None, all subjects found will be processed.",
    )

    args = parser.parse_args()

    # Resolve config path
    if args.config:
        config_path = Path(str(files("brats_evaluation").joinpath("configs", f"config_{args.config}.yaml")))
        if not config_path.exists():
            print(f"Error: Bundled config '{args.config}' not found at {config_path}")
            return
        config_path = str(config_path)
    else:
        config_path = args.config_path

    # 2. Validate Paths
    if not os.path.isdir(args.ref_path):
        print(f"Error: Reference path '{args.ref_path}' is not a valid directory.")
        return
    if not os.path.isdir(args.pred_path):
        print(f"Error: Prediction path '{args.pred_path}' is not a valid directory.")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config path '{config_path}' does not exist.")
        return

    # 3. Prepare for Evaluation
    all_evaluation_results = {
        "metrics": [],
        "missings": []
    }

    # Initialize Panoptica Evaluator once for all subjects to improve performance
    print("Initializing Panoptica Evaluator...")
    evaluator = Panoptica_Evaluator.load_from_config(config_path)

    # Get list of reference files (assuming NIfTI files)
    reference_files = sorted([f for f in os.listdir(args.ref_path) if f.endswith(".nii.gz")])
    
    if args.num_subjects:
        reference_files = reference_files[:args.num_subjects]

    print(f"Starting evaluation for {len(reference_files)} subjects...")

    # 4. Iterate and Evaluate Each Subject
    for i, ref_filename in enumerate(reference_files):
        # Construct full paths
        reference_filepath = os.path.join(args.ref_path, ref_filename)

        # Extract the 5-digit case ID (and 3-digit timepoint, if available)
        match = re.search(r"(\d{5}(?:-\d{3})?)", ref_filename)
        if not match:
            print(
                "Warning: Could not extract subject ID (e.g. 12345 or 12345-001) "
                f"from {ref_filename}. Skipping."
            )
            all_evaluation_results["missings"].append(ref_filename)
            continue
            
        subject_id = match.group(1)
        expected_suffix = f"{subject_id}.nii.gz"
        
        # Find the matching prediction file
        prediction_filepath = None
        for pred_file in os.listdir(args.pred_path):
            if pred_file.endswith(expected_suffix):
                prediction_filepath = os.path.join(args.pred_path, pred_file)
                break
                
        if prediction_filepath is None:
            print(f"Warning: No matching prediction file found for {ref_filename} (expected ending with {expected_suffix}). Skipping.")
            all_evaluation_results["missings"].append(ref_filename)
            continue

        print(f"[{i+1}/{len(reference_files)}] Evaluating {ref_filename}...")

        # Perform evaluation for the current subject
        results = evaluate_single_exam(
            prediction_filepath=prediction_filepath,
            reference_filepath=reference_filepath,
            subject_identifier=ref_filename, # Use ref_filename as identifier
            evaluator=evaluator,
        )
        all_evaluation_results["metrics"].append(results)

    # 5. Save Summary to JSON
    with open(args.summary_json, 'w') as json_file:
        json.dump(all_evaluation_results, json_file, indent=4)

    print(f"\nEvaluation complete. Results saved to {args.summary_json}")

if __name__ == "__main__":
    main()