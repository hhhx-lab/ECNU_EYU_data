#!/usr/bin/env python3
"""
BraTS 2026 Official Evaluation Runner (METs Challenge)

Uses the official Panoptica-based evaluation pipeline with config_mets.yaml.
Computes:
  - Global segmentation metrics: DSC, HD95, NSD per region (et, rc, tc, wt)
  - Instance-level metrics: lesion-wise DSC/HD95/NSD, detection F1
  - Large/small lesion split analysis

Usage:
    python evaluation/run_evaluation.py \
        --ref_path /public/home/zqchen/data/ \
        --pred_path ./prediction_results/segmamba_brats2026_exp_ce_dice/ \
        --output_dir ./evaluation_results/exp_ce_dice/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Hardcoded METs config — the only config we use
_METS_CONFIG_PATH = str(
    Path(__file__).resolve().parent / "brats_evaluation" / "configs" / "config_mets.yaml"
)


def run_brats_evaluation(
    ref_path: str,
    pred_path: str,
    summary_json: str = None,
    num_subjects: int = None,
    vol_threshold: float = 27.0,
    overlap_threshold: float = 0.1,
    output_dir: str = "./evaluation_results",
    verbose: bool = True,
) -> dict:
    """
    Run the complete BraTS official evaluation pipeline (METs config).

    Args:
        ref_path: Directory containing reference (ground truth) NIfTI files.
        pred_path: Directory containing prediction NIfTI files.
        summary_json: Path to save the raw Panoptica JSON summary.
        num_subjects: Limit evaluation to first N subjects (None = all).
        vol_threshold: Volume threshold (mm³) for large/small lesion split.
        overlap_threshold: DSC threshold for small lesion TP/FN classification.
        output_dir: Directory for output CSV files.
        verbose: Print progress information.

    Returns:
        dict with keys: 'summary_json', 'seg_csv', 'mets_csv', 'status'.
    """
    from panoptica import Panoptica_Evaluator

    resolved_config = _METS_CONFIG_PATH

    if not os.path.exists(resolved_config):
        raise FileNotFoundError(f"Config file not found: {resolved_config}")

    # Resolve summary JSON path
    if summary_json is None:
        os.makedirs(output_dir, exist_ok=True)
        summary_json = os.path.join(output_dir, "panoptica_evaluation_summary.json")

    # Validate paths
    if not os.path.isdir(ref_path):
        raise NotADirectoryError(f"Reference path not found: {ref_path}")
    if not os.path.isdir(pred_path):
        raise NotADirectoryError(f"Prediction path not found: {pred_path}")

    all_results = {"metrics": [], "missings": []}

    # Initialize Panoptica evaluator once
    if verbose:
        print(f"Initializing Panoptica Evaluator with config: {resolved_config}")
    evaluator = Panoptica_Evaluator.load_from_config(resolved_config)

    # Discover reference files
    ref_files = sorted([f for f in os.listdir(ref_path) if f.endswith(".nii.gz") and "seg" in f.lower()])
    # Also check subdirectories (BraTS directory structure: case_name/seg.nii.gz)
    if not ref_files:
        ref_files = []
        for item in sorted(os.listdir(ref_path)):
            item_path = os.path.join(ref_path, item)
            if os.path.isdir(item_path):
                seg_file = os.path.join(item_path, "seg.nii.gz")
                if os.path.exists(seg_file):
                    ref_files.append(os.path.join(item, "seg.nii.gz"))

    if num_subjects:
        ref_files = ref_files[:num_subjects]

    if verbose:
        print(f"Found {len(ref_files)} reference files")
        print(f"Prediction directory: {pred_path}")

    # Evaluate each subject
    for i, ref_file in enumerate(ref_files):
        ref_filepath = os.path.join(ref_path, ref_file)

        # Extract case name
        ref_dir = os.path.dirname(ref_file)
        if ref_dir:
            case_name = os.path.basename(ref_dir)
        else:
            case_name = os.path.splitext(ref_file)[0].replace(".nii", "")

        # Try to extract 5-digit ID
        match = re.search(r"(\d{5}(?:-\d{3})?)", ref_file)
        subject_id = match.group(1) if match else case_name

        # Find matching prediction
        pred_filepath = None
        for pred_file in os.listdir(pred_path):
            if pred_file.endswith(f"{subject_id}.nii.gz"):
                pred_filepath = os.path.join(pred_path, pred_file)
                break

        # Fallback: match by case_name
        if pred_filepath is None:
            for pred_file in os.listdir(pred_path):
                if case_name in pred_file and pred_file.endswith(".nii.gz"):
                    pred_filepath = os.path.join(pred_path, pred_file)
                    break

        if pred_filepath is None:
            if verbose:
                print(f"[{i+1}/{len(ref_files)}] WARNING: No prediction found for {case_name}, skipping")
            all_results["missings"].append(case_name)
            continue

        if verbose:
            print(f"[{i+1}/{len(ref_files)}] Evaluating {case_name}...")

        try:
            from evaluation.brats_evaluation import evaluate_single_exam

            results = evaluate_single_exam(
                prediction_filepath=pred_filepath,
                reference_filepath=ref_filepath,
                subject_identifier=case_name,
                evaluator=evaluator,
            )
            all_results["metrics"].append(results)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            all_results["missings"].append(case_name)

    # Save summary JSON
    with open(summary_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    if verbose:
        print(f"\nPanoptica summary saved to: {summary_json}")

    # Parse results to CSV
    seg_csv_path = os.path.join(output_dir, "segmentation_metrics.csv")
    mets_csv_path = os.path.join(output_dir, "mets_detailed_metrics.csv")

    from evaluation.brats_evaluation import parse_seg_results, parse_mets_results

    parse_seg_results(summary_json, seg_csv_path)
    if verbose:
        print(f"Segmentation metrics CSV saved to: {seg_csv_path}")

    parse_mets_results(
        summary_json,
        vol_threshold=vol_threshold,
        overlap_threshold=overlap_threshold,
        output_csv_path=mets_csv_path,
    )
    if verbose:
        print(f"METs detailed metrics CSV saved to: {mets_csv_path}")

    if verbose:
        print_summary(seg_csv_path)

    return {
        "summary_json": summary_json,
        "seg_csv": seg_csv_path,
        "mets_csv": mets_csv_path,
        "status": "success",
    }


def print_summary(seg_csv_path: str):
    """Print and save a human-readable summary of the evaluation results."""
    import pandas as pd

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("BraTS 2026 Official Evaluation Summary (METs)")
    lines.append("=" * 70)

    try:
        df = pd.read_csv(seg_csv_path)

        mean_row = df[df['subject_id'] == 'mean']
        if mean_row.empty:
            lines.append("(No aggregated metrics available)")
            return

        lines.append(f"\n{'Region':>8}  {'Global DSC':>12}  {'Global HD95':>12}  {'Global NSD':>12}")
        lines.append("-" * 52)

        for region in ['et', 'rc', 'tc', 'wt']:
            dsc_col = f"global_dsc_{region}"
            hd95_col = f"global_hd95_{region}"
            nsd_col = f"global_nsd_{region}"

            dsc_val = mean_row[dsc_col].values[0] if dsc_col in mean_row else float('nan')
            hd95_val = mean_row[hd95_col].values[0] if hd95_col in mean_row else float('nan')
            nsd_val = mean_row[nsd_col].values[0] if nsd_col in mean_row else float('nan')

            lines.append(f"{region:>8}  {dsc_val:>12.4f}  {hd95_val:>12.2f}  {nsd_val:>12.4f}")

        lines.append(f"\n{'Region':>8}  {'Instance F1':>12}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
        lines.append("-" * 44)
        for region in ['et', 'rc', 'tc', 'wt']:
            f1_col = f"all_instance_f1_{region}"
            tp_col = f"all_instance_tp_{region}"
            fp_col = f"all_instance_fp_{region}"
            fn_col = f"all_instance_fn_{region}"

            f1_val = mean_row[f1_col].values[0] if f1_col in mean_row else float('nan')
            tp_val = int(mean_row[tp_col].values[0]) if tp_col in mean_row else 0
            fp_val = int(mean_row[fp_col].values[0]) if fp_col in mean_row else 0
            fn_val = int(mean_row[fn_col].values[0]) if fn_col in mean_row else 0

            lines.append(f"{region:>8}  {f1_val:>12.4f}  {tp_val:>6}  {fp_val:>6}  {fn_val:>6}")

        # 保存到同目录下的 summary.txt
        import os
        summary_path = os.path.join(os.path.dirname(seg_csv_path), "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Summary saved to: {summary_path}")

    except Exception as e:
        lines.append(f"Error reading CSV: {e}")

    lines.append("-" * 70)

    # 同时打印到终端
    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="BraTS 2026 Official Evaluation Runner — METs Challenge (Panoptica)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_evaluation.py --ref_path ./gt/ --pred_path ./pred/
  python evaluation/run_evaluation.py --ref_path ./gt/ --pred_path ./pred/ --output_dir ./results/
        """,
    )

    parser.add_argument("--ref_path", type=str, required=True,
                        help="Path to directory containing reference NIfTI files.")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to directory containing prediction NIfTI files.")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory for output files (default: ./evaluation_results).")
    parser.add_argument("--summary_json", type=str, default=None,
                        help="Path for Panoptica JSON summary (default: <output_dir>/panoptica_evaluation_summary.json).")
    parser.add_argument("--num_subjects", type=int, default=None,
                        help="Limit evaluation to first N subjects.")
    parser.add_argument("--vol_threshold", type=float, default=27.0,
                        help="Volume threshold in mm³ for large/small lesion split (default: 27.0).")
    parser.add_argument("--overlap_threshold", type=float, default=0.1,
                        help="DSC threshold for small lesion TP/FN classification (default: 0.1).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args()

    try:
        result = run_brats_evaluation(
            ref_path=args.ref_path,
            pred_path=args.pred_path,
            summary_json=args.summary_json,
            num_subjects=args.num_subjects,
            vol_threshold=args.vol_threshold,
            overlap_threshold=args.overlap_threshold,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )

        print_summary(result["seg_csv"])

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()