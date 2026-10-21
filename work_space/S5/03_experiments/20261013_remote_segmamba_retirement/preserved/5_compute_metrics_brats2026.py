"""
BraTS 2026 Official Metrics Computation

Runs the official Panoptica-based BraTS evaluation pipeline on prediction results.
Outputs standard BraTS metrics: global DSC/HD95/NSD per region group,
instance-level lesion metrics, and large/small lesion split analysis.

Usage:
    python 5_compute_metrics_brats2026.py --config configs/exp_ce_dice.yaml
"""

import argparse
import os
import sys
import yaml

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='SegMamba BraTS 2026 - Official Metrics Computation'
    )
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to YAML config file.')
    parser.add_argument('--raw_data_dir', type=str,
                        default='/public/home/zqchen/data/',
                        help='Path to raw data directory containing ground truth.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results.')
    parser.add_argument('--vol_threshold', type=float, default=27.0,
                        help='Volume threshold (mm³) for large/small lesion split.')
    parser.add_argument('--overlap_threshold', type=float, default=0.1,
                        help='DSC threshold for small lesion TP/FN classification.')
    return parser.parse_known_args()


def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args, _ = parse_args()
    config = load_config(args.config) if args.config else {}

    # Determine config name and prediction folder
    config_name = ''
    if args.config:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        print(f"Config name: {config_name}")

    if args.output_dir:
        output_dir = args.output_dir
    elif config_name:
        output_dir = f'./results/{config_name}/metrics'
    else:
        output_dir = './results/default/metrics'

    # Predictions are under results/<exp>/predictions/
    if config_name:
        pred_dir = f'./results/{config_name}/predictions'
    else:
        pred_dir = './results/default/predictions'

    raw_data_dir = args.raw_data_dir

    # Validate paths
    if not os.path.isdir(pred_dir):
        print(f'ERROR: Prediction directory not found: {pred_dir}')
        print('Run 4_predict_brats2026.py first to generate predictions.')
        sys.exit(1)

    if not os.path.isdir(raw_data_dir):
        print(f'WARNING: Raw data directory not found: {raw_data_dir}')
        print('Ground truth files may be in a different location.')

    # Count predictions
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    print(f'Found {len(pred_files)} prediction files in: {pred_dir}')

    # Run official evaluation
    print('\n' + '=' * 65)
    print('BraTS 2026 Official Evaluation (Panoptica)')
    print('=' * 65)

    try:
        from evaluation.run_evaluation import run_brats_evaluation

        result = run_brats_evaluation(
            ref_path=raw_data_dir,
            pred_path=pred_dir,
            output_dir=output_dir,
            vol_threshold=args.vol_threshold,
            overlap_threshold=args.overlap_threshold,
            verbose=True,
        )

        print(f'\nEvaluation complete!')
        print(f'  Summary JSON: {result["summary_json"]}')
        print(f'  Segmentation CSV: {result["seg_csv"]}')
        print(f'  METs detailed CSV: {result["mets_csv"]}')

    except ImportError as e:
        print(f'\nERROR: {e}')
        print('Please install the evaluation dependencies:')
        print('  pip install -r evaluation/requirements.txt')
        print('  pip install -e evaluation/')
        sys.exit(1)
    except Exception as e:
        print(f'\nERROR: Evaluation failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()