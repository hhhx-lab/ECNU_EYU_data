"""Programmatic example: evaluate the bundled METs sample data via the Python API.

Run from anywhere after `pip install BraTS-evaluation` (or `poetry install`):
    python example/programmatic_example.py

To use on your own data, swap the three paths near the top of `main()`.
"""

import re
from pathlib import Path

from panoptica import Panoptica_Evaluator

from brats_evaluation import config_path, evaluate_single_exam


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ref_dir = repo_root / "example" / "sample_data" / "ref"
    pred_dir = repo_root / "example" / "sample_data" / "pred"

    evaluator = Panoptica_Evaluator.load_from_config(str(config_path("mets")))
    pred_files = list(pred_dir.glob("*.nii.gz"))

    for ref_file in sorted(ref_dir.glob("*.nii.gz")):
        match = re.search(r"(\d{5}(?:-\d{3})?)", ref_file.name)
        if not match:
            continue
        suffix = f"{match.group(1)}.nii.gz"
        pred_file = next((p for p in pred_files if p.name.endswith(suffix)), None)
        if pred_file is None:
            print(f"{ref_file.name}: no matching prediction, skipping")
            continue

        results = evaluate_single_exam(
            prediction_filepath=str(pred_file),
            reference_filepath=str(ref_file),
            subject_identifier=ref_file.name,
            evaluator=evaluator,
        )
        et = results.get("et", {})
        print(f"{ref_file.name} → ET tp={et.get('tp')} fp={et.get('fp')} fn={et.get('fn')} DSC={et.get('sq_dsc')}")

    # For richer per-region tables (DSC/HD95/NSD aggregations, small/large-lesion
    # splits for METs), import `parse_seg_results` / `parse_mets_results` from
    # `brats_evaluation` and run them against a saved JSON summary.


if __name__ == "__main__":
    main()
