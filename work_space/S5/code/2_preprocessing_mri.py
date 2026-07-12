#!/usr/bin/env python3
"""Preprocess the G2 patient-group fixed split for S5 SegMamba."""

import argparse
from pathlib import Path

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor


MODALITY_FILENAMES = [
    "t1n.nii.gz",
    "t1c.nii.gz",
    "t2w.nii.gz",
    "t2f.nii.gz",
]


def process_split(input_dir: Path, output_dir: Path, num_processes: int) -> None:
    preprocessor = MultiModalityPreprocessor(
        base_dir=str(input_dir.parent),
        image_dir=input_dir.name,
        data_filenames=MODALITY_FILENAMES,
        seg_filename="seg.nii.gz",
    )
    preprocessor.run(
        output_spacing=[1.0, 1.0, 1.0],
        output_dir=str(output_dir),
        all_labels=[1, 2, 3, 4],
        num_processes=num_processes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g2-case-folder-root",
        required=True,
        help="G2 materializer output containing train/val/test case folders.",
    )
    parser.add_argument("--output-root", default="./data/fullres")
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    source_root = Path(args.g2_case_folder_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for split_name in ("train", "val", "test"):
        input_dir = source_root / split_name
        if not input_dir.is_dir():
            raise SystemExit(f"missing G2 split directory: {input_dir}")
        process_split(input_dir, output_root / split_name, args.num_processes)
    print(f"preprocessed_fixed_split={output_root}")


if __name__ == "__main__":
    main()
