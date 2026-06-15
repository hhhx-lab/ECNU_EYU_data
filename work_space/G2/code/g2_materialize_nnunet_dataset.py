#!/usr/bin/env python3
"""Materialize G2 real/synthetic manifests into an nnU-Net raw dataset.

Default mode is manifest-only so this script is safe to run on a laptop. On the
training machine, use --mode symlink or --mode copy after checking disk space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results")
CHANNEL_ORDERS = {
    "g2_official": ["t1n", "t1c", "t2w", "t2f"],
    "s2_current": ["t1c", "t1n", "t2f", "t2w"],
}
LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_for(row: dict[str, str], modality: str) -> str:
    candidates = [
        f"{modality}_source_path",
        f"normalized_{modality}_path",
        f"raw_{modality}_path",
    ]
    if modality == "seg":
        candidates = ["seg_source_path", "normalized_seg_path", "raw_seg_path"]
    values = [row.get(key, "") for key in candidates if row.get(key, "")]
    for value in values:
        if Path(value).exists():
            return value
    return values[0] if values else ""


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    if mode == "manifest-only":
        return "planned"
    if not src.exists():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "exists"
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def build_rows(
    real_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    dataset_dir: Path,
    channel_order: list[str],
    mode: str,
    overwrite: bool,
) -> tuple[list[dict[str, object]], int]:
    materialized: list[dict[str, object]] = []
    included_cases = 0

    all_rows: list[tuple[str, dict[str, str]]] = []
    all_rows.extend(("real", row) for row in real_rows)
    all_rows.extend(("synthetic", row) for row in synthetic_rows)

    for row_type, row in all_rows:
        case_id = row.get("nnunet_case_id") or row.get("case_id") or row.get("synthetic_final_id") or row.get("source_case_id")
        if not case_id:
            continue
        if row_type == "synthetic":
            accepted = row.get("accepted_for_training", "").lower() in {"true", "1", "yes"}
            ablation = row.get("accepted_for_ablation_only", "").lower() in {"true", "1", "yes"}
            if not (accepted or ablation):
                continue
        included_cases += 1

        for channel_idx, modality in enumerate(channel_order):
            src = Path(source_for(row, modality))
            dst = dataset_dir / "imagesTr" / f"{case_id}_{channel_idx:04d}.nii.gz"
            action = link_or_copy(src, dst, mode, overwrite)
            materialized.append(
                {
                    "case_id": case_id,
                    "row_type": row_type,
                    "modality": modality,
                    "nnunet_channel": f"{channel_idx:04d}",
                    "source_path": str(src),
                    "target_path": str(dst),
                    "action": action,
                }
            )

        src = Path(source_for(row, "seg"))
        dst = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
        action = link_or_copy(src, dst, mode, overwrite)
        materialized.append(
            {
                "case_id": case_id,
                "row_type": row_type,
                "modality": "seg",
                "nnunet_channel": "label",
                "source_path": str(src),
                "target_path": str(dst),
                "action": action,
            }
        )

    return materialized, included_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an nnU-Net raw dataset from G2 real mapping plus accepted synthetic manifest."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--real-mapping", default="", help="real-only mapping CSV. Defaults to results/manifests/nnunet_case_mapping_realonly.csv")
    parser.add_argument("--synthetic-accepted-manifest", default="", help="Optional synthetic_accepted_manifest_<run_id>.csv")
    parser.add_argument("--output-root", required=True, help="nnUNet_raw root or any destination root on the training machine.")
    parser.add_argument("--dataset-id", default="261")
    parser.add_argument("--dataset-name", default="BraTS2026_MET_RealSynth_G1")
    parser.add_argument("--channel-order", choices=sorted(CHANNEL_ORDERS), default="g2_official")
    parser.add_argument("--mode", choices=["manifest-only", "symlink", "hardlink", "copy"], default="manifest-only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    real_mapping = Path(args.real_mapping) if args.real_mapping else results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    synthetic_manifest = Path(args.synthetic_accepted_manifest) if args.synthetic_accepted_manifest else None
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / f"Dataset{args.dataset_id}_{args.dataset_name}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTr").mkdir(exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(exist_ok=True)

    real_rows = read_csv(real_mapping)
    synthetic_rows = read_csv(synthetic_manifest) if synthetic_manifest else []
    channel_order = CHANNEL_ORDERS[args.channel_order]
    rows, num_training = build_rows(real_rows, synthetic_rows, dataset_dir, channel_order, args.mode, args.overwrite)

    dataset_json = {
        "channel_names": {str(idx): modality for idx, modality in enumerate(channel_order)},
        "labels": LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "g2_channel_order": args.channel_order,
        "g2_materialization_mode": args.mode,
        "g2_real_mapping": str(real_mapping),
        "g2_synthetic_manifest": str(synthetic_manifest) if synthetic_manifest else "",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = ["case_id", "row_type", "modality", "nnunet_channel", "source_path", "target_path", "action"]
    write_csv(dataset_dir / "g2_materialization_manifest.csv", rows, fieldnames)

    print(f"dataset_dir={dataset_dir}")
    print(f"numTraining={num_training}")
    print(f"channel_order={args.channel_order}:{','.join(channel_order)}")
    print(f"mode={args.mode}")


if __name__ == "__main__":
    main()
