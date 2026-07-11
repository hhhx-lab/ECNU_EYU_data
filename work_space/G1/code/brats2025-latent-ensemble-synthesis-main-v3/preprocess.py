#!/usr/bin/env python3
"""Build G1 metadata and optionally encode complete cases into VAE latents."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import configs
import synthesis.pipeline as pipeline
import synthesis.utils as utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(configs.PATH_INPUT))
    parser.add_argument(
        "--latents-dir",
        type=Path,
        default=Path(configs.PATH_DATA) / "latents",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path(configs.PATH_DATA) / "data_csv.csv",
    )
    parser.add_argument(
        "--vae-weights",
        type=Path,
        default=Path(configs.PATH_NAME_WEIGHTS_VAE),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write data_csv.csv without loading the VAE or creating latent files.",
    )
    parser.add_argument(
        "--clean-latents",
        action="store_true",
        help="Remove the existing latent directory before encoding.",
    )
    parser.add_argument(
        "--allow-missing-seg",
        action="store_true",
        help="Keep complete four-modality cases without segmentation labels.",
    )
    parser.add_argument(
        "--respect-existing-csv",
        action="store_true",
        help="Encode only case IDs already present in data_csv.csv (post-QC allowlist).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for encoding. Default: auto (CUDA when available).",
    )
    return parser.parse_args()


def modality_from_name(name: str) -> str | None:
    for modality in configs.MODALITY_LIST:
        if name.endswith(f"-{modality}.nii.gz") or name.endswith(f"-{modality}.nii"):
            return modality
    return None


def scan_subjects(input_dir: Path, allow_missing_seg: bool = False) -> list[dict]:
    """Return complete four-modality subjects in deterministic case-ID order."""
    subjects: list[dict] = []
    for folder_path in sorted(input_dir.iterdir()):
        if not folder_path.is_dir():
            continue

        modality_map: dict[str, str] = {}
        seg_file = ""
        for path in sorted(folder_path.iterdir()):
            if not path.is_file() or not (
                path.name.endswith(".nii.gz") or path.name.endswith(".nii")
            ):
                continue
            modality = modality_from_name(path.name)
            if modality is not None:
                modality_map.setdefault(modality, path.name)
            elif path.name.endswith("-seg.nii.gz") or path.name.endswith("-seg.nii"):
                seg_file = path.name

        missing = [m for m in configs.MODALITY_LIST if m not in modality_map]
        if missing:
            print(f"  Skip {folder_path.name}: missing modalities {missing}")
            continue
        if not seg_file and not allow_missing_seg:
            print(f"  Skip {folder_path.name}: missing segmentation")
            continue

        subjects.append(
            {
                "id": folder_path.name,
                "path": folder_path,
                "modality_map": modality_map,
                "seg": seg_file,
            }
        )
    return subjects


def preprocess_subject(sub_data: dict, vae, output_latents_dir: Path) -> None:
    """Normalize and encode all four modalities for one subject."""
    subject_out_dir = output_latents_dir / sub_data["id"]
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    for modality_name in configs.MODALITY_LIST:
        file_name = sub_data["modality_map"][modality_name]
        file_path = sub_data["path"] / file_name
        img, affine = utils.load_nifti(file_path)
        img, _ = utils.preprocessing(img, affine=affine)
        latent = pipeline.encode_image(img, vae)
        latent = np.expand_dims(latent, 0)
        base = file_name.removesuffix(".nii.gz").removesuffix(".nii")
        np.save(subject_out_dir / f"{base}_latent.npy", latent)


def read_existing_splits(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        return {
            str(row.get("id", "")): str(row.get("split", "train"))
            for row in csv.DictReader(handle)
            if row.get("id")
        }


def write_data_csv(csv_path: Path, subjects: list[dict], split_by_id: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "seg", *configs.MODALITY_LIST, "split"]
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subject in subjects:
            writer.writerow(
                {
                    "id": subject["id"],
                    "seg": subject["seg"],
                    **subject["modality_map"],
                    "split": split_by_id.get(subject["id"], "train"),
                }
            )
    os.replace(temp_path, csv_path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"Scanning subjects in: {input_dir}")
    subjects = scan_subjects(input_dir, allow_missing_seg=args.allow_missing_seg)
    print(f"Found {len(subjects)} complete subjects.")
    if not subjects:
        raise RuntimeError("No complete four-modality subjects were found.")

    split_by_id = read_existing_splits(args.data_csv)
    if args.respect_existing_csv:
        if not split_by_id:
            raise ValueError(
                f"--respect-existing-csv requires a populated existing CSV: {args.data_csv}"
            )
        subjects = [subject for subject in subjects if subject["id"] in split_by_id]
        print(f"QC allowlist retained {len(subjects)} subjects from the existing CSV.")
        if not subjects:
            raise RuntimeError("The existing CSV allowlist removed every subject.")
    write_data_csv(args.data_csv, subjects, split_by_id)
    print(f"CSV saved to: {args.data_csv}")

    if args.metadata_only:
        print("Metadata-only mode: VAE encoding was not run.")
        return

    vae_weights = args.vae_weights.resolve()
    if not vae_weights.is_file():
        raise FileNotFoundError(f"VAE weights not found: {vae_weights}")
    if args.clean_latents and args.latents_dir.exists():
        shutil.rmtree(args.latents_dir)
    args.latents_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading VAE: {vae_weights}")
    vae = pipeline.instantiate_vae_model(device, weights_path=vae_weights)

    failures: list[dict[str, str]] = []
    for subject in tqdm(subjects, desc="Encoding subjects"):
        try:
            preprocess_subject(subject, vae, args.latents_dir)
        except Exception as exc:
            failures.append({"id": subject["id"], "error": repr(exc)})
            print(f"ERROR: {subject['id']}: {exc}")

    failure_csv = args.data_csv.parent / "preprocess_failures.csv"
    if failures:
        with failure_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(
            f"VAE encoding failed for {len(failures)} subjects; see {failure_csv}."
        )
    failure_csv.unlink(missing_ok=True)
    print(f"Latents saved to: {args.latents_dir}")
    print(f"Encoded subjects: {len(subjects)}")


if __name__ == "__main__":
    main()
