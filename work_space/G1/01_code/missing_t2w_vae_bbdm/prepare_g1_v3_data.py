#!/usr/bin/env python3
"""Prepare G1 V3 data links from the raw BraTS-MET Task1 directory.

The training code expects:
  data/input/<case_id>/           complete t1n/t1c/t2w/t2f plus seg
  data/input_inference/<case_id>/ missing t2w, with t1n/t1c/t2f plus seg

This script builds those two folders from one raw data root without copying the
large NIfTI files by default. Corrected labels are preferred when present.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
from pathlib import Path


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
INFERENCE_MODALITIES = ("t1n", "t1c", "t2f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/input and data/input_inference for G1 V3."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Task1 raw data root, e.g. raw_task1_2026 on the server.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Code-local data directory. Default: data",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        action="append",
        default=None,
        help="Directory containing complete training cases. Can be repeated.",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        action="append",
        default=None,
        help="Directory containing cases that truly miss t2w. Can be repeated.",
    )
    parser.add_argument(
        "--corrected-labels-root",
        type=Path,
        default=None,
        help="Directory with corrected <case_id>-seg.nii.gz files.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Place files as symlinks or physical copies. Default: symlink",
    )
    parser.add_argument("--clean", action="store_true", help="Clear input folders first.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing links/files.")
    parser.add_argument(
        "--allow-training-without-seg",
        action="store_true",
        help="Allow complete four-modality cases without seg in data/input.",
    )
    parser.add_argument(
        "--expected-fake-t2w-count",
        type=int,
        default=None,
        help="Fail unless gzip-header detection finds exactly this many fake T2W cases.",
    )
    return parser.parse_args()


def is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii.gz") or path.name.endswith(".nii")


def modality_from_name(name: str) -> str | None:
    for mod in (*MODALITIES, "seg"):
        if f"-{mod}.nii" in name or name in (f"{mod}.nii", f"{mod}.nii.gz"):
            return mod
    return None


def gzip_original_filename(path: Path) -> str:
    """Read the optional original filename field from an RFC 1952 gzip header."""
    if not path.name.endswith(".gz"):
        return ""
    with path.open("rb") as handle:
        header = handle.read(10)
        if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
            raise ValueError(f"Invalid gzip header: {path}")
        flags = header[3]
        if flags & 0x04:
            raw_length = handle.read(2)
            if len(raw_length) != 2:
                raise ValueError(f"Truncated gzip extra field: {path}")
            extra_length = struct.unpack("<H", raw_length)[0]
            if len(handle.read(extra_length)) != extra_length:
                raise ValueError(f"Truncated gzip extra payload: {path}")
        if not flags & 0x08:
            return ""
        filename = bytearray()
        while len(filename) <= 4096:
            byte = handle.read(1)
            if not byte:
                raise ValueError(f"Truncated gzip filename field: {path}")
            if byte == b"\x00":
                return filename.decode("latin-1")
            filename.extend(byte)
        raise ValueError(f"Gzip filename field is unexpectedly long: {path}")


def inspect_t2w(files: dict[str, Path]) -> tuple[str, bool]:
    path = files.get("t2w")
    if path is None:
        return "", False
    original_filename = gzip_original_filename(path)
    return original_filename, "fake" in original_filename.lower()


def find_case_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    case_dirs: list[Path] = []
    for path in root.rglob("BraTS-MET-*"):
        if path.is_dir() and any(is_nifti(child) for child in path.iterdir()):
            case_dirs.append(path)
    return sorted(set(case_dirs))


def scan_case(case_dir: Path, corrected_root: Path | None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for item in sorted(case_dir.iterdir()):
        if not item.is_file() or not is_nifti(item):
            continue
        mod = modality_from_name(item.name)
        if mod is not None and mod not in files:
            files[mod] = item

    case_id = case_dir.name
    if corrected_root is not None:
        for suffix in (".nii.gz", ".nii"):
            corrected = corrected_root / f"{case_id}-seg{suffix}"
            if corrected.exists():
                files["seg"] = corrected
                break
    return files


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def place_case(
    case_id: str,
    files: dict[str, Path],
    dst_root: Path,
    required_modalities: tuple[str, ...],
    include_t2w: bool,
    mode: str,
    overwrite: bool,
) -> None:
    dst_case = dst_root / case_id
    for mod in required_modalities:
        src = files[mod]
        extension = ".nii.gz" if src.name.endswith(".nii.gz") else ".nii"
        place_file(src, dst_case / f"{case_id}-{mod}{extension}", mode, overwrite)
    if include_t2w:
        src = files["t2w"]
        extension = ".nii.gz" if src.name.endswith(".nii.gz") else ".nii"
        place_file(src, dst_case / f"{case_id}-t2w{extension}", mode, overwrite)
    if "seg" in files:
        seg_name = f"{case_id}-seg.nii.gz" if files["seg"].name.endswith(".nii.gz") else f"{case_id}-seg.nii"
        place_file(files["seg"], dst_case / seg_name, mode, overwrite)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "source_dir",
        "target",
        "status",
        "reason",
        "label_source",
        "gzip_original_t2w_filename",
        "t2w_is_fake_by_gzip_header",
        "t1n",
        "t1c",
        "t2w",
        "t2f",
        "seg",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    data_dir = args.data_dir.resolve()
    corrected_root = (
        args.corrected_labels_root.resolve()
        if args.corrected_labels_root is not None
        else raw_root / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
    )
    if not corrected_root.exists():
        corrected_root = None

    default_training_root = raw_root / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
    train_roots = args.train_root or [
        default_training_root if default_training_root.exists() else raw_root
    ]
    inference_roots = args.inference_root or [raw_root]

    input_dir = data_dir / "input"
    inference_dir = data_dir / "input_inference"
    if args.clean:
        reset_dir(input_dir)
        reset_dir(inference_dir)
    else:
        input_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    placed_training: set[str] = set()
    placed_inference: set[str] = set()
    fake_t2w_cases: set[str] = set()

    for root in train_roots:
        for case_dir in find_case_dirs(root):
            case_id = case_dir.name
            files = scan_case(case_dir, corrected_root)
            original_t2w_filename, fake_t2w = inspect_t2w(files)
            if fake_t2w:
                fake_t2w_cases.add(case_id)
            missing = [mod for mod in MODALITIES if mod not in files]
            has_inference_inputs = (
                all(mod in files for mod in INFERENCE_MODALITIES) and "seg" in files
            )
            if has_inference_inputs and ("t2w" not in files or fake_t2w):
                place_case(
                    case_id,
                    files,
                    inference_dir,
                    INFERENCE_MODALITIES,
                    include_t2w=False,
                    mode=args.mode,
                    overwrite=args.overwrite,
                )
                placed_inference.add(case_id)
                status = "placed"
                reason = "fake_t2w_removed" if fake_t2w else "missing_t2w"
                target = "input_inference"
            elif missing:
                status = "skipped"
                reason = "training_missing_" + ",".join(missing)
                target = ""
            elif "seg" not in files and not args.allow_training_without_seg:
                status = "skipped"
                reason = "training_missing_seg"
                target = ""
            else:
                place_case(
                    case_id,
                    files,
                    input_dir,
                    ("t1n", "t1c", "t2f"),
                    include_t2w=True,
                    mode=args.mode,
                    overwrite=args.overwrite,
                )
                placed_training.add(case_id)
                status = "placed"
                reason = ""
                target = "input"
            label_source = "corrected" if "seg" in files and corrected_root and files["seg"].is_relative_to(corrected_root) else "raw"
            rows.append(
                row_for_case(
                    case_id,
                    case_dir,
                    files,
                    target,
                    status,
                    reason,
                    label_source,
                    original_t2w_filename,
                    fake_t2w,
                )
            )

    for root in inference_roots:
        for case_dir in find_case_dirs(root):
            case_id = case_dir.name
            if case_id in placed_training or case_id in placed_inference:
                continue
            files = scan_case(case_dir, corrected_root)
            original_t2w_filename, fake_t2w = inspect_t2w(files)
            if fake_t2w:
                fake_t2w_cases.add(case_id)
            has_required = all(mod in files for mod in INFERENCE_MODALITIES)
            if has_required and ("t2w" not in files or fake_t2w) and "seg" in files:
                place_case(
                    case_id,
                    files,
                    inference_dir,
                    INFERENCE_MODALITIES,
                    include_t2w=False,
                    mode=args.mode,
                    overwrite=args.overwrite,
                )
                placed_inference.add(case_id)
                label_source = "corrected" if corrected_root and files["seg"].is_relative_to(corrected_root) else "raw"
                rows.append(
                    row_for_case(
                        case_id,
                        case_dir,
                        files,
                        "input_inference",
                        "placed",
                        "fake_t2w_removed" if fake_t2w else "missing_t2w",
                        label_source,
                        original_t2w_filename,
                        fake_t2w,
                    )
                )

    if (
        args.expected_fake_t2w_count is not None
        and len(fake_t2w_cases) != args.expected_fake_t2w_count
    ):
        raise RuntimeError(
            f"Detected {len(fake_t2w_cases)} fake T2W cases, expected "
            f"{args.expected_fake_t2w_count}. Refusing to build a contaminated split."
        )

    manifest_path = data_dir / "g1_v3_data_placement_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Raw root: {raw_root}")
    print(f"Training cases placed:  {len(placed_training)} -> {input_dir}")
    print(f"Inference cases placed: {len(placed_inference)} -> {inference_dir}")
    print(f"Fake T2W cases removed by gzip header: {len(fake_t2w_cases)}")
    print(f"Manifest: {manifest_path}")
    if len(placed_inference) == 0:
        print("NOTE: No true missing-T2W inference cases were found under the selected roots.")


def row_for_case(
    case_id: str,
    case_dir: Path,
    files: dict[str, Path],
    target: str,
    status: str,
    reason: str,
    label_source: str,
    original_t2w_filename: str,
    fake_t2w: bool,
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "source_dir": str(case_dir),
        "target": target,
        "status": status,
        "reason": reason,
        "label_source": label_source if "seg" in files else "",
        "gzip_original_t2w_filename": original_t2w_filename,
        "t2w_is_fake_by_gzip_header": str(fake_t2w),
        "t1n": str(files.get("t1n", "")),
        "t1c": str(files.get("t1c", "")),
        "t2w": str(files.get("t2w", "")),
        "t2f": str(files.get("t2f", "")),
        "seg": str(files.get("seg", "")),
    }


if __name__ == "__main__":
    main()
