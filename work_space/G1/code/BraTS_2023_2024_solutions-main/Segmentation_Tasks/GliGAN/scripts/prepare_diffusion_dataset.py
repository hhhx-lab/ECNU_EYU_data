#!/usr/bin/env python3
"""Prepare MET diffusion training data from the G2 real-data manifest.

This script creates a per-case directory tree under the shared workspace data
cache so both G1 lines can read from the same project-level raw data root. It
does not copy NIfTI volumes by default; it creates symlinks so the large raw
data stay outside the repo.

Default selection policy:
- use real cases that passed G2 final QC
- keep only cases with a complete t1n/t1c/t2w/t2f/seg set
- exclude the 265 fake/broken T2W cases identified by G2
- always clear any existing dataset-root content before rebuilding the tree
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "work_space" / "G1").exists() and (parent / "work_space" / "G2").exists():
            return parent
    raise RuntimeError(f"Could not locate ECNU_EYU_data project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "work_space" / "G1" / "data" / "diffusion_cache"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "work_space" / "G2" / "results"
DEFAULT_TRAIN_MANIFEST = DEFAULT_RESULTS_ROOT / "manifests" / "real_train_manifest.csv"
DEFAULT_FAKE_T2W = DEFAULT_RESULTS_ROOT / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"
MODALITIES = ("t1n", "t1c", "t2w", "t2f", "seg")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_case_ids(path: Path) -> set[str]:
    rows = read_csv(path)
    ids: set[str] = set()
    for row in rows:
        case_id = row.get("case_id") or row.get("id")
        if case_id:
            ids.add(case_id)
    return ids


def find_case_path(case_dir: str, modality: str) -> Path | None:
    base = Path(case_dir)
    candidates = [
        base / f"{base.name}-{modality}.nii.gz",
        base / f"{base.name}-scan_{modality}.nii.gz",
    ]
    if modality == "seg":
        candidates = [
            base / f"{base.name}-seg.nii.gz",
            base / f"{base.name}-label.nii.gz",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_link(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported mode: {mode}")


def assert_met_case_id(case_id: str) -> None:
    if not case_id.startswith("BraTS-MET-"):
        raise SystemExit(f"refuse non-MET case id: {case_id}")


def clean_dataset_root(out_root: Path) -> None:
    if not out_root.exists():
        return
    for child in out_root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS-MET diffusion dataset from G2 manifests.")
    parser.add_argument("--train-manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--fake-t2w-manifest", default=str(DEFAULT_FAKE_T2W))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    train_rows = read_csv(Path(args.train_manifest))
    fake_case_ids = load_case_ids(Path(args.fake_t2w_manifest))
    out_root = Path(args.dataset_root)
    clean_dataset_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str]] = []
    for row in train_rows:
        case_id = row.get("case_id", "")
        case_dir = row.get("case_dir", "")
        if not case_id or not case_dir:
            continue
        assert_met_case_id(case_id)
        if row.get("final_qc_pass", "").lower() != "true":
            continue
        if case_id in fake_case_ids:
            continue
        if row.get("has_t2w", "").lower() != "true":
            continue
        if not all(row.get(f"has_{mod}", "").lower() == "true" for mod in ("t1n", "t1c", "t2w", "t2f", "seg")):
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    manifest_path = out_root.parent / "diffusion_dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "case_dir", "t1n", "t1c", "t2w", "t2f", "seg"])
        for row in selected:
            case_id = row["case_id"]
            assert_met_case_id(case_id)
            case_dir = Path(row["case_dir"])
            case_out = out_root / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            paths = {}
            for mod in MODALITIES:
                src = find_case_path(str(case_dir), mod)
                if src is None:
                    raise SystemExit(f"missing {mod} for {case_id}: {case_dir}")
                dst = case_out / f"{case_id}-{mod}.nii.gz"
                make_link(src, dst, args.mode, args.overwrite)
                paths[mod] = str(dst)
            writer.writerow([case_id, str(case_dir), paths["t1n"], paths["t1c"], paths["t2w"], paths["t2f"], paths["seg"]])

    print(f"selected_cases={len(selected)}")
    print(f"dataset_root={out_root}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
