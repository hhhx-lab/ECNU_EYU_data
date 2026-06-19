#!/usr/bin/env python3
"""Prepare G1 T2W inpainting data from G2 manifests.

This script does not copy large NIfTI files by default. It creates a clean
folder layout expected by the new G1 missing-modality code:

data/input/<case_id>/            complete subjects with t1n/t1c/t2w/t2f/seg
data/input_inference/<case_id>/  subjects whose T2W should be synthesized;
                                 t2w is intentionally omitted

The default policy is:
- Training input: final-QC-pass cases whose T2W is not flagged as fake.
- Inference input: final-QC-pass cases whose T2W is flagged as fake by G2.

The script fails by default if any required source NIfTI is missing. This makes
stale Mac paths or incomplete server mounts fail early instead of creating a
misleading folder layout.

Use --mode copy only on a machine with enough disk space.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
G2_RESULTS = PROJECT_ROOT / "work_space" / "G2" / "results"
DEFAULT_REAL_MANIFEST = G2_RESULTS / "manifests" / "real_train_manifest.csv"
DEFAULT_FAKE_T2W = G2_RESULTS / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fake_t2w_cases(path: Path) -> set[str]:
    if not path.exists():
        audit_path = path.with_name("official_t2w_gzip_header_audit_2026-06-15.csv")
        if audit_path.exists():
            rows = read_csv(audit_path)
            cases: set[str] = set()
            for row in rows:
                flag = str(row.get("t2w_is_fake_by_gzip_header", "")).strip().lower()
                case_id = row.get("case_id") or row.get("subject_id") or row.get("id")
                if case_id and flag in {"1", "true", "yes", "y"}:
                    cases.add(case_id)
            if cases:
                return cases
        raise FileNotFoundError(f"fake T2W manifest not found: {path}")
    rows = read_csv(path)
    cases: set[str] = set()
    for row in rows:
        case_id = row.get("case_id") or row.get("subject_id") or row.get("id")
        if case_id:
            cases.add(case_id)
    return cases


def remove_existing(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"target exists; use --overwrite: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    if not src.exists():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_existing(dst, overwrite)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "manifest-only":
        return "planned"
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def case_source_path(row: dict[str, str], key: str) -> Path | None:
    candidates = []
    if key == "seg":
        candidates = ["effective_seg_path", "raw_seg_path", "seg_source_path"]
    else:
        candidates = [f"{key}_path", f"{key}_source_path"]
    for col in candidates:
        value = row.get(col, "")
        if value:
            return Path(value)
    return None


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "destination_split",
        "is_fake_t2w",
        "final_qc_pass",
        "linked_t1n",
        "linked_t1c",
        "linked_t2w",
        "linked_t2f",
        "linked_seg",
        "case_dir",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> list[dict[str, object]]:
    real_rows = read_csv(Path(args.real_manifest))
    fake_cases = fake_t2w_cases(Path(args.fake_t2w_cases))
    data_root = Path(args.data_root).expanduser().resolve()
    train_root = data_root / "input"
    infer_root = data_root / "input_inference"

    if args.clean:
        remove_existing(train_root, overwrite=True)
        remove_existing(infer_root, overwrite=True)
    train_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, object]] = []
    counts = {"train": 0, "inference": 0, "skipped": 0}

    for row in real_rows:
        case_id = row.get("case_id", "")
        if not case_id:
            continue
        final_qc_pass = boolish(row.get("final_qc_pass", ""))
        if not final_qc_pass and not args.include_failed_qc:
            counts["skipped"] += 1
            continue

        is_fake = case_id in fake_cases
        destination_split = "inference" if is_fake else "train"
        case_dir = (infer_root if is_fake else train_root) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        linked: dict[str, str] = {}
        notes: list[str] = []
        modalities_to_link = ("t1n", "t1c", "t2f") if is_fake else MODALITIES
        for mod in modalities_to_link:
            src = case_source_path(row, mod)
            dst = case_dir / f"{case_id}-{mod}.nii.gz"
            action = link_or_copy(src, dst, args.mode, args.overwrite) if src is not None else "missing_source"
            linked[mod] = action
            if action == "missing_source":
                notes.append(f"missing_{mod}")

        seg_src = case_source_path(row, "seg")
        seg_dst = case_dir / f"{case_id}-seg.nii.gz"
        linked["seg"] = link_or_copy(seg_src, seg_dst, args.mode, args.overwrite) if seg_src is not None else "missing_source"
        if linked["seg"] == "missing_source":
            notes.append("missing_seg")

        if is_fake:
            linked["t2w"] = "omitted_for_inference"

        output_rows.append(
            {
                "case_id": case_id,
                "destination_split": destination_split,
                "is_fake_t2w": is_fake,
                "final_qc_pass": final_qc_pass,
                "linked_t1n": linked.get("t1n", ""),
                "linked_t1c": linked.get("t1c", ""),
                "linked_t2w": linked.get("t2w", ""),
                "linked_t2f": linked.get("t2f", ""),
                "linked_seg": linked.get("seg", ""),
                "case_dir": str(case_dir),
                "notes": ";".join(notes),
            }
        )
        counts[destination_split] += 1

    write_manifest(data_root / "g1_data_placement_manifest.csv", output_rows)
    missing_rows = [row for row in output_rows if row["notes"]]
    print(f"data_root={data_root}")
    print(f"mode={args.mode}")
    print(f"train_complete_cases={counts['train']}")
    print(f"inference_missing_t2w_cases={counts['inference']}")
    print(f"skipped_cases={counts['skipped']}")
    print(f"manifest={data_root / 'g1_data_placement_manifest.csv'}")
    print(f"missing_source_cases={len(missing_rows)}")
    if missing_rows and not args.allow_missing_sources:
        preview = ", ".join(f"{row['case_id']}({row['notes']})" for row in missing_rows[:10])
        raise SystemExit(
            "missing source NIfTI files detected; refresh G2 manifests with server paths "
            f"or pass --allow-missing-sources for a diagnostic dry run. examples: {preview}"
        )
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare G1 missing-T2W folders from G2 manifests.")
    parser.add_argument("--real-manifest", default=str(DEFAULT_REAL_MANIFEST))
    parser.add_argument("--fake-t2w-cases", default=str(DEFAULT_FAKE_T2W))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--mode", choices=["symlink", "copy", "manifest-only"], default="symlink")
    parser.add_argument("--clean", action="store_true", help="Remove existing data/input and data/input_inference first.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files/links.")
    parser.add_argument("--include-failed-qc", action="store_true", help="Include cases that G2 final QC failed.")
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Write the placement manifest even if source NIfTI paths are missing. Default fails fast.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
