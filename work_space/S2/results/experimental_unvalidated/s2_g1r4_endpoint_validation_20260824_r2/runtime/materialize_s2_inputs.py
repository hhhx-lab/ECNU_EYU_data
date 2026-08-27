#!/usr/bin/env python3
"""Materialize audited four-channel S2 inputs without copying NIfTI payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


CHANNELS = (("t1n", "0000"), ("t1c", "0001"), ("t2w", "0002"), ("t2f", "0003"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--route", required=True, choices=("real", "synthetic"))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    input_root = args.input_root.resolve()
    reference_root = args.reference_root.resolve()
    require(source_root.is_dir(), f"missing source root: {source_root}")
    require(not input_root.exists() and not reference_root.exists(), "exclusive input/reference target exists")
    require(not args.audit_json.exists() and not args.audit_csv.exists(), "audit target exists")
    synthetic_root = args.synthetic_root.resolve() if args.synthetic_root else None
    if args.route == "synthetic":
        require(synthetic_root is not None and synthetic_root.is_dir(), "synthetic route requires an existing synthetic root")

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == args.expected_count, "manifest count drift")
    ids = [row["source_case_id"].strip() for row in rows]
    require(len(ids) == len(set(ids)), "duplicate case IDs")
    input_root.mkdir(parents=True, exist_ok=False)
    reference_root.mkdir(parents=True, exist_ok=False)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    audits: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["source_case_id"].strip()
        paths = {
            modality: Path(row[f"{modality}_source_path"]).resolve()
            for modality in ("t1n", "t1c", "t2f", "seg")
        }
        for modality, path in paths.items():
            require(path.is_relative_to(source_root), f"{modality} escaped source root: {path}")
            require(path.is_file() and path.stat().st_size > 0, f"missing {modality}: {path}")
        if args.route == "real":
            require(row["source_t2w_allowed"].strip().lower() == "true", f"real T2W forbidden for {case_id}")
            t2w = Path(row["t2w_source_path"]).resolve()
            require(t2w.is_relative_to(source_root), f"real T2W escaped source root: {t2w}")
        else:
            require(row["source_t2w_allowed"].strip().lower() in {"true", "false"}, "invalid source T2W policy")
            t2w = Path(row["synthesized_t2w_path"]).resolve()
            require(t2w.is_relative_to(synthetic_root), f"synthetic T2W escaped frozen root: {t2w}")
        require(t2w.is_file() and t2w.stat().st_size > 0, f"missing route T2W: {t2w}")
        paths["t2w"] = t2w

        reference = nib.load(str(paths["seg"]))
        reference_data = np.asanyarray(reference.dataobj)
        require(np.isfinite(reference_data).all(), f"nonfinite segmentation: {case_id}")
        labels = sorted(int(value) for value in np.unique(reference_data))
        require(set(labels).issubset({0, 1, 2, 3, 4}), f"invalid labels: {case_id}: {labels}")
        modality_hashes: dict[str, str] = {}
        for modality, _ in CHANNELS:
            image = nib.load(str(paths[modality]))
            data = np.asanyarray(image.dataobj)
            require(np.isfinite(data).all(), f"nonfinite {modality}: {case_id}")
            require(tuple(image.shape) == tuple(reference.shape), f"shape mismatch {case_id} {modality}")
            require(np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-5), f"affine mismatch {case_id} {modality}")
            modality_hashes[modality] = sha256_file(paths[modality])

        for modality, channel in CHANNELS:
            target = input_root / f"{case_id}_{channel}.nii.gz"
            target.symlink_to(paths[modality])
        reference_target = reference_root / f"{case_id}.nii.gz"
        reference_target.symlink_to(paths["seg"])
        audits.append(
            {
                "source_case_id": case_id,
                "nnunet_case_id": row["nnunet_case_id"].strip(),
                "route": args.route,
                "t2w_role": "authentic" if args.route == "real" else "r4_ensemble_synthesized",
                "t2w_path": str(t2w),
                "shape": "x".join(map(str, reference.shape)),
                "spacing": "x".join(f"{value:.8g}" for value in reference.header.get_zooms()[:3]),
                "geometry_mismatch": False,
                "geometry_repaired": False,
                "labels": ";".join(map(str, labels)),
                **{f"{modality}_sha256": digest for modality, digest in modality_hashes.items()},
                "seg_sha256": sha256_file(paths["seg"]),
            }
        )

    require(len(list(input_root.glob("*.nii.gz"))) == args.expected_count * 4, "input file count drift")
    require(len(list(reference_root.glob("*.nii.gz"))) == args.expected_count, "reference file count drift")
    write_csv_exclusive(args.audit_csv, audits)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": args.route,
        "case_count": args.expected_count,
        "channel_order": {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"},
        "input_file_count": args.expected_count * 4,
        "reference_file_count": args.expected_count,
        "geometry_mismatch_count": 0,
        "geometry_repaired_count": 0,
        "source_t2w_used": args.route == "real",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "audit_csv_sha256": sha256_file(args.audit_csv),
    }
    write_json_exclusive(args.audit_json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

