#!/usr/bin/env python3
"""Build audited source-ID symlink views for evaluation-only Table 7 r3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


SOURCE_ID_RE = re.compile(r"BraTS-MET-[0-9]{5}-[0-9]{3}")
NNUNET_ID_RE = re.compile(r"BraTSMET_[0-9]{6}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    json.loads(path.read_text(encoding="utf-8"))


def load_case_contract(
    test_list: Path,
    mapping_csv: Path,
    expected_count: int,
) -> list[tuple[str, str]]:
    nnunet_ids = [
        line.strip()
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(nnunet_ids) == expected_count, "test-list count drift")
    require(len(set(nnunet_ids)) == expected_count, "duplicate test-list ID")
    require(all(NNUNET_ID_RE.fullmatch(case_id) for case_id in nnunet_ids), "invalid nnU-Net ID")

    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1295, f"master mapping row-count drift: {len(rows)}")
    require(
        {"nnunet_case_id", "source_case_id"} <= set(rows[0]),
        "master mapping columns missing",
    )
    by_nnunet: dict[str, str] = {}
    all_source_ids: set[str] = set()
    for row in rows:
        nnunet_id = row["nnunet_case_id"]
        source_id = row["source_case_id"]
        require(nnunet_id not in by_nnunet, f"duplicate mapping nnU-Net ID: {nnunet_id}")
        require(source_id not in all_source_ids, f"duplicate mapping source ID: {source_id}")
        require(SOURCE_ID_RE.fullmatch(source_id) is not None, f"invalid source ID: {source_id}")
        by_nnunet[nnunet_id] = source_id
        all_source_ids.add(source_id)

    missing = sorted(set(nnunet_ids) - set(by_nnunet))
    require(not missing, f"test IDs missing from master mapping: {missing}")
    pairs = [(case_id, by_nnunet[case_id]) for case_id in nnunet_ids]
    require(len({source_id for _, source_id in pairs}) == expected_count, "source IDs are not bijective")
    return pairs


def discover_ids(root: Path) -> list[str]:
    require(root.is_dir(), f"missing source directory: {root}")
    paths = sorted(root.glob("*.nii.gz"))
    for path in paths:
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty source: {path}")
    return [path.name.removesuffix(".nii.gz") for path in paths]


def geometry(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    image = nib.load(str(path))
    return tuple(int(value) for value in image.shape), np.asarray(image.affine, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-list", required=True, type=Path)
    parser.add_argument("--mapping-csv", required=True, type=Path)
    parser.add_argument("--r2-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--view-root", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--expected-mapping-sha256", required=True)
    parser.add_argument("--expected-pair-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()

    test_list = args.test_list.resolve(strict=True)
    mapping_csv = args.mapping_csv.resolve(strict=True)
    r2_root = args.r2_root.resolve(strict=True)
    r2_prediction = r2_root / "prediction"
    reference_root = args.reference_root.resolve(strict=True)
    view_root = args.view_root
    audit_json = args.audit_json

    require(sha256_file(test_list) == args.expected_test_sha256, "test-list SHA drift")
    require(sha256_file(mapping_csv) == args.expected_mapping_sha256, "mapping CSV SHA drift")
    require(not view_root.exists(), f"exclusive view root exists: {view_root}")
    require(not audit_json.exists(), f"exclusive audit exists: {audit_json}")

    inference_path = r2_root / "evidence/INFERENCE_VALIDATION.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    require(inference["status"] == "pass", "r2 inference did not pass")
    require(inference["case_count"] == args.expected_count, "r2 inference count drift")
    require(
        inference["checkpoint_sha256"] == args.expected_checkpoint_sha256,
        "r2 checkpoint SHA drift",
    )

    r2_official_path = r2_root / "official_reproduction/panoptica_evaluation_summary.json"
    r2_official = json.loads(r2_official_path.read_text(encoding="utf-8"))
    require(len(r2_official.get("metrics", [])) == 16, "r2 failure metric-count drift")
    require(len(r2_official.get("missings", [])) == 88, "r2 failure missing-count drift")

    pairs = load_case_contract(test_list, mapping_csv, args.expected_count)
    pair_lines = "".join(f"{nnunet_id},{source_id}\n" for nnunet_id, source_id in sorted(pairs))
    pair_sha256 = hashlib.sha256(pair_lines.encode("utf-8")).hexdigest()
    require(pair_sha256 == args.expected_pair_sha256, "104-case mapping pair SHA drift")

    expected_internal_ids = sorted(nnunet_id for nnunet_id, _ in pairs)
    require(discover_ids(r2_prediction) == expected_internal_ids, "r2 prediction inventory drift")
    require(discover_ids(reference_root) == expected_internal_ids, "reference inventory drift")

    prediction_view = view_root / "prediction"
    reference_view = view_root / "reference"
    view_root.mkdir(parents=True)
    prediction_view.mkdir()
    reference_view.mkdir()

    rows: list[dict[str, object]] = []
    for nnunet_id, source_id in pairs:
        prediction_source = (r2_prediction / f"{nnunet_id}.nii.gz").resolve(strict=True)
        reference_source = (reference_root / f"{nnunet_id}.nii.gz").resolve(strict=True)
        prediction_shape, prediction_affine = geometry(prediction_source)
        reference_shape, reference_affine = geometry(reference_source)
        require(prediction_shape == reference_shape, f"shape mismatch after mapping: {nnunet_id}")
        require(
            np.allclose(prediction_affine, reference_affine, rtol=0, atol=1e-5),
            f"affine mismatch after mapping: {nnunet_id}",
        )

        prediction_link = prediction_view / f"{source_id}.nii.gz"
        reference_link = reference_view / f"{source_id}.nii.gz"
        os.symlink(str(prediction_source), prediction_link)
        os.symlink(str(reference_source), reference_link)
        require(prediction_link.is_symlink(), f"prediction view is not a symlink: {source_id}")
        require(reference_link.is_symlink(), f"reference view is not a symlink: {source_id}")

        rows.append(
            {
                "nnunet_case_id": nnunet_id,
                "source_case_id": source_id,
                "prediction_source": str(prediction_source),
                "reference_source": str(reference_source),
                "prediction_link": str(prediction_link.absolute()),
                "reference_link": str(reference_link.absolute()),
                "prediction_bytes": prediction_source.stat().st_size,
                "reference_bytes": reference_source.stat().st_size,
                "prediction_sha256": sha256_file(prediction_source),
                "reference_sha256": sha256_file(reference_source),
                "shape": "x".join(str(value) for value in reference_shape),
                "geometry_match": True,
            }
        )

    require(len(discover_ids(prediction_view)) == args.expected_count, "prediction view count drift")
    require(len(discover_ids(reference_view)) == args.expected_count, "reference view count drift")

    csv_path = audit_json.parent / "CASE_ID_MAPPING_AND_SYMLINKS.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_snapshot = {
        "r2_job_id": str(inference["job_id"]),
        "r2_inference_audit_sha256": sha256_file(inference_path),
        "r2_official_summary_sha256": sha256_file(r2_official_path),
        "r2_official_metrics_count": len(r2_official["metrics"]),
        "r2_official_missings_count": len(r2_official["missings"]),
        "r2_failed_evidence_present": (r2_root / "evidence/TABLE7_DISJOINT_MICROF1_FAILED.json").exists(),
        "r2_prediction_inventory_sha256": canonical_sha256(
            [
                {
                    "nnunet_case_id": row["nnunet_case_id"],
                    "bytes": row["prediction_bytes"],
                    "sha256": row["prediction_sha256"],
                }
                for row in rows
            ]
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "case_count": len(rows),
        "mapping_row_count": 1295,
        "source_id_bijection": True,
        "geometry_mismatch_count": 0,
        "test_list_sha256": sha256_file(test_list),
        "mapping_csv_sha256": sha256_file(mapping_csv),
        "case_pair_sha256": pair_sha256,
        "mapping_csv_audit_sha256": sha256_file(csv_path),
        "r2_reuse": source_snapshot,
        "view_inventory_sha256": canonical_sha256(rows),
    }
    write_json_exclusive(audit_json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
