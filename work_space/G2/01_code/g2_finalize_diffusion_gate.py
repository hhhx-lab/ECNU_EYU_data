#!/usr/bin/env python3
"""Freeze the approved four-modality Diffusion checkpoint and final G2 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODALITIES = ("t1c", "t1n", "t2w", "t2f")
PASS_DECISIONS = {"pass_technical_visual", "pass_with_documented_risk"}
SMOKE_RISK_IDS = {
    "BraTS-MET-01134-003",
    "BraTS-MET-01250-001",
    "BraTS-MET-01191-003",
    "BraTS-MET-01268-002",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def required_review_ids(
    review_rows: list[dict[str, str]], *, low_score_count: int = 10
) -> tuple[set[str], dict[str, list[str]]]:
    ordered = sorted(
        review_rows,
        key=lambda row: (float(row["min_tumour_ssim"]), row["source_case_id"]),
    )
    low_ids = {row["source_case_id"] for row in ordered[:low_score_count]}
    reasons: dict[str, list[str]] = {}
    for row in review_rows:
        case_id = row["source_case_id"]
        case_reasons: list[str] = []
        if as_bool(row.get("has_rc")):
            case_reasons.append("rc")
        if int(row.get("tiny_count", 0)) > 0:
            case_reasons.append("tiny")
        if int(row.get("large_count", 0)) > 0:
            case_reasons.append("large")
        if int(row.get("artifact_flag_count", 0)) > 0:
            case_reasons.append("artifact")
        if "large_tiled_support" in row.get("artifact_flags", ""):
            case_reasons.append("tiled")
        if case_id in low_ids:
            case_reasons.append("low_score")
        if case_id in SMOKE_RISK_IDS:
            case_reasons.append("smoke_risk")
        if case_reasons:
            reasons[case_id] = sorted(set(case_reasons))
    return set(reasons), reasons


def _check_generation_metadata(metrics: dict[str, Any]) -> None:
    metadata = metrics.get("metadata", {})
    expected = {
        "dataset": "BRATS_2024",
        "split": "val",
        "evaluation_mode": "whole_brain",
        "checkpoint_step": 150000,
        "normalization": "zscore",
        "noise_schedule": "edm",
        "sampling_method": "edm_heun",
        "sampling_steps": 18,
        "seed": 20260720,
        "large_lesion_mode": "tile",
        "crop_size": 64,
        "max_cases": 0,
        "save_support_volumes": True,
        "generation_manifest_rows": 376,
    }
    differences = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if differences:
        raise ValueError(f"Generation metadata mismatch: {differences}")
    if set(metadata.get("modalities", [])) != set(MODALITIES):
        raise ValueError("Generation metadata does not contain all four modalities")


def finalize_gate(
    qc_root: Path,
    cohort_summary_path: Path,
    noop_csv_path: Path,
    generation_metrics_path: Path,
    generation_manifest_path: Path,
    checkpoint_inventory_path: Path,
    manual_decisions_path: Path,
    human_report_path: Path,
    output_root: Path,
    *,
    low_score_count: int = 10,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    selection_path = output_root / "checkpoint_selection.json"
    gate_path = output_root / "g2_diffusion_qc_gate.json"
    sha_path = output_root / "SHA256SUMS.txt"
    existing = [path for path in (selection_path, gate_path, sha_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite frozen gate files: {existing}")

    qc_summary_path = qc_root / "summary.json"
    review_index_path = qc_root / "review_index.csv"
    artifact_metrics_path = qc_root / "artifact_metrics.csv"
    for path in (
        qc_summary_path,
        review_index_path,
        artifact_metrics_path,
        cohort_summary_path,
        noop_csv_path,
        generation_metrics_path,
        generation_manifest_path,
        checkpoint_inventory_path,
        manual_decisions_path,
        human_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    qc_summary = json.loads(qc_summary_path.read_text(encoding="utf-8"))
    if qc_summary.get("technical_gate") != "pass":
        raise ValueError("G2 technical gate is not pass")
    expected_qc = {
        "case_count": 94,
        "expected_case_count": 94,
        "modality_row_count": 376,
        "artifact_row_count": 376,
        "montage_count": 94,
        "hard_failure_count": 0,
    }
    for key, expected in expected_qc.items():
        if qc_summary.get(key) != expected:
            raise ValueError(f"QC {key}={qc_summary.get(key)} != {expected}")

    cohort = json.loads(cohort_summary_path.read_text(encoding="utf-8"))
    expected_cohort = {
        "fixed_val_count": 103,
        "generated_positive_count": 94,
        "strict_noop_negative_count": 9,
        "strict_noop_pass_count": 9,
    }
    for key, expected in expected_cohort.items():
        if cohort.get(key) != expected:
            raise ValueError(f"Cohort {key}={cohort.get(key)} != {expected}")
    if not all(cohort.get("validation_pipeline_contract", {}).values()):
        raise ValueError("Validation pipeline is not isolated from OnlineDiffusion")

    noop_rows = read_csv(noop_csv_path)
    if len(noop_rows) != 9:
        raise ValueError(f"Expected 9 no-op rows, found {len(noop_rows)}")
    for row in noop_rows:
        if as_bool(row.get("was_modified")):
            raise ValueError(f"No-op case was modified: {row.get('source_case_id')}")
        if not as_bool(row.get("image_equal")) or not as_bool(row.get("seg_equal")):
            raise ValueError(f"No-op identity failed: {row.get('source_case_id')}")
        if row.get("image_sha256_before") != row.get("image_sha256_after"):
            raise ValueError(f"No-op image hash failed: {row.get('source_case_id')}")
        if row.get("seg_sha256_before") != row.get("seg_sha256_after"):
            raise ValueError(f"No-op seg hash failed: {row.get('source_case_id')}")

    metrics = json.loads(generation_metrics_path.read_text(encoding="utf-8"))
    _check_generation_metadata(metrics)
    manifest_rows = read_csv(generation_manifest_path)
    manifest_ids = {row["source_case_id"] for row in manifest_rows}
    if len(manifest_rows) != 376 or len(manifest_ids) != 94:
        raise ValueError("Generation manifest is not exactly 94 cases x 4 modalities")
    if manifest_ids != set(cohort.get("selected_source_case_ids", [])):
        raise ValueError("Generation manifest IDs differ from frozen positive cohort")

    inventory_rows = read_csv(checkpoint_inventory_path)
    inventory = {
        row["modality"]: row
        for row in inventory_rows
        if row.get("step") == "150000" and row.get("checksum_verified") == "yes"
    }
    if set(inventory) != set(MODALITIES):
        raise ValueError("Checkpoint inventory lacks four verified 150k rows")
    checkpoint_files: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        metadata = metrics["metadata"]["checkpoints"].get(modality, {})
        row = inventory[modality]
        if int(metadata.get("step", -1)) != 150000:
            raise ValueError(f"{modality} metrics step is not 150000")
        if metadata.get("sha256") != row.get("sha256"):
            raise ValueError(f"{modality} checkpoint SHA256 differs from inventory")
        if int(metadata.get("bytes", -1)) != int(row.get("bytes", -2)):
            raise ValueError(f"{modality} checkpoint size differs from inventory")
        if qc_summary.get("checkpoint_sha256", {}).get(modality) != row.get("sha256"):
            raise ValueError(f"{modality} checkpoint SHA256 differs from QC")
        checkpoint_files[modality] = {
            "step": 150000,
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "canonical_relative_path": row.get("local_canonical_relative_path", ""),
        }

    review_rows = read_csv(review_index_path)
    if len(review_rows) != 94:
        raise ValueError(f"Expected 94 review-index rows, found {len(review_rows)}")
    mandatory_ids, mandatory_reasons = required_review_ids(
        review_rows, low_score_count=low_score_count
    )
    decisions = read_csv(manual_decisions_path)
    decision_by_id = {row["source_case_id"]: row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Duplicate manual review decisions")
    unknown_decisions = sorted(set(decision_by_id) - {row["source_case_id"] for row in review_rows})
    if unknown_decisions:
        raise ValueError(f"Manual decisions contain unknown cases: {unknown_decisions[:20]}")
    missing_reviews = sorted(mandatory_ids - set(decision_by_id))
    if missing_reviews:
        raise ValueError(f"Mandatory manual reviews are missing: {missing_reviews[:20]}")
    for case_id, row in decision_by_id.items():
        decision = row.get("manual_decision")
        if decision not in PASS_DECISIONS:
            raise ValueError(f"Unapproved manual decision: {case_id}={decision}")
        if decision == "pass_with_documented_risk" and not as_bool(
            row.get("risk_accepted")
        ):
            raise ValueError(f"Documented risk is not accepted: {case_id}")

    generated_at = datetime.now(timezone.utc).isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1,
        "status": "frozen",
        "generated_at_utc": generated_at,
        "checkpoint_steps": {modality: 150000 for modality in MODALITIES},
        "checkpoint_files": checkpoint_files,
        "normalization": "zscore",
        "noise_schedule": "edm",
        "sampling_method": "edm_heun",
        "sampling_steps": 18,
        "seed": 20260720,
        "crop_size": 64,
        "large_lesion_mode": "tile",
        "selection_decision": "keep_150000_no_rollback_comparison_required",
        "selection_rationale": (
            "150k completed the fixed 94+9 gate without technical failure; "
            "145k/140k comparison was not triggered"
        ),
        "evidence_sha256": {
            "generation_metrics": sha256_file(generation_metrics_path),
            "generation_manifest": sha256_file(generation_manifest_path),
            "cohort_summary": sha256_file(cohort_summary_path),
            "g2_qc_summary": sha256_file(qc_summary_path),
            "manual_decisions": sha256_file(manual_decisions_path),
            "human_report": sha256_file(human_report_path),
        },
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    selection_sha256 = sha256_file(selection_path)
    gate = {
        "schema_version": 1,
        "decision": "approve",
        "generated_at_utc": generated_at,
        "checkpoint_selection_sha256": selection_sha256,
        "normalization": "zscore",
        "sampling_method": "edm_heun",
        "sampling_steps": 18,
        "seed": 20260720,
        "case_count": 103,
        "generated_positive_count": 94,
        "strict_noop_negative_count": 9,
        "strict_noop_pass_count": 9,
        "reviewed_case_count": len(decision_by_id),
        "mandatory_reviewed_case_count": len(mandatory_ids),
        "mandatory_review_reasons": mandatory_reasons,
        "hard_failure_count": 0,
        "checkpoint_step": 150000,
        "rollback_comparison_required": False,
        "report": str(human_report_path.resolve()),
        "source_sha256": {
            "qc_summary": sha256_file(qc_summary_path),
            "noop_csv": sha256_file(noop_csv_path),
            "manual_decisions": sha256_file(manual_decisions_path),
        },
    }
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    sha_lines = [
        f"{sha256_file(selection_path)}  {selection_path.name}",
        f"{sha256_file(gate_path)}  {gate_path.name}",
        f"{sha256_file(human_report_path)}  {human_report_path.name}",
        f"{sha256_file(manual_decisions_path)}  {manual_decisions_path.name}",
    ]
    sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-root", required=True, type=Path)
    parser.add_argument("--cohort-summary", required=True, type=Path)
    parser.add_argument("--noop-csv", required=True, type=Path)
    parser.add_argument("--generation-metrics", required=True, type=Path)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--manual-decisions", required=True, type=Path)
    parser.add_argument("--human-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--low-score-count", type=int, default=10)
    args = parser.parse_args()
    gate = finalize_gate(
        args.qc_root,
        args.cohort_summary,
        args.noop_csv,
        args.generation_metrics,
        args.generation_manifest,
        args.checkpoint_inventory,
        args.manual_decisions,
        args.human_report,
        args.output_root,
        low_score_count=args.low_score_count,
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
