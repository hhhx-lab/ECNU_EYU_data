#!/usr/bin/env python3
"""Validate immutable inputs before submission and outputs after completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/public/home/zqchen/projects/ECNU_EYU_data")
S2 = PROJECT / "work_space/S2"
DATA = S2 / "data/ecnu_completion_emergency"
DATASET = DATA / "nnUNet_raw/Dataset264_BraTS2026_MET_Completion"
INPUT = DATASET / "imagesTs"
REFERENCE = DATASET / "labelsTs"
MODEL = DATA / "nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres"
CHECKPOINT = MODEL / "fold_0/checkpoint_final.pth"
EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_CASES = 104
EXPECTED_CHANNELS = {"0000", "0001", "0002", "0003"}
EXPECTED_PACKAGES = {"BraTS-evaluation": "0.0.8", "panoptica": "2.1.0"}
PUBLISHED = {
    "et": {"all": 0.810, "large": 0.760, "small": 0.381},
    "tc": {"all": 0.821, "large": 0.766, "small": 0.387},
    "wt": {"all": 0.801, "large": 0.757, "small": 0.299},
    "rc": {"all": 0.493, "large": 0.073, "small": 0.000},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    json.loads(path.read_text(encoding="utf-8"))


def verify_static_hashes(root: Path) -> dict[str, str]:
    manifest = root / "STATIC_SHA256SUMS.txt"
    require(manifest.is_file(), "missing STATIC_SHA256SUMS.txt")
    verified: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = root / relative
        require(path.is_file(), f"missing static artifact: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"static SHA drift: {relative}")
        verified[relative] = actual
    require(len(verified) >= 8, "static manifest unexpectedly small")
    return verified


def input_inventory() -> tuple[list[str], dict[str, set[str]]]:
    paths = sorted(INPUT.glob("*.nii.gz"))
    require(len(paths) == EXPECTED_CASES * 4, f"input count drift: {len(paths)}")
    by_case: dict[str, set[str]] = {}
    for path in paths:
        require(path.is_file() and path.stat().st_size > 0, f"broken input: {path}")
        stem = path.name.removesuffix(".nii.gz")
        case_id, channel = stem.rsplit("_", 1)
        by_case.setdefault(case_id, set()).add(channel)
    require(len(by_case) == EXPECTED_CASES, "input case count drift")
    require(all(value == EXPECTED_CHANNELS for value in by_case.values()), "input channel set drift")
    return sorted(by_case), by_case


def reference_inventory() -> list[str]:
    paths = sorted(REFERENCE.glob("*.nii.gz"))
    require(len(paths) == EXPECTED_CASES, f"reference count drift: {len(paths)}")
    for path in paths:
        require(path.is_file() and path.stat().st_size > 0, f"broken reference: {path}")
    return [path.name.removesuffix(".nii.gz") for path in paths]


def preflight(root: Path) -> None:
    target = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    require(not target.exists(), f"preflight evidence already exists: {target}")
    static = verify_static_hashes(root)
    plan_path = root / "TABLE7_DISJOINT_MICROF1_EXECUTION_PLAN_20260828_R2.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan["artifact_status"] == "experimental_unvalidated", "plan artifact status drift")
    require(plan["operator_approved"] is False, "plan operator approval drift")
    require(plan["formal_gate_status"] == "not_run_not_passed", "plan gate status drift")
    require(plan["case_count"] == EXPECTED_CASES, "plan case count drift")
    input_ids, _ = input_inventory()
    reference_ids = reference_inventory()
    require(input_ids == reference_ids, "input/reference case IDs differ")
    require(all(re.search(r"\d{5}(?:-\d{3})?", case_id) for case_id in input_ids), "case IDs incompatible with official evaluator")
    require(CHECKPOINT.is_file(), "checkpoint missing")
    require(sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA drift")
    for path in (MODEL / "plans.json", MODEL / "dataset.json"):
        require(path.is_file() and path.stat().st_size > 0, f"model metadata missing: {path}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    require(versions == EXPECTED_PACKAGES, f"official evaluation package drift: {versions}")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": __import__("socket").gethostname(),
        "case_count": len(input_ids),
        "input_file_count": EXPECTED_CASES * 4,
        "reference_file_count": len(reference_ids),
        "case_ids_sha256": hashlib.sha256(("\n".join(input_ids) + "\n").encode()).hexdigest(),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_plans_sha256": sha256_file(MODEL / "plans.json"),
        "model_dataset_sha256": sha256_file(MODEL / "dataset.json"),
        "package_versions": versions,
        "static_artifact_count": len(static),
        "static_manifest_sha256": sha256_file(root / "STATIC_SHA256SUMS.txt"),
        "execution_plan_sha256": sha256_file(plan_path),
    }
    write_json_exclusive(target, payload)
    print(json.dumps(payload, sort_keys=True))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_official(root: Path) -> dict[str, object]:
    official = root / "official_reproduction"
    summary = json.loads((official / "panoptica_evaluation_summary.json").read_text(encoding="utf-8"))
    metrics = summary.get("metrics")
    require(summary.get("missings") == [], "official evaluator reported missing cases")
    require(isinstance(metrics, list) and len(metrics) == EXPECTED_CASES, "official metric count drift")
    require(not any(isinstance(item, dict) and "error" in item for item in metrics), "official evaluator subject error")
    outputs: dict[str, object] = {}
    for threshold in (27, 50):
        path = official / f"leaderboard_metrics_threshold{threshold}.csv"
        fields, rows = read_csv(path)
        require(len(rows) == EXPECTED_CASES + 3, f"official threshold {threshold} row count drift")
        require([row["subject_id"] for row in rows[-3:]] == ["mean", "std", "median"], "official summary rows drift")
        require(len({row["subject_id"] for row in rows[:-3]}) == EXPECTED_CASES, "official subject IDs drift")
        mean = rows[-3]
        values: dict[str, dict[str, float]] = {}
        for region in ("et", "tc", "wt", "rc"):
            values[region] = {}
            for group in ("all", "large", "small"):
                column = f"{group}_instance_f1_{region}"
                require(column in fields, f"official column missing: {column}")
                value = float(mean[column])
                require(math.isfinite(value), f"nonfinite official mean: {column}")
                values[region][group] = value
                if threshold == 50:
                    require(abs(value - PUBLISHED[region][group]) <= 0.0005000001, f"Table 7 reproduction mismatch: {region}/{group}={value}")
        outputs[str(threshold)] = {"means": values, "sha256": sha256_file(path)}
    outputs["panoptica_summary_sha256"] = sha256_file(official / "panoptica_evaluation_summary.json")
    return outputs


def validate_disjoint(root: Path) -> dict[str, object]:
    output = root / "disjoint_microf1"
    summary_path = output / "DISJOINT_MICROF1_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["status"] == "pass" and summary["case_count"] == EXPECTED_CASES, "disjoint summary status/count drift")
    require(summary["artifact_status"] == "experimental_unvalidated", "disjoint artifact status drift")
    require(summary["operator_approved"] is False, "disjoint approval drift")
    require(summary["formal_gate_status"] == "not_run_not_passed", "disjoint gate status drift")
    contract = summary["contract"]
    require(contract["thresholds_voxels"] == [27, 50], "disjoint threshold drift")
    require(contract["overlap_threshold_dsc"] == 0.2, "disjoint overlap threshold drift")
    require(contract["size_rule"] == "small < T; large >= T", "disjoint size rule drift")
    fields, rows = read_csv(output / "case_level_disjoint_counts.csv")
    require(len(rows) == EXPECTED_CASES * 4 * 2, "case-level disjoint row count drift")
    required = {f"{group}_{event}" for group in ("all", "small", "large") for event in ("tp", "fp", "fn")}
    require(required <= set(fields), "case-level count columns missing")
    pooled: dict[tuple[int, str, str, str], int] = {}
    all_by_threshold: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in rows:
        threshold = int(row["size_threshold_voxels"])
        region = row["region"]
        for event in ("tp", "fp", "fn"):
            all_value = int(row[f"all_{event}"])
            small_value = int(row[f"small_{event}"])
            large_value = int(row[f"large_{event}"])
            require(all_value == small_value + large_value, f"case non-additivity: {row['case_id']}/{region}/{threshold}/{event}")
            for group, value in (("all", all_value), ("small", small_value), ("large", large_value)):
                key = (threshold, region, group, event)
                pooled[key] = pooled.get(key, 0) + value
    event_fields, event_rows = read_csv(output / "component_events.csv")
    require(
        {"case_id", "region", "outcome", "reference_component", "prediction_component"}
        <= set(event_fields),
        "component event columns missing",
    )
    event_totals: dict[tuple[str, str], int] = {}
    seen_reference: set[tuple[str, str, str]] = set()
    seen_prediction: set[tuple[str, str, str]] = set()
    for row in event_rows:
        outcome = row["outcome"]
        require(outcome in {"tp", "fp", "fn"}, f"unknown component outcome: {outcome}")
        key = (row["case_id"], row["region"])
        event_totals[(row["region"], outcome)] = event_totals.get((row["region"], outcome), 0) + 1
        if outcome in {"tp", "fn"}:
            reference_key = (*key, row["reference_component"])
            require(reference_key not in seen_reference, f"reference component reused: {reference_key}")
            seen_reference.add(reference_key)
        if outcome in {"tp", "fp"}:
            prediction_key = (*key, row["prediction_component"])
            require(prediction_key not in seen_prediction, f"prediction component reused: {prediction_key}")
            seen_prediction.add(prediction_key)
    for region, region_data in summary["regions"].items():
        for threshold_text, threshold_data in region_data["thresholds"].items():
            threshold = int(threshold_text)
            require(threshold_data["additivity_pass"] is True, "summary additivity flag failed")
            groups = threshold_data["groups"]
            all_tuple = tuple(groups["all"][event] for event in ("tp", "fp", "fn"))
            previous = all_by_threshold.setdefault((region, "all"), all_tuple)
            require(previous == all_tuple, f"all counts changed across size thresholds: {region}")
            for event in ("tp", "fp", "fn"):
                require(groups["all"][event] == groups["small"][event] + groups["large"][event], "summary non-additivity")
                require(groups["all"][event] == event_totals.get((region, event), 0), "component event total mismatch")
            for group in ("all", "small", "large"):
                for event in ("tp", "fp", "fn"):
                    require(groups[group][event] == pooled[(threshold, region, group, event)], "pooled count mismatch")
                denominator = 2 * groups[group]["tp"] + groups[group]["fp"] + groups[group]["fn"]
                expected = 2 * groups[group]["tp"] / denominator if denominator else None
                actual = groups[group]["micro_f1"]
                require((expected is None and actual is None) or abs(expected - actual) < 1e-15, "micro-F1 formula mismatch")
    for name, expected_sha in summary["artifacts"].items():
        require(sha256_file(output / name) == expected_sha, f"disjoint artifact SHA drift: {name}")
    return {
        "summary_sha256": sha256_file(summary_path),
        "case_counts_sha256": sha256_file(output / "case_level_disjoint_counts.csv"),
        "component_events_sha256": sha256_file(output / "component_events.csv"),
        "summary_csv_sha256": sha256_file(output / "disjoint_microf1_summary.csv"),
        "regions": summary["regions"],
    }


def final(root: Path, job_id: str) -> None:
    target = root / "evidence/TABLE7_DISJOINT_MICROF1_VALIDATION.json"
    require(not target.exists(), f"final validation already exists: {target}")
    verify_static_hashes(root)
    preflight_data = json.loads((root / "evidence/REMOTE_STATIC_PREFLIGHT.json").read_text(encoding="utf-8"))
    started = json.loads((root / "evidence/TABLE7_DISJOINT_MICROF1_STARTED.json").read_text(encoding="utf-8"))
    inference = json.loads((root / "evidence/INFERENCE_VALIDATION.json").read_text(encoding="utf-8"))
    require(preflight_data["status"] == started["status"] == inference["status"] == "pass", "stage evidence failed")
    require(started["job_id"] == job_id and inference["case_count"] == EXPECTED_CASES, "job/inference evidence drift")
    require("V100" in started["device_name"] and "V100" in inference["device_name"], "GPU contract drift")
    require(inference["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "inference checkpoint drift")
    prediction_ids = sorted(path.name.removesuffix(".nii.gz") for path in (root / "prediction").glob("*.nii.gz"))
    require(len(prediction_ids) == EXPECTED_CASES and prediction_ids == reference_inventory(), "final prediction coverage drift")
    official = validate_official(root)
    disjoint = validate_disjoint(root)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "case_count": EXPECTED_CASES,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "published_table7_reproduced_at_three_decimals": True,
        "official_reproduction": official,
        "disjoint_microf1": disjoint,
    }
    write_json_exclusive(target, payload)
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("preflight", "final"))
    parser.add_argument("--job-id", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_dir(), f"missing run root: {root}")
    if args.mode == "preflight":
        preflight(root)
    else:
        require(bool(args.job_id), "--job-id is required for final validation")
        final(root, args.job_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
