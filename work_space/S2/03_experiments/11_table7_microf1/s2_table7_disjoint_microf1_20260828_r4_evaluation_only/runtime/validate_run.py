#!/usr/bin/env python3
"""Validate the Table 7 evaluation-only r4 before and after execution."""

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
REFERENCE_INTERNAL = DATA / "nnUNet_raw/Dataset264_BraTS2026_MET_Completion/labelsTs"
TEST_LIST = S2 / "BraTS2026_S2_RC_v1.0/repository/data/splits/completion_warmstart/test_locked.txt"
MAPPING_CSV = PROJECT / "work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
R2_ROOT = S2 / "results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100"
R2_PREDICTION = R2_ROOT / "prediction"
R3_ROOT = S2 / "results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only"
EXPECTED_R3_FAILURE_SHA256 = "342f501ba9a4e89235ab4ec567867fbc8c40f4540710ca92bc00ed9c3c14a030"
EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_R2_FAILURE_SHA256 = "a2c608e941265e5cccb70b8786e6dfd67e5a762d91c43a2bae807b76f0bf64aa"
EXPECTED_CASES = 104
EXPECTED_TEST_SHA256 = "d763fb4d1eb798a46670079a0e148ba5ce0e0cc15de174f93effb1c741cb114b"
EXPECTED_MAPPING_SHA256 = "b1ff36ea927a3d5c8d4aa976ce5533d320e16c7801a9a30db22e8ecbb6d64672"
EXPECTED_PAIR_SHA256 = "78dd2710043171dd1de75043ff5d9f7be7ff9a226b71191a11c17a6ec493ffaf"
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


def internal_ids() -> list[str]:
    values = [line.strip() for line in TEST_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(values) == EXPECTED_CASES, f"test-list count drift: {len(values)}")
    require(len(set(values)) == EXPECTED_CASES, "duplicate test-list IDs")
    require(all(re.fullmatch(r"BraTSMET_[0-9]{6}", value) for value in values), "invalid test-list ID")
    return values


def source_mapping() -> dict[str, str]:
    with MAPPING_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1295, f"mapping row-count drift: {len(rows)}")
    result: dict[str, str] = {}
    all_sources: set[str] = set()
    for row in rows:
        internal = row["nnunet_case_id"]
        source = row["source_case_id"]
        require(internal not in result, f"duplicate mapping ID: {internal}")
        require(source not in all_sources, f"duplicate mapping source ID: {source}")
        result[internal] = source
        all_sources.add(source)
    return result


def expected_source_ids() -> list[str]:
    mapping = source_mapping()
    pairs = [(internal, mapping[internal]) for internal in internal_ids()]
    require(all(re.fullmatch(r"BraTS-MET-[0-9]{5}-[0-9]{3}", source) for _, source in pairs), "source ID format drift")
    require(len({source for _, source in pairs}) == EXPECTED_CASES, "source mapping is not bijective")
    pair_lines = "".join(f"{internal},{source}\n" for internal, source in sorted(pairs))
    require(hashlib.sha256(pair_lines.encode("utf-8")).hexdigest() == EXPECTED_PAIR_SHA256, "case-pair SHA drift")
    return sorted(source for _, source in pairs)


def nifti_inventory(root: Path) -> list[str]:
    require(root.is_dir(), f"missing NIfTI directory: {root}")
    paths = sorted(root.glob("*.nii.gz"))
    for path in paths:
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty NIfTI: {path}")
    return [path.name.removesuffix(".nii.gz") for path in paths]


def preflight(root: Path) -> None:
    target = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    require(not target.exists(), f"preflight evidence already exists: {target}")
    static = verify_static_hashes(root)
    plan_path = root / "TABLE7_DISJOINT_MICROF1_EXECUTION_PLAN_20260828_R4.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan["artifact_status"] == "experimental_unvalidated", "plan artifact status drift")
    require(plan["operator_approved"] is False, "plan operator approval drift")
    require(plan["formal_gate_status"] == "not_run_not_passed", "plan gate status drift")
    require(plan["case_count"] == EXPECTED_CASES, "plan case count drift")
    require(plan["execution"]["evaluation_only"] is True, "plan is not evaluation-only")
    require(plan["execution"]["training"] is False, "plan unexpectedly enables training")
    require(plan["execution"]["inference"] is False, "plan unexpectedly enables inference")
    require(sha256_file(TEST_LIST) == EXPECTED_TEST_SHA256, "test-list SHA drift")
    require(sha256_file(MAPPING_CSV) == EXPECTED_MAPPING_SHA256, "mapping CSV SHA drift")
    internal = sorted(internal_ids())
    sources = expected_source_ids()
    require(nifti_inventory(R2_PREDICTION) == internal, "r2 prediction inventory drift")
    require(nifti_inventory(REFERENCE_INTERNAL) == internal, "reference inventory drift")
    r2_inference_path = R2_ROOT / "evidence/INFERENCE_VALIDATION.json"
    r2_inference = json.loads(r2_inference_path.read_text(encoding="utf-8"))
    require(r2_inference["status"] == "pass", "r2 inference evidence failed")
    require(r2_inference["case_count"] == EXPECTED_CASES, "r2 inference count drift")
    require(r2_inference["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "r2 checkpoint SHA drift")
    r2_started_path = R2_ROOT / "evidence/TABLE7_DISJOINT_MICROF1_STARTED.json"
    r2_started = json.loads(r2_started_path.read_text(encoding="utf-8"))
    require(r2_started["status"] == "pass", "r2 STARTED evidence failed")
    require(str(r2_started["job_id"]) == "3542339", "r2 job identity drift")
    r2_official_path = R2_ROOT / "official_reproduction/panoptica_evaluation_summary.json"
    r2_official = json.loads(r2_official_path.read_text(encoding="utf-8"))
    require(len(r2_official.get("metrics", [])) == 16, "r2 failure metrics-count drift")
    require(len(r2_official.get("missings", [])) == 88, "r2 failure missings-count drift")
    r2_failure_path = R2_ROOT / "evidence/TABLE7_DISJOINT_MICROF1_FAILED.json"
    require(sha256_file(r2_failure_path) == EXPECTED_R2_FAILURE_SHA256, "r2 failure evidence drift")
    r3_failure_path = R3_ROOT / "evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json"
    require(sha256_file(r3_failure_path) == EXPECTED_R3_FAILURE_SHA256, "r3 failure evidence drift")
    r3_failure = json.loads(r3_failure_path.read_text(encoding="utf-8"))
    require(r3_failure["status"] == "failed", "r3 failure status drift")
    require(r3_failure["job_submitted"] is False, "r3 unexpectedly submitted a job")
    slurm_text = (root / "slurm/01_evaluate_only_table7.slurm").read_text(encoding="utf-8")
    require("#SBATCH --partition=cpu_192G" in slurm_text, "r4 is not on CPU partition")
    require("#SBATCH --gres" not in slurm_text, "r4 unexpectedly requests a GPU")
    require("frozen_s2_inference" not in slurm_text, "r4 unexpectedly invokes inference")
    for relative in ("views", "official_reproduction", "disjoint_microf1"):
        require(not (root / relative).exists(), f"exclusive output already exists: {relative}")
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
        "evaluation_only": True,
        "case_count": len(internal),
        "r2_job_id": str(r2_started["job_id"]),
        "r2_prediction_count": len(internal),
        "r2_inference_audit_sha256": sha256_file(r2_inference_path),
        "r2_started_audit_sha256": sha256_file(r2_started_path),
        "r2_official_summary_sha256": sha256_file(r2_official_path),
        "r2_official_metrics_count": len(r2_official["metrics"]),
        "r2_official_missings_count": len(r2_official["missings"]),
        "r2_failure_evidence_sha256": sha256_file(r2_failure_path),
        "r3_failure_evidence_sha256": sha256_file(r3_failure_path),
        "source_case_count": len(sources),
        "source_case_ids_sha256": hashlib.sha256(("\n".join(sources) + "\n").encode()).hexdigest(),
        "test_list_sha256": sha256_file(TEST_LIST),
        "mapping_csv_sha256": sha256_file(MAPPING_CSV),
        "case_pair_sha256": EXPECTED_PAIR_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
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
        expected_subjects = {f"{case_id}.nii.gz" for case_id in expected_source_ids()}
        require({row["subject_id"] for row in rows[:-3]} == expected_subjects, "official source-ID set drift")
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


def validate_mapping_audit(root: Path) -> dict[str, object]:
    audit_path = root / "evidence/CASE_ID_MAPPING_AND_SYMLINK_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit["status"] == "pass", "mapping audit did not pass")
    require(audit["artifact_status"] == "experimental_unvalidated", "mapping artifact status drift")
    require(audit["operator_approved"] is False, "mapping approval drift")
    require(audit["formal_gate_status"] == "not_run_not_passed", "mapping gate status drift")
    require(audit["evaluation_only"] is True, "mapping audit is not evaluation-only")
    require(audit["case_count"] == EXPECTED_CASES, "mapping audit count drift")
    require(audit["source_id_bijection"] is True, "mapping is not bijective")
    require(audit["geometry_mismatch_count"] == 0, "mapped geometry mismatch")
    require(audit["test_list_sha256"] == EXPECTED_TEST_SHA256, "mapping audit test SHA drift")
    require(audit["mapping_csv_sha256"] == EXPECTED_MAPPING_SHA256, "mapping audit CSV SHA drift")
    require(audit["case_pair_sha256"] == EXPECTED_PAIR_SHA256, "mapping audit pair SHA drift")
    reuse = audit["r2_reuse"]
    require(reuse["r2_job_id"] == "3542339", "r2 job identity drift")
    require(reuse["r2_official_metrics_count"] == 16, "r2 preserved metrics count drift")
    require(reuse["r2_official_missings_count"] == 88, "r2 preserved missings count drift")
    require(reuse["r2_failed_evidence_sha256"] == EXPECTED_R2_FAILURE_SHA256, "r2 failure evidence SHA drift")
    require(
        reuse["r2_inference_audit_sha256"]
        == sha256_file(R2_ROOT / "evidence/INFERENCE_VALIDATION.json"),
        "r2 inference evidence changed during r4",
    )
    require(
        reuse["r2_started_audit_sha256"]
        == sha256_file(R2_ROOT / "evidence/TABLE7_DISJOINT_MICROF1_STARTED.json"),
        "r2 STARTED evidence changed during r4",
    )
    require(
        reuse["r2_official_summary_sha256"]
        == sha256_file(R2_ROOT / "official_reproduction/panoptica_evaluation_summary.json"),
        "r2 official failure evidence changed during r4",
    )
    require(
        reuse["r2_failed_evidence_sha256"]
        == sha256_file(R2_ROOT / "evidence/TABLE7_DISJOINT_MICROF1_FAILED.json"),
        "r2 failure evidence changed during r4",
    )

    csv_path = root / "evidence/CASE_ID_MAPPING_AND_SYMLINKS.csv"
    fields, rows = read_csv(csv_path)
    required_fields = {
        "nnunet_case_id", "source_case_id", "prediction_source", "reference_source",
        "prediction_link", "reference_link", "prediction_sha256", "reference_sha256",
        "geometry_match",
    }
    require(required_fields <= set(fields), "mapping audit CSV columns missing")
    require(len(rows) == EXPECTED_CASES, "mapping audit CSV row-count drift")
    mapping = source_mapping()
    expected_internal = set(internal_ids())
    require({row["nnunet_case_id"] for row in rows} == expected_internal, "mapping audit internal-ID drift")
    require({row["source_case_id"] for row in rows} == set(expected_source_ids()), "mapping audit source-ID drift")
    for row in rows:
        internal = row["nnunet_case_id"]
        source = row["source_case_id"]
        require(mapping[internal] == source, f"mapping row mismatch: {internal}")
        expected_prediction = (R2_PREDICTION / f"{internal}.nii.gz").resolve(strict=True)
        expected_reference = (REFERENCE_INTERNAL / f"{internal}.nii.gz").resolve(strict=True)
        prediction_link = root / "views/prediction" / f"{source}.nii.gz"
        reference_link = root / "views/reference" / f"{source}.nii.gz"
        require(prediction_link.is_symlink(), f"prediction link replaced: {source}")
        require(reference_link.is_symlink(), f"reference link replaced: {source}")
        require(prediction_link.resolve(strict=True) == expected_prediction, f"prediction link target drift: {source}")
        require(reference_link.resolve(strict=True) == expected_reference, f"reference link target drift: {source}")
        require(Path(row["prediction_source"]) == expected_prediction, f"prediction provenance drift: {source}")
        require(Path(row["reference_source"]) == expected_reference, f"reference provenance drift: {source}")
        require(row["geometry_match"] == "True", f"geometry audit failed: {source}")
        require(sha256_file(expected_prediction) == row["prediction_sha256"], f"r2 prediction changed: {source}")
        require(sha256_file(expected_reference) == row["reference_sha256"], f"reference changed: {source}")
    require(sha256_file(csv_path) == audit["mapping_csv_audit_sha256"], "mapping audit CSV SHA drift")
    return {
        "audit_sha256": sha256_file(audit_path),
        "csv_sha256": sha256_file(csv_path),
        "case_pair_sha256": audit["case_pair_sha256"],
        "r2_reuse": reuse,
    }


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
    started = json.loads((root / "evidence/TABLE7_EVALUATION_ONLY_STARTED.json").read_text(encoding="utf-8"))
    require(preflight_data["status"] == started["status"] == "pass", "stage evidence failed")
    require(started["job_id"] == job_id, "job identity drift")
    require(started["evaluation_only"] is True, "job was not evaluation-only")
    require(started["gpu_requested"] is False, "evaluation-only job requested a GPU")
    mapping_audit = validate_mapping_audit(root)
    sources = expected_source_ids()
    require(nifti_inventory(root / "views/prediction") == sources, "mapped prediction view drift")
    require(nifti_inventory(root / "views/reference") == sources, "mapped reference view drift")
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
        "evaluation_only": True,
        "case_count": EXPECTED_CASES,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "published_table7_reproduced_at_three_decimals": True,
        "mapping_and_r2_reuse": mapping_audit,
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
