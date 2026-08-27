#!/usr/bin/env python3
"""Validate and finalize the experimental Fix-v3 versus original-E decision."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any


REGIONS = ("et", "rc", "tc", "wt")
PAIR_METRICS = (
    "lesionwise_dsc_mean",
    "lesionwise_nsd_mean",
    "all_instance_f1",
    "small_instance_f1",
    "large_instance_f1",
)
CORE_REGIONS = ("ET", "TC", "WT")
PRIMARY_METRICS = (
    "lesionwise_dsc_mean",
    "lesionwise_nsd_mean",
    "all_instance_f1",
)
SUMMARY_IDS = ("mean", "std", "median")
OUTPUT_NAMES = (
    "FIX_V3_VS_E_PAIRED_ANALYSIS.json",
    "SELECTION_VALIDATION.json",
    "MODEL_SELECTION.json",
    "MODEL_SELECTION.md",
    "MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt",
    "MODEL_SELECTION_COMPLETE.ok",
)


def _official_columns() -> tuple[str, ...]:
    columns = ["subject_id"]
    for region in REGIONS:
        columns.extend(
            [
                f"all_instance_tp_{region}",
                f"all_instance_fp_{region}",
                f"all_instance_fn_{region}",
                f"all_instance_f1_{region}",
                f"large_instance_tp_{region}",
                f"large_instance_fp_{region}",
                f"large_instance_fn_{region}",
                f"large_instance_f1_{region}",
                f"lesionwise_dsc_mean_{region}",
                f"lesionwise_dsc_std_{region}",
                f"lesionwise_hd95_mean_{region}",
                f"lesionwise_hd95_std_{region}",
                f"lesionwise_nsd_mean_{region}",
                f"lesionwise_nsd_std_{region}",
                f"small_instance_tp_{region}",
                f"small_instance_fn_{region}",
                f"small_instance_fp_{region}",
                f"small_instance_f1_{region}",
            ]
        )
    return tuple(columns)


OFFICIAL_COLUMNS = _official_columns()


class SelectionError(RuntimeError):
    """Raised when fixed-103 evidence violates the selection contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


def parse_number(value: str, *, field: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"invalid numeric value for {field}: {value!r}") from exc
    require(math.isfinite(parsed), f"non-finite numeric value for {field}: {value!r}")
    return parsed


def _index_rows(rows: Sequence[Mapping[str, str]], *, model: str) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        subject = str(row.get("subject_id", "")).strip()
        require(bool(subject), f"{model} row {row_number} lacks subject_id")
        require(subject not in indexed, f"{model} contains duplicate subject_id: {subject}")
        indexed[subject] = row
    require(bool(indexed), f"{model} contains no subject rows")
    return indexed


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    require(bool(sorted_values), "cannot compute percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    require(bool(values), "cannot bootstrap an empty paired sample")
    require(resamples > 0, "bootstrap resamples must be positive")
    require(all(math.isfinite(value) for value in values), "bootstrap values must be finite")
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _numeric_sum(rows: Sequence[Mapping[str, str]], field: str) -> float:
    total = 0.0
    for row in rows:
        require(field in row, f"missing count field: {field}")
        value = parse_number(row[field], field=field)
        if value is not None:
            total += value
    return total


def paired_analysis(
    e_rows: Sequence[Mapping[str, str]],
    fix_v3_rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Compute common-defined paired deltas without imputing missing metrics."""

    e_index = _index_rows(e_rows, model="E")
    fix_index = _index_rows(fix_v3_rows, model="Fix_v3")
    require(set(e_index) == set(fix_index), "E and Fix_v3 subject sets differ")
    subjects = sorted(e_index)
    regions: dict[str, Any] = {}

    for region in REGIONS:
        paired_metrics: dict[str, Any] = {}
        for metric in PAIR_METRICS:
            field = f"{metric}_{region}"
            for model, indexed in (("E", e_index), ("Fix_v3", fix_index)):
                missing_fields = [subject for subject in subjects if field not in indexed[subject]]
                require(not missing_fields, f"{model} lacks {field}: {missing_fields[:3]}")
            e_defined = {
                subject for subject in subjects if e_index[subject][field] != ""
            }
            fix_defined = {
                subject for subject in subjects if fix_index[subject][field] != ""
            }
            common = sorted(e_defined & fix_defined)
            require(bool(common), f"no common-defined values for {field}")
            deltas = [
                float(parse_number(fix_index[subject][field], field=f"Fix_v3:{subject}:{field}"))
                - float(parse_number(e_index[subject][field], field=f"E:{subject}:{field}"))
                for subject in common
            ]
            metric_seed = int(
                hashlib.sha256(f"{seed}:{field}".encode("ascii")).hexdigest()[:16],
                16,
            )
            ci_low, ci_high = bootstrap_mean_ci(
                deltas,
                seed=metric_seed,
                resamples=resamples,
            )
            tolerance = 1e-12
            mean_delta = round(sum(deltas) / len(deltas), 15)
            sorted_deltas = sorted(deltas)
            paired_metrics[metric] = {
                "bootstrap_95_ci_mean_delta": [ci_low, ci_high],
                "bootstrap_resamples": resamples,
                "bootstrap_seed": metric_seed,
                "common_defined_cases": len(common),
                "e_defined_cases": len(e_defined),
                "fix_v3_defined_cases": len(fix_defined),
                "e_only_subjects": sorted(e_defined - fix_defined),
                "fix_v3_only_subjects": sorted(fix_defined - e_defined),
                "losses": sum(delta < -tolerance for delta in deltas),
                "mean_delta_fix_v3_minus_e": mean_delta,
                "median_delta_fix_v3_minus_e": _percentile(sorted_deltas, 0.5),
                "ties": sum(abs(delta) <= tolerance for delta in deltas),
                "wins": sum(delta > tolerance for delta in deltas),
            }

        count_deltas: dict[str, Any] = {}
        for metric in ("tp", "fp", "fn"):
            field = f"all_instance_{metric}_{region}"
            e_total = _numeric_sum(e_rows, field)
            fix_total = _numeric_sum(fix_v3_rows, field)
            count_deltas[metric] = {
                "e": e_total,
                "fix_v3": fix_total,
                "delta_fix_v3_minus_e": fix_total - e_total,
            }
        regions[region.upper()] = {
            "paired_metrics": paired_metrics,
            "all_instance_counts": count_deltas,
        }

    return {
        "schema_version": 1,
        "route_status": "experimental_unvalidated",
        "generated_at_utc": utc_now(),
        "comparison": "Fix_v3 minus E",
        "case_count": len(subjects),
        "missing_values_are_not_imputed": True,
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap of the common-defined mean delta",
            "base_seed": seed,
            "resamples_per_metric": resamples,
            "confidence_level": 0.95,
        },
        "regions": regions,
    }


def choose_model(analysis: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the frozen conservative fallback policy to paired fixed-103 evidence."""

    regressions: list[dict[str, Any]] = []
    robust_improvements: list[dict[str, Any]] = []
    false_positive_increases: list[dict[str, Any]] = []

    for region in CORE_REGIONS:
        region_data = analysis["regions"][region]
        for metric in PRIMARY_METRICS:
            metric_data = region_data["paired_metrics"][metric]
            delta = float(metric_data["mean_delta_fix_v3_minus_e"])
            if delta < 0:
                regressions.append(
                    {"region": region, "metric": metric, "mean_delta": delta}
                )
            ci_low, ci_high = metric_data["bootstrap_95_ci_mean_delta"]
            if float(ci_low) > 0:
                robust_improvements.append(
                    {
                        "region": region,
                        "metric": metric,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }
                )
        fp_delta = float(
            region_data["all_instance_counts"]["fp"]["delta_fix_v3_minus_e"]
        )
        if fp_delta > 0:
            false_positive_increases.append(
                {"region": region, "fp_delta": fp_delta}
            )

    candidate_pass = (
        not regressions
        and not false_positive_increases
        and bool(robust_improvements)
    )
    selected = "Fix_v3" if candidate_pass else "E"
    return selected, {
        "name": "conservative_fallback_v1",
        "eligible_models": ["E", "Fix_v3"],
        "candidate_pass_conditions": [
            "No negative common-defined mean delta in ET/TC/WT lesionwise DSC, lesionwise NSD, or all-instance F1.",
            "No increase in total ET/TC/WT all-instance false positives.",
            "At least one ET/TC/WT primary metric has a paired-bootstrap 95% CI strictly above zero.",
        ],
        "candidate_pass": candidate_pass,
        "observed_regressions": regressions,
        "observed_false_positive_increases": false_positive_increases,
        "observed_robust_improvements": robust_improvements,
        "decision": selected,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_metric_table(
    path: Path,
    *,
    expected_count: int,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    require(path.is_file() and path.stat().st_size > 0, f"missing metric CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    require(len(columns) == 73, f"official metric CSV must contain 73 columns: {path}")
    require(set(columns) == set(OFFICIAL_COLUMNS), f"official metric CSV schema drift: {path}")
    require(
        len(rows) == expected_count + len(SUMMARY_IDS),
        f"official metric CSV row count mismatch: {path}",
    )
    require(
        tuple(row["subject_id"] for row in rows[-3:]) == SUMMARY_IDS,
        f"official metric summary rows drift: {path}",
    )
    subject_rows = rows[:-3]
    subjects = [row["subject_id"] for row in subject_rows]
    require(len(set(subjects)) == expected_count, f"duplicate metric subjects: {path}")
    for row in rows:
        for field, value in row.items():
            if field != "subject_id" and value != "":
                parse_number(value, field=f"{path.name}:{row['subject_id']}:{field}")
    return columns, rows, subject_rows


def _parse_contract(path: Path) -> dict[str, str]:
    require(path.is_file() and path.stat().st_size > 0, f"missing evaluation contract: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        require("=" in line, f"invalid contract row {line_number}: {path}")
        key, value = line.split("=", 1)
        require(key not in result, f"duplicate contract key {key}: {path}")
        result[key] = value
    return result


def _file_manifest_digest(paths: Sequence[Path]) -> tuple[str, int, int]:
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.name):
        require(path.stat().st_size > 0, f"empty evaluation file: {path}")
        size = path.stat().st_size
        rows.append(f"{path.name}\t{size}\t{sha256_file(path)}")
        total_bytes += size
    encoded = ("\n".join(rows) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(rows), total_bytes


def validate_evaluation(
    root: Path,
    *,
    model: str,
    expected_checkpoint_sha: str,
    expected_count: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    root = root.expanduser().resolve()
    require(root.is_dir(), f"missing {model} evaluation root: {root}")
    paths = {
        "preparation": root / "preparation_summary.json",
        "summary": root / "panoptica_evaluation_summary.json",
        "metrics": root / "leaderboard_metrics.csv",
        "marker": root / "EVALUATION_COMPLETE.ok",
        "contract": root / "evaluation_contract.txt",
        "environment": root / "evaluation_environment.txt",
        "evaluate_log": root / "brats_evaluate.log",
        "parse_log": root / "brats_parse_metrics.log",
        "mapping": root / "nnunet_to_source_id.tsv",
    }
    for name, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0, f"missing {model} {name}: {path}")

    prep = read_json(paths["preparation"])
    require(prep.get("status") == "pass", f"{model} preparation did not pass")
    for key in ("case_count", "prediction_count", "reference_count", "mapping_count"):
        require(prep.get(key) == expected_count, f"{model} preparation {key} mismatch")
    require(prep.get("materialization_mode") == "hardlink", f"{model} was not hardlinked")
    require(prep.get("evaluation_config") == "mets", f"{model} evaluation config drift")
    require(prep.get("vol_threshold") == 27, f"{model} volume threshold drift")
    require(prep.get("overlap_threshold") == 0.2, f"{model} overlap threshold drift")
    require(prep.get("checkpoint_sha256") == expected_checkpoint_sha, f"{model} checkpoint binding drift")
    checkpoint = Path(str(prep.get("checkpoint_path", "")))
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"missing {model} checkpoint")
    require(checkpoint.stat().st_size == prep.get("checkpoint_bytes"), f"{model} checkpoint size drift")
    require(sha256_file(checkpoint) == expected_checkpoint_sha, f"{model} checkpoint SHA drift")

    columns, all_rows, subject_rows = read_metric_table(
        paths["metrics"],
        expected_count=expected_count,
    )
    subject_names = {row["subject_id"] for row in subject_rows}
    summary = read_json(paths["summary"])
    metrics = summary.get("metrics")
    require(summary.get("missings") == [], f"{model} evaluator reported missing cases")
    require(isinstance(metrics, list) and len(metrics) == expected_count, f"{model} summary count mismatch")
    require(
        not any(isinstance(item, dict) and "error" in item for item in metrics),
        f"{model} evaluator reported subject errors",
    )
    summary_subjects = {
        str(item.get("subject_name"))
        for item in metrics
        if isinstance(item, dict)
    }
    require(summary_subjects == subject_names, f"{model} summary subject coverage mismatch")

    prediction_files = sorted((root / "prediction").glob("*.nii.gz"))
    reference_files = sorted((root / "reference").glob("*.nii.gz"))
    require({path.name for path in prediction_files} == subject_names, f"{model} prediction coverage mismatch")
    require({path.name for path in reference_files} == subject_names, f"{model} reference coverage mismatch")
    prediction_digest, prediction_count, prediction_bytes = _file_manifest_digest(prediction_files)
    reference_digest, reference_count, reference_bytes = _file_manifest_digest(reference_files)
    require(prediction_count == expected_count, f"{model} prediction count mismatch")
    require(reference_count == expected_count, f"{model} reference count mismatch")

    contract = _parse_contract(paths["contract"])
    require(contract.get("config") == "mets", f"{model} contract config drift")
    require(contract.get("vol_threshold") == "27", f"{model} contract volume threshold drift")
    require(contract.get("overlap_threshold") == "0.2", f"{model} contract overlap threshold drift")
    require(contract.get("expected_count") == str(expected_count), f"{model} contract count drift")
    environment = paths["environment"].read_text(encoding="utf-8")
    for token in ("Python 3.10", "BraTS-evaluation=0.0.8", "panoptica=2.1.0"):
        require(token in environment, f"{model} evaluation environment drift: {token}")
    logs = "\n".join(
        paths[name].read_text(encoding="utf-8", errors="replace")
        for name in ("evaluate_log", "parse_log")
    )
    red_flags = re.findall(
        r"Traceback|out of memory|OutOfMemory|Segmentation fault|worker failure",
        logs,
        flags=re.IGNORECASE,
    )
    require(not red_flags, f"{model} evaluation log red flags: {red_flags[:3]}")
    summary_sha = sha256_file(paths["summary"])
    metrics_sha = sha256_file(paths["metrics"])
    marker = paths["marker"].read_text(encoding="utf-8")
    require("status=pass\n" in marker, f"{model} evaluation marker did not pass")
    require(summary_sha in marker and metrics_sha in marker, f"{model} marker SHA binding drift")

    evidence = {
        "root": str(root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "preparation_summary_sha256": sha256_file(paths["preparation"]),
        "summary_sha256": summary_sha,
        "metrics_sha256": metrics_sha,
        "completion_marker_sha256": sha256_file(paths["marker"]),
        "mapping_sha256": sha256_file(paths["mapping"]),
        "environment_sha256": sha256_file(paths["environment"]),
        "case_count": expected_count,
        "prediction_manifest_sha256": prediction_digest,
        "prediction_total_bytes": prediction_bytes,
        "reference_manifest_sha256": reference_digest,
        "reference_total_bytes": reference_bytes,
    }
    return evidence, columns, all_rows


def _selection_markdown(
    selection: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> str:
    lines = [
        "# Fix-v3 versus original E on fixed 103",
        "",
        f"- Route status: `{selection['route_status']}`",
        f"- Selected model: `{selection['selected_model']}`",
        "- Inference: segmentation checkpoint only; MET-AUG, G1, G2, and donor generation are disabled.",
        "- Official 179-case inference has not been started by this decision.",
        "",
        "| Region | DSC mean delta | NSD mean delta | all-F1 mean delta | FP delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in ("ET", "RC", "TC", "WT"):
        values = analysis["regions"][region]
        metrics = values["paired_metrics"]
        lines.append(
            f"| {region} | {metrics['lesionwise_dsc_mean']['mean_delta_fix_v3_minus_e']:+.6f} "
            f"| {metrics['lesionwise_nsd_mean']['mean_delta_fix_v3_minus_e']:+.6f} "
            f"| {metrics['all_instance_f1']['mean_delta_fix_v3_minus_e']:+.6f} "
            f"| {values['all_instance_counts']['fp']['delta_fix_v3_minus_e']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "The conservative fallback policy selects Fix-v3 only when no ET/TC/WT primary mean regresses, ",
            "no ET/TC/WT false-positive total increases, and at least one primary paired-bootstrap interval ",
            "is strictly above zero. Otherwise original E remains selected.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_selection(
    e_eval_root: Path,
    fix_v3_eval_root: Path,
    output_root: Path,
    *,
    expected_e_checkpoint_sha: str,
    expected_fix_v3_checkpoint_sha: str,
    expected_count: int = 103,
    bootstrap_seed: int = 20260729,
    bootstrap_resamples: int = 20000,
    dry_run: bool = False,
) -> dict[str, Any]:
    require(expected_count > 0, "expected_count must be positive")
    require(bootstrap_resamples > 0, "bootstrap_resamples must be positive")
    for label, digest in (
        ("E", expected_e_checkpoint_sha),
        ("Fix_v3", expected_fix_v3_checkpoint_sha),
    ):
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"invalid {label} checkpoint SHA")
    output_root = output_root.expanduser().resolve()
    require(not output_root.exists(), f"refusing to overwrite selection output: {output_root}")

    e_evidence, e_columns, e_rows = validate_evaluation(
        e_eval_root,
        model="E",
        expected_checkpoint_sha=expected_e_checkpoint_sha,
        expected_count=expected_count,
    )
    fix_evidence, fix_columns, fix_rows = validate_evaluation(
        fix_v3_eval_root,
        model="Fix_v3",
        expected_checkpoint_sha=expected_fix_v3_checkpoint_sha,
        expected_count=expected_count,
    )
    require(e_columns == fix_columns, "E and Fix_v3 metric schemas differ")
    require(
        {row["subject_id"] for row in e_rows[:-3]}
        == {row["subject_id"] for row in fix_rows[:-3]},
        "E and Fix_v3 metric subject sets differ",
    )
    require(
        e_evidence["reference_manifest_sha256"]
        == fix_evidence["reference_manifest_sha256"],
        "E and Fix_v3 reference content differs",
    )
    require(
        e_evidence["environment_sha256"] == fix_evidence["environment_sha256"],
        "E and Fix-v3 evaluator environments differ",
    )

    analysis = paired_analysis(
        e_rows[:-3],
        fix_rows[:-3],
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    selected_model, policy = choose_model(analysis)
    selected_evidence = e_evidence if selected_model == "E" else fix_evidence
    selection = {
        "schema_version": 1,
        "status": "pass",
        "route_status": "experimental_unvalidated",
        "generated_at_utc": utc_now(),
        "selected_model": selected_model,
        "selected_checkpoint_path": selected_evidence["checkpoint_path"],
        "selected_checkpoint_sha256": selected_evidence["checkpoint_sha256"],
        "eligible_models": ["E", "Fix_v3"],
        "decision_policy": policy,
        "inference_contract": "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
        "immutable_boundaries": {
            "reference_development_holdout_and_gate_0_1a_1b_2_skipped_by_user": True,
            "official_179_started": False,
            "zip_creation_allowed": False,
            "synapse_upload_allowed": False,
        },
    }
    validation = {
        "schema_version": 1,
        "status": "pass",
        "route_status": "experimental_unvalidated",
        "generated_at_utc": utc_now(),
        "expected_count": expected_count,
        "csv_schema_identical": True,
        "metric_subject_sets_identical": True,
        "reference_content_identical": True,
        "evaluator_environment_identical": True,
        "models": {"E": e_evidence, "Fix_v3": fix_evidence},
    }
    validation["validation_audit_sha256"] = canonical_sha256(validation)
    result = {
        "status": "dry_run_pass" if dry_run else "pass",
        "route_status": "experimental_unvalidated",
        "selected_model": selected_model,
        "selected_checkpoint_sha256": selected_evidence["checkpoint_sha256"],
        "policy": policy,
    }
    if dry_run:
        return result

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fix_v3_selection_stage_", dir=output_root.parent))
    try:
        write_json(stage / "FIX_V3_VS_E_PAIRED_ANALYSIS.json", analysis)
        write_json(stage / "SELECTION_VALIDATION.json", validation)
        selection["supporting_artifacts_sha256"] = {
            "FIX_V3_VS_E_PAIRED_ANALYSIS.json": sha256_file(
                stage / "FIX_V3_VS_E_PAIRED_ANALYSIS.json"
            ),
            "SELECTION_VALIDATION.json": sha256_file(stage / "SELECTION_VALIDATION.json"),
        }
        write_json(stage / "MODEL_SELECTION.json", selection)
        (stage / "MODEL_SELECTION.md").write_text(
            _selection_markdown(selection, analysis),
            encoding="utf-8",
        )
        sum_names = OUTPUT_NAMES[:-2]
        sums = "".join(
            f"{sha256_file(stage / name)}  {name}\n"
            for name in sum_names
        )
        (stage / "MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt").write_text(
            sums,
            encoding="utf-8",
        )
        marker = (
            "status=pass\n"
            "route_status=experimental_unvalidated\n"
            f"completed_at_utc={utc_now()}\n"
            f"selected_model={selected_model}\n"
            f"selected_checkpoint_sha256={selected_evidence['checkpoint_sha256']}\n"
            f"model_selection_sha256={sha256_file(stage / 'MODEL_SELECTION.json')}\n"
            f"artifact_sums_sha256={sha256_file(stage / 'MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt')}\n"
        )
        (stage / "MODEL_SELECTION_COMPLETE.ok").write_text(marker, encoding="utf-8")
        os.replace(stage, output_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    result["model_selection_sha256"] = sha256_file(output_root / "MODEL_SELECTION.json")
    result["completion_marker_sha256"] = sha256_file(
        output_root / "MODEL_SELECTION_COMPLETE.ok"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-eval-root", required=True, type=Path)
    parser.add_argument("--fix-v3-eval-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-e-checkpoint-sha", required=True)
    parser.add_argument("--expected-fix-v3-checkpoint-sha", required=True)
    parser.add_argument("--expected-count", type=int, default=103)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = finalize_selection(
        args.e_eval_root,
        args.fix_v3_eval_root,
        args.output_root,
        expected_e_checkpoint_sha=args.expected_e_checkpoint_sha,
        expected_fix_v3_checkpoint_sha=args.expected_fix_v3_checkpoint_sha,
        expected_count=args.expected_count,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
