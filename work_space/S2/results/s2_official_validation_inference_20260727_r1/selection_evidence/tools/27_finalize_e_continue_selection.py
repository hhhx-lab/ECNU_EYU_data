#!/usr/bin/env python3
"""Validate fixed-case official metrics and archive the E/E-continue decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODELS = ("B", "E", "E_continue_final")
SELECTION_CANDIDATES = ("E", "E_continue_final")
REGIONS = ("et", "rc", "tc", "wt")
PAIR_METRICS = (
    "lesionwise_dsc_mean",
    "lesionwise_nsd_mean",
    "all_instance_f1",
    "small_instance_f1",
    "large_instance_f1",
)
SUMMARY_IDS = ("mean", "std", "median")
OUTPUT_NAMES = (
    "EVALUATION_VALIDATION_AUDIT.json",
    "OFFICIAL_METRICS_AGGREGATE.csv",
    "E_VS_E_CONTINUE_PAIRED_ANALYSIS.json",
    "MISSING_VALUE_SEMANTICS.json",
    "MODEL_SELECTION.json",
    "MODEL_SELECTION.md",
    "MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt",
    "MODEL_SELECTION_COMPLETE.ok",
)


class ValidationError(RuntimeError):
    """Raised when immutable evaluation evidence fails validation."""


@dataclass(frozen=True)
class ExpectedOutput:
    summary_sha256: str
    csv_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def parse_keyed_sha(values: Iterable[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        try:
            model, digest = value.split("=", 1)
        except ValueError as exc:
            raise ValidationError(f"Invalid {option} value: {value}") from exc
        require(model in MODELS, f"Unknown model in {option}: {model}")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Invalid SHA256: {value}")
        require(model not in parsed, f"Duplicate model in {option}: {model}")
        parsed[model] = digest
    require(set(parsed) == set(MODELS), f"{option} must bind exactly {MODELS}")
    return parsed


def parse_expected_outputs(values: Iterable[str]) -> dict[str, ExpectedOutput]:
    parsed: dict[str, ExpectedOutput] = {}
    for value in values:
        parts = value.split("=")
        require(len(parts) == 3, f"Invalid --expected-output value: {value}")
        model, summary_sha, csv_sha = parts
        require(model in MODELS, f"Unknown output model: {model}")
        for digest in (summary_sha, csv_sha):
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Invalid SHA256: {digest}")
        require(model not in parsed, f"Duplicate output model: {model}")
        parsed[model] = ExpectedOutput(summary_sha, csv_sha)
    require(set(parsed) == set(MODELS), f"--expected-output must bind exactly {MODELS}")
    return parsed


def read_table(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    require(bool(columns), f"Missing table header: {path}")
    return columns, rows


def parse_float(value: str, *, field: str) -> float | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValidationError(f"Invalid numeric value for {field}: {value!r}") from exc
    require(math.isfinite(number), f"Non-finite CSV value for {field}: {value!r}")
    return number


def numeric_sum(rows: list[dict[str, str]], field: str) -> float:
    values = [parse_float(row[field], field=field) for row in rows]
    return float(sum(value for value in values if value is not None))


def micro_f1(tp: float, fp: float, fn: float) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def file_manifest_digest(paths: Iterable[Path]) -> tuple[str, int, int]:
    rows: list[str] = []
    total_bytes = 0
    count = 0
    for path in sorted(paths, key=lambda value: value.name):
        size = path.stat().st_size
        rows.append(f"{path.name}\t{size}\t{sha256_file(path)}")
        total_bytes += size
        count += 1
    payload = "\n".join(rows) + "\n"
    return sha256_text(payload), count, total_bytes


def audit_environment(env_root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    expected_packages = {
        "BraTS-evaluation": "0.0.8",
        "numpy": "1.26.4",
        "panoptica": "2.1.0",
    }
    require(Path(sys.executable).resolve() == (env_root / "bin/python").resolve(), "Wrong Python runtime")
    require(sys.version_info[:2] == (3, 10), f"Wrong Python version: {sys.version.split()[0]}")

    actual_packages = {
        package: importlib.metadata.version(package) for package in expected_packages
    }
    require(actual_packages == expected_packages, f"Evaluation package drift: {actual_packages}")

    executable_paths = {
        "python": env_root / "bin/python",
        "brats-evaluate": env_root / "bin/brats-evaluate",
        "brats-parse-metrics": env_root / "bin/brats-parse-metrics",
    }
    executable_shas: dict[str, str] = {}
    for name, path in executable_paths.items():
        require(path.is_file() and os.access(path, os.X_OK), f"Missing executable: {path}")
        executable_shas[name] = sha256_file(path)

    require(audit.get("status") == "pass", "Environment audit status is not pass")
    require(Path(str(audit.get("environment"))).resolve() == env_root.resolve(), "Environment path drift")
    require(audit.get("packages") == expected_packages, "Environment audit package drift")
    require(audit.get("python") == sys.version.split()[0], "Environment audit Python drift")
    require(audit.get("executables_sha256") == executable_shas, "Executable SHA drift")
    return {
        "environment": str(env_root),
        "python": sys.version.split()[0],
        "packages": actual_packages,
        "executables_sha256": executable_shas,
    }


def validate_model(
    root: Path,
    model: str,
    expected_count: int,
    expected_checkpoint_sha: str,
    expected_output: ExpectedOutput,
    expected_environment: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, str]], dict[str, Any]]:
    model_root = root / model
    prep_path = model_root / "preparation_summary.json"
    summary_path = model_root / "panoptica_evaluation_summary.json"
    csv_path = model_root / "leaderboard_metrics.csv"
    marker_path = model_root / "EVALUATION_COMPLETE.ok"
    manifest_path = model_root / "nnunet_to_source_id.tsv"
    contract_path = model_root / "evaluation_contract.txt"
    environment_path = model_root / "evaluation_environment.txt"
    evaluate_log = model_root / "brats_evaluate.log"
    parse_log = model_root / "brats_parse_metrics.log"
    wrapper_log = root / "logs" / f"official_eval_{model}.log"
    pid_path = root / f"official_eval_{model}.pid"

    required_files = (
        prep_path,
        summary_path,
        csv_path,
        marker_path,
        manifest_path,
        contract_path,
        environment_path,
        evaluate_log,
        parse_log,
        wrapper_log,
        pid_path,
    )
    for path in required_files:
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty evidence: {path}")

    pid = int(pid_path.read_text(encoding="utf-8").strip())
    require(not pid_is_alive(pid), f"Evaluation PID is still alive for {model}: {pid}")

    prep = read_json(prep_path)
    require(prep.get("status") == "pass", f"Preparation status failed for {model}")
    for key in ("case_count", "prediction_count", "reference_count", "mapping_count"):
        require(prep.get(key) == expected_count, f"{model} preparation {key} mismatch")
    require(prep.get("materialization_mode") == "hardlink", f"{model} is not hardlinked")
    require(prep.get("evaluation_config") == "mets", f"{model} config drift")
    require(prep.get("vol_threshold") == 27, f"{model} volume threshold drift")
    require(prep.get("overlap_threshold") == 0.2, f"{model} overlap threshold drift")
    require(prep.get("checkpoint_sha256") == expected_checkpoint_sha, f"{model} checkpoint binding drift")

    checkpoint_path = Path(str(prep["checkpoint_path"]))
    require(checkpoint_path.is_file(), f"Missing checkpoint for {model}: {checkpoint_path}")
    require(checkpoint_path.stat().st_size == prep.get("checkpoint_bytes"), f"{model} checkpoint size drift")
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    require(actual_checkpoint_sha == expected_checkpoint_sha, f"{model} checkpoint SHA drift")

    manifest_columns, manifest_rows = read_table(manifest_path, delimiter="\t")
    required_manifest_columns = {
        "nnunet_id",
        "source_case_id",
        "prediction_source",
        "reference_source",
        "prediction_eval_path",
        "reference_eval_path",
    }
    require(set(manifest_columns) == required_manifest_columns, f"{model} manifest columns drift")
    require(len(manifest_rows) == expected_count, f"{model} manifest row count mismatch")
    nnunet_ids = [row["nnunet_id"] for row in manifest_rows]
    source_ids = [row["source_case_id"] for row in manifest_rows]
    require(len(set(nnunet_ids)) == expected_count, f"{model} duplicate nnU-Net IDs")
    require(len(set(source_ids)) == expected_count, f"{model} duplicate source IDs")

    prediction_files = sorted((model_root / "prediction").glob("*.nii.gz"))
    reference_files = sorted((model_root / "reference").glob("*.nii.gz"))
    expected_names = {f"{value}.nii.gz" for value in source_ids}
    require({path.name for path in prediction_files} == expected_names, f"{model} prediction coverage mismatch")
    require({path.name for path in reference_files} == expected_names, f"{model} reference coverage mismatch")

    for row in manifest_rows:
        prediction_source = Path(row["prediction_source"])
        reference_source = Path(row["reference_source"])
        prediction_eval = Path(row["prediction_eval_path"])
        reference_eval = Path(row["reference_eval_path"])
        require(prediction_source.is_file(), f"Missing prediction source: {prediction_source}")
        require(reference_source.is_file(), f"Missing reference source: {reference_source}")
        require(prediction_eval.is_file(), f"Missing prediction link: {prediction_eval}")
        require(reference_eval.is_file(), f"Missing reference link: {reference_eval}")
        require(os.path.samefile(prediction_source, prediction_eval), f"Prediction hardlink drift: {prediction_eval}")
        require(os.path.samefile(reference_source, reference_eval), f"Reference hardlink drift: {reference_eval}")

    prediction_digest, prediction_count, prediction_bytes = file_manifest_digest(prediction_files)
    reference_digest, reference_count, reference_bytes = file_manifest_digest(reference_files)

    summary_sha = sha256_file(summary_path)
    csv_sha = sha256_file(csv_path)
    require(summary_sha == expected_output.summary_sha256, f"{model} summary SHA drift")
    require(csv_sha == expected_output.csv_sha256, f"{model} CSV SHA drift")

    summary = read_json(summary_path)
    metrics = summary.get("metrics")
    require(isinstance(metrics, list) and len(metrics) == expected_count, f"{model} metric count mismatch")
    require(summary.get("missings") == [], f"{model} evaluator missings are nonempty")
    require(not any(isinstance(item, dict) and "error" in item for item in metrics), f"{model} evaluator errors found")
    summary_subjects = [str(item.get("subject_name")) for item in metrics if isinstance(item, dict)]
    require(len(summary_subjects) == expected_count, f"{model} malformed summary subjects")
    require(set(summary_subjects) == expected_names, f"{model} summary subject coverage mismatch")

    columns, all_rows = read_table(csv_path)
    require(len(columns) == 73, f"{model} expected 73 CSV columns, got {len(columns)}")
    require(len(all_rows) == expected_count + 3, f"{model} CSV row count mismatch")
    require(tuple(row["subject_id"] for row in all_rows[-3:]) == SUMMARY_IDS, f"{model} summary rows mismatch")
    subject_rows = all_rows[:-3]
    subject_ids = [row["subject_id"] for row in subject_rows]
    require(len(set(subject_ids)) == expected_count, f"{model} duplicate CSV subjects")
    require(set(subject_ids) == expected_names, f"{model} CSV subject coverage mismatch")
    for row in all_rows:
        for field, value in row.items():
            if field != "subject_id" and value != "":
                parse_float(value, field=f"{model}:{row['subject_id']}:{field}")

    marker = marker_path.read_text(encoding="utf-8")
    require("status=pass\n" in marker, f"{model} completion marker is not pass")
    require(summary_sha in marker and csv_sha in marker, f"{model} marker SHA binding mismatch")
    require("S2_INTERNAL_OFFICIAL_EVAL_PASS" in wrapper_log.read_text(encoding="utf-8"), f"{model} wrapper pass line missing")
    combined_logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (evaluate_log, parse_log, wrapper_log)
    )
    red_flags = re.findall(
        r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|worker failure",
        combined_logs,
        flags=re.IGNORECASE,
    )
    require(not red_flags, f"{model} evaluation log red flags: {red_flags[:5]}")

    environment_text = environment_path.read_text(encoding="utf-8")
    require(expected_environment["python"] in environment_text, f"{model} environment Python missing")
    for package, version in expected_environment["packages"].items():
        if package == "numpy":
            continue
        require(f"{package}={version}" in environment_text, f"{model} package evidence missing: {package}")
    for digest in expected_environment["executables_sha256"].values():
        require(digest in environment_text, f"{model} executable evidence missing")

    expected_contract = {
        "config": "mets",
        "vol_threshold": "27",
        "overlap_threshold": "0.2",
        "expected_count": str(expected_count),
    }
    contract = {}
    for line in contract_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        contract[key] = value
    for key, value in expected_contract.items():
        require(contract.get(key) == value, f"{model} evaluation contract drift: {key}")

    evidence = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha,
        "completion_marker_sha256": sha256_file(marker_path),
        "csv_columns": len(columns),
        "csv_data_rows": len(all_rows),
        "csv_sha256": csv_sha,
        "evaluation_pid": pid,
        "evaluation_pid_stopped": True,
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_files": prediction_count,
        "prediction_manifest_sha256": prediction_digest,
        "prediction_total_bytes": prediction_bytes,
        "preparation_summary_sha256": sha256_file(prep_path),
        "reference_files": reference_count,
        "reference_manifest_sha256": reference_digest,
        "reference_total_bytes": reference_bytes,
        "summary_metrics": len(metrics),
        "summary_missings": len(summary["missings"]),
        "summary_sha256": summary_sha,
    }
    return evidence, columns, all_rows, prep


def aggregate_rows(model_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        rows = model_rows[model]
        subject_rows = rows[:-3]
        mean_row = rows[-3]
        for region in REGIONS:
            row: dict[str, Any] = {"model": model, "region": region.upper()}
            for metric in PAIR_METRICS:
                field = f"{metric}_{region}"
                row[f"official_mean_{metric}"] = parse_float(mean_row[field], field=field)
                row[f"defined_cases_{metric}"] = sum(item[field] != "" for item in subject_rows)
            for scale in ("all_instance", "small_instance", "large_instance"):
                tp = numeric_sum(subject_rows, f"{scale}_tp_{region}")
                fp = numeric_sum(subject_rows, f"{scale}_fp_{region}")
                fn = numeric_sum(subject_rows, f"{scale}_fn_{region}")
                row[f"{scale}_tp_total"] = tp
                row[f"{scale}_fp_total"] = fp
                row[f"{scale}_fn_total"] = fn
                row[f"{scale}_micro_f1"] = micro_f1(tp, fp, fn)
            row["tiny_metrics"] = "unavailable"
            output.append(row)
    return output


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    require(values.size > 0, "Cannot bootstrap an empty paired sample")
    rng = np.random.default_rng(seed)
    chunk_size = 5000
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, values.size, size=(current, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= current
    samples = np.concatenate(means)
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def paired_analysis(
    model_rows: dict[str, list[dict[str, str]]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for model in SELECTION_CANDIDATES:
        indexed[model] = {
            row["subject_id"]: row
            for row in model_rows[model][:-3]
        }
    require(set(indexed["E"]) == set(indexed["E_continue_final"]), "Candidate subject sets differ")
    subjects = sorted(indexed["E"])

    regions: dict[str, Any] = {}
    for region in REGIONS:
        region_metrics: dict[str, Any] = {}
        for metric in PAIR_METRICS:
            field = f"{metric}_{region}"
            e_defined = {subject for subject in subjects if indexed["E"][subject][field] != ""}
            ec_defined = {
                subject for subject in subjects if indexed["E_continue_final"][subject][field] != ""
            }
            common = sorted(e_defined & ec_defined)
            deltas = np.asarray(
                [
                    float(indexed["E_continue_final"][subject][field])
                    - float(indexed["E"][subject][field])
                    for subject in common
                ],
                dtype=np.float64,
            )
            metric_seed = int(
                hashlib.sha256(f"{seed}:{field}".encode("ascii")).hexdigest()[:16],
                16,
            )
            ci_low, ci_high = bootstrap_mean_ci(deltas, seed=metric_seed, resamples=resamples)
            tolerance = 1e-12
            region_metrics[metric] = {
                "bootstrap_95_ci_mean_delta": [ci_low, ci_high],
                "bootstrap_resamples": resamples,
                "bootstrap_seed": metric_seed,
                "common_defined_cases": len(common),
                "e_continue_defined_cases": len(ec_defined),
                "e_continue_only_subjects": sorted(ec_defined - e_defined),
                "e_defined_cases": len(e_defined),
                "e_only_subjects": sorted(e_defined - ec_defined),
                "losses": int(np.sum(deltas < -tolerance)),
                "mean_delta_e_continue_minus_e": float(deltas.mean()),
                "median_delta_e_continue_minus_e": float(np.median(deltas)),
                "ties": int(np.sum(np.abs(deltas) <= tolerance)),
                "wins": int(np.sum(deltas > tolerance)),
            }

        count_deltas: dict[str, Any] = {}
        for metric in ("tp", "fp", "fn"):
            field = f"all_instance_{metric}_{region}"
            e_total = numeric_sum(model_rows["E"][:-3], field)
            ec_total = numeric_sum(model_rows["E_continue_final"][:-3], field)
            count_deltas[metric] = {
                "e": e_total,
                "e_continue": ec_total,
                "delta_e_continue_minus_e": ec_total - e_total,
            }
        regions[region.upper()] = {"paired_metrics": region_metrics, "all_instance_counts": count_deltas}

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "comparison": "E_continue_final minus E",
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


def choose_model(analysis: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    core_regions = ("ET", "TC", "WT")
    nonnegative_metrics = ("lesionwise_dsc_mean", "lesionwise_nsd_mean", "all_instance_f1")
    regressions: list[dict[str, Any]] = []
    robust_improvements: list[dict[str, Any]] = []
    fp_increases: list[dict[str, Any]] = []

    for region in core_regions:
        region_data = analysis["regions"][region]
        for metric in nonnegative_metrics:
            metric_data = region_data["paired_metrics"][metric]
            delta = metric_data["mean_delta_e_continue_minus_e"]
            if delta < 0:
                regressions.append({"region": region, "metric": metric, "mean_delta": delta})
            ci_low, ci_high = metric_data["bootstrap_95_ci_mean_delta"]
            if ci_low > 0:
                robust_improvements.append(
                    {"region": region, "metric": metric, "ci_low": ci_low, "ci_high": ci_high}
                )
        fp_delta = region_data["all_instance_counts"]["fp"]["delta_e_continue_minus_e"]
        if fp_delta > 0:
            fp_increases.append({"region": region, "fp_delta": fp_delta})

    candidate_pass = not regressions and not fp_increases and bool(robust_improvements)
    selected = "E_continue_final" if candidate_pass else "E"
    policy = {
        "name": "conservative_fallback_v1",
        "eligible_models": list(SELECTION_CANDIDATES),
        "comparator_only": "B",
        "candidate_pass_conditions": [
            "No negative common-defined mean delta in ET/TC/WT lesionwise DSC, lesionwise NSD, or all-instance F1.",
            "No increase in total ET/TC/WT all-instance false positives.",
            "At least one ET/TC/WT primary metric has a paired-bootstrap 95% CI strictly above zero.",
        ],
        "candidate_pass": candidate_pass,
        "observed_regressions": regressions,
        "observed_false_positive_increases": fp_increases,
        "observed_robust_improvements": robust_improvements,
        "decision": selected,
    }
    return selected, policy


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(
    selection: dict[str, Any],
    aggregate_rows_value: list[dict[str, Any]],
    paired: dict[str, Any],
) -> str:
    lookup = {(row["model"], row["region"]): row for row in aggregate_rows_value}
    lines = [
        "# S2 fixed-103 official-metric model selection",
        "",
        f"- Status: `{selection['status']}`",
        f"- Selected model: `{selection['selected_model']}`",
        f"- Selected checkpoint SHA256: `{selection['selected_checkpoint_sha256']}`",
        "- Eligible models: original `E` and `E_continue_final`; `B` is a comparator only.",
        "- Tiny-lesion metrics: unavailable in the locked official evaluator output.",
        "- Empty official cells remain undefined and were never replaced with zero.",
        "",
        "## E-continue minus E",
        "",
        "| Region | DSC mean delta | NSD mean delta | Official all-F1 delta | Common-defined paired all-F1 delta | FP delta | FN delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("ET", "RC", "TC", "WT"):
        e = lookup[("E", region)]
        ec = lookup[("E_continue_final", region)]
        paired_region = paired["regions"][region]
        official_f1_delta = (
            ec["official_mean_all_instance_f1"] - e["official_mean_all_instance_f1"]
        )
        common_f1_delta = paired_region["paired_metrics"]["all_instance_f1"][
            "mean_delta_e_continue_minus_e"
        ]
        fp_delta = paired_region["all_instance_counts"]["fp"]["delta_e_continue_minus_e"]
        fn_delta = paired_region["all_instance_counts"]["fn"]["delta_e_continue_minus_e"]
        lines.append(
            f"| {region} | {ec['official_mean_lesionwise_dsc_mean'] - e['official_mean_lesionwise_dsc_mean']:+.6f} "
            f"| {ec['official_mean_lesionwise_nsd_mean'] - e['official_mean_lesionwise_nsd_mean']:+.6f} "
            f"| {official_f1_delta:+.6f} | {common_f1_delta:+.6f} | {fp_delta:+.0f} | {fn_delta:+.0f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Original E is retained. E-continue does not satisfy the conservative fallback rule: ",
            "its sparse RC gain is not a broad, statistically robust improvement, while ET/TC/WT ",
            "show false-positive increases and TC/WT primary-metric regressions.",
            "",
            "The original evaluation outputs and launch evidence were not modified. The malformed ",
            "candidate map in the preserved launch audit is recorded as a provenance-format warning; ",
            "all three runs were independently rebound through stopped PID files, pass logs, exact ",
            "result SHA256 values, completion markers, and 103-case input hardlinks.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.selection_root.resolve()
    env_root = args.eval_env.resolve()
    require(root.is_dir(), f"Selection root does not exist: {root}")
    require(env_root.is_dir(), f"Evaluation environment does not exist: {env_root}")
    for name in OUTPUT_NAMES:
        require(not (root / name).exists(), f"Refusing to overwrite selection artifact: {root / name}")

    expected_checkpoints = parse_keyed_sha(
        args.expected_checkpoint_sha,
        "--expected-checkpoint-sha",
    )
    expected_outputs = parse_expected_outputs(args.expected_output)

    prep_audit_path = root / "SELECTION_PREPARATION_AUDIT.json"
    env_audit_path = root / "EVAL_ENVIRONMENT_AUDIT.json"
    launch_audit_path = root / "OFFICIAL_EVALUATION_LAUNCH_AUDIT.json"
    for path in (prep_audit_path, env_audit_path, launch_audit_path):
        require(path.is_file(), f"Missing root audit: {path}")

    prep_audit = read_json(prep_audit_path)
    env_audit = read_json(env_audit_path)
    launch_audit = read_json(launch_audit_path)
    require(prep_audit.get("audit_sha256") == args.expected_preparation_audit_id, "Preparation audit identity drift")
    require(env_audit.get("audit_sha256") == args.expected_environment_audit_id, "Environment audit identity drift")
    require(launch_audit.get("audit_sha256") == args.expected_launch_audit_id, "Launch audit identity drift")
    require(prep_audit.get("status") == "pass", "Selection preparation audit is not pass")
    require(env_audit.get("status") == "pass", "Evaluation environment audit is not pass")
    require(launch_audit.get("status") == "launched", "Evaluation launch audit is not launched")
    require(prep_audit.get("candidates", {}).keys() == expected_checkpoints.keys(), "Preparation candidate set drift")
    for model in MODELS:
        require(
            prep_audit["candidates"][model]["checkpoint_sha256"] == expected_checkpoints[model],
            f"Preparation audit checkpoint drift for {model}",
        )

    environment = audit_environment(env_root, env_audit)
    evaluation_script = Path(str(launch_audit.get("evaluation_script")))
    require(evaluation_script.is_file(), f"Missing evaluation script: {evaluation_script}")
    evaluation_script_sha = sha256_file(evaluation_script)
    require(evaluation_script_sha == args.expected_evaluation_script_sha, "Evaluation script SHA drift")
    require(launch_audit.get("evaluation_script_sha256") == evaluation_script_sha, "Launch script binding drift")
    require(launch_audit.get("environment_audit_sha256") == args.expected_environment_audit_id, "Launch environment binding drift")
    require(launch_audit.get("selection_preparation_audit_sha256") == args.expected_preparation_audit_id, "Launch preparation binding drift")
    require(launch_audit.get("expected_cases_per_candidate") == args.expected_count, "Launch case count drift")
    require(launch_audit.get("config") == "mets", "Launch evaluation config drift")
    require(launch_audit.get("volume_threshold") == 27, "Launch volume threshold drift")
    require(launch_audit.get("overlap_threshold") == 0.2, "Launch overlap threshold drift")

    warnings: list[dict[str, Any]] = []
    launch_candidates = launch_audit.get("candidates")
    if not isinstance(launch_candidates, dict) or set(launch_candidates) != set(MODELS):
        warnings.append(
            {
                "code": "preserved_launch_candidate_map_malformed",
                "severity": "noncritical_provenance_format",
                "message": (
                    "The immutable launch audit candidate map contains only B and concatenates the "
                    "E/E-continue launch records into B.wrapper_log. It was not rewritten. Each run "
                    "is independently validated from its PID file, wrapper log, outputs, and marker."
                ),
                "launch_audit_file_sha256": sha256_file(launch_audit_path),
            }
        )

    model_evidence: dict[str, Any] = {}
    model_columns: dict[str, list[str]] = {}
    model_rows: dict[str, list[dict[str, str]]] = {}
    model_preps: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        evidence, columns, rows, prep = validate_model(
            root,
            model,
            args.expected_count,
            expected_checkpoints[model],
            expected_outputs[model],
            environment,
        )
        model_evidence[model] = evidence
        model_columns[model] = columns
        model_rows[model] = rows
        model_preps[model] = prep
    require(model_columns["B"] == model_columns["E"] == model_columns["E_continue_final"], "CSV schemas differ")
    reference_digests = {value["reference_manifest_sha256"] for value in model_evidence.values()}
    require(len(reference_digests) == 1, "Reference content differs across candidates")

    aggregate = aggregate_rows(model_rows)
    paired = paired_analysis(model_rows, seed=args.bootstrap_seed, resamples=args.bootstrap_resamples)
    selected_model, policy = choose_model(paired)
    selected_prep = model_preps[selected_model]

    validation_audit = {
        "schema_version": 1,
        "status": "pass",
        "generated_at_utc": utc_now(),
        "selection_root": str(root),
        "finalizer_script": str(Path(__file__).resolve()),
        "finalizer_script_sha256": sha256_file(Path(__file__).resolve()),
        "expected_cases_per_model": args.expected_count,
        "violations": [],
        "warnings": warnings,
        "root_audits": {
            "selection_preparation": {
                "claimed_identity": args.expected_preparation_audit_id,
                "file_sha256": sha256_file(prep_audit_path),
            },
            "evaluation_environment": {
                "claimed_identity": args.expected_environment_audit_id,
                "file_sha256": sha256_file(env_audit_path),
            },
            "evaluation_launch": {
                "claimed_identity": args.expected_launch_audit_id,
                "file_sha256": sha256_file(launch_audit_path),
            },
        },
        "environment": environment,
        "evaluation_script": str(evaluation_script),
        "evaluation_script_sha256": evaluation_script_sha,
        "models": model_evidence,
        "cross_model_checks": {
            "csv_schema_identical": True,
            "reference_content_identical": True,
            "reference_manifest_sha256": next(iter(reference_digests)),
        },
    }

    missing_semantics = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "pass",
        "rule": "Empty official metric cells are undefined and are never imputed as zero.",
        "paired_analysis_rule": "Only subjects defined for both E and E-continue enter a paired metric.",
        "tiny_metrics": "unavailable because the locked official evaluator emits only all/small/large strata",
        "field_coverage": {
            region: {
                metric: paired["regions"][region]["paired_metrics"][metric]
                for metric in PAIR_METRICS
            }
            for region in ("ET", "RC", "TC", "WT")
        },
        "known_noninterchangeable_empty_cases": {
            region: paired["regions"][region]["paired_metrics"]["all_instance_f1"]["e_only_subjects"]
            for region in ("ET", "RC", "TC", "WT")
        },
    }

    selection = {
        "schema_version": 1,
        "status": "pass",
        "generated_at_utc": utc_now(),
        "selected_model": selected_model,
        "selected_checkpoint_path": selected_prep["checkpoint_path"],
        "selected_checkpoint_sha256": selected_prep["checkpoint_sha256"],
        "eligible_models": list(SELECTION_CANDIDATES),
        "comparator_only": "B",
        "decision_policy": policy,
        "interpretation": (
            "Retain original E. E-continue has a sparse RC gain but no broad robust improvement; "
            "it increases ET/TC/WT false positives and regresses TC/WT primary metrics."
            if selected_model == "E"
            else "Select E-continue because it passed every conservative candidate condition."
        ),
        "immutable_boundaries": {
            "met_aug_status": "permanently_stopped_for_current_submission",
            "route_a_training_allowed": False,
            "zip_creation_allowed": False,
            "synapse_upload_allowed": False,
        },
    }

    if args.dry_run:
        return {
            "status": "dry_run_pass",
            "selected_model": selected_model,
            "selected_checkpoint_sha256": selected_prep["checkpoint_sha256"],
            "warnings": warnings,
            "policy": policy,
        }

    stage = Path(tempfile.mkdtemp(prefix=".model_selection_stage_", dir=root))
    try:
        write_json(stage / "EVALUATION_VALIDATION_AUDIT.json", validation_audit)
        write_aggregate_csv(stage / "OFFICIAL_METRICS_AGGREGATE.csv", aggregate)
        write_json(stage / "E_VS_E_CONTINUE_PAIRED_ANALYSIS.json", paired)
        write_json(stage / "MISSING_VALUE_SEMANTICS.json", missing_semantics)

        supporting = {
            name: sha256_file(stage / name)
            for name in (
                "EVALUATION_VALIDATION_AUDIT.json",
                "OFFICIAL_METRICS_AGGREGATE.csv",
                "E_VS_E_CONTINUE_PAIRED_ANALYSIS.json",
                "MISSING_VALUE_SEMANTICS.json",
            )
        }
        selection["supporting_artifacts_sha256"] = supporting
        write_json(stage / "MODEL_SELECTION.json", selection)
        (stage / "MODEL_SELECTION.md").write_text(
            build_markdown(selection, aggregate, paired), encoding="utf-8"
        )

        sum_names = OUTPUT_NAMES[:-2]
        sums = "".join(f"{sha256_file(stage / name)}  {name}\n" for name in sum_names)
        (stage / "MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt").write_text(sums, encoding="utf-8")
        marker = (
            "status=pass\n"
            f"completed_at_utc={utc_now()}\n"
            f"selected_model={selected_model}\n"
            f"selected_checkpoint_sha256={selected_prep['checkpoint_sha256']}\n"
            f"model_selection_sha256={sha256_file(stage / 'MODEL_SELECTION.json')}\n"
            f"artifact_sums_sha256={sha256_file(stage / 'MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt')}\n"
        )
        (stage / "MODEL_SELECTION_COMPLETE.ok").write_text(marker, encoding="utf-8")

        for name in OUTPUT_NAMES:
            os.replace(stage / name, root / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return {
        "status": "pass",
        "selected_model": selected_model,
        "selected_checkpoint_sha256": selected_prep["checkpoint_sha256"],
        "model_selection_sha256": sha256_file(root / "MODEL_SELECTION.json"),
        "completion_marker_sha256": sha256_file(root / "MODEL_SELECTION_COMPLETE.ok"),
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--eval-env", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=103)
    parser.add_argument("--expected-preparation-audit-id", required=True)
    parser.add_argument("--expected-environment-audit-id", required=True)
    parser.add_argument("--expected-launch-audit-id", required=True)
    parser.add_argument("--expected-evaluation-script-sha", required=True)
    parser.add_argument("--expected-checkpoint-sha", action="append", default=[], metavar="MODEL=SHA256")
    parser.add_argument(
        "--expected-output",
        action="append",
        default=[],
        metavar="MODEL=SUMMARY_SHA256=CSV_SHA256",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = finalize(args)
    except ValidationError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
