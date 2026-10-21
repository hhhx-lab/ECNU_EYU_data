#!/usr/bin/env python3
"""Compare official MET CSVs with paired complete-case bootstrap statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_PREFIXES = (
    "all_instance_f1_",
    "large_instance_f1_",
    "small_instance_f1_",
    "lesionwise_dsc_mean_",
    "lesionwise_hd95_mean_",
    "lesionwise_nsd_mean_",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    require(bool(separator and name and path), f"invalid --input: {value}")
    return name, Path(path).expanduser().resolve()


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def direction(metric: str) -> str:
    lowered = metric.lower()
    if any(token in lowered for token in ("_fp_", "_fn_", "hd95", "_std_", "_mse", "_mae")):
        return "lower_is_better"
    return "higher_is_better"


def read_metrics(path: Path, expected: int) -> dict[str, Any]:
    require(path.is_file(), f"missing metrics CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    require(fields and fields[0] == "subject_id", "subject_id column/order drift")
    require(len(rows) == expected + 3, f"row count drift: {path}")
    require([row["subject_id"] for row in rows[-3:]] == ["mean", "std", "median"], "summary row drift")
    case_rows = rows[:-3]
    ids = [row["subject_id"] for row in case_rows]
    require(len(ids) == len(set(ids)) == expected, "case ID coverage drift")
    metrics = fields[1:]
    return {
        "path": path,
        "sha256": sha256_file(path),
        "ids": ids,
        "rows": {row["subject_id"]: row for row in case_rows},
        "aggregate_mean": {metric: to_float(rows[-3][metric]) for metric in metrics},
        "metrics": metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    require(args.bootstrap == 20000 and args.seed == 20260824, "bootstrap contract drift")
    inputs = [parse_input(value) for value in args.input]
    require(2 <= len(inputs) <= 4, "comparison requires two to four methods")
    names = [name for name, _ in inputs]
    require(len(names) == len(set(names)), "duplicate method name")
    output_root = args.output_root.resolve()
    require(not output_root.exists(), f"exclusive comparison output exists: {output_root}")

    data = {name: read_metrics(path, args.expected_count) for name, path in inputs}
    reference_ids = data[names[0]]["ids"]
    metrics = data[names[0]]["metrics"]
    for name in names[1:]:
        require(data[name]["ids"] == reference_ids, f"case order differs for {name}")
        require(data[name]["metrics"] == metrics, f"metric columns differ for {name}")

    output_root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    pair_rows: list[dict[str, Any]] = []
    pair_payloads: list[dict[str, Any]] = []
    for method_a, method_b in itertools.combinations(names, 2):
        pair_metrics: list[dict[str, Any]] = []
        for metric in metrics:
            a = np.asarray([to_float(data[method_a]["rows"][case_id][metric]) for case_id in reference_ids], dtype=np.float64)
            b = np.asarray([to_float(data[method_b]["rows"][case_id][metric]) for case_id in reference_ids], dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            paired_n = int(mask.sum())
            excluded = int(len(mask) - paired_n)
            metric_direction = direction(metric)
            record: dict[str, Any] = {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "primary_endpoint": metric.startswith(PRIMARY_PREFIXES),
                "direction": metric_direction,
                "aggregate_mean_a": data[method_a]["aggregate_mean"][metric] if math.isfinite(data[method_a]["aggregate_mean"][metric]) else None,
                "aggregate_mean_b": data[method_b]["aggregate_mean"][metric] if math.isfinite(data[method_b]["aggregate_mean"][metric]) else None,
                "paired_complete_case_count": paired_n,
                "excluded_nonfinite_pair_count": excluded,
                "bootstrap_replicates": args.bootstrap,
                "seed": args.seed,
            }
            if paired_n >= 2:
                raw_delta = a[mask] - b[mask]
                benefit = raw_delta if metric_direction == "higher_is_better" else -raw_delta
                indices = rng.integers(0, paired_n, size=(args.bootstrap, paired_n))
                bootstrap_means = benefit[indices].mean(axis=1)
                ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
                probability_a = float(np.mean(bootstrap_means > 0))
                probability_b = float(np.mean(bootstrap_means < 0))
                probability_nonpositive = float(np.mean(bootstrap_means <= 0))
                probability_nonnegative = float(np.mean(bootstrap_means >= 0))
                wins_a = int(np.sum(benefit > 0))
                ties = int(np.sum(benefit == 0))
                std = float(np.std(benefit, ddof=1))
                record.update(
                    {
                        "status": "available",
                        "raw_mean_a_minus_b": float(raw_delta.mean()),
                        "benefit_mean_a_over_b": float(benefit.mean()),
                        "benefit_ci95_low": float(ci_low),
                        "benefit_ci95_high": float(ci_high),
                        "bootstrap_probability_a_better": probability_a,
                        "bootstrap_probability_b_better": probability_b,
                        "paired_two_sided_bootstrap_p": min(
                            1.0,
                            2.0 * min(probability_nonpositive, probability_nonnegative),
                        ),
                        "paired_effect_dz": float(benefit.mean() / std) if std > 0 else None,
                        "wins_a": wins_a,
                        "ties": ties,
                        "wins_b": paired_n - wins_a - ties,
                    }
                )
            else:
                record.update(
                    {
                        "status": "unavailable_insufficient_complete_pairs",
                        "raw_mean_a_minus_b": None,
                        "benefit_mean_a_over_b": None,
                        "benefit_ci95_low": None,
                        "benefit_ci95_high": None,
                        "bootstrap_probability_a_better": None,
                        "bootstrap_probability_b_better": None,
                        "paired_two_sided_bootstrap_p": None,
                        "paired_effect_dz": None,
                        "wins_a": None,
                        "ties": None,
                        "wins_b": None,
                    }
                )
            pair_rows.append(record)
            pair_metrics.append(record)
        pair_payloads.append({"method_a": method_a, "method_b": method_b, "metrics": pair_metrics})

    ranking_rows: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for metric in metrics:
        finite = [name for name in names if math.isfinite(data[name]["aggregate_mean"][metric])]
        reverse = direction(metric) == "higher_is_better"
        ordered = sorted(finite, key=lambda name: data[name]["aggregate_mean"][metric], reverse=reverse)
        rankings[metric] = ordered
        for rank, name in enumerate(ordered, start=1):
            ranking_rows.append(
                {
                    "metric": metric,
                    "primary_endpoint": metric.startswith(PRIMARY_PREFIXES),
                    "direction": direction(metric),
                    "rank": rank,
                    "method": name,
                    "aggregate_mean": data[name]["aggregate_mean"][metric],
                }
            )

    case_rows: list[dict[str, Any]] = []
    for case_id in reference_ids:
        row: dict[str, Any] = {"subject_id": case_id}
        for name in names:
            for metric in metrics:
                row[f"{name}__{metric}"] = data[name]["rows"][case_id][metric]
        case_rows.append(row)
    write_csv(output_root / "PAIRWISE_BOOTSTRAP.csv", pair_rows)
    write_csv(output_root / "METRIC_RANKINGS.csv", ranking_rows)
    write_csv(output_root / "PAIRED_CASE_METRICS.csv", case_rows)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": args.scope,
        "case_count": args.expected_count,
        "methods": {name: {"metrics_csv": str(data[name]["path"]), "metrics_csv_sha256": data[name]["sha256"]} for name in names},
        "metric_count": len(metrics),
        "primary_metric_count": sum(metric.startswith(PRIMARY_PREFIXES) for metric in metrics),
        "nonfinite_policy": "paired_complete_case_per_metric",
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed},
        "rankings": rankings,
        "pairs": pair_payloads,
        "interpretation_boundary": "Official MET segmentation evidence only; no reconstruction-quality or nnU-Net training superiority claim is implied.",
    }
    with (output_root / "PAIRED_COMPARISON.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (output_root / "PAIRED_COMPARISON_COMPLETE.ok").write_text(
        "status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "methods": names, "case_count": args.expected_count}, sort_keys=True))


if __name__ == "__main__":
    main()
