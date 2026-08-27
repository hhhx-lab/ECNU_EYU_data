#!/usr/bin/env python3
"""Validate and compare the r4 EncDec, BBDM, and ensemble evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


METHODS = {
    "ensemble_r4": "ensamble",
    "encdec_only": "encdec",
    "bbdm_only": "bbdm",
}
METRICS = (
    "whole_SSIM",
    "whole_PSNR",
    "whole_MSE",
    "whole_MAE",
    "brain_SSIM",
    "brain_PSNR",
    "brain_MSE",
    "brain_MAE",
)
HIGHER_IS_BETTER = {"whole_SSIM", "whole_PSNR", "brain_SSIM", "brain_PSNR"}
PAIR_ORDER = (
    ("ensemble_r4", "encdec_only"),
    ("ensemble_r4", "bbdm_only"),
    ("encdec_only", "bbdm_only"),
)
EXPECTED_SPATIAL_PREPROCESSING = "foreground_centered_isotropic_resample_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--case-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--vae-sha256", required=True)
    parser.add_argument("--encdec-sha256", required=True)
    parser.add_argument("--bbdm-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any], *, exclude: tuple[str, ...] = ()) -> str:
    filtered = {key: value for key, value in payload.items() if key not in exclude}
    encoded = json.dumps(filtered, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_case_ids(path: Path) -> list[str]:
    rows = load_csv(path)
    require(rows and rows[0], "fixed case list is empty")
    first_column = next(iter(rows[0]))
    case_ids = [str(row[first_column]).strip() for row in rows]
    require(len(case_ids) == 103, f"fixed case count is {len(case_ids)}, expected 103")
    require(len(case_ids) == len(set(case_ids)), "fixed case list contains duplicates")
    return case_ids


def validate_method(
    method: str,
    run_dir: Path,
    case_ids: list[str],
    case_list: Path,
) -> dict[str, Any]:
    record = load_json(run_dir / "evaluation_run.json")
    require(record.get("case_count") == 103, f"{method}: evaluation case count drifted")
    require(record.get("synthesis_type") == METHODS[method], f"{method}: synthesis type drifted")
    require(record.get("spatial_preprocessing") == EXPECTED_SPATIAL_PREPROCESSING, f"{method}: spatial preprocessing drifted")
    require(record.get("operator_approved") is False, f"{method}: operator_approved must remain false")
    require(Path(str(record.get("case_list", ""))).resolve() == case_list, f"{method}: fixed case list binding drifted")
    require(math.isclose(float(record.get("bbdm_s")), 0.01, rel_tol=0.0, abs_tol=1e-12), f"{method}: bbdm_s drifted")

    metric_rows = load_csv(run_dir / "metrics.csv")
    require(len(metric_rows) == 103, f"{method}: metrics row count drifted")
    by_subject = {row["subject"]: row for row in metric_rows}
    require(len(by_subject) == 103, f"{method}: duplicate metric subjects")
    require(set(by_subject) == set(case_ids), f"{method}: metric subject coverage drifted")
    values: dict[str, np.ndarray] = {}
    recomputed_means: dict[str, float] = {}
    for metric in METRICS:
        array = np.asarray([float(by_subject[case_id][metric]) for case_id in case_ids], dtype=np.float64)
        require(bool(np.isfinite(array).all()), f"{method}: nonfinite {metric}")
        values[metric] = array
        recomputed_means[metric] = float(array.mean())
        recorded = float(record["mean_metrics"][metric])
        require(math.isclose(recomputed_means[metric], recorded, rel_tol=0.0, abs_tol=1e-12), f"{method}: mean {metric} drifted")

    spatial_rows = load_csv(run_dir / "spatial_audit.csv")
    require(len(spatial_rows) == 103, f"{method}: spatial audit row count drifted")
    spatial_by_subject = {row["subject"]: row for row in spatial_rows}
    require(len(spatial_by_subject) == 103, f"{method}: duplicate spatial subjects")
    require(set(spatial_by_subject) == set(case_ids), f"{method}: spatial subject coverage drifted")
    require(all(int(row["foreground_outside_voxel_count"]) == 0 for row in spatial_rows), f"{method}: foreground escaped model FOV")
    require(all(int(row["lesion_outside_voxel_count"]) == 0 for row in spatial_rows), f"{method}: lesion escaped model FOV")

    geometry = load_json(run_dir / "geometry_audit" / "geometry_audit.json")
    expected_geometry = {
        "case_count": 103,
        "geometry_mismatch_before_count": 0,
        "repaired_count": 0,
        "repair_mode": False,
        "voxel_resampling_performed": False,
        "max_abs_voxel_difference_after_repair": 0.0,
    }
    for key, expected in expected_geometry.items():
        require(geometry.get(key) == expected, f"{method}: geometry contract drifted for {key}")

    outputs = sorted((run_dir / "synthesized").glob("*.nii.gz"))
    output_subjects = {path.name[: -len("-t2w.nii.gz")] for path in outputs}
    require(len(outputs) == 103 and output_subjects == set(case_ids), f"{method}: synthesized output coverage drifted")

    return {
        "record": record,
        "metrics": values,
        "means": recomputed_means,
        "spatial": spatial_by_subject,
        "files": {
            "evaluation_run_sha256": sha256_file(run_dir / "evaluation_run.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.csv"),
            "spatial_audit_sha256": sha256_file(run_dir / "spatial_audit.csv"),
            "geometry_audit_sha256": sha256_file(run_dir / "geometry_audit" / "geometry_audit.json"),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    require(args.bootstrap == 20_000, "comparison is locked to 20,000 paired bootstrap replicates")
    root = Path(args.root).expanduser().resolve()
    case_list = Path(args.case_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require(root.is_dir(), f"missing r4 root: {root}")
    require(not output_dir.exists(), f"paired comparison target already exists: {output_dir}")
    case_ids = read_case_ids(case_list)

    method_data = {
        method: validate_method(method, root / method, case_ids, case_list)
        for method in METHODS
    }
    reference_spatial = method_data["encdec_only"]["spatial"]
    for method in METHODS:
        require(method_data[method]["spatial"] == reference_spatial, f"{method}: spatial audit differs from r4 contract")

    checkpoint_fields = ("vae_weights", "encdec_checkpoint", "bbdm_checkpoint")
    expected_hashes = {
        "vae_weights": args.vae_sha256,
        "encdec_checkpoint": args.encdec_sha256,
        "bbdm_checkpoint": args.bbdm_sha256,
    }
    checkpoint_bindings: dict[str, dict[str, str]] = {}
    for field in checkpoint_fields:
        paths = {str(method_data[method]["record"][field]) for method in METHODS}
        require(len(paths) == 1, f"checkpoint path differs across methods: {field}")
        path = Path(paths.pop()).resolve()
        require(path.is_file(), f"missing checkpoint: {path}")
        observed_sha = sha256_file(path)
        require(observed_sha == expected_hashes[field], f"checkpoint SHA drifted: {field}")
        checkpoint_bindings[field] = {"path": str(path), "sha256": observed_sha}

    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(case_ids), size=(args.bootstrap, len(case_ids)))
    pair_rows: list[dict[str, Any]] = []
    pair_payloads: list[dict[str, Any]] = []
    for method_a, method_b in PAIR_ORDER:
        metric_payloads: list[dict[str, Any]] = []
        for metric in METRICS:
            raw_delta = method_data[method_a]["metrics"][metric] - method_data[method_b]["metrics"][metric]
            benefit_delta = raw_delta if metric in HIGHER_IS_BETTER else -raw_delta
            bootstrap_means = benefit_delta[indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
            wins = int(np.sum(benefit_delta > 0))
            ties = int(np.sum(benefit_delta == 0))
            losses = int(len(benefit_delta) - wins - ties)
            probability_a = float(np.mean(bootstrap_means > 0))
            probability_b = float(np.mean(bootstrap_means < 0))
            std = float(np.std(benefit_delta, ddof=1))
            row = {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
                "mean_a": method_data[method_a]["means"][metric],
                "mean_b": method_data[method_b]["means"][metric],
                "raw_mean_a_minus_b": float(raw_delta.mean()),
                "benefit_mean_a_over_b": float(benefit_delta.mean()),
                "benefit_ci95_low": float(ci_low),
                "benefit_ci95_high": float(ci_high),
                "bootstrap_probability_a_better": probability_a,
                "bootstrap_probability_b_better": probability_b,
                "paired_two_sided_bootstrap_p": min(1.0, 2.0 * min(probability_a, probability_b)),
                "paired_effect_dz": float(benefit_delta.mean() / std) if std > 0 else None,
                "wins_a": wins,
                "ties": ties,
                "wins_b": losses,
                "bootstrap_replicates": args.bootstrap,
                "seed": args.seed,
            }
            pair_rows.append(row)
            metric_payloads.append(row)
        pair_payloads.append({"method_a": method_a, "method_b": method_b, "metrics": metric_payloads})

    summary_rows: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for metric in METRICS:
        reverse = metric in HIGHER_IS_BETTER
        ordered = sorted(METHODS, key=lambda name: method_data[name]["means"][metric], reverse=reverse)
        rankings[metric] = ordered
        for rank, method in enumerate(ordered, start=1):
            summary_rows.append({
                "metric": metric,
                "direction": "higher_is_better" if reverse else "lower_is_better",
                "rank": rank,
                "method": method,
                "mean": method_data[method]["means"][metric],
            })

    paired_subject_rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        row: dict[str, Any] = {"subject": case_id}
        for method in METHODS:
            for metric in METRICS:
                row[f"{method}__{metric}"] = float(method_data[method]["metrics"][metric][index])
        paired_subject_rows.append(row)

    first_place_counts = {
        method: sum(ranking[0] == method for ranking in rankings.values())
        for method in METHODS
    }
    unanimous_winner = next((method for method, count in first_place_counts.items() if count == len(METRICS)), None)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "experimental_unvalidated",
        "contract_validation": "pass",
        "operator_approved": False,
        "comparison_scope": "missing_t2w_image_quality_only",
        "downstream_nnunet_segmentation_evaluated": False,
        "case_count": len(case_ids),
        "case_list": str(case_list),
        "case_list_sha256": sha256_file(case_list),
        "spatial_preprocessing": EXPECTED_SPATIAL_PREPROCESSING,
        "spatial_audits_identical": True,
        "checkpoint_bindings": checkpoint_bindings,
        "methods": {
            method: {
                "synthesis_type": METHODS[method],
                "mean_metrics": method_data[method]["means"],
                "files": method_data[method]["files"],
            }
            for method in METHODS
        },
        "metric_rankings": rankings,
        "first_place_metric_counts": first_place_counts,
        "unanimous_image_quality_winner": unanimous_winner,
        "paired_bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "pairs": pair_payloads,
        },
        "interpretation_boundary": (
            "This comparison ranks reconstruction metrics under the matched r4 contract. "
            "It does not establish which completion method benefits nnU-Net segmentation."
        ),
    }
    payload["comparison_audit_sha256"] = canonical_sha256(payload, exclude=("comparison_audit_sha256",))

    comparison_json = output_dir / "THREE_WAY_PAIRED_COMPARISON.json"
    comparison_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "PAIRWISE_BOOTSTRAP.csv", pair_rows)
    write_csv(output_dir / "METRIC_RANKINGS.csv", summary_rows)
    write_csv(output_dir / "PAIRED_SUBJECT_METRICS.csv", paired_subject_rows)

    lines = [
        "# G1 r4 three-way paired comparison",
        "",
        "Status: `experimental_unvalidated`; `operator_approved=false`.",
        "",
        "All three methods use the same 103 cases, frozen checkpoints, and spatial preprocessing contract.",
        "",
        "| Metric | Direction | 1st | 2nd | 3rd |",
        "|---|---|---|---|---|",
    ]
    for metric in METRICS:
        order = rankings[metric]
        lines.append(
            f"| {metric} | {'higher' if metric in HIGHER_IS_BETTER else 'lower'} | "
            f"{order[0]} | {order[1]} | {order[2]} |"
        )
    lines.extend([
        "",
        f"Unanimous image-quality winner: `{unanimous_winner or 'none'}`.",
        "",
        "This result compares missing-T2W reconstruction quality only. It does not show which method is best for downstream nnU-Net segmentation.",
    ])
    (output_dir / "THREE_WAY_PAIRED_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    with (output_dir / "SHA256SUMS.txt").open("x", encoding="utf-8") as handle:
        for path in manifest_paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    (output_dir / "THREE_WAY_PAIRED_COMPLETE.ok").write_text("experimental_unvalidated\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "contract_validation": payload["contract_validation"],
        "output_dir": str(output_dir),
        "unanimous_image_quality_winner": unanimous_winner,
        "first_place_metric_counts": first_place_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
