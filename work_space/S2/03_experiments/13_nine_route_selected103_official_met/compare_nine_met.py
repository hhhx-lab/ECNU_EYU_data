#!/usr/bin/env python3
"""Read-only nine-route comparison of previously validated official MET CSVs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


S2 = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
SCREEN = S2 / "03_experiments/07_screen125_official_met"
SCREEN_R2 = SCREEN / "s2_met_aug_btof_selected103_official_met_20260907_r2"
SCREEN_R4 = SCREEN / "s2_met_aug_btof_selected103_official_met_20260907_r4"
EXT = S2 / "03_experiments/09_epoch175_official_met/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact"
KROOT = S2 / "03_experiments/12_k1245_rc_aware"
K1 = KROOT / "s2_met_aug_k1_selected103_official_met_20261008_r1"
K24 = KROOT / "s2_met_aug_k24_selected103_official_met_20261002_r1"
ORIGINAL = S2 / "07_submissions/original_e_20260727_20260728/s2_official_validation_inference_20260727_r1/selection_evidence"
FROZEN = S2 / "03_experiments/09_epoch175_official_met/s2_met_aug_bef_epoch175_official_met_20260911_r2/runtime/compare_paired_met.py"
FROZEN_SHA = "9ff245240c438beb998652ee59f130fc8c012be29ac3b689c862d04135df6e12"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"empty output: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    require(digest(FROZEN) == FROZEN_SHA, "frozen comparator SHA drift")
    spec = importlib.util.spec_from_file_location("frozen_compare_paired_met", FROZEN)
    require(spec is not None and spec.loader is not None, "frozen comparator import failed")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    sources = {
        "K1": (K1 / "cohort/fixed103/models/K1/evaluation/leaderboard_metrics.csv", K1 / "evidence/K1_VALIDATION.json", 175),
        "K2": (K24 / "cohort/fixed103/models/K2/evaluation/leaderboard_metrics.csv", K24 / "evidence/K2_VALIDATION.json", 175),
        "K4": (K24 / "cohort/fixed103/models/K4/evaluation/leaderboard_metrics.csv", K24 / "evidence/K4_VALIDATION.json", 175),
        "B175": (EXT / "sources/B/epoch175/leaderboard_metrics.csv", EXT / "evidence/B_CPU_VALIDATION.json", 175),
        "C40": (SCREEN_R4 / "cohort/fixed103/models/C40/evaluation/leaderboard_metrics.csv", SCREEN_R4 / "evidence/C40_RECOVERY_VALIDATION.json", 50),
        "D50": (SCREEN_R4 / "cohort/fixed103/models/D/evaluation/leaderboard_metrics.csv", SCREEN_R4 / "evidence/D_RECOVERY_VALIDATION.json", 50),
        "E_SNFH175": (EXT / "sources/E_SNFH/epoch175/leaderboard_metrics.csv", EXT / "evidence/E_SNFH_CPU_VALIDATION.json", 175),
        "F175": (EXT / "sources/F/epoch175/leaderboard_metrics.csv", EXT / "evidence/F_CPU_VALIDATION.json", 175),
        "original_E": (ORIGINAL / "E/leaderboard_metrics.csv", ORIGINAL / "EVALUATION_VALIDATION_AUDIT.json", 200),
    }
    original_audit = load_json(ORIGINAL / "EVALUATION_VALIDATION_AUDIT.json")
    require(original_audit["status"] == "pass", "original E audit failed")
    require(original_audit["models"]["E"]["checkpoint_sha256"] == "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267", "original E identity drift")
    contract = (ORIGINAL / "E/evaluation_contract.txt").read_text(encoding="utf-8")
    for line in ("config=mets", "vol_threshold=27", "overlap_threshold=0.2", "expected_count=103"):
        require(line in contract.splitlines(), f"original E MET contract drift: {line}")
    require((ORIGINAL / "E/EVALUATION_COMPLETE.ok").is_file(), "original E completion marker missing")
    require(load_json(EXT / "evidence/CPU_RECOVERY_VALIDATION.json")["status"] == "pass", "175 CPU recovery failed")

    data = {}
    identities = []
    for name, (path, evidence_path, epoch) in sources.items():
        item = frozen.read_metrics(path, 103)
        require(len(item["metrics"]) == 72, f"metric count drift: {name}")
        evidence = load_json(evidence_path)
        if name == "original_E":
            expected_sha = evidence["models"]["E"]["csv_sha256"]
            checkpoint_sha = evidence["models"]["E"]["checkpoint_sha256"]
        else:
            require(evidence["status"] == "pass", f"validation failed: {name}")
            require(evidence["selected_epoch"] == epoch, f"selected epoch drift: {name}")
            checkpoint_sha = evidence["checkpoint_sha256"]
            validation_root = path.parent if (path.parent / "EVALUATION_VALIDATION.json").is_file() else path.parent / "evaluation"
            validation = validation_root / "EVALUATION_VALIDATION.json"
            require(validation.is_file(), f"evaluation validation missing: {name}")
            eval_result = load_json(validation)
            require(eval_result["status"] == "pass" and eval_result["case_count"] == 103 and eval_result["metric_column_count"] == 72, f"evaluation validation drift: {name}")
            expected_sha = eval_result["metrics_csv_sha256"]
            if "metrics_csv_sha256" in evidence:
                require(evidence["metrics_csv_sha256"] == expected_sha, f"recovery evidence SHA drift: {name}")
            require((validation_root / "EVALUATION_COMPLETE.ok").is_file(), f"evaluation marker missing: {name}")
        require(item["sha256"] == expected_sha, f"CSV SHA drift: {name}")
        require(len(checkpoint_sha) == 64, f"checkpoint SHA missing: {name}")
        for metric in item["metrics"]:
            values = np.asarray([frozen.to_float(item["rows"][case][metric]) for case in item["ids"]])
            finite = values[np.isfinite(values)]
            reported = item["aggregate_mean"][metric]
            require((not len(finite) and not math.isfinite(reported)) or (len(finite) > 0 and math.isclose(float(finite.mean()), reported, rel_tol=1e-9, abs_tol=1e-9)), f"summary mean drift: {name}/{metric}")
        data[name] = item
        identities.append({"method": name, "selected_epoch": epoch, "checkpoint_sha256": checkpoint_sha, "metrics_csv": str(path), "metrics_csv_sha256": expected_sha, "validation": str(evidence_path)})
    names = list(sources)
    ids = data[names[0]]["ids"]
    metrics = data[names[0]]["metrics"]
    for name in names[1:]:
        require(data[name]["ids"] == ids, f"case order drift: {name}")
        require(data[name]["metrics"] == metrics, f"metric order drift: {name}")

    means = []
    for name in names:
        for metric in metrics:
            count = sum(math.isfinite(frozen.to_float(data[name]["rows"][case][metric])) for case in ids)
            means.append({"method": name, "selected_epoch": sources[name][2], "metric": metric, "direction": frozen.direction(metric), "official_mean": data[name]["aggregate_mean"][metric], "finite_case_count": count})

    rng = np.random.default_rng(20260831)
    pairs = []
    for a_name, b_name in itertools.combinations(names, 2):
        for metric in metrics:
            a = np.asarray([frozen.to_float(data[a_name]["rows"][case][metric]) for case in ids], dtype=np.float64)
            b = np.asarray([frozen.to_float(data[b_name]["rows"][case][metric]) for case in ids], dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            n = int(mask.sum())
            row = {"method_a": a_name, "method_b": b_name, "metric": metric, "direction": frozen.direction(metric), "official_mean_a": data[a_name]["aggregate_mean"][metric], "official_mean_b": data[b_name]["aggregate_mean"][metric], "paired_n": n, "excluded_nonfinite": 103 - n, "bootstrap_replicates": 20000, "seed": 20260831}
            if n >= 2:
                raw = a[mask] - b[mask]
                benefit = raw if row["direction"] == "higher_is_better" else -raw
                draws = rng.integers(0, n, size=(20000, n))
                sample = benefit[draws].mean(axis=1)
                low, high = np.quantile(sample, (0.025, 0.975))
                wins = int(np.sum(benefit > 0))
                ties = int(np.sum(benefit == 0))
                row.update({"status": "available", "paired_mean_a": float(a[mask].mean()), "paired_mean_b": float(b[mask].mean()), "raw_mean_a_minus_b": float(raw.mean()), "benefit_mean_a_over_b": float(benefit.mean()), "benefit_ci95_low": float(low), "benefit_ci95_high": float(high), "wins_a": wins, "ties": ties, "wins_b": n - wins - ties, "paired_two_sided_bootstrap_p": min(1.0, 2 * min(float(np.mean(sample <= 0)), float(np.mean(sample >= 0))))})
            else:
                row.update({"status": "unavailable_insufficient_complete_pairs", "paired_mean_a": None, "paired_mean_b": None, "raw_mean_a_minus_b": None, "benefit_mean_a_over_b": None, "benefit_ci95_low": None, "benefit_ci95_high": None, "wins_a": None, "ties": None, "wins_b": None, "paired_two_sided_bootstrap_p": None})
            pairs.append(row)

    require(len(means) == 648 and len(pairs) == 2592, "output row count drift")
    with (K1 / "paired_k124_20261008/PAIRWISE_BOOTSTRAP.csv").open(encoding="utf-8", newline="") as handle:
        previous = list(csv.DictReader(handle))
    checked = 0
    for prior in previous:
        prior_a = prior["method_a"].removesuffix("_175")
        prior_b = prior["method_b"].removesuffix("_175")
        found = next((row for row in pairs if row["method_a"] == prior_a and row["method_b"] == prior_b and row["metric"] == prior["metric"]), None)
        require(found is not None, "prior K-pair missing")
        require(found["paired_n"] == int(prior["paired_complete_case_count"]), "prior K-pair n drift")
        if found["status"] == "available":
            require(math.isclose(found["benefit_mean_a_over_b"], float(prior["benefit_mean_a_over_b"]), abs_tol=1e-12), "prior K-pair benefit drift")
            require((found["wins_a"], found["ties"], found["wins_b"]) == (int(prior["wins_a"]), int(prior["ties"]), int(prior["wins_b"])), "prior K-pair W/T/L drift")
        checked += 1
    require(checked == 216, "prior K-pair record count drift")

    output = ROOT / "results"
    require(not output.exists(), "exclusive results directory already exists")
    output.mkdir()
    write_csv(output / "OFFICIAL_MEANS_648.csv", means)
    write_csv(output / "PAIRED_ALL_2592.csv", pairs)
    write_csv(output / "PAIRED_VS_ORIGINAL_E_576.csv", [row for row in pairs if row["method_b"] == "original_E"])
    payload = {"schema_version": 1, "status": "pass", "artifact_status": "experimental_unvalidated", "operator_approved": False, "formal_gate_status": "not_run_not_passed", "generated_at_local_utc": datetime.now(timezone.utc).isoformat(), "case_count": 103, "metric_count": 72, "method_count": 9, "pair_count": 36, "paired_metric_record_count": 2592, "input_identities": identities, "frozen_comparator_sha256": FROZEN_SHA, "cross_check_prior_k_pair_records": checked, "bootstrap": {"replicates": 20000, "seed": 20260831, "policy": "paired_complete_case_per_metric", "p_values": "two_sided_uncorrected"}, "scope": "Same fixed103 authentic-T2W development cases used for checkpoint selection and reporting; exploratory only. Different training budgets and recipe combinations are displayed together, not an equal-budget causal comparison."}
    with (output / "VALIDATION.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    (output / "COMPLETE.ok").write_text("status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "methods": names, "cases": len(ids), "means": len(means), "pairs": len(pairs), "cross_checked_k_records": checked, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
