#!/usr/bin/env python3
"""Validate the preregistered engineering fallback against immutable r2 evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


PLAN = Path(__file__).resolve().parents[1] / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
ORDER = ("1", "2_3", "4_8", "9_16", "17_24", "gt24")
EXPECTED_CASES = {"1": 265, "2_3": 259, "4_8": 269, "9_16": 148, "17_24": 41, "gt24": 53}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2_3"
    if count <= 8:
        return "4_8"
    if count <= 16:
        return "9_16"
    if count <= 24:
        return "17_24"
    return "gt24"


def probabilities(means: dict[str, float], counts: dict[str, int]) -> tuple[dict[str, float], float]:
    overall = sum(means[key] * counts[key] for key in ORDER) / sum(counts.values())
    if not math.isfinite(overall) or overall <= 0:
        raise ValueError("invalid case-weighted small-lesion fraction")
    normalized = {key: means[key] / overall for key in ORDER}

    def weighted_mean(scale: float) -> float:
        return sum(
            min(0.60, 0.20 * scale * normalized[key]) * counts[key]
            for key in ORDER
        ) / sum(counts.values())

    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if weighted_mean(middle) < 0.20:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    return ({key: min(0.60, 0.20 * scale * normalized[key]) for key in ORDER}, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    receipt = args.r2_receipt.resolve()
    if sha256(receipt) != plan["fallback_gate"]["source_snapshot_receipt_sha256"]:
        raise SystemExit("r2 snapshot receipt SHA mismatch")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    result = data.get("result", {})
    files = result.get("files", {})
    if not result.get("complete") or result.get("failed") is not None:
        raise SystemExit("r2 snapshot does not attest a completed, nonfailed pool build")
    for name, expected in (
        ("K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json", plan["parent_contract"]["r2_experiment_plan_sha256"]),
        ("evidence/REMOTE_STATIC_PREFLIGHT.json", plan["fallback_gate"]["source_remote_static_preflight_sha256"]),
        ("evidence/POOL_BUILD_COMPLETE.ok", plan["fallback_gate"]["source_pool_complete_marker_sha256"]),
    ):
        if files.get(name) != expected:
            raise SystemExit(f"r2 source identity mismatch: {name}")
    pool = result.get("pool_validation")
    if not isinstance(pool, dict) or pool.get("status") != "blocked_f_gate":
        raise SystemExit("r2 pool result missing or not the expected strict-gate failure")
    manifests = pool.get("manifests", {})
    for name, expected in (
        ("rc27", plan["fallback_gate"]["source_rc27_manifest_identity_sha256"]),
        ("small8", plan["fallback_gate"]["source_small8_manifest_identity_sha256"]),
    ):
        if manifests.get(name, {}).get("sha256") != expected:
            raise SystemExit(f"frozen {name} manifest identity mismatch")

    empirical = pool["empirical"]
    case_counts = empirical.get("target_counts", {})
    if len(case_counts) != 1035 or any(not isinstance(v, int) or v < 0 for v in case_counts.values()):
        raise SystemExit("frozen training case-count denominator is invalid")
    observed_counts = {key: 0 for key in ORDER}
    for count in case_counts.values():
        observed_counts[bucket(count)] += 1
    means = {key: float(empirical["burden_small_fraction_mean"][key]) for key in ORDER}
    intervals = {key: [float(v) for v in empirical["burden_small_fraction_ci95"][key]] for key in ORDER}
    probs, scale = probabilities(means, observed_counts)
    p_mean = sum(probs[bucket(count)] for count in case_counts.values()) / len(case_counts)
    tol = float(plan["fallback_gate"]["criteria"]["fraction_order_tolerance"])
    p_tol = float(plan["fallback_gate"]["criteria"]["probability_order_tolerance"])
    checks = {
        "strict_gate_remains_false": empirical.get("f_gate_pass") is False,
        "fixed_case_denominator": len(case_counts) == 1035,
        "burden_bin_counts": observed_counts == EXPECTED_CASES,
        "fraction_non_decreasing": all(means[b] + tol >= means[a] for a, b in zip(ORDER, ORDER[1:])),
        "probability_non_decreasing": all(probs[b] + p_tol >= probs[a] for a, b in zip(ORDER, ORDER[1:])),
        "probability_mean_calibrated": abs(p_mean - 0.20) <= 0.0001,
        "probability_bounds": min(probs.values()) >= 0.0 and max(probs.values()) <= 0.60,
        "source_rc_and_small8_manifests_bound": True,
    }
    passed = all(checks.values())
    payload = {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "pass" if passed else "fail",
        "gate_name": "F_FALLBACK_ENGINEERING_ELIGIBILITY_GATE",
        "is_scientific_gate": False,
        "is_significance_test": False,
        "F_STRICT_PREREG_GATE": False,
        "checks": checks,
        "case_count": len(case_counts),
        "burden_bin_case_counts": observed_counts,
        "burden_small_fraction_mean": means,
        "burden_small_fraction_ci95_descriptive_only": intervals,
        "burden_probabilities_recomputed": probs,
        "probability_mean_recomputed": p_mean,
        "probability_scale_bisection_0_10_200_iterations": scale,
        "source_limitation": "R2 snapshot includes per-case lesion counts and frozen bin-level fraction summaries, but not the 1,035 per-case fraction values. This gate validates the preregistered frozen summaries and recomputes case-weighted probabilities; it does not rederive lesion fractions from voxel labels.",
        "source_receipt_sha256": sha256(receipt),
        "r2_plan_sha256": plan["parent_contract"]["r2_experiment_plan_sha256"],
        "r2_pool_status": pool["status"],
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
