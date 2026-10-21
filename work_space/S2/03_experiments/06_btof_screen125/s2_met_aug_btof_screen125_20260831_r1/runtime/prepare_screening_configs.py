#!/usr/bin/env python3
"""Materialize train-only burden counts and five immutable screening configs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import blosc2
except ImportError:  # pragma: no cover - required by the remote nnU-Net cache
    blosc2 = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: dict, *, exclude: tuple[str, ...] = ()) -> str:
    payload = {key: value[key] for key in value if key not in set(exclude)}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"JSON round-trip failed: {path}")


def load_segmentation(path: Path) -> np.ndarray:
    b2nd_path = path.with_name(f"{path.stem}_seg.b2nd")
    if b2nd_path.is_file():
        if blosc2 is None:
            raise RuntimeError("the configured environment lacks blosc2")
        segmentation = np.asarray(blosc2.open(str(b2nd_path), mode="r")[:])
        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]
        if segmentation.ndim != 3:
            raise RuntimeError(f"unexpected segmentation shape for {b2nd_path}: {segmentation.shape}")
        return segmentation
    with np.load(path, allow_pickle=False) as payload:
        if "seg" in payload:
            segmentation = payload["seg"]
        elif "data" in payload and payload["data"].ndim == 4 and payload["data"].shape[0] >= 5:
            segmentation = payload["data"][-1:]
        else:
            raise RuntimeError(f"cannot locate segmentation in {path}; keys={list(payload.keys())}")
    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    if segmentation.ndim != 3:
        raise RuntimeError(f"unexpected segmentation shape for {path}: {segmentation.shape}")
    return segmentation


def solve_probabilities(counts: dict[str, int]) -> dict[str, float]:
    weights = {"0_1": 2.0, "2_4": 1.25, "5_9": 0.75, "10_plus": 0.375}

    def bucket(value: int) -> str:
        return "0_1" if value <= 1 else "2_4" if value <= 4 else "5_9" if value <= 9 else "10_plus"

    def mean_for(scale: float) -> float:
        return float(np.mean([min(0.8, scale * weights[bucket(value)] * 0.2) for value in counts.values()]))

    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if mean_for(middle) < 0.20:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    probabilities = {key: min(0.8, scale * value * 0.2) for key, value in weights.items()}
    observed = float(np.mean([probabilities[bucket(value)] for value in counts.values()]))
    if abs(observed - 0.20) > 1e-12:
        raise RuntimeError(f"adaptive probability calibration failed: {observed}")
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--base-route-config", required=True, type=Path)
    parser.add_argument("--component-manifest", required=True, type=Path)
    parser.add_argument("--preprocessed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("experiment_id") != "s2_met_aug_btof_screen125_20260831_r1":
        raise RuntimeError("unexpected screening plan")
    manifest = json.loads(args.component_manifest.read_text(encoding="utf-8"))
    groups_path = args.component_manifest.parent / manifest["target_groups_file"]
    groups = json.loads(groups_path.read_text(encoding="utf-8"))["case_to_patient_group"]
    counts: dict[str, int] = {}
    structure = np.ones((3, 3, 3), dtype=bool)
    for case_id in sorted(groups):
        segmentation = load_segmentation(args.preprocessed / f"{case_id}.npz")
        _, count = ndimage.label(np.isin(segmentation, (1, 2, 3, 4)), structure=structure)
        counts[case_id] = int(count)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    counts_payload = {
        "artifact_status": "experimental_unvalidated",
        "case_count": len(counts),
        "case_to_native_lesion_count": counts,
        "formal_gate_status": "not_run_not_passed",
        "operator_approved": False,
        "segmentation_definition": "26-connected components of labels 1/2/3/4 in frozen train preprocessed segmentation",
        "split": "train_only",
        "status": "pass",
        "timestamp_utc": timestamp,
    }
    counts_path = args.output / "TRAIN_NATIVE_LESION_COUNTS.json"
    write_exclusive(counts_path, counts_payload)
    probabilities = solve_probabilities(counts)
    common = {
        "artifact_status": "experimental_unvalidated",
        "base_route_config_sha256": sha256_file(args.base_route_config),
        "formal_gate_status": "not_run_not_passed",
        "operator_approved": False,
        "plan_sha256": sha256_file(args.plan),
        "schema_version": 1,
    }
    route_payloads = {
        "B": {"route_id": "MET-AUG-B", "p_select": 0.20, "scale_policy": "core_volume_controlled_v1"},
        "C40": {"route_id": "MET-AUG-C40", "p_select": 0.40},
        "D": {"route_id": "MET-AUG-D", "p_select": 0.20, "second_tumour_probability": 0.20},
        "E_SNFH": {
            "route_id": "MET-AUG-E-SNFH",
            "p_select": 0.20,
            "topology_policy": "pure_et_snfh_shell_1mm_v1",
            "topology_probability": 0.10,
        },
        "F": {
            "route_id": "MET-AUG-F",
            "p_select": 0.20,
            "target_counts_sha256": sha256_file(counts_path),
            "burden_probabilities": probabilities,
        },
    }
    for label, route_values in route_payloads.items():
        payload = {
            **common,
            "burden_probabilities": {},
            "p_select": 0.20,
            "route_id": "",
            "scale_policy": "none",
            "second_tumour_probability": 0.0,
            "target_counts_sha256": None,
            "topology_policy": "none",
            "topology_probability": 0.0,
            **route_values,
        }
        payload["screen_config_sha256"] = canonical_sha(payload)
        write_exclusive(args.output / f"SCREEN_ROUTE_{label}.json", payload)
    summary = {
        "artifact_status": "experimental_unvalidated",
        "burden_probabilities": probabilities,
        "config_count": len(route_payloads),
        "formal_gate_status": "not_run_not_passed",
        "operator_approved": False,
        "plan_sha256": sha256_file(args.plan),
        "status": "pass",
        "target_count": len(counts),
        "target_counts_sha256": sha256_file(counts_path),
        "timestamp_utc": timestamp,
    }
    write_exclusive(args.output / "SCREEN_CONFIG_PREPARATION.json", summary)


if __name__ == "__main__":
    main()
