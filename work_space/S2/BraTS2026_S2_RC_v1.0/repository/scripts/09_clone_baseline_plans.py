#!/usr/bin/env python3
"""Clone the fixed real-only nnU-Net plans for Dataset264 preprocessing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(content)
    temporary.replace(path)


def validate_dataset_contract(payload: object, expected_num_training: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError("source dataset.json must contain a JSON object")
    if payload.get("channel_names") != {
        "0": "t1n",
        "1": "t1c",
        "2": "t2w",
        "3": "t2f",
    }:
        raise ValueError("source dataset.json must use G2 channel order t1n/t1c/t2w/t2f")
    labels = payload.get("labels")
    if not isinstance(labels, dict) or sorted(labels.values()) != [0, 1, 2, 3, 4]:
        raise ValueError("source dataset.json must define labels 0/1/2/3/4 exactly once")
    if payload.get("numTraining") != expected_num_training:
        raise ValueError(
            "source dataset.json numTraining mismatch: "
            f"expected={expected_num_training} actual={payload.get('numTraining')}"
        )
    if payload.get("file_ending") != ".nii.gz":
        raise ValueError("source dataset.json file_ending must be .nii.gz")


def load_baseline_plans(
    baseline_plans: Path | None,
    baseline_checkpoint: Path | None,
) -> tuple[dict, Path, str]:
    if baseline_plans is not None:
        source = baseline_plans.expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"baseline plans not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        return payload, source, "plans_file"

    if baseline_checkpoint is None:
        raise ValueError("baseline plans or checkpoint is required")
    source = baseline_checkpoint.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"baseline checkpoint not found: {source}")

    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    init_args = checkpoint.get("init_args")
    if isinstance(init_args, dict):
        payload = init_args.get("plans")
    elif isinstance(init_args, (list, tuple)) and init_args:
        payload = init_args[0]
    else:
        payload = None
    if not isinstance(payload, dict):
        raise ValueError("baseline checkpoint does not contain init_args.plans")
    return copy.deepcopy(payload), source, "checkpoint_init_args"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-plans", type=Path)
    baseline.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--source-dataset-json", required=True, type=Path)
    parser.add_argument("--target-preprocessed-dir", required=True, type=Path)
    parser.add_argument("--target-dataset-name", required=True)
    parser.add_argument("--expected-num-training", type=int, default=1138)
    args = parser.parse_args()

    payload, baseline_source, plans_source_kind = load_baseline_plans(
        args.baseline_plans,
        args.baseline_checkpoint,
    )
    configurations = payload.get("configurations", {})
    fullres = configurations.get("3d_fullres")
    if not isinstance(fullres, dict):
        raise ValueError("baseline plans do not contain 3d_fullres")
    if not isinstance(fullres.get("batch_size"), int) or fullres["batch_size"] <= 0:
        raise ValueError(f"unexpected baseline batch size: {fullres.get('batch_size')}")
    patch_size = fullres.get("patch_size")
    if not isinstance(patch_size, list) or len(patch_size) != 3 or min(patch_size) <= 0:
        raise ValueError(f"unexpected baseline patch size: {patch_size}")
    if len(fullres.get("normalization_schemes", [])) != 4:
        raise ValueError("baseline plans must define four modality normalizers")
    if not isinstance(fullres.get("architecture"), dict):
        raise ValueError("baseline plans must define a 3d_fullres architecture")

    source_dataset_json = args.source_dataset_json.expanduser().resolve()
    if not source_dataset_json.is_file():
        raise SystemExit(f"source dataset.json not found: {source_dataset_json}")
    dataset_payload = json.loads(source_dataset_json.read_text(encoding="utf-8"))
    validate_dataset_contract(dataset_payload, args.expected_num_training)

    source_dataset_name = payload.get("dataset_name", "")
    payload["dataset_name"] = args.target_dataset_name
    target_dir = args.target_preprocessed_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "nnUNetPlans.json"
    atomic_write(target, (json.dumps(payload, indent=4) + "\n").encode("utf-8"))
    target_dataset_json = target_dir / "dataset.json"
    atomic_write(target_dataset_json, source_dataset_json.read_bytes())

    audit = {
        "plans_source_kind": plans_source_kind,
        "plans_source": str(baseline_source),
        "plans_source_sha256": sha256(baseline_source),
        "source_dataset_name": source_dataset_name,
        "target_dataset_name": args.target_dataset_name,
        "target_plans": str(target),
        "target_sha256": sha256(target),
        "source_dataset_json": str(source_dataset_json),
        "source_dataset_json_sha256": sha256(source_dataset_json),
        "target_dataset_json": str(target_dataset_json),
        "target_dataset_json_sha256": sha256(target_dataset_json),
        "expected_num_training": args.expected_num_training,
        "configuration": "3d_fullres",
        "batch_size": fullres["batch_size"],
        "patch_size": fullres["patch_size"],
        "architecture": fullres["architecture"],
    }
    if plans_source_kind == "plans_file":
        audit["baseline_plans"] = str(baseline_source)
        audit["baseline_sha256"] = audit["plans_source_sha256"]
    else:
        audit["baseline_checkpoint"] = str(baseline_source)
        audit["baseline_checkpoint_sha256"] = audit["plans_source_sha256"]
    atomic_write(
        target_dir / "completion_plans_audit.json",
        (json.dumps(audit, indent=2) + "\n").encode("utf-8"),
    )
    print(
        "COMPLETION_PLANS_PASS "
        f"source={source_dataset_name} target={args.target_dataset_name} "
        f"num_training={args.expected_num_training}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
