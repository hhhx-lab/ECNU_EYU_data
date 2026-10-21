#!/usr/bin/env python3
"""Assemble a self-contained nnU-Net inference model root from a frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-trainer", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    require(sha256_file(checkpoint) == args.expected_sha256, "checkpoint SHA drift")
    require(not output_root.exists(), f"model snapshot already exists: {output_root}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(payload.get("trainer_name") == args.expected_trainer, "trainer name drift")
    init_args = payload.get("init_args")
    require(isinstance(init_args, dict), "checkpoint init_args missing")
    plans = init_args.get("plans")
    dataset = init_args.get("dataset_json")
    require(isinstance(plans, dict) and isinstance(dataset, dict), "embedded plans/dataset missing")
    require(dataset.get("channel_names") in ({"0": "T1N", "1": "T1C", "2": "T2W", "3": "T2F"}, {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}), "channel order drift")

    fold_root = output_root / "fold_0"
    fold_root.mkdir(parents=True, exist_ok=False)
    target_checkpoint = fold_root / "checkpoint_final.pth"
    try:
        os.link(checkpoint, target_checkpoint)
        mode = "hardlink"
    except OSError:
        shutil.copy2(checkpoint, target_checkpoint)
        mode = "copy"
    (output_root / "plans.json").write_text(
        json.dumps(plans, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "model_name": args.model_name,
        "trainer_name": args.expected_trainer,
        "checkpoint_source": str(checkpoint),
        "checkpoint_materialization_mode": mode,
        "checkpoint_sha256": sha256_file(target_checkpoint),
        "plans_sha256": sha256_file(output_root / "plans.json"),
        "dataset_sha256": sha256_file(output_root / "dataset.json"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_root / "MODEL_SNAPSHOT.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
