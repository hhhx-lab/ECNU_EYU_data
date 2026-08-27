#!/usr/bin/env python3
"""Validate the remote static bundle, source cohorts, and frozen model roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    require(root.is_dir() and not target.exists(), "root missing or remote preflight already exists")
    check = subprocess.run(
        ["sha256sum", "-c", "STATIC_SHA256SUMS.txt"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(check.returncode == 0, f"static SHA validation failed:\n{check.stdout}")
    plan_path = root / "ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan["artifact_status"] == "experimental_unvalidated", "artifact status drift")
    require(plan["operator_approved"] is False, "operator approval drift")
    require(plan["formal_gate_status"] == "not_run_not_passed", "formal gate drift")
    require(plan["stages"] == ["val27_r4_synthesis", "val27_four_models", "fixed103_real_vs_synthetic", "test26_locked_endpoint"], "stage order drift")

    cohort_files = {
        "val27": (root / "manifests/VAL27_MISSING_T2W_CASES.csv", 27),
        "fixed103": (root / "manifests/FIXED103_REAL_SYNTHETIC_CASES.csv", 103),
        "test26": (root / "manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv", 26),
    }
    ids: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}
    for cohort, (path, expected) in cohort_files.items():
        rows = read_csv(path)
        require(len(rows) == expected, f"{cohort} count drift")
        ids[cohort] = {row["source_case_id"] for row in rows}
        require(len(ids[cohort]) == expected, f"{cohort} duplicate IDs")
        source_files = 0
        for row in rows:
            modalities = ("t1n", "t1c", "t2f", "seg") if cohort != "fixed103" else ("t1n", "t1c", "t2w", "t2f", "seg")
            for modality in modalities:
                source = Path(row[f"{modality}_source_path"])
                require(source.is_file() and source.stat().st_size > 0, f"missing source: {source}")
                source_files += 1
            if cohort != "fixed103":
                require(not row["t2w_source_path"].strip(), f"old T2W exposed in {cohort}")
                require(row["source_t2w_allowed"].lower() == "false", f"source T2W allowed in {cohort}")
        source_counts[cohort] = source_files
    require(not ids["val27"] & ids["fixed103"], "val27/fixed103 overlap")
    require(not ids["test26"] & ids["fixed103"], "test26/fixed103 overlap")
    require(not ids["val27"] & ids["test26"], "val27/test26 overlap")

    r4_root = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4")
    r4_outputs = sorted((r4_root / "ensemble_r4/synthesized").glob("*.nii.gz"))
    require(len(r4_outputs) == 103, "r4 fixed103 synthesis count drift")
    r4_ids = {path.name.removesuffix("-t2w.nii.gz") for path in r4_outputs}
    require(r4_ids == ids["fixed103"], "r4 fixed103 case coverage drift")
    geometry = json.loads((r4_root / "ensemble_r4/geometry_audit/geometry_audit.json").read_text())
    require(geometry["case_count"] == 103 and geometry["geometry_mismatch_before_count"] == 0 and geometry["repaired_count"] == 0, "r4 geometry audit drift")

    checkpoints: dict[str, dict[str, Any]] = {}
    for name, binding in plan["checkpoints"].items():
        model_root = Path(binding["remote_model_root"])
        checkpoint = model_root / "fold_0/checkpoint_final.pth"
        require((model_root / "dataset.json").is_file(), f"missing dataset.json for {name}")
        require((model_root / "plans.json").is_file(), f"missing plans.json for {name}")
        require(checkpoint.is_file(), f"missing checkpoint for {name}")
        observed = sha256_file(checkpoint)
        require(observed == binding["sha256"], f"checkpoint SHA drift for {name}")
        checkpoints[name] = {"root": str(model_root), "checkpoint_sha256": observed}
    for name in ("R", "B"):
        snapshot = json.loads((Path(plan["checkpoints"][name]["remote_model_root"]) / "MODEL_SNAPSHOT.json").read_text())
        require(snapshot["status"] == "pass" and snapshot["checkpoint_sha256"] == plan["checkpoints"][name]["sha256"], f"model snapshot invalid: {name}")

    g1_checkpoints: dict[str, dict[str, str]] = {}
    for name in ("vae", "encdec", "bbdm"):
        binding = plan["g1_r4_checkpoints"][name]
        checkpoint = Path(binding["remote_path"])
        require(checkpoint.is_file(), f"missing G1 {name} checkpoint: {checkpoint}")
        observed = sha256_file(checkpoint)
        require(observed == binding["sha256"], f"G1 {name} checkpoint SHA drift")
        g1_checkpoints[name] = {"path": str(checkpoint), "sha256": observed}
    require(plan["g1_r4_checkpoints"]["bbdm_s"] == 0.01, "G1 BBDM s drift")

    g1_code = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3")
    for relative, expected_sha in plan["g1_runtime_code_sha256"].items():
        require(sha256_file(g1_code / relative) == expected_sha, f"G1 runtime code drift: {relative}")
    seg_python = Path("/public/home/zqchen/.conda/envs/segmamba/bin/python")
    eval_python = Path("/public/home/zqchen/.conda/envs/brats_eval/bin/python")
    for executable in (
        seg_python,
        eval_python,
        Path("/public/home/zqchen/.conda/envs/brats_eval/bin/brats-evaluate"),
        Path("/public/home/zqchen/.conda/envs/brats_eval/bin/brats-parse-metrics"),
    ):
        require(executable.is_file() and os.access(executable, os.X_OK), f"missing executable: {executable}")
    version_check = subprocess.run(
        [
            str(eval_python),
            "-c",
            (
                "import importlib.metadata as m,json;"
                "print(json.dumps({'BraTS-evaluation':m.version('BraTS-evaluation'),"
                "'panoptica':m.version('panoptica'),'numpy':m.version('numpy')},sort_keys=True))"
            ),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(version_check.returncode == 0, f"evaluation version probe failed: {version_check.stdout}")
    evaluation_versions = json.loads(version_check.stdout)
    require(evaluation_versions == {"BraTS-evaluation": "0.0.8", "panoptica": "2.1.0", "numpy": "1.26.4"}, f"evaluation environment drift: {evaluation_versions}")

    disk = os.statvfs("/public")
    available_bytes = disk.f_bavail * disk.f_frsize
    require(available_bytes >= 100 * 1024**3, "less than 100 GiB available on /public")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "execution_plan_sha256": sha256_file(plan_path),
        "static_sha256sum_check": "pass",
        "cohort_counts": {name: count for name, (_, count) in cohort_files.items()},
        "source_file_counts": source_counts,
        "cohorts_pairwise_disjoint": True,
        "r4_fixed103_nifti_count": 103,
        "r4_geometry_mismatch_count": 0,
        "r4_geometry_repaired_count": 0,
        "checkpoints": checkpoints,
        "g1_checkpoints": g1_checkpoints,
        "g1_runtime_code_file_count": len(plan["g1_runtime_code_sha256"]),
        "evaluation_versions": evaluation_versions,
        "available_bytes_public": available_bytes,
        "old_synthesized_t2w_reuse_allowed": False,
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
