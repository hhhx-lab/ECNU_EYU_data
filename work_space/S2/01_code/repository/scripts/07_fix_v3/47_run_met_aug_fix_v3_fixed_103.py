#!/usr/bin/env python3
"""Run the frozen Fix-v3 fixed-103 evaluation and conservative E comparison."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


ROUTE_STATUS = "experimental_unvalidated"
PLACEHOLDER = "<bind_after_training_validation>"
EXPECTED_E_CHECKPOINT_SHA256 = (
    "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
)


class PipelineError(RuntimeError):
    """Raised when the frozen fixed-103 pipeline contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any], *, exclude: Sequence[str] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in set(exclude)}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing JSON evidence: {path}")

    def reject_constant(value: str) -> None:
        raise PipelineError(f"non-finite JSON constant {value}: {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _runtime_files(freeze: Mapping[str, Any]) -> dict[str, str]:
    files = freeze.get("runtime_files_sha256")
    require(isinstance(files, dict) and files, "evaluation runtime file map is missing")
    require(
        all(isinstance(name, str) and re.fullmatch(r"[0-9a-f]{64}", str(digest)) for name, digest in files.items()),
        "evaluation runtime file SHA map is invalid",
    )
    return {str(name): str(digest) for name, digest in files.items()}


def validate_static_plan(plan_path: Path, *, expected_plan_sha: str) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha) is not None, "invalid expected plan SHA")
    require(sha256_file(plan_path) == expected_plan_sha, "fixed-103 evaluation plan SHA drift")
    plan = read_json(plan_path)
    require(plan.get("status") == "awaiting_training_completion", "evaluation plan status drift")
    require(plan.get("route_status") == ROUTE_STATUS, "evaluation route status drift")
    require(
        plan.get("inference_contract") == "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
        "evaluation inference contract drift",
    )
    require(
        plan.get("immutable_boundaries")
        == {
            "official_179_started": False,
            "synapse_upload_allowed": False,
            "zip_creation_allowed": False,
        },
        "evaluation immutable boundary drift",
    )

    freeze_path = Path(str(plan.get("runtime_freeze_path", ""))).resolve()
    require(
        sha256_file(freeze_path) == plan.get("runtime_freeze_sha256"),
        "evaluation runtime freeze SHA drift",
    )
    freeze = read_json(freeze_path)
    require(freeze.get("status") == "pass", "evaluation runtime freeze did not pass")
    require(freeze.get("route_status") == ROUTE_STATUS, "evaluation runtime route drift")
    runtime_root = Path(str(freeze.get("runtime_root", ""))).resolve()
    files = _runtime_files(freeze)
    for name, expected_sha in sorted(files.items()):
        path = runtime_root / name
        require(path.is_file() and path.stat().st_size > 0, f"missing evaluation runtime file: {name}")
        require(sha256_file(path) == expected_sha, f"evaluation runtime SHA drift: {name}")

    evaluator = freeze.get("evaluator")
    require(isinstance(evaluator, dict), "evaluator freeze is missing")
    evaluator_root = Path(str(evaluator.get("root", ""))).resolve()
    executable_names = {
        "python": "python",
        "brats-evaluate": "brats-evaluate",
        "brats-parse-metrics": "brats-parse-metrics",
    }
    executable_evidence: dict[str, str] = {}
    frozen_executables = evaluator.get("executables_sha256")
    require(isinstance(frozen_executables, dict), "evaluator executable SHA map is missing")
    for key, filename in executable_names.items():
        path = evaluator_root / "bin" / filename
        require(path.is_file() and os.access(path, os.X_OK), f"missing evaluator executable: {path}")
        digest = sha256_file(path)
        require(digest == frozen_executables.get(key), f"evaluator executable drift: {key}")
        executable_evidence[key] = digest

    baseline = plan.get("baseline_e")
    require(isinstance(baseline, dict), "baseline E evidence is missing")
    require(baseline.get("checkpoint_sha256") == EXPECTED_E_CHECKPOINT_SHA256, "baseline E checkpoint drift")
    baseline_root = Path(str(baseline.get("root", ""))).resolve()
    baseline_files = {
        "completion_marker_sha256": baseline_root / "EVALUATION_COMPLETE.ok",
        "metrics_sha256": baseline_root / "leaderboard_metrics.csv",
        "preparation_summary_sha256": baseline_root / "preparation_summary.json",
        "summary_sha256": baseline_root / "panoptica_evaluation_summary.json",
    }
    for key, path in baseline_files.items():
        require(path.is_file() and path.stat().st_size > 0, f"missing baseline E evidence: {path}")
        require(sha256_file(path) == baseline.get(key), f"baseline E evidence drift: {key}")
    require(baseline.get("prediction_count") == 103, "baseline E prediction count drift")
    require(baseline.get("reference_count") == 103, "baseline E reference count drift")

    for key in ("prepare_command", "evaluate_command", "selection_command_template"):
        command = plan.get(key)
        require(isinstance(command, list) and command, f"missing plan command: {key}")
        require(all(isinstance(item, str) and item for item in command), f"invalid plan command: {key}")
    selection_template = plan["selection_command_template"]
    require(selection_template.count(PLACEHOLDER) == 1, "selection checkpoint placeholder drift")

    return {
        "plan": plan,
        "plan_sha256": expected_plan_sha,
        "runtime_freeze_sha256": sha256_file(freeze_path),
        "runtime_file_count": len(files),
        "evaluator_executables_sha256": executable_evidence,
        "baseline_e_root": str(baseline_root),
        "baseline_e_checkpoint_sha256": EXPECTED_E_CHECKPOINT_SHA256,
    }


def validate_training_evidence(
    path: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    required_path = Path(str(plan["prerequisite"]["training_validation_path"])).resolve()
    require(path == required_path, "training validation path drift")
    evidence = read_json(path)
    require(evidence.get("status") == "pass", "training validation did not pass")
    require(evidence.get("route_status") == ROUTE_STATUS, "training validation route drift")
    require(evidence.get("training_complete") is True, "training is not complete")
    require(evidence.get("expected_epochs") == 200, "training epoch evidence drift")
    require(evidence.get("fixed_validation_count") == 103, "training validation count drift")
    require(evidence.get("official_179_started") is False, "official 179 was already started")
    require(evidence.get("zip_created") is False, "training route created a ZIP")
    require(evidence.get("synapse_uploaded") is False, "training route uploaded to Synapse")
    audit = evidence.get("validation_audit_sha256")
    require(
        audit == canonical_sha256(evidence, exclude=("validation_audit_sha256",)),
        "training validation audit SHA drift",
    )

    checkpoint = evidence.get("checkpoint_final")
    require(isinstance(checkpoint, dict), "training checkpoint evidence is missing")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    require(checkpoint_path == Path(str(plan["checkpoint"])).resolve(), "evaluation checkpoint path drift")
    require(checkpoint.get("current_epoch") == 200, "evaluation checkpoint epoch drift")
    require(re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is not None, "invalid Fix-v3 checkpoint SHA")
    require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, "Fix-v3 checkpoint is missing")
    require(sha256_file(checkpoint_path) == checkpoint_sha, "Fix-v3 checkpoint SHA drift")

    validation = evidence.get("validation")
    require(isinstance(validation, dict), "training prediction evidence is missing")
    prediction_root = Path(str(validation.get("root", ""))).resolve()
    require(prediction_root == Path(str(plan["prediction_root"])).resolve(), "prediction root drift")
    require(validation.get("prediction_count") == 103, "prediction count evidence drift")
    predictions = sorted(prediction_root.glob("*.nii.gz"))
    require(len(predictions) == 103, "fixed-103 prediction file count drift")
    require((prediction_root / "summary.json").is_file(), "nnU-Net validation summary is missing")
    require(
        sha256_file(prediction_root / "summary.json") == validation.get("summary_sha256"),
        "nnU-Net validation summary SHA drift",
    )

    artifact_manifest = evidence.get("artifact_manifest")
    require(isinstance(artifact_manifest, dict), "training artifact manifest evidence is missing")
    manifest_path = Path(str(artifact_manifest.get("path", ""))).resolve()
    require(manifest_path.is_file() and manifest_path.stat().st_size > 0, "training artifact manifest is missing")
    require(sha256_file(manifest_path) == artifact_manifest.get("sha256"), "training artifact manifest drift")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "validation_audit_sha256": audit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_root": str(prediction_root),
        "prediction_count": 103,
    }


def build_selection_command(
    template: Sequence[str],
    *,
    checkpoint_sha256: str,
) -> list[str]:
    require(re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is not None, "invalid checkpoint SHA")
    require(list(template).count(PLACEHOLDER) == 1, "selection checkpoint placeholder drift")
    command = [checkpoint_sha256 if item == PLACEHOLDER else item for item in template]
    require(PLACEHOLDER not in command, "selection checkpoint placeholder remains unresolved")
    return command


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _run_stage(
    name: str,
    command: Sequence[str],
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    started = utc_now()
    monotonic_start = time.monotonic()
    result = runner(command)
    output = str(result.stdout or "")
    record = {
        "name": name,
        "command": list(command),
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - monotonic_start,
        "returncode": int(result.returncode),
        "stdout_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "stdout_tail": output[-4000:],
    }
    require(result.returncode == 0, f"pipeline stage failed: {name}, exit={result.returncode}")
    return record


def _freeze_success_outputs(evaluation_root: Path, selection_root: Path) -> None:
    for root in (evaluation_root, selection_root):
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".nii.gz"):
                path.chmod(0o444)
        for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o555)
        root.chmod(0o555)


def run_pipeline(
    *,
    plan_path: Path,
    expected_plan_sha: str,
    training_validation: Path,
    report: Path,
    failure_report: Path,
    preflight_only: bool = False,
    dry_run: bool = False,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _default_runner,
) -> dict[str, Any]:
    report = report.expanduser().resolve()
    failure_report = failure_report.expanduser().resolve()
    require(not report.exists(), f"refusing to overwrite pipeline report: {report}")
    require(not failure_report.exists(), f"refusing to overwrite failure report: {failure_report}")
    static = validate_static_plan(plan_path, expected_plan_sha=expected_plan_sha)
    plan = static.pop("plan")
    if preflight_only:
        return {"status": "preflight_pass", "route_status": ROUTE_STATUS, "static": static}

    stages: list[dict[str, Any]] = []
    try:
        training = validate_training_evidence(training_validation, plan=plan)
        evaluation_root = Path(str(plan["fix_v3_evaluation_root"])).resolve()
        selection_root = Path(str(plan["selection_output_root"])).resolve()
        require(not evaluation_root.exists(), f"refusing to reuse evaluation root: {evaluation_root}")
        require(not selection_root.exists(), f"refusing to reuse selection root: {selection_root}")
        selection_command = build_selection_command(
            plan["selection_command_template"],
            checkpoint_sha256=training["checkpoint_sha256"],
        )
        commands = {
            "prepare": list(plan["prepare_command"]),
            "official_evaluation": list(plan["evaluate_command"]),
            "paired_selection": selection_command,
        }
        if dry_run:
            return {
                "status": "dry_run_pass",
                "route_status": ROUTE_STATUS,
                "static": static,
                "training": training,
                "commands": commands,
            }

        stages.append(_run_stage("prepare", commands["prepare"], runner=runner))
        preparation = read_json(evaluation_root / "preparation_summary.json")
        require(preparation.get("status") == "pass", "evaluation preparation did not pass")
        require(preparation.get("checkpoint_sha256") == training["checkpoint_sha256"], "prepared checkpoint drift")
        require(preparation.get("prediction_count") == 103, "prepared prediction count drift")
        require(preparation.get("reference_count") == 103, "prepared reference count drift")

        stages.append(
            _run_stage("official_evaluation", commands["official_evaluation"], runner=runner)
        )
        marker = evaluation_root / "EVALUATION_COMPLETE.ok"
        require(marker.is_file() and "status=pass" in marker.read_text(), "official evaluation marker failed")

        stages.append(_run_stage("paired_selection", commands["paired_selection"], runner=runner))
        selection_path = selection_root / "MODEL_SELECTION.json"
        selection = read_json(selection_path)
        require(selection.get("status") == "pass", "model selection did not pass")
        require(selection.get("route_status") == ROUTE_STATUS, "model selection route drift")
        require(selection.get("selected_model") in {"E", "Fix_v3"}, "invalid selected model")
        require(selection.get("immutable_boundaries", {}).get("official_179_started") is False, "179 started early")
        require(not list(evaluation_root.rglob("*.zip")), "ZIP output is forbidden")
        require(not list(selection_root.rglob("*.zip")), "selection ZIP output is forbidden")

        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "pass",
            "route_status": ROUTE_STATUS,
            "completed_at_utc": utc_now(),
            "static_preflight": static,
            "training_validation": training,
            "stages": stages,
            "fix_v3_evaluation_root": str(evaluation_root),
            "evaluation_completion_marker_sha256": sha256_file(marker),
            "evaluation_metrics_sha256": sha256_file(evaluation_root / "leaderboard_metrics.csv"),
            "evaluation_summary_sha256": sha256_file(
                evaluation_root / "panoptica_evaluation_summary.json"
            ),
            "selection_root": str(selection_root),
            "model_selection_sha256": sha256_file(selection_path),
            "selected_model": selection["selected_model"],
            "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "official_179_started": False,
            "zip_created": False,
            "synapse_uploaded": False,
            "orchestrator_sha256": sha256_file(Path(__file__)),
        }
        result["pipeline_audit_sha256"] = canonical_sha256(result)
        write_exclusive(report, result)
        _freeze_success_outputs(evaluation_root, selection_root)
        return result
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "fail",
            "route_status": ROUTE_STATUS,
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_stages": stages,
            "official_179_started": False,
            "zip_created": False,
            "synapse_uploaded": False,
            "orchestrator_sha256": sha256_file(Path(__file__)),
        }
        failure["failure_audit_sha256"] = canonical_sha256(failure)
        write_exclusive(failure_report, failure)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha", required=True)
    parser.add_argument("--training-validation", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--failure-report", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        plan_path=args.plan,
        expected_plan_sha=args.expected_plan_sha,
        training_validation=args.training_validation,
        report=args.report,
        failure_report=args.failure_report,
        preflight_only=args.preflight_only,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
