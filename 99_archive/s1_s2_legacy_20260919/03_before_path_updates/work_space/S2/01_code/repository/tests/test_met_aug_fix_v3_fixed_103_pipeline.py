from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "47_run_met_aug_fix_v3_fixed_103.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fix_v3_fixed_103_pipeline", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fixture(root: Path, module):
    runtime = root / "runtime"
    runtime_files = {
        "scripts/10_prepare_internal_official_eval.py": "prepare\n",
        "scripts/11_run_internal_official_eval.sh": "evaluate\n",
        "scripts/45_finalize_met_aug_fix_v3_selection.py": "select\n",
        "tests/test_prepare_internal_official_eval.py": "test prepare\n",
        "tests/test_met_aug_fix_v3_selection.py": "test select\n",
        "tools/mapping_fixed.tsv": "a\tb\n",
    }
    for name, content in runtime_files.items():
        _write(runtime / name, content)

    evaluator = root / "evaluator"
    executable_shas = {}
    for name in ("python", "brats-evaluate", "brats-parse-metrics"):
        path = evaluator / "bin" / name
        _write(path, f"{name}\n")
        path.chmod(0o755)
        executable_shas[name] = _sha(path)

    baseline = root / "E"
    baseline_files = {
        "completion_marker_sha256": baseline / "EVALUATION_COMPLETE.ok",
        "metrics_sha256": baseline / "leaderboard_metrics.csv",
        "preparation_summary_sha256": baseline / "preparation_summary.json",
        "summary_sha256": baseline / "panoptica_evaluation_summary.json",
    }
    for key, path in baseline_files.items():
        _write(path, key + "\n")

    freeze = {
        "status": "pass",
        "route_status": module.ROUTE_STATUS,
        "runtime_root": str(runtime),
        "runtime_files_sha256": {
            name: _sha(runtime / name) for name in runtime_files
        },
        "evaluator": {
            "root": str(evaluator),
            "executables_sha256": {
                "python": executable_shas["python"],
                "brats-evaluate": executable_shas["brats-evaluate"],
                "brats-parse-metrics": executable_shas["brats-parse-metrics"],
            },
        },
    }
    freeze_path = root / "freeze.json"
    _write(freeze_path, json.dumps(freeze))

    training_validation = root / "TRAINING_VALIDATION.json"
    checkpoint = root / "checkpoint_final.pth"
    checkpoint.write_bytes(b"fix-v3-checkpoint")
    prediction_root = root / "validation"
    prediction_root.mkdir()
    for index in range(103):
        (prediction_root / f"case-{index:03d}.nii.gz").write_bytes(b"prediction")
    summary_path = prediction_root / "summary.json"
    _write(summary_path, "{}\n")
    artifact_manifest = root / "TRAINING_ARTIFACTS.sha256"
    _write(artifact_manifest, "artifact manifest\n")
    evaluation_root = root / "fix_v3_evaluation"
    selection_root = root / "selection"
    plan = {
        "status": "awaiting_training_completion",
        "route_status": module.ROUTE_STATUS,
        "inference_contract": "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
        "immutable_boundaries": {
            "official_179_started": False,
            "synapse_upload_allowed": False,
            "zip_creation_allowed": False,
        },
        "runtime_freeze_path": str(freeze_path),
        "runtime_freeze_sha256": _sha(freeze_path),
        "baseline_e": {
            "root": str(baseline),
            "checkpoint_sha256": module.EXPECTED_E_CHECKPOINT_SHA256,
            "prediction_count": 103,
            "reference_count": 103,
            **{key: _sha(path) for key, path in baseline_files.items()},
        },
        "prerequisite": {
            "training_validation_path": str(training_validation),
        },
        "checkpoint": str(checkpoint),
        "prediction_root": str(prediction_root),
        "fix_v3_evaluation_root": str(evaluation_root),
        "selection_output_root": str(selection_root),
        "prepare_command": [str(evaluator / "bin/python"), str(runtime / "scripts/10_prepare_internal_official_eval.py")],
        "evaluate_command": ["bash", str(runtime / "scripts/11_run_internal_official_eval.sh")],
        "selection_command_template": [
            str(evaluator / "bin/python"),
            str(runtime / "scripts/45_finalize_met_aug_fix_v3_selection.py"),
            "--expected-fix-v3-checkpoint-sha",
            module.PLACEHOLDER,
        ],
    }
    plan_path = root / "plan.json"
    _write(plan_path, json.dumps(plan))
    training = {
        "status": "pass",
        "route_status": module.ROUTE_STATUS,
        "training_complete": True,
        "expected_epochs": 200,
        "fixed_validation_count": 103,
        "official_179_started": False,
        "zip_created": False,
        "synapse_uploaded": False,
        "checkpoint_final": {
            "path": str(checkpoint),
            "sha256": _sha(checkpoint),
            "current_epoch": 200,
        },
        "validation": {
            "root": str(prediction_root),
            "prediction_count": 103,
            "summary_sha256": _sha(summary_path),
        },
        "artifact_manifest": {
            "path": str(artifact_manifest),
            "sha256": _sha(artifact_manifest),
        },
    }
    training["validation_audit_sha256"] = module.canonical_sha256(training)
    _write(training_validation, json.dumps(training))
    return plan_path, training_validation, plan


class FixV3Fixed103PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script()

    def test_static_preflight_verifies_runtime_evaluator_and_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, _training, _plan = _fixture(root, self.module)
            result = self.module.validate_static_plan(
                plan_path,
                expected_plan_sha=_sha(plan_path),
            )
            self.assertEqual(result["runtime_file_count"], 6)
            self.assertEqual(
                result["baseline_e_checkpoint_sha256"],
                self.module.EXPECTED_E_CHECKPOINT_SHA256,
            )

    def test_static_preflight_rejects_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, _training, plan = _fixture(root, self.module)
            runtime_script = root / "runtime/scripts/10_prepare_internal_official_eval.py"
            runtime_script.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(self.module.PipelineError, "runtime SHA drift"):
                self.module.validate_static_plan(
                    plan_path,
                    expected_plan_sha=_sha(plan_path),
                )

    def test_preflight_only_does_not_require_finished_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, training, _plan = _fixture(root, self.module)
            result = self.module.run_pipeline(
                plan_path=plan_path,
                expected_plan_sha=_sha(plan_path),
                training_validation=training,
                report=root / "report.json",
                failure_report=root / "failed.json",
                preflight_only=True,
            )
            self.assertEqual(result["status"], "preflight_pass")
            self.assertFalse((root / "report.json").exists())
            self.assertFalse((root / "failed.json").exists())

    def test_selection_command_binds_exact_checkpoint(self):
        digest = "a" * 64
        command = self.module.build_selection_command(
            ["python", "select.py", self.module.PLACEHOLDER],
            checkpoint_sha256=digest,
        )
        self.assertEqual(command, ["python", "select.py", digest])

    def test_selection_command_rejects_missing_placeholder(self):
        with self.assertRaisesRegex(self.module.PipelineError, "placeholder"):
            self.module.build_selection_command(
                ["python", "select.py"],
                checkpoint_sha256="a" * 64,
            )

    def test_static_preflight_rejects_plan_sha_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, _training, _plan = _fixture(root, self.module)
            with self.assertRaisesRegex(self.module.PipelineError, "plan SHA drift"):
                self.module.validate_static_plan(
                    plan_path,
                    expected_plan_sha="0" * 64,
                )

    def test_runs_all_three_stages_and_freezes_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, training, plan = _fixture(root, self.module)
            evaluation_root = Path(plan["fix_v3_evaluation_root"])
            selection_root = Path(plan["selection_output_root"])

            def fake_runner(command):
                script = Path(command[1]).name
                if script == "10_prepare_internal_official_eval.py":
                    evaluation_root.mkdir()
                    (evaluation_root / "prediction").mkdir()
                    (evaluation_root / "reference").mkdir()
                    _write(
                        evaluation_root / "preparation_summary.json",
                        json.dumps(
                            {
                                "status": "pass",
                                "checkpoint_sha256": _sha(Path(plan["checkpoint"])),
                                "prediction_count": 103,
                                "reference_count": 103,
                            }
                        ),
                    )
                elif script == "11_run_internal_official_eval.sh":
                    _write(evaluation_root / "EVALUATION_COMPLETE.ok", "status=pass\n")
                    _write(evaluation_root / "leaderboard_metrics.csv", "metrics\n")
                    _write(
                        evaluation_root / "panoptica_evaluation_summary.json",
                        "{}\n",
                    )
                elif script == "45_finalize_met_aug_fix_v3_selection.py":
                    selection_root.mkdir()
                    _write(
                        selection_root / "MODEL_SELECTION.json",
                        json.dumps(
                            {
                                "status": "pass",
                                "route_status": self.module.ROUTE_STATUS,
                                "selected_model": "E",
                                "selected_checkpoint_sha256": (
                                    self.module.EXPECTED_E_CHECKPOINT_SHA256
                                ),
                                "immutable_boundaries": {
                                    "official_179_started": False,
                                },
                            }
                        ),
                    )
                else:
                    raise AssertionError(f"unexpected command: {command}")
                return subprocess.CompletedProcess(command, 0, stdout=f"{script} pass\n")

            report = root / "PIPELINE.json"
            result = self.module.run_pipeline(
                plan_path=plan_path,
                expected_plan_sha=_sha(plan_path),
                training_validation=training,
                report=report,
                failure_report=root / "PIPELINE_FAILED.json",
                runner=fake_runner,
            )

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["selected_model"], "E")
            self.assertEqual(
                [stage["name"] for stage in result["stages"]],
                ["prepare", "official_evaluation", "paired_selection"],
            )
            self.assertTrue(report.is_file())
            self.assertEqual(report.stat().st_mode & 0o777, 0o444)
            self.assertEqual(selection_root.stat().st_mode & 0o777, 0o555)
            self.assertFalse((root / "PIPELINE_FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
