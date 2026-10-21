from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runtime_module(name: str):
    path = ROOT / "runtime" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, case_rows: list[dict[str, str]]) -> None:
    fields = ["subject_id", "lesionwise_dsc_mean_et", "lesionwise_hd95_mean_et"]
    summaries = [
        {"subject_id": "mean", "lesionwise_dsc_mean_et": "0.6", "lesionwise_hd95_mean_et": "21"},
        {"subject_id": "std", "lesionwise_dsc_mean_et": "0.1", "lesionwise_hd95_mean_et": "8"},
        {"subject_id": "median", "lesionwise_dsc_mean_et": "0.6", "lesionwise_hd95_mean_et": "20"},
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*case_rows, *summaries])


class StaticRuntimeTests(unittest.TestCase):
    def test_r2_stage_paths_job_names_and_runtime_route_are_consistent(self) -> None:
        advancer = load_runtime_module("advance_pipeline")
        expected_jobs = {
            "val27_r4_synthesis": "s2r4v27s2",
            "val27_four_models": "s2r4v27m2",
            "fixed103_real_vs_synthetic": "s2r4103p2",
            "test26_locked_endpoint": "s2r4t26e2",
        }
        self.assertEqual(
            {stage["key"]: stage["job_name"] for stage in advancer.STAGES},
            expected_jobs,
        )
        for stage in advancer.STAGES:
            script = (ROOT / stage["slurm"]).read_text(encoding="utf-8")
            self.assertIn("s2_g1r4_endpoint_validation_20260824_r2", script)
            self.assertNotIn("s2_g1r4_endpoint_validation_20260824_r1", script)
            self.assertIn(f"#SBATCH --job-name={stage['job_name']}", script)
        for script_name in ("01_val27_synthesize_r4.slurm", "04_test26_locked_endpoint.slurm"):
            script = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")
            self.assertIn('G1_RUNTIME="${ROOT}/runtime/g1_r4_frozen"', script)
            self.assertIn("${G1_RUNTIME}", script)
        self.assertIn(
            "ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R2.json",
            (ROOT / "runtime/advance_pipeline.py").read_text(encoding="utf-8"),
        )

    def test_exclusive_json_writes_are_valid_and_newline_terminated(self) -> None:
        builder = load_runtime_module("build_static_bundle")
        advancer = load_runtime_module("advance_pipeline")
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            for index, writer in enumerate((builder.write_json_exclusive, advancer.write_exclusive)):
                target = temp_root / f"audit_{index}.json"
                writer(target, {"status": "pass", "operator_approved": False})
                raw = target.read_bytes()
                self.assertTrue(raw.endswith(b"\n"))
                self.assertFalse(raw.endswith(b"\\n"))
                self.assertEqual(json.loads(raw), {"operator_approved": False, "status": "pass"})
                with self.assertRaises(FileExistsError):
                    writer(target, {"status": "overwritten"})

    def test_missing_t2w_manifest_row_never_exposes_source_t2w(self) -> None:
        builder = load_runtime_module("build_static_bundle")
        row = builder.cohort_row(
            0,
            {
                "source_case_id": "BraTS-Test-00001-000",
                "nnunet_case_id": "BraTS2026_0001",
                "t2w_status": "fake_or_broken",
                "completion_required": "true",
            },
            source_split="test",
            t2w_role="r4_ensemble_synthesized",
            synthesized_root=Path("/frozen/synthesized"),
        )
        self.assertFalse(row["source_t2w_allowed"])
        self.assertEqual(row["t2w_source_path"], "")
        self.assertEqual(row["synthesized_t2w_path"], "/frozen/synthesized/BraTS-Test-00001-000-t2w.nii.gz")

    def test_paired_comparison_excludes_nonfinite_pairs_and_tie_p_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            method_a = temp_root / "a.csv"
            method_b = temp_root / "b.csv"
            write_metrics(
                method_a,
                [
                    {"subject_id": "case1", "lesionwise_dsc_mean_et": "0.5", "lesionwise_hd95_mean_et": "10"},
                    {"subject_id": "case2", "lesionwise_dsc_mean_et": "0.6", "lesionwise_hd95_mean_et": "nan"},
                    {"subject_id": "case3", "lesionwise_dsc_mean_et": "0.7", "lesionwise_hd95_mean_et": "30"},
                ],
            )
            write_metrics(
                method_b,
                [
                    {"subject_id": "case1", "lesionwise_dsc_mean_et": "0.5", "lesionwise_hd95_mean_et": "12"},
                    {"subject_id": "case2", "lesionwise_dsc_mean_et": "0.6", "lesionwise_hd95_mean_et": "20"},
                    {"subject_id": "case3", "lesionwise_dsc_mean_et": "0.7", "lesionwise_hd95_mean_et": "inf"},
                ],
            )
            output = temp_root / "comparison"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "runtime/compare_paired_met.py"),
                    "--input",
                    f"A={method_a}",
                    "--input",
                    f"B={method_b}",
                    "--output-root",
                    str(output),
                    "--expected-count",
                    "3",
                    "--bootstrap",
                    "20000",
                    "--seed",
                    "20260824",
                    "--scope",
                    "unit_test",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            raw = (output / "PAIRED_COMPARISON.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            payload = json.loads(raw)
            metrics = {item["metric"]: item for item in payload["pairs"][0]["metrics"]}
            tie_metric = metrics["lesionwise_dsc_mean_et"]
            self.assertEqual(tie_metric["wins_a"], 0)
            self.assertEqual(tie_metric["ties"], 3)
            self.assertEqual(tie_metric["wins_b"], 0)
            self.assertEqual(tie_metric["paired_two_sided_bootstrap_p"], 1.0)
            sparse_metric = metrics["lesionwise_hd95_mean_et"]
            self.assertEqual(sparse_metric["paired_complete_case_count"], 1)
            self.assertEqual(sparse_metric["excluded_nonfinite_pair_count"], 2)
            self.assertEqual(sparse_metric["status"], "unavailable_insufficient_complete_pairs")


if __name__ == "__main__":
    unittest.main()
