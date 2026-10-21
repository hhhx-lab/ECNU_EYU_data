from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "45_finalize_met_aug_fix_v3_selection.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("met_aug_fix_v3_selection", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test target: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    subject: str,
    *,
    value: float,
    false_positives: float = 0.0,
    empty_et_dsc: bool = False,
) -> dict[str, str]:
    row = {"subject_id": subject}
    for region in ("et", "rc", "tc", "wt"):
        for metric in (
            "lesionwise_dsc_mean",
            "lesionwise_nsd_mean",
            "all_instance_f1",
            "small_instance_f1",
            "large_instance_f1",
        ):
            row[f"{metric}_{region}"] = str(value)
        row[f"all_instance_tp_{region}"] = "1"
        row[f"all_instance_fp_{region}"] = str(false_positives)
        row[f"all_instance_fn_{region}"] = "0"
    if empty_et_dsc:
        row["lesionwise_dsc_mean_et"] = ""
    return row


def _official_row(
    subject: str,
    *,
    value: float,
    false_positives: float = 0.0,
) -> dict[str, str]:
    row = _row(
        subject,
        value=value,
        false_positives=false_positives,
    )
    for region in ("et", "rc", "tc", "wt"):
        row.update(
            {
                f"large_instance_tp_{region}": "1",
                f"large_instance_fp_{region}": str(false_positives),
                f"large_instance_fn_{region}": "0",
                f"lesionwise_dsc_std_{region}": "0",
                f"lesionwise_hd95_mean_{region}": "1",
                f"lesionwise_hd95_std_{region}": "0",
                f"lesionwise_nsd_std_{region}": "0",
                f"small_instance_tp_{region}": "0",
                f"small_instance_fn_{region}": "0",
                f"small_instance_fp_{region}": "0",
            }
        )
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evaluation_fixture(
    root: Path,
    *,
    value: float,
    reference_suffix: bytes = b"",
) -> tuple[Path, str]:
    root.mkdir()
    prediction = root / "prediction"
    reference = root / "reference"
    prediction.mkdir()
    reference.mkdir()
    checkpoint = root / "checkpoint_final.pth"
    checkpoint.write_bytes(f"checkpoint:{root.name}".encode("ascii"))
    checkpoint_sha = _sha256(checkpoint)
    names = ["case-a.nii.gz", "case-b.nii.gz"]
    rows = []
    for name in names:
        (prediction / name).write_bytes(f"prediction:{root.name}:{name}".encode("ascii"))
        (reference / name).write_bytes(b"shared-reference:" + name.encode("ascii") + reference_suffix)
        rows.append(_official_row(name, value=value))
    fieldnames = list(rows[0])
    csv_path = root / "leaderboard_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        for summary_name in ("mean", "std", "median"):
            summary = dict(rows[0])
            summary["subject_id"] = summary_name
            writer.writerow(summary)
    summary_path = root / "panoptica_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "missings": [],
                "metrics": [{"subject_name": name} for name in names],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "preparation_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "case_count": 2,
                "prediction_count": 2,
                "reference_count": 2,
                "mapping_count": 2,
                "materialization_mode": "hardlink",
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_config": "mets",
                "vol_threshold": 27,
                "overlap_threshold": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "evaluation_contract.txt").write_text(
        "config=mets\nvol_threshold=27\noverlap_threshold=0.2\nexpected_count=2\n",
        encoding="utf-8",
    )
    (root / "evaluation_environment.txt").write_text(
        "Python 3.10.14\nBraTS-evaluation=0.0.8\npanoptica=2.1.0\n",
        encoding="utf-8",
    )
    (root / "brats_evaluate.log").write_text("evaluation complete\n", encoding="utf-8")
    (root / "brats_parse_metrics.log").write_text("parse complete\n", encoding="utf-8")
    (root / "nnunet_to_source_id.tsv").write_text(
        "nnunet_id\tsource_case_id\n"
        "nnunet-a\tcase-a\n"
        "nnunet-b\tcase-b\n",
        encoding="utf-8",
    )
    (root / "EVALUATION_COMPLETE.ok").write_text(
        "status=pass\n"
        f"{_sha256(summary_path)}  {summary_path}\n"
        f"{_sha256(csv_path)}  {csv_path}\n",
        encoding="utf-8",
    )
    return root, checkpoint_sha


class FixV3SelectionPresenceTests(unittest.TestCase):
    def test_selection_script_exists(self):
        self.assertTrue(SCRIPT.is_file(), f"missing Fix-v3 selection script: {SCRIPT}")


@unittest.skipUnless(SCRIPT.is_file(), "Fix-v3 selection script is not implemented yet")
class FixV3SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script()

    def test_selects_fix_v3_only_for_nonregressing_robust_improvement(self):
        e_rows = [_row(f"case-{index}", value=0.5) for index in range(8)]
        fix_rows = [_row(f"case-{index}", value=0.6) for index in range(8)]

        analysis = self.module.paired_analysis(
            e_rows,
            fix_rows,
            seed=20260729,
            resamples=500,
        )
        selected, policy = self.module.choose_model(analysis)

        self.assertEqual(selected, "Fix_v3")
        self.assertTrue(policy["candidate_pass"])
        self.assertEqual(policy["observed_regressions"], [])
        self.assertEqual(policy["observed_false_positive_increases"], [])
        self.assertTrue(policy["observed_robust_improvements"])

    def test_falls_back_to_e_for_any_core_metric_regression(self):
        e_rows = [_row(f"case-{index}", value=0.5) for index in range(8)]
        fix_rows = [_row(f"case-{index}", value=0.6) for index in range(8)]
        for row in fix_rows:
            row["lesionwise_nsd_mean_tc"] = "0.49"

        analysis = self.module.paired_analysis(e_rows, fix_rows, seed=7, resamples=200)
        selected, policy = self.module.choose_model(analysis)

        self.assertEqual(selected, "E")
        self.assertFalse(policy["candidate_pass"])
        self.assertIn(
            {"region": "TC", "metric": "lesionwise_nsd_mean", "mean_delta": -0.01},
            policy["observed_regressions"],
        )

    def test_falls_back_to_e_when_false_positives_increase(self):
        e_rows = [_row(f"case-{index}", value=0.5) for index in range(8)]
        fix_rows = [
            _row(f"case-{index}", value=0.6, false_positives=1.0)
            for index in range(8)
        ]

        analysis = self.module.paired_analysis(e_rows, fix_rows, seed=11, resamples=200)
        selected, policy = self.module.choose_model(analysis)

        self.assertEqual(selected, "E")
        self.assertFalse(policy["candidate_pass"])
        self.assertEqual(
            policy["observed_false_positive_increases"],
            [
                {"region": "ET", "fp_delta": 8.0},
                {"region": "TC", "fp_delta": 8.0},
                {"region": "WT", "fp_delta": 8.0},
            ],
        )

    def test_pairs_only_cases_defined_for_both_models(self):
        e_rows = [
            _row("case-a", value=0.5),
            _row("case-b", value=0.5, empty_et_dsc=True),
        ]
        fix_rows = [
            _row("case-a", value=0.6),
            _row("case-b", value=0.6),
        ]

        analysis = self.module.paired_analysis(e_rows, fix_rows, seed=13, resamples=200)
        metric = analysis["regions"]["ET"]["paired_metrics"]["lesionwise_dsc_mean"]

        self.assertEqual(metric["common_defined_cases"], 1)
        self.assertEqual(metric["e_defined_cases"], 1)
        self.assertEqual(metric["fix_v3_defined_cases"], 2)
        self.assertEqual(metric["fix_v3_only_subjects"], ["case-b"])
        self.assertAlmostEqual(metric["mean_delta_fix_v3_minus_e"], 0.1)

    def test_rejects_different_subject_sets(self):
        with self.assertRaisesRegex(self.module.SelectionError, "subject sets differ"):
            self.module.paired_analysis(
                [_row("case-a", value=0.5)],
                [_row("case-b", value=0.6)],
                seed=17,
                resamples=100,
            )

    def test_finalizes_audited_experimental_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e_root, e_sha = _write_evaluation_fixture(root / "E", value=0.5)
            fix_root, fix_sha = _write_evaluation_fixture(root / "Fix_v3", value=0.6)
            output = root / "selection"

            result = self.module.finalize_selection(
                e_root,
                fix_root,
                output,
                expected_e_checkpoint_sha=e_sha,
                expected_fix_v3_checkpoint_sha=fix_sha,
                expected_count=2,
                bootstrap_seed=20260729,
                bootstrap_resamples=200,
            )

            self.assertEqual(result["selected_model"], "Fix_v3")
            decision = json.loads((output / "MODEL_SELECTION.json").read_text())
            self.assertEqual(decision["route_status"], "experimental_unvalidated")
            self.assertEqual(decision["selected_model"], "Fix_v3")
            self.assertFalse(decision["immutable_boundaries"]["official_179_started"])
            self.assertEqual(
                decision["inference_contract"],
                "segmentation_checkpoint_only_no_met_aug_g1_g2_or_donor",
            )
            self.assertTrue((output / "MODEL_SELECTION_COMPLETE.ok").is_file())

    def test_rejects_reference_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e_root, e_sha = _write_evaluation_fixture(root / "E", value=0.5)
            fix_root, fix_sha = _write_evaluation_fixture(
                root / "Fix_v3",
                value=0.6,
                reference_suffix=b"drift",
            )

            with self.assertRaisesRegex(self.module.SelectionError, "reference content differs"):
                self.module.finalize_selection(
                    e_root,
                    fix_root,
                    root / "selection",
                    expected_e_checkpoint_sha=e_sha,
                    expected_fix_v3_checkpoint_sha=fix_sha,
                    expected_count=2,
                    bootstrap_seed=20260729,
                    bootstrap_resamples=100,
                )


if __name__ == "__main__":
    unittest.main()
