import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_s2_v3_teacher_eval.py"


def load_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"S2 V3 teacher implementation is missing: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("g2_s2_v3_teacher_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class G2S2V3TeacherEvalTest(unittest.TestCase):
    def test_install_model_uses_current_checkpoint_metadata(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint_final.pth"
            torch.save(
                {
                    "trainer_name": "nnUNetTrainerBraTS2026RC",
                    "current_epoch": 1000,
                    "init_args": {
                        "plans": {
                            "dataset_name": "Dataset263_BraTS2026_MET_RealOnly_Current",
                            "plans_name": "nnUNetPlans",
                            "configurations": {"3d_fullres": {}},
                        },
                        "dataset_json": {
                            "channel_names": {"0": "T1N", "1": "T1C", "2": "T2W", "3": "T2F"},
                            "labels": {"background": 0, "ET": 3},
                            "file_ending": ".nii.gz",
                        },
                        "configuration": "3d_fullres",
                    },
                },
                checkpoint,
            )

            summary = mod.install_s2_model(
                checkpoint_path=checkpoint,
                nnunet_results_root=root / "nnunet_results",
                expected_dataset_name="Dataset263_BraTS2026_MET_RealOnly_Current",
                expected_trainer="nnUNetTrainerBraTS2026RC",
                configuration="3d_fullres",
                overwrite=False,
            )

            model_root = Path(summary["model_root"])
            self.assertEqual(summary["current_epoch"], 1000)
            self.assertTrue((model_root / "plans.json").is_file())
            self.assertTrue((model_root / "dataset.json").is_file())
            self.assertEqual(
                (model_root / "fold_0" / "checkpoint_final.pth").resolve(),
                checkpoint.resolve(),
            )

    def test_prepare_uses_generated_t2w_and_real_protected_modalities(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "BraTS-MET-00001-000"
            nnunet_id = "BraTSMET_000001"
            source_dir = root / "source" / case_id
            paths = {}
            for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f"), start=1):
                path = source_dir / f"{case_id}-{modality}.nii.gz"
                save_nifti(path, np.full((8, 8, 8), index, dtype=np.float32))
                paths[modality] = path
            seg_path = source_dir / f"{case_id}-seg.nii.gz"
            seg = np.zeros((8, 8, 8), dtype=np.int16)
            seg[2:6, 2:6, 2:6] = 3
            save_nifti(seg_path, seg)
            generated_path = root / "generated" / f"{case_id}-t2w.nii.gz"
            save_nifti(generated_path, np.full((8, 8, 8), 99, dtype=np.float32))

            mapping = root / "mapping.csv"
            write_csv(
                mapping,
                [
                    {
                        "nnunet_case_id": nnunet_id,
                        "source_case_id": case_id,
                        "eligible_for_realonly": "True",
                        "t1n_source_path": root / "unavailable" / "t1n.nii.gz",
                        "t1c_source_path": root / "unavailable" / "t1c.nii.gz",
                        "t2w_source_path": root / "unavailable" / "t2w.nii.gz",
                        "t2f_source_path": root / "unavailable" / "t2f.nii.gz",
                        "seg_source_path": root / "unavailable" / "seg.nii.gz",
                    }
                ],
            )
            metrics = root / "metrics.csv"
            write_csv(metrics, [{"subject": case_id}])
            split = root / "split.json"
            split.write_text(
                json.dumps([{"train": [], "val": [nnunet_id], "test": []}]),
                encoding="utf-8",
            )

            summary = mod.prepare_teacher_input(
                project_root=root,
                real_root=root / "source",
                mapping_csv=mapping,
                split_json=split,
                stage5_metrics=metrics,
                synthetic_root=root / "generated",
                input_root=root / "teacher_input",
                reference_root=root / "reference",
                case_map_path=root / "case_map.tsv",
                expected_cases=1,
                materialize_mode="symlink",
                clean=False,
            )

            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["nifti_count"], 4)
            self.assertEqual(
                (root / "teacher_input" / f"{nnunet_id}_0002.nii.gz").resolve(),
                generated_path.resolve(),
            )
            self.assertEqual(
                (root / "teacher_input" / f"{nnunet_id}_0000.nii.gz").resolve(),
                paths["t1n"].resolve(),
            )
            self.assertEqual(
                (root / "reference" / f"{case_id}.nii.gz").resolve(),
                seg_path.resolve(),
            )

    def test_compare_reports_real_to_generated_teacher_degradation(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "BraTS-MET-00001-000"
            nnunet_id = "BraTSMET_000001"
            reference = np.zeros((12, 12, 12), dtype=np.int16)
            reference[2:8, 2:8, 2:8] = 3
            baseline = reference.copy()
            generated = reference.copy()
            generated[2:5, 2:8, 2:8] = 0
            save_nifti(root / "reference" / f"{case_id}.nii.gz", reference)
            save_nifti(root / "baseline" / f"{case_id}.nii.gz", baseline)
            save_nifti(root / "generated" / f"{nnunet_id}.nii.gz", generated)
            (root / "case_map.tsv").write_text(
                f"{nnunet_id}\t{case_id}\n", encoding="utf-8"
            )

            summary = mod.compare_teacher_predictions(
                baseline_prediction_root=root / "baseline",
                generated_prediction_root=root / "generated",
                reference_root=root / "reference",
                case_map_path=root / "case_map.tsv",
                output_root=root / "report",
                expected_cases=1,
                max_macro_dice_drop=0.02,
                max_region_dice_drop=0.03,
                max_missing_large_fraction=0.05,
                overwrite=False,
            )

            self.assertLess(summary["metrics"]["macro_region_dice_generated"], 1.0)
            self.assertLess(summary["metrics"]["macro_region_dice_delta"], -0.02)
            self.assertEqual(summary["teacher_technical_gate"], "fail")
            self.assertEqual(summary["stage6_gate"], "hold_for_review")


if __name__ == "__main__":
    unittest.main()
