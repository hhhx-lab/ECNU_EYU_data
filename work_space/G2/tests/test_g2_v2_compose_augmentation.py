import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_v2_compose_augmentation.py"
CODE_DIR = SCRIPT_PATH.parent


def load_module():
    spec = importlib.util.spec_from_file_location("g2_v2_compose_augmentation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_nifti(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, affine), str(path))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_audit_module():
    sys.path.insert(0, str(CODE_DIR))
    path = CODE_DIR / "g2_pretraining_audit.py"
    spec = importlib.util.spec_from_file_location("g2_pretraining_audit_v2_e2e", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class G2V2ComposeTest(unittest.TestCase):
    def test_composes_full_case_and_preserves_nonroi_and_geometry(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            generated_dir = root / "generated"
            output_dir = root / "composed"
            source_dir.mkdir()
            generated_dir.mkdir()
            case_id = "BraTS-MET-00001-000"
            shape = (12, 12, 12)
            affine = np.array(
                [[1.2, 0, 0, 10], [0, 1.1, 0, -4], [0, 0, 1.5, 7], [0, 0, 0, 1]],
                dtype=float,
            )
            grid = np.indices(shape).sum(axis=0).astype(np.float32)
            support = np.zeros(shape, dtype=bool)
            support[3:9, 3:9, 3:9] = True
            generated_paths = {}
            source_row = {
                "source_case_id": case_id,
                "split": "train",
                "allowed_as_v2_source": "True",
            }
            for index, modality in enumerate(mod.MODALITIES, start=1):
                source = grid + index * 10
                source_path = source_dir / f"{case_id}-{modality}.nii.gz"
                save_nifti(source_path, source, affine)
                source_row[f"{modality}_path"] = str(source_path)
                generated = np.zeros(shape, dtype=np.float32)
                generated[support] = np.linspace(-1, 1, support.sum(), dtype=np.float32)
                generated_path = generated_dir / f"{case_id}-{modality}.nii.gz"
                save_nifti(generated_path, generated, np.eye(4))
                generated_paths[modality] = generated_path
            seg = np.zeros(shape, dtype=np.int16)
            seg[4:8, 4:8, 4:8] = 3
            seg_path = source_dir / f"{case_id}-seg.nii.gz"
            save_nifti(seg_path, seg, affine)
            source_row["seg_path"] = str(seg_path)

            row = mod.compose_case(
                case_id,
                generated_paths,
                source_row,
                output_dir,
                blend_width=2.0,
                support_epsilon=1e-6,
                overwrite=False,
            )

            self.assertEqual(row["status"], "success")
            raw_id = row["synthetic_raw_id"]
            composed_path = output_dir / raw_id / f"{raw_id}-t1n.nii.gz"
            composed_image = nib.load(str(composed_path))
            composed = np.asanyarray(composed_image.dataobj)
            source = np.asanyarray(nib.load(str(source_row["t1n_path"])).dataobj)
            self.assertTrue(np.array_equal(composed[~support], source[~support]))
            self.assertFalse(np.array_equal(composed[support], source[support]))
            self.assertTrue(np.allclose(composed_image.affine, affine))
            self.assertTrue((output_dir / raw_id / f"{raw_id}-seg.nii.gz").is_file())
            self.assertTrue((output_dir / raw_id / f"{raw_id}-generation_support.nii.gz").is_file())

    def test_discovers_flat_outputs_by_case(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id in ("BraTS-MET-00001-000", "BraTS-MET-00002-000"):
                for modality in mod.MODALITIES:
                    (root / f"{case_id}-{modality}.nii.gz").touch()
            discovered = mod.discover_v2_outputs(root)
            self.assertEqual(set(discovered), {"BraTS-MET-00001-000", "BraTS-MET-00002-000"})
            self.assertTrue(all(set(files) == set(mod.MODALITIES) for files in discovered.values()))

    def test_v2_composer_to_g2_qc_end_to_end(self):
        composer = load_module()
        audit = load_audit_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "BraTS-MET-00001-000"
            nnunet_id = "BraTSMET_000001"
            source_dir = root / "source" / case_id
            raw_dir = root / "v2_raw"
            composed_dir = root / "v2_composed"
            results_root = root / "results"
            source_dir.mkdir(parents=True)
            raw_dir.mkdir()
            shape = (12, 12, 12)
            affine = np.diag([1.0, 1.0, 1.2, 1.0])
            grid = np.indices(shape).sum(axis=0).astype(np.float32) + 10
            support = np.zeros(shape, dtype=bool)
            support[3:9, 3:9, 3:9] = True
            source_row = {
                "source_case_id": case_id,
                "patient_group": "BraTS-MET-00001",
                "nnunet_case_id": nnunet_id,
                "split": "train",
                "t2w_status": "authentic",
                "allowed_as_v2_source": True,
                "label_source": "raw",
            }
            real_manifest_row = {"case_id": case_id, "final_qc_pass": True, "shape_seg": "12x12x12"}
            for index, modality in enumerate(composer.MODALITIES, start=1):
                source = grid + index * 5
                if modality == "t1c":
                    source = source.copy()
                    source[support] += 20
                source_path = source_dir / f"{case_id}-{modality}.nii.gz"
                save_nifti(source_path, source, affine)
                source_row[f"{modality}_path"] = str(source_path)
                real_manifest_row[f"{modality}_path"] = str(source_path)
                generated = np.zeros(shape, dtype=np.float32)
                generated[support] = np.linspace(-0.2, 0.2, support.sum(), dtype=np.float32)
                save_nifti(raw_dir / f"{case_id}-{modality}.nii.gz", generated, np.eye(4))
            seg = np.zeros(shape, dtype=np.int16)
            seg[4:8, 4:8, 4:8] = 3
            seg_path = source_dir / f"{case_id}-seg.nii.gz"
            save_nifti(seg_path, seg, affine)
            source_row["seg_path"] = str(seg_path)
            real_manifest_row["effective_seg_path"] = str(seg_path)
            real_manifest_row["raw_seg_path"] = str(seg_path)

            source_manifest = results_root / "manifests" / "g1_v2_source_manifest.csv"
            write_csv(source_manifest, [source_row])
            config = {
                "generation_run_id": "v2_e2e",
                "generator_name": "g1_diffusion_v2",
                "seed": 42,
                "source_csv": str(source_manifest),
                "diffusion_checkpoint_dir": "/checkpoints/v2_e2e",
                "sampling_method": "edm_heun",
                "sampling_steps": 18,
                "eta": 0.0,
                "crop_size": 64,
            }
            (raw_dir / "generation_config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_PATH),
                    "--v2-output-root", str(raw_dir),
                    "--source-manifest", str(source_manifest),
                    "--output-run-root", str(composed_dir),
                ],
            ):
                self.assertEqual(composer.main(), 0)

            write_csv(results_root / "manifests" / "real_train_manifest.csv", [real_manifest_row])
            write_csv(
                results_root / "manifests" / "nnunet_case_mapping_master.csv",
                [{"source_case_id": case_id, "nnunet_case_id": nnunet_id}],
            )
            (results_root / "splits").mkdir()
            (results_root / "splits" / "splits_master_train_val_test.json").write_text(
                json.dumps([{"train": [nnunet_id], "val": [], "test": []}]), encoding="utf-8"
            )
            (results_root / "qc").mkdir()
            write_csv(
                results_root / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv",
                [{"case_id": "BraTS-MET-99999-000"}],
            )
            raw_id = f"{case_id}_v2aug_label_0"
            write_csv(
                composed_dir / "g2_approval_manifest.csv",
                [{
                    "synthetic_raw_id": raw_id,
                    "approved_for_training": True,
                    "approved_for_evaluation": False,
                    "reviewer": "unit-test",
                    "reason": "e2e fixture",
                }],
            )
            dirs = audit.ensure_dirs(results_root)
            audit.ingest_synthetic_run(
                composed_dir,
                results_root,
                argparse.Namespace(synthetic_run_id="", generation_mode="full_generation"),
                dirs,
            )
            qc_path = results_root / "qc" / "qc_metrics_v2_e2e.csv"
            with qc_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["qc_status"], "accepted_for_training")
            self.assertTrue(row["synthetic_final_id"].startswith("SYN-MET-"))
            self.assertNotEqual(row["synthetic_final_id"], case_id)
            self.assertGreaterEqual(
                float(row["roi_boundary_p95_jump"]),
                float(row["roi_boundary_gradient_jump"]),
            )
            with (results_root / "qc" / "diffusion_quality_metrics_v2_e2e.csv").open(newline="") as handle:
                diffusion_row = next(csv.DictReader(handle))
            self.assertIn("label_source_seg_dice", diffusion_row)
            self.assertNotIn("label_source_synth_roi_ssim", diffusion_row)

    def test_output_root_requires_explicit_full_clean(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "composed"
            output.mkdir()
            (output / "stale.txt").write_text("stale")
            with self.assertRaises(FileExistsError):
                mod.prepare_output_root(output, overwrite=False)
            mod.prepare_output_root(output, overwrite=True)
            self.assertEqual(list(output.iterdir()), [])

    def test_zero_seed_and_eta_are_valid_metadata(self):
        mod = load_module()
        mod.validate_generation_config({
            "generation_run_id": "run-zero",
            "generator_name": "g1_diffusion_v2",
            "seed": 0,
            "sampling_method": "ddim",
            "sampling_steps": 50,
            "eta": 0.0,
            "crop_size": 64,
            "diffusion_checkpoint_dir": "/checkpoints/run-zero",
        })


if __name__ == "__main__":
    unittest.main()
