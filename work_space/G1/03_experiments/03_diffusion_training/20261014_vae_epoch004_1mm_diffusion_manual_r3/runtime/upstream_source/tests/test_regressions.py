"""Synthetic, isolated regressions. Run: python -m unittest discover -s tests -v."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preprocess_dataset as prep
from src.preprocess.nnunet_like import physical_grid, resample_physical, preprocess_case
from src.train import csv_creator, tumour_main_diffusion as trainer
from src.infer import evaluate_generation as evaluation
from src.utils.checkpoints import unwrap_model, clean_state_dict
from src.utils.splits import patient_group, select_split, validate_group_splits
from src.utils.volume_validation import validate_case


def make_case(root, name="BraTS-MET-00001-000", spacing=(1., 1., 1.)):
    folder = Path(root) / name
    folder.mkdir(parents=True)
    shape = (24, 28, 32)
    ramp = (1 + np.indices(shape).sum(axis=0)).astype(np.float32)
    seg = np.zeros(shape, np.float32)
    seg[3:9, 11:17, 21:27] = 4  # off-centre RC: all 216 voxels must survive cropping
    affine = np.diag([*spacing, 1.])
    paths = []
    for mod in prep.MODALITIES + ["seg"]:
        path = folder / f"{name}-{mod}.nii.gz"
        nib.save(nib.Nifti1Image(seg if mod == "seg" else ramp, affine), path)
        paths.append(path)
    return folder, paths


def csv_args(**kwargs):
    values = dict(seg_ending="seg.nii.gz", t1c_ending="t1c.nii.gz", t1n_ending="t1n.nii.gz",
                  t2w_ending="t2w.nii.gz", t2f_ending="t2f.nii.gz", val_patients="",
                  val_list="", t2w_invalid_list="", crop_size=16, merge_dist=0)
    values.update(kwargs)
    return SimpleNamespace(**values)


class DataRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tumour-regression-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_reuse_validates_outputs_and_provenance(self):
        case, paths = make_case(self.root / "raw")
        job = (case.name, str(case), str(self.root / "out"), (1., 1., 1.), False, (0, 1, 2, 3, 4), .1)
        self.assertEqual(prep.process(job)["status"], "ok")
        self.assertEqual(prep.process(job)["status"], "skipped (verified)")
        output = self.root / "out" / case.name / paths[0].name
        output.write_bytes(b"truncated output")
        self.assertIn("error:", prep.process(job)["status"])
        fixed_job = (*job[:4], True, *job[5:])
        self.assertEqual(prep.process(fixed_job)["status"], "ok")
        image = nib.load(paths[0])
        nib.save(nib.Nifti1Image(image.get_fdata() + 1, image.affine), paths[0])
        self.assertIn("error:", prep.process(job)["status"])

    def test_preprocess_cli_fails_for_rejected_missing_and_empty(self):
        raw = self.root / "raw"
        case, paths = make_case(raw)
        paths[0].unlink()
        for case_list in (None, "BraTS-MET-99999-000", ""):
            command = [sys.executable, str(ROOT / "scripts/preprocess_dataset.py"),
                       "--raw_dirs", str(raw), "--out_dir", str(self.root / "out"), "--num_workers", "1"]
            if case_list is not None:
                listing = self.root / "cases.txt"
                listing.write_text(case_list, encoding="utf-8")
                command += ["--case_list", str(listing)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        # All cases rejected, rather than just missing a file.
        nib.save(nib.Nifti1Image(np.full((24, 28, 32), np.nan), np.eye(4)), paths[0])
        result = subprocess.run(command[:-2], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_volumes_rejected(self):
        case, paths = make_case(self.root)
        original = nib.load(paths[-1]).get_fdata()
        for invalid in (np.nan, np.inf, 2.25, 6):
            label = original.copy()
            label[0, 0, 0] = invalid
            nib.save(nib.Nifti1Image(label, np.eye(4)), paths[-1])
            with self.assertRaises(ValueError):
                validate_case(paths)
        nib.save(nib.Nifti1Image(original, np.eye(4)), paths[-1])
        moved = np.eye(4)
        moved[0, 3] = 1
        nib.save(nib.Nifti1Image(original, moved), paths[0])
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            validate_case(paths)

    def test_csv_failure_preserves_previous_file(self):
        case, paths = make_case(self.root / "raw")
        paths[0].unlink()
        destination = self.root / "cases.csv"
        destination.write_text("previous verified data", encoding="utf-8")
        with self.assertRaises(ValueError):
            csv_creator.create_csv(csv_args(), "BRATS_2024", str(destination), str(case.parent))
        self.assertEqual(destination.read_text(), "previous verified data")

    def test_csv_groups_visits_and_actual_tile_counts(self):
        raw = self.root / "raw"
        make_case(raw, "BraTS-MET-00001-000")
        make_case(raw, "BraTS-MET-00001-001")
        case, paths = make_case(raw, "BraTS-MET-00002-000")
        seg = np.zeros((24, 28, 32), np.uint8)
        seg[1:23, 2:26, 2:30] = 3
        nib.save(nib.Nifti1Image(seg, np.eye(4)), paths[-1])
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(val_patients="00001-000"), "BRATS_2024", str(destination), str(raw))
        df = pd.read_csv(destination)
        self.assertEqual(set(df[df.patient_id.str.startswith("00001")]["split"]), {"val"})
        train = select_split(df, "train")
        self.assertGreater(len(train), 1)
        self.assertEqual(set(train.patient_n_crops), {len(train)})
        covered = np.zeros_like(seg, bool)
        for _, row in train.iterrows():
            sl = tuple(slice(int(row[f"{a}_extreme_min"]), int(row[f"{a}_extreme_max"])) for a in "xyz")
            covered[sl] = True
        self.assertTrue(covered[seg > 0].all())

    def test_real_csv_loader_preserves_rc_and_small_batch(self):
        from src.utils.data_utils import get_loader
        case, paths = make_case(self.root / "raw")
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(), "BRATS_2024", str(destination), str(case.parent))
        args = SimpleNamespace(csv_path=str(destination), num_workers=0, modality="t1c",
                               crop_size=16, dataset="BRATS_2024", in_channels=5,
                               normalization="nnunet", noise_type="gaussian_tumour",
                               batch_size=4, cache_rate=1., loader_workers=0)
        loader = get_loader(args)
        self.assertEqual(len(loader), 1)  # no drop_last starvation
        for _ in range(24):
            batch = next(iter(loader))
            self.assertEqual(int(batch["label_crop_pad"][:, 3].sum()), 216)
            self.assertTrue(torch.isfinite(batch["scan_t1c_crop_pad"]).all())

    def test_evaluation_cli_self_test_and_missing_scan(self):
        case, paths = make_case(self.root / "raw")
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(val_patients="00001-000"), "BRATS_2024", str(destination), str(case.parent))
        output = self.root / "eval"
        command = [sys.executable, str(ROOT / "src/infer/evaluate_generation.py"),
                   "--self_test", "--csv_path", str(destination), "--dataset", "BRATS_2024",
                   "--modality", "t1c", "--crop_size", "16", "--normalization", "nnunet",
                   "--output_dir", str(output), "--device", "cpu"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["summary"]["t1c"]["mse"], 0)
        paths[0].unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_smoke_refuses_existing_work_directory(self):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        sentinel = self.root / "keep.txt"
        sentinel.write_text("must survive", encoding="utf-8")
        env = {**os.environ, "RAW_DIRS": str(self.root / "absent"), "WORK": str(self.root)}
        result = subprocess.run([bash, str(ROOT / "scripts/smoke_test.sh")],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing existing WORK path", result.stderr)
        self.assertEqual(sentinel.read_text(), "must survive")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_training_cli_two_steps_then_resume_one(self):
        case, _ = make_case(self.root / "raw")
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(), "BRATS_2024", str(destination), str(case.parent))
        command = [sys.executable, str(ROOT / "src/train/tumour_main_diffusion.py"),
                   "--csv_path", str(destination), "--dataset", "BRATS_2024", "--logdir", "synthetic",
                   "--modality", "t1c", "--crop_size", "16", "--normalization", "nnunet",
                   "--batch_size", "4", "--num_workers", "0", "--loader_workers", "0",
                   "--planner_gpu_gb", "8", "--small_lesion_weight", "0", "--sigma_data", "1"]
        for limit, extra in ((2, []), (3, ["--resume_iter", "2"])):
            result = subprocess.run(command + ["--num_steps", str(limit)] + extra,
                                    cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            path = self.root / f"Checkpoint/synthetic/t1c/weights/diffusion_{limit}.pt"
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["global_step"], limit)
            self.assertEqual(checkpoint["step_semantics"], "completed_optimizer_updates_v2")
            self.assertTrue(all(int(item["step"]) == limit for item in checkpoint["optimizer"]["state"].values()))

    def training_command(self, destination, logdir, steps, *extra):
        return [sys.executable, str(ROOT / "src/train/tumour_main_diffusion.py"),
                "--csv_path", str(destination), "--dataset", "BRATS_2024", "--logdir", logdir,
                "--modality", "t1c", "--crop_size", "16", "--normalization", "nnunet",
                "--batch_size", "4", "--num_workers", "0", "--loader_workers", "0",
                "--planner_gpu_gb", "8", "--small_lesion_weight", "0", "--sigma_data", "1",
                "--num_steps", str(steps), *extra]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_adam_beta1_default_and_resume_enforcement(self):
        case, _ = make_case(self.root / "raw")
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(), "BRATS_2024", str(destination), str(case.parent))

        def train(logdir, steps, *extra):
            return subprocess.run(self.training_command(destination, logdir, steps, *extra),
                                  cwd=self.root, capture_output=True, text=True)

        def weights(logdir, step):
            return self.root / f"Checkpoint/{logdir}/t1c/weights/diffusion_{step}.pt"

        result = train("new", 2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checkpoint = torch.load(weights("new", 2), map_location="cpu")
        self.assertEqual(checkpoint["adam_beta1"], 0.9)
        self.assertEqual(tuple(checkpoint["optimizer"]["param_groups"][0]["betas"]), (0.9, 0.999))
        # A checkpoint from before the change: beta1 0.5 in the optimizer state, no adam_beta1 field.
        result = train("legacy", 2, "--adam_beta1", "0.5")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        legacy = torch.load(weights("legacy", 2), map_location="cpu")
        del legacy["adam_beta1"]
        torch.save(legacy, weights("legacy", 2))
        result = train("legacy", 3, "--resume_iter", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pass --adam_beta1 0.5", result.stdout + result.stderr)
        self.assertFalse(weights("legacy", 3).exists())
        result = train("legacy", 3, "--resume_iter", "2", "--adam_beta1", "0.5")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        resumed = torch.load(weights("legacy", 3), map_location="cpu")
        self.assertEqual(resumed["optimizer"]["param_groups"][0]["betas"][0], 0.5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_evaluation_samples_with_the_checkpoint_schedule(self):
        case, _ = make_case(self.root / "raw")
        destination = self.root / "cases.csv"
        csv_creator.create_csv(csv_args(), "BRATS_2024", str(destination), str(case.parent))
        result = subprocess.run(self.training_command(destination, "sched", 1),
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "eval"
        # No --sigma_data: the command-line default (0.5) differs from the trained value (1.0).
        command = [sys.executable, str(ROOT / "src/infer/evaluate_generation.py"),
                   "--diffusion_ckpt_dir", str(self.root / "Checkpoint/sched"), "--csv_path", str(destination),
                   "--dataset", "BRATS_2024", "--modality", "t1c", "--crop_size", "16",
                   "--normalization", "nnunet", "--eval_split", "all", "--sampling_steps", "2",
                   "--output_dir", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--sigma_data 0.5 -> 1.0", result.stdout)
        sampling = json.loads((output / "metrics.json").read_text())["sampling"]
        self.assertEqual((sampling["noise_schedule"], sampling["sigma_data"]), ("edm", 1.0))
        result = subprocess.run(command + ["--sampling_method", "ddim"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("use edm_heun", result.stdout + result.stderr)


class SplitRegressionTests(unittest.TestCase):
    def test_missing_split_fails_closed(self):
        df = pd.DataFrame({"patient_id": ["00001-000"]})
        for split in ("train", "val"):
            with self.assertRaises(ValueError):
                select_split(df, split)
        self.assertEqual(len(select_split(df, "all")), 1)

    def test_visits_cannot_cross_splits(self):
        self.assertEqual(patient_group("BraTS-MET-00001-000"), patient_group("00001-001"))
        df = pd.DataFrame({"patient_id": ["00001-000", "00001-001"], "split": ["train", "val"]})
        for split in ("train", "val", "all"):
            with self.assertRaisesRegex(ValueError, "both"):
                select_split(df, split)

    def test_list_export_does_not_partially_write_on_leakage(self):
        from scripts import make_case_lists
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "split.json"
            source.write_text(json.dumps({"train": ["BraTS-MET-00001-000"], "val": ["BraTS-MET-00001-001"]}))
            out = root / "lists"
            with patch.object(sys, "argv", ["make_case_lists", "--split_json", str(source), "--out_dir", str(out)]):
                with self.assertRaises(ValueError):
                    make_case_lists.main()
            self.assertFalse(out.exists())


class CacheRegressionTests(unittest.TestCase):
    def test_eval_cache_requires_matching_inputs_and_intact_metrics(self):
        from scripts.eval_checkpoints import cached_evaluation_matches
        from src.utils.volume_validation import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({"summary": {"t1c": {"mse": 1.}}}))
            request = {"csv_sha256": "abc", "checkpoint_sha256": "def", "command": ["evaluate"]}
            self.assertFalse(cached_evaluation_matches(directory, request, "t1c"))
            (root / "evaluation_request.json").write_text(json.dumps({"request": request,
                                                                      "metrics_sha256": sha256_file(metrics)}))
            self.assertTrue(cached_evaluation_matches(directory, request, "t1c"))
            self.assertFalse(cached_evaluation_matches(directory, {**request, "csv_sha256": "changed"}, "t1c"))
            metrics.write_text("broken")
            self.assertFalse(cached_evaluation_matches(directory, request, "t1c"))

    def test_identity_changes_when_preprocessed_data_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, paths = make_case(root / "data")
            csv_path = root / "cases.csv"
            pd.DataFrame({"label": [str(paths[-1])]}).to_csv(csv_path, index=False)
            marker = case / "preprocess_complete.json"
            marker.write_text('{"version": 1}')
            original = trainer.csv_identity(csv_path)
            marker.write_text('{"version": 2}')
            updated = trainer.csv_identity(csv_path)
            self.assertEqual(original["csv_sha256"], updated["csv_sha256"])
            self.assertNotEqual(original["preprocessing_sha256"], updated["preprocessing_sha256"])


class GeometryRegressionTests(unittest.TestCase):
    def test_sampling_coordinates_match_affine_not_rounded_shape(self):
        shape, spacing, target = (31, 37, 43), (1.17, .83, 1.31), (1., 1., 1.)
        new_shape, factor, offset = physical_grid(shape, spacing, target)
        # Three coordinate-valued images expose shifts invisible to shape/header tests.
        data = np.indices(shape).astype(np.float32)
        sampled = resample_physical(data, new_shape, factor, offset, spacing, target)
        affine = np.diag([*spacing, 1.])
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, -1]])
        affine[:3, :3] = rotation @ affine[:3, :3]
        bbox = [[3, 34], [5, 42], [7, 50]]
        affine[:3, 3] = [-20, 7, 11]
        output_affine, actual_spacing = prep.new_affine(affine, bbox, shape, new_shape, spacing, target)
        np.testing.assert_allclose(np.linalg.norm(output_affine[:3, :3], axis=0), target)
        indices = np.indices(new_shape).astype(float)
        input_coords = sampled + np.array([3, 5, 7])[:, None, None, None]
        actual_world = np.einsum("ij,jxyz->ixyz", affine[:3, :3], input_coords) + affine[:3, 3, None, None, None]
        expected_world = np.einsum("ij,jxyz->ixyz", output_affine[:3, :3], indices) + output_affine[:3, 3, None, None, None]
        error = np.linalg.norm(actual_world - expected_world, axis=0)[5:-5, 5:-5, 5:-5].max()
        print(f"physical coordinate ramp: max interior error {error:.6f} mm")
        self.assertLess(error, .002)

    def test_anisotropic_grid_keeps_legal_labels(self):
        data = np.indices((15, 17, 7)).astype(np.float32) + 1
        label = np.zeros((1, 15, 17, 7), np.int16)
        label[:, 4:9, 5:10, 2:5] = 4
        result, seg, info = preprocess_case(data, label, (.8, .9, 4.2))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(set(np.unique(seg)).issubset({-1, 0, 4}))
        self.assertEqual(info["grid_version"], "physical_spacing_v2")


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 1)

    def forward(self, x, t, cond=None):
        return self.conv(x)


class TrainingRegressionTests(unittest.TestCase):
    def test_amp_skipped_attempt_does_not_count_as_update(self):
        class FirstAttemptSkipped:
            def __init__(self):
                self.attempt = 0
                self.value = 8
            def get_scale(self):
                return self.value
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                self.attempt += 1
                if self.attempt > 1:
                    optimizer.step()
            def update(self):
                if self.attempt == 1:
                    self.value /= 2
        model = TinyDenoiser()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = FirstAttemptSkipped()
        args = SimpleNamespace(num_steps=2, modality="t1c", small_lesion_weight=0,
                               patient_balance_mode="none", p_uncond=0)
        batch = {"scan_t1c_crop_pad": torch.randn(1, 1, 4, 4, 4),
                 "label_crop_pad": torch.zeros(1, 4, 4, 4, 4)}
        cfg = trainer._diffusion_utils.make_noise_schedule(schedule="edm", n_steps=10)
        result = trainer.train(args, 0, [batch] * 4, model, optimizer, scaler, cfg, "unused")
        self.assertEqual(result[0], 2)
        self.assertEqual(scaler.attempt, 3)
        self.assertEqual(int(next(iter(optimizer.state.values()))["step"]), 2)

    def test_network_cpu_forward_backward(self):
        from src.networks.DiffusionNetwork import get_diffusion_network
        torch.set_num_threads(1)
        args = SimpleNamespace(in_channels=5, out_channels=1, noise_embedding_mode="continuous",
                               crop_size=16, time_ch_count=8, generator_type="Unet_NnU", planner_gpu_gb=8)
        model = get_diffusion_network(args)
        output = model(torch.randn(1, 1, 16, 16, 16), torch.zeros(1), torch.zeros(1, 4, 16, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16, 16))
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_failed_atomic_save_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diffusion_2.pt"
            path.write_bytes(b"old checkpoint")
            with patch("src.utils.checkpoints.torch.save", side_effect=RuntimeError("disk full")):
                with self.assertRaises(RuntimeError):
                    trainer.save_ckp({}, str(path))
            self.assertEqual(path.read_bytes(), b"old checkpoint")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_update_and_scaler_restore(self):
        model = TinyDenoiser().cuda()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.cuda.amp.GradScaler(init_scale=8.)
        args = SimpleNamespace(num_steps=2, modality="t1c", small_lesion_weight=0,
                               patient_balance_mode="none", p_uncond=0)
        batch = {"scan_t1c_crop_pad": torch.randn(1, 1, 4, 4, 4),
                 "label_crop_pad": torch.zeros(1, 4, 4, 4, 4)}
        cfg = trainer._diffusion_utils.make_noise_schedule(schedule="edm", n_steps=10)
        result = trainer.train(args, 0, [batch] * 4, model, optimizer, scaler, cfg, "unused")
        self.assertEqual(result[0], 2)
        self.assertEqual(int(next(iter(optimizer.state.values()))["step"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "Checkpoint/exp/t1c/weights"
            location.mkdir(parents=True)
            trainer.save_ckp({"global_step": 2, "epoch": 1, "state_dict": model.state_dict(),
                              "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
                             str(location / "diffusion_2.pt"))
            restored = torch.cuda.amp.GradScaler(init_scale=1024.)
            resume = SimpleNamespace(logdir="exp", modality="t1c", resume_iter="2")
            original = os.getcwd()
            try:
                os.chdir(directory)
                trainer.load_ckp(resume, model, optimizer, restored)
            finally:
                os.chdir(original)
            self.assertEqual(restored.state_dict(), scaler.state_dict())

    def test_exact_updates_and_resume_boundary_for_all_schedules(self):
        torch.set_num_threads(1)
        for schedule in ("edm", "lognsr", "cosine"):
            with self.subTest(schedule=schedule):
                model = TinyDenoiser()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                batch = {"scan_t1c_crop_pad": torch.randn(1, 1, 4, 4, 4),
                         "label_crop_pad": torch.zeros(1, 4, 4, 4, 4)}
                args = SimpleNamespace(num_steps=2, modality="t1c", small_lesion_weight=0,
                                       patient_balance_mode="none", p_uncond=0)
                cfg = trainer._diffusion_utils.make_noise_schedule(schedule=schedule, n_steps=10)
                result = trainer.train(args, 0, [batch] * 5, model, optimizer, scaler, cfg, "unused")
                self.assertEqual(result[0], 2)
                self.assertEqual(len(result[3]), 2)
                self.assertEqual(int(next(iter(optimizer.state.values()))["step"]), 2)
                result = trainer.train(args, 2, [batch], model, optimizer, scaler, cfg, "unused")
                self.assertEqual(result[3], [])
                args.num_steps = 3
                result = trainer.train(args, 2, [batch] * 5, model, optimizer, scaler, cfg, "unused")
                self.assertEqual(int(next(iter(optimizer.state.values()))["step"]), 3)

    def test_compiled_and_eager_checkpoint_roundtrip(self):
        # Real torch.compile wrapper, with an identity compiler context: no CUDA/Triton required.
        from torch._dynamo.eval_frame import OptimizedModule
        with tempfile.TemporaryDirectory() as directory:
            model = TinyDenoiser()
            compiled = OptimizedModule(model, lambda function: function)
            optimizer = torch.optim.AdamW(model.parameters())
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            location = Path(directory) / "Checkpoint/exp/t1c/weights"
            location.mkdir(parents=True)
            args = SimpleNamespace(logdir="exp", modality="t1c", resume_iter="2")
            for source, target in ((compiled, TinyDenoiser()), (model, OptimizedModule(TinyDenoiser(), lambda f: f))):
                state = {"state_dict": source.state_dict(), "optimizer": optimizer.state_dict(),
                         "global_step": 2, "epoch": 1, "scaler": scaler.state_dict()}
                trainer.save_ckp(state, str(location / "diffusion_2.pt"))
                target_optimizer = torch.optim.AdamW(target.parameters())
                original_cwd = os.getcwd()
                try:
                    os.chdir(directory)
                    trainer.load_ckp(args, target, target_optimizer, scaler)
                finally:
                    os.chdir(original_cwd)
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(unwrap_model(target).state_dict()[key], tensor)
                self.assertFalse(list(location.glob("*.tmp")))

    def test_label_preview_background_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "t1c/checkpoint_scans").mkdir(parents=True)
            trainer.save_sample(SimpleNamespace(modality="t1c"), torch.zeros(4, 5, 5, 5),
                                "label", 1, directory, label=True)
            data = nib.load(Path(directory) / "t1c/checkpoint_scans/1_label.nii.gz").get_fdata()
            self.assertTrue(np.equal(data, 0).all())


class EvaluationScheduleTests(unittest.TestCase):
    @staticmethod
    def command_line():
        return SimpleNamespace(noise_schedule="edm", n_steps=1000, sigma_data=0.5, sigma_min=0.002,
                               sigma_max=50.0, rho=7.0, gamma_min=-10.0, gamma_max=10.0, snr_shift=0.0)

    @staticmethod
    def recorded(sigma_data=1.0):
        return evaluation.checkpoint_schedule({
            "noise_schedule": "edm", "n_steps": 1000,
            "schedule_config": {"name": "edm", "sigma_data": sigma_data, "sigma_min": 0.002,
                                "sigma_max": 50.0, "rho": 7.0, "snr_shift": 0.0, "n_steps": 1000}})

    def test_checkpoint_schedule_replaces_command_line_values(self):
        args = self.command_line()
        sources = {mod: (f"{mod}.pt", self.recorded()) for mod in ("t1c", "t2w")}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            evaluation.reconcile_schedule(args, sources)
        self.assertEqual(args.sigma_data, 1.0)
        self.assertIn("--sigma_data 0.5 -> 1.0", out.getvalue())

    def test_modalities_with_different_schedules_are_refused(self):
        sources = {"t1c": ("a.pt", self.recorded(1.0)), "t2w": ("b.pt", self.recorded(0.5))}
        with self.assertRaisesRegex(ValueError, "different noise schedules"):
            evaluation.reconcile_schedule(self.command_line(), sources)

    def test_checkpoint_without_schedule_keeps_command_line(self):
        args = self.command_line()
        evaluation.reconcile_schedule(args, {"t1c": ("old.pt", evaluation.checkpoint_schedule({}))})
        self.assertEqual(vars(args), vars(self.command_line()))

    def test_sampler_must_match_trained_schedule(self):
        for schedule, method in (("edm", "edm_heun"), ("lognsr", "lognsr_ode"), ("cosine", "ddpm"), ("cosine", "ddim")):
            evaluation.check_sampling_method(schedule, method)
        for schedule, method in (("edm", "ddim"), ("edm", "lognsr_ode"), ("lognsr", "edm_heun"), ("cosine", "edm_heun")):
            with self.assertRaises(ValueError):
                evaluation.check_sampling_method(schedule, method)


if __name__ == "__main__":
    unittest.main()
