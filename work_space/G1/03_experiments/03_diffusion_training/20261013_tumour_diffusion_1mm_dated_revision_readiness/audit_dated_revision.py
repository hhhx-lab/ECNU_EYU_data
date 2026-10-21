"""Independent CPU regression audit of the unchanged staged or published revision."""
import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = HERE / "incoming_source"
if not SOURCE.exists():
    SOURCE = PROJECT / "work_space/G1/01_code/tumour_diffusion_1mm"
STATE = dict(artifact_status="experimental_unvalidated", operator_approved=False,
             formal_gate_status="not_run_not_passed")


def snapshot():
    return {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE.rglob("*") if p.is_file()}


def main():
    destination = HERE / "DATED_REVISION_READINESS_EVIDENCE.json"
    if destination.exists():
        raise ValueError("Refuse to overwrite completed audit")
    before, checks = snapshot(), []
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import torch
    torch.set_num_threads(2)

    def probe(name, function):
        try:
            checks.append(dict(name=name, **function()))
        except Exception as exc:
            checks.append(dict(name=name, status="probe_error", error=str(exc), traceback=traceback.format_exc()))
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)

    def syntax_import():
        files = list(SOURCE.rglob("*.py"))
        for path in files:
            ast.parse(path.read_text(), filename=str(path))
        shells = {p.relative_to(SOURCE).as_posix(): subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode
                  for p in SOURCE.rglob("*.sh")}
        modules = ["scripts.preprocess_dataset", "src.train.csv_creator", "src.train.tumour_main_diffusion",
                   "src.train.label_main_diffusion", "src.utils.data_utils", "src.utils.splits",
                   "src.utils.volume_validation", "src.infer.evaluate_generation",
                   "src.infer.generate_from_label", "src.infer.on_the_fly_augmentation"]
        for name in modules:
            importlib.import_module(name)
        return dict(status="pass" if not any(shells.values()) else "fail",
                    python_files=len(files), shell_exit_codes=shells, real_imports=modules)

    probe("syntax_and_real_import", syntax_import)

    class Results(unittest.TextTestResult):
        records = []
        def addSuccess(self, test):
            super().addSuccess(test)
            self.records.append(dict(test=test.id(), status="pass"))
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.records.append(dict(test=test.id(), status="skipped", reason=reason))
        def addFailure(self, test, error):
            super().addFailure(test, error)
            self.records.append(dict(test=test.id(), status="fail", traceback=self._exc_info_to_string(error, test)))
        def addError(self, test, error):
            super().addError(test, error)
            self.records.append(dict(test=test.id(), status="error", traceback=self._exc_info_to_string(error, test)))

    def upstream_suite():
        suite = unittest.TestLoader().discover(str(SOURCE / "tests"))
        with (HERE / "UPSTREAM_REGRESSION_TEST_LOG.txt").open("x") as log:
            runner = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Results)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = runner.run(suite)
        return dict(status="pass" if result.wasSuccessful() else "fail", tests_run=result.testsRun,
                    passed=sum(r["status"] == "pass" for r in result.records),
                    skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
                    records=result.records, scope="local_CPU_only_GPU_tests_skipped")

    probe("upstream_isolated_regressions", upstream_suite)
    prep = importlib.import_module("scripts.preprocess_dataset")
    grid = importlib.import_module("src.preprocess.nnunet_like")
    splits = importlib.import_module("src.utils.splits")
    trainer = importlib.import_module("src.train.tumour_main_diffusion")
    evaluation = importlib.import_module("src.infer.evaluate_generation")

    def independent_grid():
        shape = np.array([48, 52, 56])
        spacing = np.array([1.2, .9, 1.7])
        angle = .31
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, -1.]])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag(spacing)
        affine[:3, 3] = [9., -12., 20.]
        indices = np.indices(shape)
        world = affine[:3, :3] @ indices.reshape(3, -1) + affine[:3, 3:4]
        data = world.reshape(3, *shape).astype(np.float32)
        new_shape, factor, offset = grid.physical_grid(shape, spacing, (1., 1., 1.))
        out = grid.resample_physical(data, new_shape, factor, offset, spacing, (1., 1., 1.))
        output_affine, actual_spacing = prep.new_affine(affine, [[0, int(n)] for n in shape], shape,
                                                       out.shape[1:], spacing, np.ones(3))
        sample = np.indices(out.shape[1:])[:, 6:-6, 6:-6, 6:-6]
        expected = (output_affine[:3, :3] @ sample.reshape(3, -1) + output_affine[:3, 3:4])
        error = float(np.abs(out[:, 6:-6, 6:-6, 6:-6].reshape(3, -1)-expected).max())
        return dict(status="pass" if error < .003 and np.allclose(actual_spacing, 1.) else "fail",
                    maximum_interior_coordinate_error_mm=error, tolerance_mm=.003,
                    saved_spacing=list(actual_spacing), shape=list(out.shape[1:]),
                    scope="analytic_world_ramps_rotated_and_reversed_affine_noninteger_shape_ratio")

    probe("independent_oblique_physical_grid", independent_grid)

    def independent_split():
        results = {}
        inputs = dict(missing_column=pd.DataFrame({"patient_id": ["00001-000"]}),
                      train_only=pd.DataFrame({"patient_id": ["00001-000"], "split": ["train"]}),
                      crossing=pd.DataFrame({"patient_id": ["00001-000", "00001-001"], "split": ["train", "val"]}))
        for key, frame in inputs.items():
            try:
                splits.select_split(frame, "val")
                results[key] = False
            except ValueError:
                results[key] = True
        valid = pd.DataFrame({"patient_id": ["00001-000", "00002-001"], "split": ["train", "val"]})
        result = splits.select_split(valid, "val")
        return dict(status="pass" if all(results.values()) and len(result) == 1 else "fail",
                    rejected=results, isolated_validation_rows=len(result),
                    braTS_patient_group=splits.patient_group("BraTS-MET-00001-001"))

    probe("independent_fail_closed_patient_split", independent_split)

    def resumed_budget():
        class Tiny(torch.nn.Conv3d):
            def forward(self, x, time, cond=None):
                return super().forward(x)
        model = Tiny(1, 1, 1)
        opt = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        args = SimpleNamespace(num_steps=2, modality="t1c", small_lesion_weight=0,
                               patient_balance_mode="none", p_uncond=0)
        batch = dict(scan_t1c_crop_pad=torch.ones(1, 1, 4, 4, 4), label_crop_pad=torch.zeros(1, 4, 4, 4, 4))
        cfg = trainer._diffusion_utils.make_noise_schedule("edm", sigma_data=1.)
        first = trainer.train(args, 0, [batch]*5, model, opt, scaler, cfg, "unused")
        state = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     global_step=first[0], epoch=1, step_semantics="completed_optimizer_updates_v2")
        with tempfile.TemporaryDirectory(prefix="ecnu-dated-budget-") as temp:
            path = Path(temp) / "Checkpoint/check/t1c/weights"
            path.mkdir(parents=True)
            trainer.save_ckp(state, str(path / "diffusion_2.pt"))
            fresh = Tiny(1, 1, 1)
            fresh_opt = torch.optim.AdamW(fresh.parameters())
            old = os.getcwd()
            try:
                os.chdir(temp)
                _, _, _, step = trainer.load_ckp(SimpleNamespace(logdir="check", modality="t1c", resume_iter="2"),
                                                 fresh, fresh_opt, scaler)
            finally:
                os.chdir(old)
            args.num_steps = 3
            second = trainer.train(args, step, [batch]*5, fresh, fresh_opt, scaler, cfg, "unused")
        updates = int(next(iter(fresh_opt.state.values()))["step"])
        return dict(status="pass" if first[0] == 2 and second[0] == 3 and len(second[3]) == 1 and updates == 3 else "fail",
                    first_completed_updates=first[0], resumed_completed_updates=second[0], resumed_new_updates=len(second[3]),
                    restored_optimizer_update_count=updates, scope="real_checkpoint_roundtrip_then_actual_CPU_train_loop")

    probe("independent_saved_resume_exact_updates", resumed_budget)

    def changed_image_identity():
        with tempfile.TemporaryDirectory(prefix="ecnu-dated-identity-") as temp:
            folder = Path(temp)
            image = folder / "case-t1c.nii.gz"
            label = folder / "case-seg.nii.gz"
            image.write_bytes(b"image version one")
            label.write_bytes(b"synthetic label")
            marker = folder / "preprocess_complete.json"
            marker.write_text(json.dumps(dict(status="complete", outputs={image.name: hashlib.sha256(image.read_bytes()).hexdigest()})))
            csv = folder / "cases.csv"
            csv.write_text(f"label,scan_t1c,split,patient_id\n{label},{image},train,00001-000\n")
            before_identity = trainer.csv_identity(str(csv))
            image.write_bytes(b"image version two")
            mutation_identity = trainer.csv_identity(str(csv))
            marker.write_text(json.dumps(dict(status="complete", outputs={image.name: hashlib.sha256(image.read_bytes()).hexdigest()})))
            new_marker_identity = trainer.csv_identity(str(csv))
        return dict(status="partial" if before_identity == mutation_identity else "pass",
                    in_place_image_mutation_detected=before_identity != mutation_identity,
                    changed_completion_marker_detected=before_identity != new_marker_identity,
                    scope="actual_csv_identity_fingerprint_requires_external_revalidation_of_marker_output_SHAs")

    probe("referenced_image_content_vs_marker_fingerprint", changed_image_identity)

    def remaining_contracts():
        training = (SOURCE / "src/train/tumour_main_diffusion.py").read_text()
        evaluation_text = (SOURCE / "src/infer/evaluate_generation.py").read_text()
        smoke = (SOURCE / "scripts/smoke_test.sh").read_text()
        launch = (SOURCE / "scripts/train_4gpu.sh").read_text()
        invalid = {line.strip() for line in (SOURCE / "lists/t2w_invalid_cases.txt").read_text().splitlines()
                   if line.strip() and not line.startswith("#")}
        checkpoint_keys = next(n.value for n in ast.walk(ast.parse(training)) if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in n.targets))
        keys = [k.value for k in checkpoint_keys.keys if isinstance(k, ast.Constant)]
        return dict(status="partial", checkpoint_keys=keys, rng_or_sampler_state_saved=any("rng" in k or "sampler" in k for k in keys),
                    term_handler_present="signal.signal" in training,
                    eval_csv_mismatch_still_warning="its val rows may overlap" in evaluation_text,
                    smoke_no_recursive_delete="rm -rf" not in smoke,
                    smoke_trained_modalities=["t1c", "t2w"], invalid_t2w_count=len(invalid),
                    train_launcher_still_overwrites_logs_and_status='> "logs/train_${mod}.log"' in launch,
                    server_slurm_contract_test="not_run", full_A100_64cube_batch16_amp_compile="not_run",
                    raw_data_and_patient_manifest_compatibility="not_run", S2_native_space_roundtrip="not_run",
                    adam_beta1_default=.9, previous_beta1=.5)

    probe("remaining_scientific_and_long_run_contracts", remaining_contracts)

    def reference_network():
        module = importlib.import_module("src.networks.DiffusionNetwork")
        args = SimpleNamespace(in_channels=5, out_channels=1, noise_embedding_mode="continuous", crop_size=64,
                               time_ch_count=8, generator_type="Unet_NnU", planner_gpu_gb=48)
        model = module.get_diffusion_network(args)
        result = model(torch.randn(1, 1, 16, 16, 16), torch.zeros(1), torch.zeros(1, 4, 16, 16, 16))
        result.square().mean().backward()
        finite_grad = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            full = model(torch.zeros(1, 1, 64, 64, 64), torch.zeros(1), torch.zeros(1, 4, 64, 64, 64))
        return dict(status="pass" if finite_grad and bool(torch.isfinite(full).all()) else "fail",
                    parameters=sum(p.numel() for p in model.parameters()), forward_shape=list(full.shape),
                    backward_shape=list(result.shape), scope="CPU_reference_architecture_not_full_size_GPU_backprop")

    probe("reference_network_64cube_forward_16cube_backward", reference_network)
    evidence = dict(**STATE, reviewed_at=datetime.now(timezone.utc).isoformat(), source_root=str(SOURCE),
                    source_unchanged=before == snapshot(), source_sha256=before,
                    status="not_ready_for_full_training", checks=checks,
                    environment=dict(executable=sys.executable, python=sys.version, cuda_available=torch.cuda.is_available(),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "monai", "numpy", "nibabel", "nnunetv2")}),
                    boundaries=dict(source_edits=False, server_access=False, dependency_install=False,
                                    real_data_download=False, training_submitted=False, upstream_gpu_smoke_run=False))
    with destination.open("x") as output:
        json.dump(evidence, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return 1 if not evidence["source_unchanged"] or any(c["status"] == "probe_error" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
