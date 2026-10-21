#!/usr/bin/env python3
"""Read-only revision audit. Synthetic outputs live only in a temporary directory."""

import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = PROJECT / "work_space/G1/01_code/tumour_diffusion_1mm"
STATE = dict(artifact_status="experimental_unvalidated", operator_approved=False,
             formal_gate_status="not_run_not_passed")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot():
    return {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE.rglob("*") if p.is_file()}


@contextlib.contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main():
    before = snapshot()
    checks = []

    def probe(name, function):
        try:
            checks.append(dict(name=name, **function()))
        except Exception as exc:
            checks.append(dict(name=name, status="probe_error", error=str(exc), traceback=traceback.format_exc()))
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)

    def syntax():
        python = list(SOURCE.rglob("*.py"))
        for p in python:
            ast.parse(p.read_text(), filename=str(p))
        shell = {p.name: subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode
                 for p in SOURCE.rglob("*.sh")}
        return dict(status="pass" if all(v == 0 for v in shell.values()) else "fail",
                    python_files=len(python), shell_exit_codes=shell)

    probe("syntax", syntax)
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    creator = importlib.import_module("src.train.csv_creator")
    transforms = importlib.import_module("src.utils.gaussian_noise_tumour")
    converter = importlib.import_module("src.utils.convert_to_multi_channel_based_on_brats_classes")
    networks = importlib.import_module("src.networks.DiffusionNetwork")
    trainer = importlib.import_module("src.train.tumour_main_diffusion")
    diffusion = load("revision_diffusion", SOURCE / "model.py")
    preprocess = load("revision_preprocess", SOURCE / "scripts/preprocess_dataset.py")
    label_transform = converter.ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024()

    def real_import():
        names = ["src.utils.data_utils", "src.infer.evaluate_generation",
                 "src.infer.generate_from_label", "src.infer.on_the_fly_augmentation"]
        for name in names:
            importlib.import_module(name)
        return dict(status="pass", modules=names + [trainer.__name__])

    probe("real_import", real_import)

    def xyz():
        mask = np.zeros((36, 24, 14), dtype=bool)
        mask[10:34, 4:9, 2:7] = True
        cc = creator.connected_component_analysis(mask)[0]
        tiles = list(creator.tile_oversized_lesion(cc, mask, crop_size=16, stride=12))
        good = cc["bbox"] == (10, 34, 4, 9, 2, 7) and cc["centroid"] == (22, 6, 4)
        return dict(status="pass" if good and len(tiles) == 2 else "fail",
                    bbox=list(map(int, cc["bbox"])), centroid=list(map(int, cc["centroid"])),
                    tile_count=len(tiles), tile_voxel_counts=[r["n_voxels"] for r in tiles])

    probe("xyz_and_tiles", xyz)

    def rc_crop():
        seg = np.zeros((96, 96, 96), dtype=np.uint8)
        seg[70:76, 20:26, 10:16] = 4
        cc = creator.connected_component_analysis(seg > 0)[0]
        row = dict(scan_t1c=torch.ones((1, *seg.shape)), label=torch.from_numpy(seg[None]), n_voxels=216)
        for i, ax in enumerate("xyz"):
            row[f"{ax}_extreme_min"], row[f"{ax}_extreme_max"] = map(int, cc["bbox"][2*i:2*i+2])
            row[f"center_{ax}"] = int(cc["centroid"][i])
        roi = transforms.CropLesionROId(["scan_t1c", "label"], target_size=64)(row)
        baseline = roi["label"].clone()
        transform = transforms.GaussianNoiseTumour("scan_t1c", normalization="nnunet", target_size=64,
                                                   label_transform=label_transform)
        transform.set_random_state(seed=20261013)
        counts = [int(np.asarray(transform(roi)["label_crop_pad"])[3].sum()) for _ in range(24)]
        return dict(status="pass" if all(n == 216 for n in counts) else "fail",
                    expected_rc_voxels=216, actual_rc_voxels_24_offsets=counts,
                    cache_unchanged=bool(torch.equal(baseline, roi["label"])))

    probe("rc_crop_24_offsets", rc_crop)

    with tempfile.TemporaryDirectory(prefix="ecnu_revision3_") as temporary:
        root = Path(temporary)

        def create_case(name, mode="valid"):
            raw = root / "raw" / name
            raw.mkdir(parents=True)
            shape = (12, 14, 16)
            affine = np.diag([1.2, .9, 1.7, 1.])
            rng = np.random.default_rng(13)
            for modality in preprocess.MODALITIES:
                a = rng.uniform(1, 20, shape).astype(np.float32)
                world = affine.copy()
                if modality == "t2w" and mode == "affine":
                    world[0, 3] = 10
                if modality == "t1c" and mode == "nan":
                    a[4, 4, 4] = np.nan
                nib.save(nib.Nifti1Image(a, world), raw / f"{name}-{modality}.nii.gz")
            seg = np.zeros(shape, dtype=np.float32)
            seg[3:7, 3:7, 3:7] = 4
            if mode == "label9":
                seg[8:10, 8:10, 8:10] = 9
            if mode == "fractional":
                seg[8, 8, 8] = 2.5
            nib.save(nib.Nifti1Image(seg, affine), raw / f"{name}-seg.nii.gz")
            return raw

        valid_name = "BraTS-MET-AUDIT-000"
        valid = create_case(valid_name)
        outputs = root / "outputs"

        def input_rejection():
            results = {}
            for mode in ("affine", "nan", "label9", "fractional"):
                name = f"BraTS-MET-{mode}-000"
                raw = create_case(name, mode)
                result = preprocess.process((name, str(raw), str(outputs), (1., 1., 1.), False, (0, 1, 2, 3, 4), .1))
                results[mode] = result["status"]
            return dict(status="pass" if all(v.startswith("rejected") for v in results.values()) else "fail", results=results)

        probe("input_rejection", input_rejection)

        def grid():
            result = preprocess.process((valid_name, str(valid), str(outputs), (1., 1., 1.), False,
                                         (0, 1, 2, 3, 4), .1))
            img = nib.load(outputs / valid_name / f"{valid_name}-seg.nii.gz")
            old_shape = np.array((12, 14, 16), float)
            new_shape = np.array(img.shape, float)
            old_affine = nib.load(valid / f"{valid_name}-seg.nii.gz").affine
            mapping = np.eye(4)
            mapping[:3, :3] = np.diag(old_shape / new_shape)
            mapping[:3, 3] = .5 * old_shape / new_shape - .5
            actual_sample_affine = old_affine @ mapping
            drift = preprocess.corner_shift_mm(actual_sample_affine, img.affine, img.shape)
            return dict(status="pass" if drift < 1e-5 else "fail", process_status=result["status"],
                        header_spacing=list(map(float, img.header.get_zooms()[:3])),
                        reported_resize_spacing=result["true_spacing"],
                        max_sample_location_vs_saved_world_mm=drift,
                        saved_output_finite=bool(np.isfinite(img.get_fdata()).all()))

        probe("physical_sampling_grid", grid)

        def exit_contract():
            script = SOURCE / "scripts/preprocess_dataset.py"
            common = [sys.executable, "-B", str(script), "--raw_dirs", str(root / "raw"), "--num_workers", "1"]
            cases = root / "reject_only.txt"
            cases.write_text("BraTS-MET-nan-000\n")
            reject = subprocess.run(common + ["--case_list", str(cases), "--out_dir", str(root / "rejected_cli")],
                                    capture_output=True, text=True, timeout=90)
            cases.write_text("BraTS-MET-absent-000\n")
            absent = subprocess.run(common + ["--case_list", str(cases), "--out_dir", str(root / "absent_cli")],
                                    capture_output=True, text=True, timeout=90)
            corrupt = outputs / valid_name / f"{valid_name}-t1c.nii.gz"
            corrupt.write_bytes(b"broken nifti")
            skipped = preprocess.process((valid_name, str(valid), str(outputs), (1., 1., 1.), False,
                                           (0, 1, 2, 3, 4), .1))
            return dict(status="fail" if reject.returncode == 0 or absent.returncode == 0 or skipped["status"].startswith("skipped") else "pass",
                        rejected_case_cli_exit=reject.returncode, absent_required_case_cli_exit=absent.returncode,
                        corrupt_existing_output_status=skipped["status"],
                        rejected_cli_tail=reject.stdout[-1500:], absent_cli_tail=absent.stdout[-800:])

        probe("strict_complete_and_existing_output", exit_contract)

        def eval_split():
            tree = ast.parse((SOURCE / "src/infer/evaluate_generation.py").read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                        and "args.eval_split" in ast.unparse(n.test) and "df.columns" in ast.unparse(n.test))
            code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "eval_split_isolated", "exec")
            results = {}
            for label, df in (("only_train", pd.DataFrame({"split": ["train"]})),
                              ("missing_split_column", pd.DataFrame({"patient_id": ["p1"]})),
                              ("has_val", pd.DataFrame({"split": ["train", "val"]}))):
                env = dict(df=df, args=SimpleNamespace(eval_split="val", csv_path="synthetic.csv"))
                try:
                    exec(code, env)
                    results[label] = dict(rejected=False, rows=len(env["df"]))
                except SystemExit as exc:
                    results[label] = dict(rejected=True, error=str(exc))
            return dict(status="pass" if results["only_train"]["rejected"] and results["missing_split_column"]["rejected"] else "partial",
                        scope="actual_ast_branch_executed_no_sampling", results=results)

        probe("validation_split_guard", eval_split)

        def csv_and_loader():
            first, second = "BraTS-MET-00001-000", "BraTS-MET-00001-001"
            # Same patient, different visits: case-level val selection must not be mistaken for group isolation.
            group_root = root / "group_dataset"
            group_root.mkdir()
            for name in (first, second):
                raw = create_case(name)
                raw.rename(group_root / name)
            csv_path = root / "group.csv"
            args = SimpleNamespace(seg_ending="seg.nii.gz", t1c_ending="t1c.nii.gz",
                 t1n_ending="t1n.nii.gz", t2w_ending="t2w.nii.gz", t2f_ending="t2f.nii.gz",
                 val_patients=second, val_list="", t2w_invalid_list="", crop_size=64, merge_dist=16)
            creator.create_csv(args, "BRATS_2024", str(csv_path), str(group_root))
            frame = pd.read_csv(csv_path)
            data_utils = importlib.import_module("src.utils.data_utils")
            loader_args = SimpleNamespace(csv_path=str(csv_path), logdir="unused", num_workers=0,
                crop_size=64, modality="t1c", normalization="nnunet", dataset="BRATS_2024",
                in_channels=5, noise_type="gaussian_tumour", no_roi_cache=False, use_all_rows=False,
                cache_rate=1., loader_workers=0, batch_size=1)
            batch = next(iter(data_utils.get_loader(loader_args)))
            groups = frame.groupby(frame["patient_id"].str.rsplit("-", n=1).str[0])["split"].nunique()
            return dict(status="partial" if any(groups > 1) else "pass",
                        real_csv_and_loader_success=True, generated_rows=len(frame),
                        train_rows=loader_args.n_train_rows, batch_shape=list(batch["scan_t1c_crop_pad"].shape),
                        batch_rc_voxels=int(batch["label_crop_pad"][:, 3].sum()),
                        same_patient_cross_visit_split_leak=bool(any(groups > 1)),
                        split_assignment=frame[["patient_id", "split"]].to_dict("records"))

        probe("actual_csv_loader_and_patient_group_split", csv_and_loader)

        def atomic_and_scaler():
            destination = root / "atomic.pt"
            trainer.save_ckp({"value": torch.tensor([1.])}, str(destination))
            good = destination.read_bytes()
            def interrupted(state, path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("synthetic_save_interruption")
            with patch.object(trainer.torch, "save", interrupted):
                try:
                    trainer.save_ckp({"value": 2}, str(destination))
                except RuntimeError:
                    pass
            scaler = torch.amp.GradScaler("cpu", init_scale=256)
            scaler.scale(torch.tensor(1., requires_grad=True))
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            weight_dir = root / "Checkpoint" / "resume" / "t1c" / "weights"
            weight_dir.mkdir(parents=True)
            trainer.save_ckp(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                 scaler=scaler.state_dict(), epoch=1, global_step=2,
                                 data_identity={"csv_sha256": "synthetic"}), str(weight_dir / "diffusion_2.pt"))
            restored_scaler = torch.amp.GradScaler("cpu", init_scale=65536)
            with cwd(root):
                trainer.load_ckp(SimpleNamespace(logdir="resume", modality="t1c", resume_iter="2"),
                                 model, optimizer, restored_scaler)
            return dict(status="pass" if destination.read_bytes() == good and restored_scaler.get_scale() == 256 else "fail",
                        existing_checkpoint_survives_interrupted_save=destination.read_bytes() == good,
                        amp_scale_restored=restored_scaler.get_scale(),
                        incomplete_tmp_retained=(root / "atomic.pt.tmp").exists(),
                        scope="CPU_scaler_roundtrip_not_cuda_recovery")

        probe("atomic_save_and_amp_restore", atomic_and_scaler)

        def exact_steps():
            class CountingSGD(torch.optim.SGD):
                def __init__(self, params):
                    super().__init__(params, lr=.01)
                    self.updates = 0
                def step(self, closure=None):
                    self.updates += 1
                    return super().step(closure)
            class TinyConditionalModel(torch.nn.Conv3d):
                def forward(self, x, time, cond=None):
                    return super().forward(x)
            model = TinyConditionalModel(1, 1, 1)
            optimizer = CountingSGD(model.parameters())
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            args = SimpleNamespace(modality="t1c", small_lesion_weight=0., patient_balance_mode="none",
                                   p_uncond=0., num_steps=2)
            batch = {"scan_t1c_crop_pad": torch.ones(1, 1, 4, 4, 4), "label_crop_pad": torch.zeros(1, 4, 4, 4, 4)}
            with patch.object(torch.Tensor, "cuda", lambda tensor: tensor), \
                 patch.object(torch.amp, "autocast", lambda *a, **k: contextlib.nullcontext()):
                outcome = trainer.train(args, 0, [batch]*5, model, optimizer, scaler,
                                        diffusion.make_noise_schedule("edm", sigma_data=1.), str(root))
            return dict(status="pass" if optimizer.updates == args.num_steps else "fail",
                        requested_updates=args.num_steps, actual_updates=optimizer.updates,
                        returned_global_step=outcome[0],
                        scope="actual_train_loop_CPU_with_cuda_and_autocast_adapters_only")

        probe("exact_optimizer_step_count", exact_steps)

        def compiled_restore():
            model = torch.nn.Linear(2, 1)
            wrapper = torch.nn.Module()
            wrapper.add_module("_orig_mod", model)
            optimizer = torch.optim.AdamW(model.parameters())
            target = root / "Checkpoint" / "compiled" / "t1c" / "weights"
            target.mkdir(parents=True)
            trainer.save_ckp(dict(state_dict=wrapper.state_dict(), optimizer=optimizer.state_dict(),
                                 epoch=1, global_step=2), str(target / "diffusion_2.pt"))
            try:
                with cwd(root):
                    trainer.load_ckp(SimpleNamespace(logdir="compiled", modality="t1c", resume_iter="2"), model, optimizer)
            except RuntimeError as exc:
                return dict(status="fail", restore_error=str(exc),
                            scope="actual_loader_test_using_compile_state_dict_prefix_no_compiler_invoked")
            return dict(status="pass")

        probe("compile_to_uncompiled_restore", compiled_restore)

        def unchanged_csv_changed_image():
            csv_path = root / "fingerprint.csv"
            image_path = root / "fingerprint_image.nii.gz"
            image_path.write_bytes(b"image version one")
            csv_path.write_text(f"image,split\n{image_path},train\n")
            identity1 = trainer.csv_identity(str(csv_path))
            image_path.write_bytes(b"image version two")
            identity2 = trainer.csv_identity(str(csv_path))
            return dict(status="partial" if identity1 == identity2 else "pass",
                        csv_fingerprint_survives_image_mutation=identity1 == identity2,
                        scope="CSV_fingerprint_only_does_not_bind_referenced_files")

        probe("csv_hash_vs_underlying_data_hash", unchanged_csv_changed_image)

    def network():
        model = networks.TimeConditionedUNet_NnU(in_channels=1, cond_channels=4, input_size=64,
                    gpu_memory_gb=48, noise_embedding_mode="continuous")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        seg = torch.zeros(1, 16, 16, 16, dtype=torch.int64)
        seg[:, 5:9, 5:9, 5:9] = 4
        cond = label_transform(seg).float()[None]
        loss = diffusion.edm_loss_fn(model, torch.randn(1, 1, 16, 16, 16), cond,
                                     diffusion.make_noise_schedule("edm", sigma_data=1.), "cpu")
        loss.backward()
        finite_grad = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 1, 64, 64, 64), torch.zeros(1), torch.zeros(1, 4, 64, 64, 64))
        return dict(status="pass" if bool(torch.isfinite(loss)) and finite_grad and bool(torch.isfinite(result).all()) else "fail",
                    params=sum(p.numel() for p in model.parameters()), forward_shape=list(result.shape),
                    backward_shape=[1, 1, 16, 16, 16], loss=float(loss.detach()), finite_gradients=finite_grad,
                    cuda_available=torch.cuda.is_available(), full_size_gpu_batch16_test="not_run")

    probe("reference_edm_network_cpu", network)

    def remaining_static():
        text = (SOURCE / "src/train/tumour_main_diffusion.py").read_text()
        invalid = [s.strip() for s in (SOURCE / "lists/t2w_invalid_cases.txt").read_text().splitlines()
                   if s.strip() and not s.startswith("#")]
        return dict(status="unresolved", rng_saved='"rng"' in text, signal_handler="signal.signal" in text,
                    lognsr_called_function_exists=hasattr(diffusion, "lognsr_loss_fn"),
                    csv_sha_not_image_sha="csv_sha256" in text,
                    invalid_t2w_count=len(set(invalid)),
                    smoke_script_deletes_existing_work_and_checkpoint='rm -rf "$WORK" "$CKPT"' in (SOURCE / "scripts/smoke_test.sh").read_text())

    probe("remaining_static_contracts", remaining_static)
    evidence = dict(**STATE, reviewed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    active_root=str(SOURCE), source_unchanged=before == snapshot(),
                    status="not_ready_for_full_training", checks=checks,
                    environment=dict(python=sys.version, executable=sys.executable,
                         packages={name: importlib.metadata.version(name) for name in ("torch", "monai", "numpy", "nibabel", "nnunetv2")}),
                    boundaries=dict(server_access=False, source_edits=False, installed_dependencies=False,
                                    training_submitted=False, real_data_download=False, upstream_destructive_smoke_run=False))
    (HERE / "REVISION_READINESS_EVIDENCE.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return 1 if not evidence["source_unchanged"] or any(c["status"] == "probe_error" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
