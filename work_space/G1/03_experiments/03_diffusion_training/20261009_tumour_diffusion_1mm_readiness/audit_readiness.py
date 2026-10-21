#!/usr/bin/env python3
"""Read-only imported-source inspection with disposable synthetic CPU probes."""

import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = PROJECT / "work_space/G1/01_code/tumour_diffusion_1mm"
OLD = PROJECT / "work_space/G1/01_code/diffusion_gliGAN"
STATE = dict(artifact_status="experimental_unvalidated", operator_approved=False,
             formal_gate_status="not_run_not_passed")


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest(root):
    files, empty = [], []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            blob = path.read_bytes()
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=len(blob),
                              sha256=hashlib.sha256(blob).hexdigest(),
                              symlink=path.is_symlink()))
        elif path.is_dir() and not any(path.iterdir()):
            empty.append(path.relative_to(root).as_posix())
    return dict(files=files, empty_directories=empty,
                total_bytes=sum(row["bytes"] for row in files))


def main():
    before = manifest(SOURCE)
    checks = []

    def probe(name, function):
        try:
            result = function()
            checks.append(dict(name=name, **result))
        except Exception as exc:
            checks.append(dict(name=name, status="probe_error", error=str(exc),
                               traceback=traceback.format_exc()))
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)

    sys.path.insert(0, str(SOURCE))

    def syntax():
        sources = sorted(SOURCE.rglob("*.py"))
        for path in sources:
            ast.parse(path.read_text(), filename=str(path))
        shell = subprocess.run(["bash", "-n", str(SOURCE / "scripts/train_4gpu.sh")],
                               capture_output=True, text=True)
        return dict(status="pass" if shell.returncode == 0 else "fail",
                    python_files=len(sources), bash_exit=shell.returncode,
                    stderr=shell.stderr)

    probe("syntax", syntax)
    import numpy as np
    import nibabel as nib
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261009)
    np.random.seed(20261009)
    creator = importlib.import_module("src.train.csv_creator")
    transform = importlib.import_module("src.utils.gaussian_noise_tumour")
    converter = importlib.import_module("src.utils.convert_to_multi_channel_based_on_brats_classes")
    preprocess = load("audit_preprocess", SOURCE / "scripts/preprocess_dataset.py")
    networks = importlib.import_module("src.networks.DiffusionNetwork")
    diffusion = load("audit_diffusion", SOURCE / "model.py")
    old_creator = load("audit_old_csv", OLD / "src/train/csv_creator.py")
    label_transform = converter.ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024()

    def imports():
        names = ["src.train.tumour_main_diffusion", "src.utils.data_utils",
                 "src.infer.diffusion_inference_utils", "src.infer.generate_from_label",
                 "src.infer.evaluate_generation", "src.infer.on_the_fly_augmentation"]
        for name in names:
            importlib.import_module(name)
        return dict(status="pass", modules=names)

    probe("real_imports", imports)

    def axis():
        mask = np.zeros((36, 24, 14), dtype=bool)
        mask[10:34, 4:9, 2:7] = True
        new_cc = creator.connected_component_analysis(mask)[0]
        old_cc = old_creator.connected_component_analysis(mask)[0]
        return dict(status="pass" if new_cc["bbox"] == old_cc["bbox"] else "fail",
                    expected_bbox=list(map(int, old_cc["bbox"])), actual_bbox=list(map(int, new_cc["bbox"])),
                    expected_centroid=list(map(int, old_cc["centroid"])),
                    actual_centroid=list(map(int, new_cc["centroid"])))

    probe("nifti_xyz_axis", axis)

    def tiled_axis():
        mask = np.zeros((36, 24, 14), dtype=bool)
        mask[10:34, 4:9, 2:7] = True
        correct = old_creator.connected_component_analysis(mask)[0]
        actual = list(creator.tile_oversized_lesion(correct, mask, crop_size=16, stride=12))
        expected = list(old_creator.tile_oversized_lesion(correct, mask, crop_size=16, stride=12))
        return dict(status="pass" if len(actual) == len(expected) else "fail",
                    actual_tiles=len(actual), expected_tiles=len(expected),
                    actual_voxel_counts=[r["n_voxels"] for r in actual],
                    expected_voxel_counts=[r["n_voxels"] for r in expected])

    probe("tile_xyz_axis", tiled_axis)

    def crop():
        seg = np.zeros((96, 96, 96), dtype=np.uint8)
        seg[70:76, 20:26, 10:16] = 4
        scan = torch.from_numpy(np.random.default_rng(1).normal(size=(1, *seg.shape)).astype(np.float32))
        values = {}
        for name, module in (("imported", creator), ("correct_xyz_control", old_creator)):
            cc = module.connected_component_analysis(seg > 0)[0]
            row = dict(scan_t1c=scan, label=torch.from_numpy(seg[None]))
            for i, ax in enumerate("xyz"):
                row[f"{ax}_extreme_min"], row[f"{ax}_extreme_max"] = map(int, cc["bbox"][2*i:2*i+2])
                row[f"center_{ax}"] = int(cc["centroid"][i])
            roi = transform.CropLesionROId(["scan_t1c", "label"], target_size=64)(row)
            original_roi = roi["label"].clone()
            random_crop = transform.GaussianNoiseTumour("scan_t1c", normalization="nnunet",
                                                        target_size=64, label_transform=label_transform)
            random_crop.set_random_state(seed=20261009)
            output = random_crop(roi)
            values[name] = dict(shape=list(output["scan_t1c_crop_pad"].shape),
                                rc_voxels=int(np.asarray(output["label_crop_pad"])[3].sum()),
                                cached_label_unchanged=bool(torch.equal(original_roi, roi["label"])))
        return dict(status="pass" if values["imported"]["rc_voxels"] > 0 else "fail", **values)

    probe("actual_roi_crop_rc", crop)

    def geometry_and_labels():
        with tempfile.TemporaryDirectory(prefix="ecnu_1mm_audit_") as temporary:
            root = Path(temporary)
            case = "BraTS-MET-AUDIT-000"
            raw = root / case
            raw.mkdir()
            shape = (12, 14, 16)
            affine = np.diag([1.2, 0.9, 1.7, 1.0])
            rng = np.random.default_rng(4)
            for modality in preprocess.MODALITIES:
                shifted = affine.copy()
                if modality == "t2w":
                    shifted[0, 3] = 10.0
                nib.save(nib.Nifti1Image(rng.uniform(1, 20, shape).astype(np.float32), shifted),
                         raw / f"{case}-{modality}.nii.gz")
            seg = np.zeros(shape, dtype=np.int16)
            seg[3:7, 3:7, 3:7] = 4
            seg[8:10, 8:10, 8:10] = 9
            nib.save(nib.Nifti1Image(seg, affine), raw / f"{case}-seg.nii.gz")
            result = preprocess.process((case, str(raw), str(root / "output"), (1., 1., 1.), False))
            output = root / "output" / case
            saved = nib.load(output / f"{case}-seg.nii.gz")
            zooms = list(map(float, saved.header.get_zooms()[:3]))
            labels = list(map(int, np.unique(np.asanyarray(saved.dataobj))))
            return dict(status="fail", backend=preprocess.BACKEND, process_status=result["status"],
                        target_spacing=[1, 1, 1], actual_header_spacing=zooms,
                        exact_1mm=bool(np.allclose(zooms, 1.0, atol=1e-6, rtol=0)),
                        same_shape_t2w_affine_shift_mm=10,
                        affine_mismatch_rejected=result["status"] != "ok",
                        illegal_input_label=9, output_labels=labels,
                        illegal_label_rejected=result["status"] != "ok")

    probe("preprocess_geometry_and_label_contract", geometry_and_labels)

    def finite_contract():
        with tempfile.TemporaryDirectory(prefix="ecnu_1mm_finite_") as temporary:
            root = Path(temporary)
            case = "BraTS-MET-FINITE-000"
            raw = root / case
            raw.mkdir()
            affine = np.eye(4)
            for modality in preprocess.MODALITIES:
                data = np.ones((8, 8, 8), dtype=np.float32)
                if modality == "t1c":
                    data[4, 4, 4] = np.nan
                nib.save(nib.Nifti1Image(data, affine), raw / f"{case}-{modality}.nii.gz")
            seg = np.zeros((8, 8, 8), dtype=np.uint8)
            seg[2:4, 2:4, 2:4] = 4
            nib.save(nib.Nifti1Image(seg, affine), raw / f"{case}-seg.nii.gz")
            result = preprocess.process((case, str(raw), str(root / "output"), (1., 1., 1.), False))
            path = root / "output" / case / f"{case}-t1c.nii.gz"
            finite = bool(np.isfinite(nib.load(path).get_fdata()).all()) if path.exists() else None
            return dict(status="pass" if result["status"] != "ok" else "fail",
                        process_status=result["status"], output_all_finite=finite)

    probe("preprocess_nonfinite_rejection", finite_contract)

    def network_cpu():
        model = networks.TimeConditionedUNet_NnU(in_channels=1, cond_channels=4,
                    input_size=64, gpu_memory_gb=48, noise_embedding_mode="continuous")
        model.train()
        seg = torch.zeros((1, 16, 16, 16), dtype=torch.int64)
        seg[:, 5:9, 5:9, 5:9] = 4
        cond = label_transform(seg).float()[None]
        x = torch.randn(1, 1, 16, 16, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        schedule = diffusion.make_noise_schedule("edm", sigma_data=1.0)
        loss = diffusion.edm_loss_fn(model, x, cond, schedule, "cpu")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite_gradients = all(torch.isfinite(g).all().item() for g in gradients)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 64, 64, 64), torch.zeros(1), torch.zeros(1, 4, 64, 64, 64))
        return dict(status="pass" if bool(torch.isfinite(loss)) and finite_gradients and bool(torch.isfinite(output).all()) else "fail",
                    params=sum(p.numel() for p in model.parameters()),
                    planned_channels=list(networks.plan_network_params(64, 48)[0]),
                    backward_shape=list(x.shape), forward_shape=list(output.shape),
                    loss=float(loss.detach()), finite_gradients=finite_gradients,
                    rc_condition_voxels=int(cond[:, 3].sum()), cuda_available=torch.cuda.is_available(),
                    full_size_batch16_amp_compile_stress_test="not_run")

    probe("edm_reference_network_cpu", network_cpu)

    def recovery():
        tree = ast.parse((SOURCE / "src/train/tumour_main_diffusion.py").read_text())
        checkpoint = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in node.targets))
        keys = [key.value for key in checkpoint.keys]
        trainer = (SOURCE / "src/train/tumour_main_diffusion.py").read_text()
        return dict(status="fail", checkpoint_keys=keys,
                    saves_amp_scaler="scaler" in keys, saves_rng=any("rng" in k for k in keys),
                    atomic_replace="os.replace" in trainer, term_handler="signal.signal" in trainer,
                    dataset_sha_locked=any("sha" in k or "hash" in k for k in keys),
                    lognsr_requested_exists=hasattr(diffusion, "lognsr_loss_fn"),
                    correct_logsnr_function_exists=hasattr(diffusion, "logsnr_loss_fn"))

    probe("recovery_and_optional_schedule", recovery)

    def split_contract():
        import pandas as pd
        frame = pd.DataFrame({"split": ["train", "train"]})
        requested = "val"
        if requested != "all" and "split" in frame.columns:
            if (frame["split"] == requested).any():
                frame = frame[frame["split"] == requested]
        invalid = {line.strip() for line in (SOURCE / "lists/t2w_invalid_cases.txt").read_text().splitlines()
                   if line.strip() and not line.startswith("#")}
        membership = PROJECT / "work_space/G1/03_experiments/03_diffusion_training/20260720_v3_multiserver/canonical/splits/current/dataset_membership.csv"
        with membership.open(newline="") as stream:
            original = list(csv.DictReader(stream))
        overlap = sorted(invalid & {row["source_case_id"] for row in original})
        plan_path = PROJECT / "work_space/S2/03_experiments/01_baselines_and_original_e/realonly_fold0/nnUNetPlans.json"
        plan = json.loads(plan_path.read_text())["configurations"]["3d_fullres"]
        return dict(status="fail", requested_split=requested, actual_evaluated_splits=frame["split"].tolist(),
                    imported_invalid_t2w_count=len(invalid),
                    invalid_overlap_with_original_complete926=overlap,
                    old_complete926_splits=dict(Counter(r["split"] for r in original)),
                    s2_original_e_spacing=plan["spacing"],
                    imported_preprocessor_spacing=[1, 1, 1],
                    s2_exact_grid_match=False)

    probe("split_and_s2_grid", split_contract)
    after = manifest(SOURCE)
    versions = {}
    for package in ("torch", "monai", "numpy", "nibabel", "scipy", "nnunetv2", "scikit-image"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    evidence = dict(**STATE, reviewed_at=utc(), status="not_ready_for_full_training",
                    code_root=str(SOURCE), source_unchanged=before == after,
                    environment=dict(executable=sys.executable, python=platform.python_version(),
                                     packages=versions), checks=checks,
                    boundaries=dict(server_access=False, training_submitted=False,
                                    real_dataset_download=False, installed_dependencies=False))
    comparison = []
    for path in sorted(SOURCE.rglob("*.py")):
        relative = path.relative_to(SOURCE)
        old_path = OLD / relative
        if old_path.is_file():
            new_blob, old_blob = path.read_bytes(), old_path.read_bytes()
            comparison.append(dict(path=relative.as_posix(), identical=new_blob == old_blob,
                                   imported_sha256=hashlib.sha256(new_blob).hexdigest(),
                                   previous_active_sha256=hashlib.sha256(old_blob).hexdigest()))
    (HERE / "SOURCE_COMPARISON.json").write_text(json.dumps(dict(**STATE, reviewed_at=utc(),
        previous_active_root=str(OLD), previous_scope="current_active_code_not_frozen_150k_runtime",
        files=comparison), indent=2, ensure_ascii=False) + "\n")
    (HERE / "READINESS_EVIDENCE.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    (HERE / "IMPORT_MANIFEST.json").write_text(json.dumps(dict(**STATE, captured_at=utc(),
                 source=str(SOURCE), **before), indent=2, ensure_ascii=False) + "\n")
    receipt = dict(**STATE, recorded_at=utc(), origin="/Users/hwaigc/Downloads/TumourDiffusion_1mm",
                   destination=str(SOURCE), origin_currently_exists=Path("/Users/hwaigc/Downloads/TumourDiffusion_1mm").exists(),
                   operation="directory_move", original_source_edits=False,
                   pre_move_digest_available=False, manifest_scope="post_move_import_baseline",
                   imported_source_unchanged_during_audit=before == after,
                   transfer_method_preserves_contents=True,
                   pre_move_byte_equality_independently_verified=False,
                   empty_original_directories_preserved=before["empty_directories"])
    (HERE / "IMPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return 1 if before != after or any(c["status"] == "probe_error" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
