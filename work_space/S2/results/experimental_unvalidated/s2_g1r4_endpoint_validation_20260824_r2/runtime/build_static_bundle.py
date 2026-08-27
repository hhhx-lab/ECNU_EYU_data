#!/usr/bin/env python3
"""Build and freeze the 2026-08-24 endpoint-validation static bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_STATUS = "experimental_unvalidated"
FORMAL_GATE_STATUS = "not_run_not_passed"
REMOTE_PROJECT_ROOT = Path("/public/home/zqchen/projects/ECNU_EYU_data")
REMOTE_RESULT_ROOT = REMOTE_PROJECT_ROOT / (
    "work_space/S2/results/experimental_unvalidated/"
    "s2_g1r4_endpoint_validation_20260824_r2"
)
REMOTE_REAL_ROOT = Path("/public/home/zqchen/data")
EXPECTED = {
    "r4_case_list": "53080017b900e0bf2b09fcb129d58d0e2383b7b1090f50eb6328f1247848f2c6",
    "r4_plan": "26dc9ca8d730d13d021756db0e45ac5aaf65dfc5d6cc742e6b136e2c8bf93516",
    "master_mapping": "b1ff36ea927a3d5c8d4aa976ce5533d320e16c7801a9a30db22e8ecbb6d64672",
    "master_split": "5b77bf4ae92e08df2495a17ace104d5e4160c77ed718bba1a7f2cdde87adfcc6",
    "missing_eval_manifest": "24c84e1b90cb4ebfd1f034e311130dfd2ea714c288f09728c8da46c7aeadbc37",
    "R": "bec0beead1b6ab6beda5e10c9c92e192af833fa82f794277d50f81d277a922a1",
    "B": "78eccc59f9217a529cafdd522733de9a1578f0e96d8765ee7c48731027824db5",
    "E": "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267",
    "FixV3": "b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e",
    "G1_VAE": "004dbaa00669685b1f24fbaf6044d5f592f4a7ae6156a482be34235724175294",
    "G1_ENCDEC": "aed9bc96d6cb285b60d092fd48f572c60f76dba2d654dba4e60754ad55c0c558",
    "G1_BBDM": "969eef82241a9500d843bb38b4fcf3230eb0b55ecfb1876aa83c65b59ec05541",
    "r1_remote_preflight_failure": "dabd787ced86ad2ab2131a76bc31f0dc8b42cfc680f394409d76f6153456d0cb",
}
EXPECTED_G1_RUNTIME = {
    "configs.py": "105da207a98a3195fa131679ba7ba3e960e5fb03d2441f878e57a40f58917863",
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/bbdm/bb_scheduler.py": "2b701c696b4bf19628caf5635cd806075b640239519b710ad43b0b59449cf13b",
    "models/bbdm/condition_tokens.py": "729f2a1759fa0b8fbe63ec8837528e080ce68391b7f1e0ee1c8cb6c1f670282b",
    "models/bbdm/unet.py": "caca4abbfba78fe51dde364747786659c1593ce8a93ffff475c879ec3964135d",
    "models/encdec/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/encdec/unet.py": "2de249accbab3ef7897527b662c5f64fe89ed3ee41a85ce2d1578bffbdcd3187",
    "synthesis/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "synthesis/pipeline.py": "0097c284d5d636dfc93f1dfd8461f5d02d54b666aa337a8f635a1c5b4853d396",
    "synthesis/segment_brain_mask.py": "79d42739db131778ba4d39c99b91cff0fdddcb77741e28cb25137d192c90a866",
    "synthesis/spatial.py": "b16b80697fd0a9394eca81833356c79b2ba81a1cfe92669d1b414ce3b6569afb",
    "synthesis/utils.py": "5d5dd28de7dfed96fe6c3988da52563f37940e89850fc2b84accafd8bd55d3ab",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    json.loads(path.read_text(encoding="utf-8"))


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def remote_source_paths(case_id: str) -> dict[str, str]:
    case_root = REMOTE_REAL_ROOT / case_id
    return {
        f"{modality}_source_path": str(case_root / f"{modality}.nii.gz")
        for modality in ("t1n", "t1c", "t2w", "t2f", "seg")
    }


def cohort_row(
    index: int,
    mapping: dict[str, str],
    *,
    source_split: str,
    t2w_role: str,
    synthesized_root: Path,
) -> dict[str, Any]:
    case_id = mapping["source_case_id"].strip()
    values: dict[str, Any] = {
        "cohort_index": index,
        "source_case_id": case_id,
        "nnunet_case_id": mapping["nnunet_case_id"].strip(),
        "source_split": source_split,
        "t2w_status": mapping["t2w_status"].strip(),
        "completion_required": mapping["completion_required"].strip(),
        "t2w_role": t2w_role,
        "source_t2w_allowed": t2w_role == "authentic",
        "synthesized_t2w_path": str(synthesized_root / f"{case_id}-t2w.nii.gz"),
    }
    values.update(remote_source_paths(case_id))
    if t2w_role != "authentic":
        values["t2w_source_path"] = ""
    return values


def local_paths(project_root: Path) -> dict[str, Path]:
    return {
        "r4_case_list": project_root / (
            "work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/"
            "val_fixed_103_case_list.csv"
        ),
        "r4_plan": project_root / (
            "work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/"
            "ENSEMBLE_R4_EXECUTION_PLAN.json"
        ),
        "master_mapping": project_root / "work_space/G2/results/manifests/nnunet_case_mapping_master.csv",
        "master_split": project_root / "work_space/G2/results/splits/splits_master_train_val_test.json",
        "missing_eval_manifest": project_root / (
            "work_space/G2/results/manifests/synthetic_accepted_evaluation_manifest_run_3104668.csv"
        ),
        "selection": project_root / "work_space/S2/results/s2_small_lesion_ablation_20260721/checkpoint_selection.json",
        "R": project_root / "work_space/S2/results/checkpoint_final.pth",
        "B": project_root / (
            "work_space/S2/results/s2_completion_dataset264_t2w_20260720/"
            "fold_0/checkpoint_final.pth"
        ),
        "E": project_root / (
            "work_space/S2/results/s2_small_lesion_ablation_20260721/"
            "remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth"
        ),
        "FixV3": project_root / (
            "work_space/S2/results/experimental_unvalidated/"
            "s2_met_aug_fix_v3_full200_attempt11_20260823_r1/artifacts/"
            "nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/"
            "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__"
            "nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
        ),
    }


def verify_hashes(paths: dict[str, Path]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for key in ("r4_case_list", "r4_plan", "master_mapping", "master_split", "missing_eval_manifest", "R", "B", "E", "FixV3"):
        path = paths[key]
        require(path.is_file(), f"missing frozen input: {key}: {path}")
        observed[key] = sha256_file(path)
        require(observed[key] == EXPECTED[key], f"SHA drift for {key}: {observed[key]}")
    return observed


def static_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "STATIC_SHA256SUMS.txt":
            continue
        if any(part in {"logs", "uploads", "model_snapshots", "cohorts"} for part in path.relative_to(root).parts):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    root = args.experiment_root.expanduser().resolve()
    require(root.is_dir(), f"experiment root does not exist: {root}")
    manifests = root / "manifests"
    evidence = root / "evidence"
    manifests.mkdir(exist_ok=False)
    evidence.mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=False)

    paths = local_paths(project_root)
    observed_hashes = verify_hashes(paths)
    r1_failure = root.parent / (
        "s2_g1r4_endpoint_validation_20260824_r1/evidence/"
        "REMOTE_STATIC_PREFLIGHT_FAILED.json"
    )
    require(r1_failure.is_file(), "r1 remote preflight failure evidence missing")
    require(
        sha256_file(r1_failure) == EXPECTED["r1_remote_preflight_failure"],
        "r1 remote preflight failure evidence drift",
    )
    r4_plan = json.loads(paths["r4_plan"].read_text(encoding="utf-8"))
    require(r4_plan["spatial_preprocessing"] == "foreground_centered_isotropic_resample_v1", "r4 spatial contract drift")
    require(r4_plan["case_count"] == 103, "r4 case count drift")
    require(r4_plan["operator_approved"] is False, "r4 approval drift")
    require(r4_plan["bbdm_s"] == 0.01, "r4 BBDM s drift")
    for name in ("vae", "encdec", "bbdm"):
        binding = r4_plan["checkpoints"][name]
        require(binding["sha256"] == EXPECTED[f"G1_{name.upper()}"], f"r4 {name} checkpoint SHA drift")
        checkpoint_path = Path(binding["path"])
        require(checkpoint_path.is_absolute(), f"r4 {name} checkpoint path is not absolute")
        require(checkpoint_path.is_relative_to(REMOTE_PROJECT_ROOT), f"r4 {name} checkpoint escaped project root")

    mapping_rows = read_csv(paths["master_mapping"])
    by_source = {row["source_case_id"].strip(): row for row in mapping_rows}
    require(len(by_source) == len(mapping_rows) == 1295, "master mapping duplicate/count drift")
    r4_ids = [row["id"].strip() for row in read_csv(paths["r4_case_list"])]
    require(len(r4_ids) == len(set(r4_ids)) == 103, "r4 fixed case list drift")
    require(not set(r4_ids) - set(by_source), "r4 IDs missing from master mapping")

    missing_rows = read_csv(paths["missing_eval_manifest"])
    val_rows = [row for row in missing_rows if row["source_split"].strip() == "val"]
    test_rows = [row for row in missing_rows if row["source_split"].strip() == "test"]
    require(len(val_rows) == 27 and len(test_rows) == 26 and len(missing_rows) == 53, "missing cohort count drift")
    for row in missing_rows:
        require(truthy(row["accepted_for_evaluation"]), "unaccepted missing-T2W case")
        require(not truthy(row["accepted_for_training"]), "evaluation case marked for training")
        require(truthy(row["source_is_fake_t2w_case"]), "case is not missing/fake T2W")
        require(truthy(row["source_completion_mode"]), "case is not completion mode")
        require(row["source_case_id"] in by_source, "missing case absent from master mapping")

    val_ids = {row["source_case_id"] for row in val_rows}
    test_ids = {row["source_case_id"] for row in test_rows}
    require(not (set(r4_ids) & val_ids), "fixed103 overlaps val27")
    require(not (set(r4_ids) & test_ids), "fixed103 overlaps test26")
    require(not (val_ids & test_ids), "val27 overlaps test26")

    r4_synth_root = REMOTE_PROJECT_ROOT / (
        "work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/"
        "ensemble_r4/synthesized"
    )
    val_synth_root = REMOTE_RESULT_ROOT / "cohorts/val27/synthesis_r4/synthesized"
    test_synth_root = REMOTE_RESULT_ROOT / "cohorts/test26/synthesis_r4/synthesized"
    fixed103 = [
        cohort_row(i, by_source[case_id], source_split="fixed_validation", t2w_role="authentic", synthesized_root=r4_synth_root)
        for i, case_id in enumerate(r4_ids)
    ]
    val27 = [
        cohort_row(i, by_source[row["source_case_id"]], source_split="val", t2w_role="r4_ensemble_synthesized", synthesized_root=val_synth_root)
        for i, row in enumerate(sorted(val_rows, key=lambda item: item["source_case_id"]))
    ]
    test26 = [
        cohort_row(i, by_source[row["source_case_id"]], source_split="test", t2w_role="r4_ensemble_synthesized", synthesized_root=test_synth_root)
        for i, row in enumerate(sorted(test_rows, key=lambda item: item["source_case_id"]))
    ]
    for row in fixed103:
        require(row["t2w_status"] == "authentic" and not truthy(row["completion_required"]), "fixed103 T2W composition drift")
    for row in [*val27, *test26]:
        require(row["t2w_status"] == "fake_or_broken" and truthy(row["completion_required"]), "missing cohort T2W composition drift")

    manifest_paths = {
        "fixed103": manifests / "FIXED103_REAL_SYNTHETIC_CASES.csv",
        "val27": manifests / "VAL27_MISSING_T2W_CASES.csv",
        "test26": manifests / "TEST26_LOCKED_MISSING_T2W_CASES.csv",
    }
    write_csv_exclusive(manifest_paths["fixed103"], fixed103)
    write_csv_exclusive(manifest_paths["val27"], val27)
    write_csv_exclusive(manifest_paths["test26"], test26)
    manifest_hashes = {key: sha256_file(path) for key, path in manifest_paths.items()}

    g1_code_root = root / "runtime/g1_r4_frozen"
    g1_code_files = tuple(
        str(path.relative_to(g1_code_root))
        for path in sorted(g1_code_root.rglob("*.py"))
    )
    require(set(g1_code_files) == set(EXPECTED_G1_RUNTIME), "frozen G1 runtime file set drift")
    g1_code_hashes: dict[str, str] = {}
    for relative in g1_code_files:
        path = g1_code_root / relative
        require(path.is_file(), f"missing G1 runtime source: {path}")
        g1_code_hashes[relative] = sha256_file(path)
        require(
            g1_code_hashes[relative] == EXPECTED_G1_RUNTIME[relative],
            f"historical G1 runtime SHA drift: {relative}",
        )

    runtime_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for folder in (root / "runtime", root / "slurm")
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    created_at = datetime.now(timezone.utc).isoformat()
    plan = {
        "schema_version": 1,
        "artifact_status": ARTIFACT_STATUS,
        "operator_approved": False,
        "formal_gate_status": FORMAL_GATE_STATUS,
        "created_at_utc": created_at,
        "experiment": "g1_r4_to_s2_endpoint_validation_20260824_r2",
        "training_required": False,
        "remote_result_root": str(REMOTE_RESULT_ROOT),
        "spatial_contract": {
            "name": "foreground_centered_isotropic_resample_v1",
            "target_shape": [256, 256, 160],
            "base_spacing_mm": 1.0,
            "margin_mm": 5.0,
            "native_geometry_restored": True,
            "reference_segmentation_used_for_offline_support_audit": True,
            "reference_segmentation_is_model_input": False,
        },
        "channel_contract": {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"},
        "cohorts": {
            "val27": {"count": 27, "role": "internal_missing_t2w_endpoint_validation", "manifest": str(REMOTE_RESULT_ROOT / "manifests/VAL27_MISSING_T2W_CASES.csv"), "manifest_sha256": manifest_hashes["val27"], "authentic_t2w_counterfactual_available": False},
            "fixed103": {"count": 103, "role": "authentic_vs_synthesized_t2w_paired_validation", "manifest": str(REMOTE_RESULT_ROOT / "manifests/FIXED103_REAL_SYNTHETIC_CASES.csv"), "manifest_sha256": manifest_hashes["fixed103"], "same_cases_as_fix_v3_training_validation": True},
            "test26": {"count": 26, "role": "locked_missing_t2w_endpoint_test", "manifest": str(REMOTE_RESULT_ROOT / "manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv"), "manifest_sha256": manifest_hashes["test26"], "unlocked_only_after_val27_and_fixed103_pass": True},
        },
        "checkpoints": {
            "R": {"role": "real_only_reference", "sha256": EXPECTED["R"], "remote_model_root": str(REMOTE_RESULT_ROOT / "model_snapshots/R")},
            "B": {"role": "completion_standard_loss", "sha256": EXPECTED["B"], "remote_model_root": str(REMOTE_RESULT_ROOT / "model_snapshots/B")},
            "E": {"role": "completion_focal_loss", "sha256": EXPECTED["E"], "remote_model_root": str(REMOTE_PROJECT_ROOT / "work_space/S2/data/ecnu_completion_emergency/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres")},
            "FixV3": {"role": "online_diffusion_augmented_focal_full200", "sha256": EXPECTED["FixV3"], "remote_model_root": str(REMOTE_PROJECT_ROOT / "work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/data/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres")},
        },
        "g1_r4_checkpoints": {
            name: {
                "remote_path": r4_plan["checkpoints"][name]["path"],
                "sha256": r4_plan["checkpoints"][name]["sha256"],
            }
            for name in ("vae", "encdec", "bbdm")
        }
        | {"bbdm_s": 0.01, "ensemble": "voxelwise_mean_encdec_bbdm"},
        "g1_runtime_code_sha256": g1_code_hashes,
        "g1_runtime_provenance": {
            "mode": "bundled_historical_r4_runtime_snapshot",
            "remote_original_root": str(
                REMOTE_PROJECT_ROOT
                / "work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3"
            ),
            "bundled_remote_root": str(REMOTE_RESULT_ROOT / "runtime/g1_r4_frozen"),
            "latest_original_mtime": "2026-07-17T10:58:15+08:00",
            "r4_job_id": "3391396",
            "r4_job_date_utc": "2026-08-13",
            "shared_runtime_used_directly": False,
            "r1_remote_preflight_failure_sha256": EXPECTED["r1_remote_preflight_failure"],
        },
        "evaluation": {"package": "BraTS-evaluation==0.0.8", "panoptica": "2.1.0", "numpy": "1.26.4", "configuration": "mets", "volume_threshold_voxels": 27, "overlap_threshold": 0.2, "paired_bootstrap_replicates": 20000, "paired_bootstrap_seed": 20260824},
        "stages": ["val27_r4_synthesis", "val27_four_models", "fixed103_real_vs_synthetic", "test26_locked_endpoint"],
        "freeze_rules": ["no_retraining", "no_post_result_tuning", "no_checkpoint_selection_from_val27_or_fixed103", "test26_locked_until_prior_gates_pass", "no_old_synthesized_t2w_for_val27_or_test26", "same_fixv3_checkpoint_for_fixed103_real_and_synthetic"],
        "interpretation_boundaries": ["g1_image_quality_is_not_segmentation_benefit", "val27_has_no_authentic_t2w_counterfactual", "fixed103_answers_t2w_replacement_effect_only", "online_diffusion_augmentation_and_offline_completion_are_reported_separately", "no_nnunet_superiority_claim_from_g1_reconstruction_metrics"],
        "source_artifact_sha256": observed_hashes,
        "runtime_sha256": runtime_hashes,
    }
    plan_path = root / "ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R2.json"
    write_json_exclusive(plan_path, plan)
    preflight = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": ARTIFACT_STATUS,
        "operator_approved": False,
        "formal_gate_status": FORMAL_GATE_STATUS,
        "generated_at_utc": created_at,
        "counts": {"fixed103": 103, "val27": 27, "test26": 26},
        "cohorts_pairwise_disjoint": True,
        "fixed103_matches_r4_case_list": True,
        "fixed103_authentic_t2w_count": 103,
        "val27_completion_required_count": 27,
        "test26_completion_required_count": 26,
        "old_synthesized_t2w_reuse_allowed": False,
        "channel_contract_validated": True,
        "g1_runtime_code_file_count": len(g1_code_hashes),
        "g1_runtime_bundled_snapshot": True,
        "r1_remote_preflight_failure_sha256": EXPECTED["r1_remote_preflight_failure"],
        "checkpoint_sha256_validated_locally": {key: observed_hashes[key] for key in ("R", "B", "E", "FixV3")},
        "manifest_sha256": manifest_hashes,
        "plan_sha256": sha256_file(plan_path),
    }
    preflight_path = evidence / "STATIC_PREFLIGHT.json"
    write_json_exclusive(preflight_path, preflight)

    sums_path = root / "STATIC_SHA256SUMS.txt"
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in static_files(root)]
    with sums_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(preflight, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
