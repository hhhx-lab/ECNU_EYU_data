#!/bin/bash
# Derived verbatim exports from the frozen source Slurm; outputs alone are redirected.
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k1_175_r3
PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1
REPO="$ROOT/runtime/repository"
RESULTS="$RESUME_ROOT/results/K1"
FOLD="$RESULTS/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175__nnUNetPlans__3d_fullres/fold_0"
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:$PATH"
export MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_extTrainer="$REPO"
export nnUNet_raw="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw"
export nnUNet_preprocessed="$PARENT/data/nnUNet_preprocessed"
export nnUNet_results="$RESULTS"
export BRATS_S2_REPO_DIR="$REPO"
export BRATS_SPLIT_DIR="$PARENT/inputs/splits"
export S2_EXPERIMENT_MODE="met_aug_k1245_20260929_k1_175_r3_K1"
export S2_CONTINUE=0 S2_SKIP_COMPLETED=0 S2_MET_AUG_ENABLE=1
export S2_MET_AUG_COMPONENT_MANIFEST="$ROOT/pools/rc27/component_manifest.json"
export S2_MET_AUG_ROUTE_CONFIG="$ROOT/config/K1245_ROUTE_K1.json"
export S2_MET_AUG_VALID_MASK_MANIFEST="$PARENT/route_rebuild_20260812_r1/valid_masks/valid_mask_manifest.json"
export S2_MET_AUG_FIX_V3_CALIBRATION="$PARENT/calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
export S2_MET_AUG_ORIGINAL_E_CHECKPOINT="$PARENT/frozen/original_e/checkpoint_final.pth"
export S2_MET_AUG_G1_CODE_DIR="$PARENT/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="$PARENT/frozen/g1"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="$PARENT/frozen/g1/checkpoint_selection.json"
export S2_MET_AUG_G2_QC_GATE="$PARENT/frozen/g1/g2_diffusion_qc_gate.json"
export S2_MET_AUG_AUDIT_PATH="$FOLD/met_aug_events.jsonl"
export S2_MET_AUG_K1245_PLAN="$ROOT/K1245_K1_175_EXECUTION_PLAN_20260929_R3.json"
export S2_K1245_COMPONENT_MANIFEST="$ROOT/pools/rc27/component_manifest.json"
export S2_K1245_CONFIG="$ROOT/config/K1245_ROUTE_K1.json"
export S2_K1245_PLAN="$ROOT/K1245_K1_175_EXECUTION_PLAN_20260929_R3.json"
export S2_K1245_SMOKE_AUTHORIZATION="$ROOT/evidence/K1_175_TRAINING_AUTHORIZATION.json"
export S2_K1245_LEGACY_COMPONENT_MANIFEST="$PARENT/route_rebuild_20260812_r1/component_pool/component_manifest.json"
export S2_K1245_EPOCHS=175 S2_K1245_INITIAL_LR=0.001 S2_K1245_SAVE_EVERY=25
export S2_COMPLETION_EPOCHS=175 S2_COMPLETION_INITIAL_LR=0.001 S2_COMPLETION_SAVE_EVERY=25
export S2_SCREEN_EPOCHS=175 S2_SCREEN_INITIAL_LR=0.001 S2_SCREEN_SAVE_EVERY=25
export S2_PAIRED_TRAINING_SEED=20260724
export S2_FOCAL_GAMMA=2.0 CUBLAS_WORKSPACE_CONFIG=:4096:8 nnUNet_compile=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export S2_CONTINUE=1
export S2_EXPERIMENT_MODE=met_aug_k1_175_resume_20261002_r1_K1
