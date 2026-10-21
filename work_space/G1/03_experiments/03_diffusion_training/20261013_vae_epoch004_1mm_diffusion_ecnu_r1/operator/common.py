"""Shared constants and the existing audited ECNU transport; no credentials here."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
REMOTE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261013_r1')
RECON = REMOTE.parent / 'g1_vae_epoch004_full_reconstruction_20261008_r2'
SOURCE_CODE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3')
PLAN_DIR = PROJECT / 'work_space/G1/05_documents/02_plans/20261013_epoch004_recon_1mm_diffusion_ecnu'
PLAN_NAME = 'G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md'
MEMBERSHIP = PROJECT / 'work_space/G2/02_dataset_release/20260919_archived_local_release/splits/splits_final_train_val_test_membership.csv'
OVERLAY = PROJECT / 'work_space/G2/02_dataset_release/20260919_archived_local_release/manifests/corrected_label_overlay.csv'
EXPECTED_PLAN = 'fe538efa6c4d1c2fae4aed2003270a5dd785366061e8ddede0d71d17c87c5353'
EXPECTED_MEMBERSHIP = '0a533791617d44d1ffe1afefe04b1154db2e7f8cf6db9f3421a8c1ff1aec7f9a'
CHECKPOINT_SHA = '5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'
SOURCE_CSV_SHA = '0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'
spec = importlib.util.spec_from_file_location('ecnu_access_g1_1mm', PROJECT / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = ROOT


def transport(action):
    return m.Access(action)
