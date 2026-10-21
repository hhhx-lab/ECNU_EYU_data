"""Scoped audited transport for the user-authorized R7 only."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
REMOTE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9')
PARENT_REMOTE = REMOTE.parent / 'g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8'
R6_REMOTE = REMOTE.parent / 'g1_diffusion_1mm_mask_diagnostics_20261015_r6'
RECON = REMOTE.parent / 'g1_vae_epoch004_full_reconstruction_20261008_r2'
SOURCE_CODE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3')
spec = importlib.util.spec_from_file_location('g1_r7_access', PROJECT / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = ROOT


def transport(action):
    return m.Access(action)
