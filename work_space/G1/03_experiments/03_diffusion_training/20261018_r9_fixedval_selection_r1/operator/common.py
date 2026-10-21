"""Use the established strict-host ECNU transport without changing it."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[4]
REMOTE=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_r9_fixedval_selection_20261018_r1')
spec=importlib.util.spec_from_file_location('r9_eval_access',PROJECT/'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=ROOT


def transport(action):
    return m.Access(action)
