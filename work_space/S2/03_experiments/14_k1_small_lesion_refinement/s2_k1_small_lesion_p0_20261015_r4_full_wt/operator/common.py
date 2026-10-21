"""Use the existing audited credential/VPN transport without modifying it."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/' + ROOT.name
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
HELPER = S2 / '03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'


def transport():
    spec = importlib.util.spec_from_file_location('audited_transport', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module
