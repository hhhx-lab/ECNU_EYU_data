"""Independent evaluation evidence; reference the immutable completed R9 package."""
import hashlib
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODS = ('t1c', 't1n', 't2w', 't2f')
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
R9_SHA = '80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix='.pending_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(name)


def settings():
    import sys
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    assert cfg['remote_root'] == str(ROOT) and cfg['inference'] == dict(steps=18,cfg=1.,churn=0.,dtype='float32',crop=64)
    assert cfg['candidate_steps'] == list(range(20000,200001,20000)) and cfg['seeds'] == list(range(20261013,20261017))
    r9 = Path(cfg['training_root'])
    assert r9.resolve() == r9 and sha(r9 / 'plan/PACKAGE_MANIFEST.json') == R9_SHA
    package = json.loads((r9 / 'plan/PACKAGE_MANIFEST.json').read_text())
    for n, h in package['files'].items():
        assert sha(r9 / n) == h, 'R9 frozen package drift: ' + n
    sys.path.insert(0, str(r9 / 'runtime/upstream_source'))
    own = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    for n, h in own['files'].items():
        assert sha(ROOT / n) == h, 'Evaluation package drift: ' + n
    return cfg, r9


def checkpoint(path, contract, step, modality):
    import torch
    v = torch.load(path, map_location='cpu', weights_only=False)
    assert v['checkpoint_schema'] == 'g1_complete_boundary_v1' and v['safe_boundary'] is True
    assert v['global_step'] == step and v['modality'] == modality
    assert v['training_config'] == contract['config'] and v['data_identity'] == contract['identity']
    assert v['step_semantics'] == 'completed_optimizer_updates_v2'
    assert v['n_steps'] == contract['config']['n_steps'] == v['schedule_config']['n_steps']
    for key in ('state_dict', 'optimizer', 'scaler', 'rng', 'sampler', 'schedule_config'):
        assert key in v
    assert all(torch.isfinite(t).all() for t in v['state_dict'].values())
    s = v['schedule_config']; c = contract['config']
    assert s['name'] == c['noise_schedule'] == 'edm'
    for key in ('sigma_data', 'sigma_min', 'sigma_max', 'rho', 'snr_shift', 'n_steps'):
        assert s[key] == c[key], 'Schedule drift: ' + key
    return v


def model_from(v, device):
    from types import SimpleNamespace
    from src.networks.DiffusionNetwork import get_diffusion_network
    model = get_diffusion_network(SimpleNamespace(**v['training_config']), n_steps=v['n_steps'])
    model.load_state_dict(v['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 4475999
    return model.to(device).eval()


def frozen_model_module(r9):
    spec = importlib.util.spec_from_file_location('r9_edm_frozen', r9 / 'runtime/upstream_source/model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
