"""Strict frozen identities for the independent comparison."""
import hashlib
import importlib.util
import json
import os
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODS = ('t1c', 't1n', 't2w', 't2f')
SEEDS = (20261013, 20261014, 20261015, 20261016)
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix='.pending_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)
        d = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        os.unlink(name)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def cfg():
    return json.loads((ROOT / 'RUN.json').read_text())


def verify_package(root, manifest, expected):
    p = Path(root) / manifest
    assert sha(p) == expected, 'Package manifest identity drift'
    v = json.loads(p.read_text())
    for n, h in v['files'].items():
        assert not Path(n).is_absolute() and '..' not in Path(n).parts
        assert sha(Path(root) / n) == h, 'Frozen source drift: ' + n


def parent():
    c = cfg()
    p = Path(c['reference_root'])
    verify_package(p, 'PACKAGE_MANIFEST.json', c['reference_package_sha256'])
    verify_package(c['training_root'], 'plan/PACKAGE_MANIFEST.json', c['training_package_sha256'])
    assert sha(p / 'reference/REPORT.json') == c['reference_report_sha256']
    assert sha(p / 'selection/SELECTION.json') == c['selection_sha256']
    assert not (p / 'reference/FAILED.json').exists()
    done = json.loads((p / 'reference/COMPLETE.ok').read_text())
    assert done['status'] == 'pass' and done['report_sha256'] == c['reference_report_sha256']
    v = json.loads((p / 'reference/REPORT.json').read_text())
    assert v['cases'] == 103 and v['positive_cases'] == 94 and v['negative_cases'] == 9 and v['errors'] == []
    records = v['records']
    assert len(records) == len({r['case'] for r in records}) == 103
    selected = json.loads((p / 'selection/SELECTION.json').read_text())['modalities']
    positive = {r['case'] for r in records if r['positive']}
    for m in MODS:
        assert set(selected[m]['cases']) == positive
        assert selected[m]['selected_step'] == c['models']['r9'][m]['step']
        assert selected[m]['selected_checkpoint_sha256'] == c['models']['r9'][m]['sha256']
    return v


def own_package():
    assert str(ROOT) == cfg()['remote_root']
    expected = os.environ.get('G1_PAIR_PACKAGE_SHA256', '')
    assert len(expected) == 64 and all(v in '0123456789abcdef' for v in expected), 'Expected package SHA missing'
    verify_package(ROOT, 'PACKAGE_MANIFEST.json', expected)
    return expected


def completed(phases):
    jobs = json.loads((ROOT / 'evidence/JOBS.json').read_text())['jobs']
    ids = [jobs[p] for p in phases]
    p = subprocess.run(['sacct','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode'],
                       capture_output=True,text=True,timeout=30)
    assert p.returncode == 0
    rows = {s.split('|')[0]:s.split('|')[1:3] for s in p.stdout.splitlines() if len(s.split('|')) >= 3}
    assert all(rows.get(j)==['COMPLETED','0:0'] and rows.get(j+'.batch')==['COMPLETED','0:0'] for j in ids), 'Dependency accounting not completed0:0'


def load_model(info, version, device='cpu'):
    import torch
    c = cfg()
    p = Path(info['path'])
    assert p.is_file() and not p.is_symlink() and sha(p) == info['sha256'] and p.stat().st_size == info['bytes']
    v = torch.load(p, map_location='cpu', weights_only=False)
    assert v['global_step'] == info['step'] and v['modality'] == info['modality']
    assert v['crop_size'] == 64 and v['in_channels'] == 5 and v['out_channels'] == 1
    assert v['noise_embedding_mode'] == 'continuous' and v['time_ch_count'] == 8
    s = v['schedule_config']
    for key, value in c['schedule'].items():
        assert s[key] == value, 'Schedule identity drift: ' + key
    if version == 'r9':
        sys.path.insert(0, str(Path(c['training_root']) / 'runtime/upstream_source'))
        helper = module('shared_r9_helper', Path(c['reference_root']) / 'runtime/common.py')
        contract_path = Path(info.get('contract_path', str(Path(c['training_root']) / 'training' / info['modality'] / 'RUN_CONTRACT.json')))
        contract = json.loads(contract_path.read_text())
        v = helper.checkpoint(p, contract, info['step'], info['modality'])
        model = helper.model_from(v, device)
        assert v['normalization'] == 'nnunet'
    else:
        assert version == 'july150k' and v['normalization'] == 'zscore' and v['generator_type'] == 'Unet_NnU'
        assert v['network_channels'] == [48, 96, 192, 384] and v['network_strides'] == [2, 2, 2]
        network = module('shared_old_network', ROOT / 'runtime/legacy_network.py')
        model = network.TimeConditionedUNet_NnU(in_channels=1, cond_channels=4, out_channels=1,
            input_size=64, gpu_memory_gb=48, network_channels=v['network_channels'],
            network_strides=v['network_strides'], time_ch_count=8, noise_embedding_mode='continuous')
        state = {k.removeprefix('_orig_mod.'): t for k, t in v['state_dict'].items()}
        assert len(state) == len(v['state_dict'])
        assert all(torch.isfinite(t).all() for t in state.values())
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
    assert sum(t.numel() for t in model.parameters()) == 4475999
    return model


def sampler():
    c = cfg()
    edm = module('shared_frozen_edm', Path(c['training_root']) / 'runtime/upstream_source/model.py')
    schedule = edm.make_noise_schedule(schedule='edm', **{k:v for k,v in c['schedule'].items() if k != 'name'})
    return edm, schedule
