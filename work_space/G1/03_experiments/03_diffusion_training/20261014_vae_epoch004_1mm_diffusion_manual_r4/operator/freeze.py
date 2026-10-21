"""Freeze independent R4 from R3's immutable payload, not by rewriting old evidence."""
import json
import shutil
from common import ROOT, REMOTE, PARENT, m


def main():
    package = json.loads((PARENT / 'plan/PACKAGE_MANIFEST.json').read_text())
    assert len(package['files']) == 55
    assert all(m.sha(PARENT / p) == v for p, v in package['files'].items())
    for relative in package['files']:
        if relative == 'README.md' or relative in ('config/RUN.json', 'plan/EXECUTION_AUTHORIZATION.json') or relative.startswith('slurm/'):
            continue
        dest = ROOT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        shutil.copy2(PARENT / relative, dest)
    cfg = json.loads((PARENT / 'config/RUN.json').read_text())
    cfg.update(attempt='manual_r4', remote_root=str(REMOTE))
    m.exclusive(ROOT / 'config/RUN.json', cfg)
    diagnosis = next((ROOT / 'operator').glob('diagnose_parent_*/DIAGNOSIS.json'))
    result = json.loads(diagnosis.read_text())
    assert result['probe']['returncode'] == 0 and not result['probe']['bounded_timeout']
    assert result['parent_canonical_jobs'] == [] and not result['root_exists']
    auth = json.loads((PARENT / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
    auth.update(attempt='manual_r4', retry_authorization='修复预检超时并建立独立新批次提交',
                retry_scope='one new immutable R4 deployment, one preprocessing and four dependent training jobs',
                parent_package_sha256=m.sha(PARENT / 'plan/PACKAGE_MANIFEST.json'),
                parent_failure_local_sha256=m.sha(PARENT / 'operator/submit_20261014T065242Z/FAILED.json'),
                parent_failure_remote_sha256=result['parent_failure']['sha256'],
                diagnosis_sha256=m.sha(diagnosis), training_config_sha256=m.sha(ROOT / 'config/RUN.json'))
    m.exclusive(ROOT / 'plan/EXECUTION_AUTHORIZATION.json', auth)
    m.exclusive(ROOT / 'plan/PARENT_R3_AND_DIAGNOSIS.json', {**m.META,
        'parent_package_sha256': auth['parent_package_sha256'], 'parent_files': package['files'],
        'diagnosis_sha256':m.sha(diagnosis), 'probe_elapsed_seconds': result['probe']['elapsed_seconds'],
        'remote_test_scope':'7 original R3 CPU tests, no acceptance claim',
        'fix':'OMP/MKL/OPENBLAS/NUMEXPR=1 and CUDA disabled only in CPU probes; unbuffered phase logs and bounded300s',
        'causal_limit':'old timeout lacks stage output; no proof of a single causal library'})
    slurm = ROOT / 'slurm'
    slurm.mkdir(exist_ok=False)
    for path in (PARENT / 'slurm').glob('*.slurm'):
        original = path.read_text()
        value = original.replace(str(json.loads((PARENT / 'config/RUN.json').read_text())['remote_root']), str(REMOTE))
        value = value.replace('g1d1m3', 'g1d1m4').replace('G1_R3_BASH_U_DRY_RUN_PASS', 'G1_R4_BASH_U_DRY_RUN_PASS')
        if path.stem == 'prepare':
            value = value.replace('unset PYTHONPATH\n', "unset PYTHONPATH\nexport CUDA_VISIBLE_DEVICES=''\n")
        with (slurm / path.name).open('x') as h:
            h.write(value)
    print(json.dumps({'status':'R4_frozen','unchanged_training': cfg['training'], 'parent_files':55}))


if __name__ == '__main__':
    main()
