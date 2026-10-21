"""Freeze once, upload one lightweight payload, independently hold-audit-release."""
import ast
import base64
import json
import os
import shlex
import subprocess

from common import REMOTE, ROOT, m, run_remote


def main():
    assert not (ROOT / 'plan/PACKAGE_MANIFEST.json').exists(), 'Frozen attempt cannot submit again'
    assert not (ROOT / 'plan/SUBMISSION_RESERVATION.json').exists(), 'Reserved attempt cannot retry automatically'
    assert (ROOT / 'config/SOURCE_RECORDS.json').is_file(), 'Full original source audit required'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    test = subprocess.run(['conda', 'run', '--no-capture-output', '-n', 'g1_t2w_bbdm', 'python', '-B',
                           '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', 'test*.py', '-v'],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert test.returncode == 0, test.stdout + test.stderr
    assert 'Ran 0 tests' not in test.stderr
    for p in (ROOT / 'slurm').glob('*.slurm'):
        subprocess.run(['bash', '-n', str(p)], check=True)
    m.exclusive(ROOT / 'plan/LOCAL_TESTS.json', dict(m.META, status='pass', output=test.stdout + test.stderr,
                scope='Synthetic CPU, no A100 or scientific quality claim', utc=m.datetime.now(m.timezone.utc).isoformat()))
    m.exclusive(ROOT / 'plan/SUBMISSION_RESERVATION.json', dict(m.META, utc=m.datetime.now(m.timezone.utc).isoformat(),
                authorization='User: additional raw-real training, three master1035/130 and T2W authentic823/103; no VAE reconstruction',
                scope='one CPU and four afterok single-A100 jobs, held audit before release, 200000 successful updates each',
                no_automatic_retry=True, source_records_sha256=m.sha(ROOT / 'config/SOURCE_RECORDS.json')))
    files = {str(p.relative_to(ROOT)): p for folder in ('runtime', 'config', 'plan', 'slurm', 'tests')
             for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['README.md'] = ROOT / 'README.md'
    for p in (ROOT / 'operator').glob('*.py'):
        files[str(p.relative_to(ROOT))] = p
    for p in files.values():
        if p.suffix == '.py':
            ast.parse(p.read_text())
    m.exclusive(ROOT / 'plan/PACKAGE_MANIFEST.json', dict(m.META, attempt='raw_real_modality_specific_1mm_r2',
                files={n: m.sha(p) for n, p in files.items()}, source_kind='original_authentic_native'))
    files['plan/PACKAGE_MANIFEST.json'] = ROOT / 'plan/PACKAGE_MANIFEST.json'
    payload = ROOT / 'plan/DEPLOYMENT_PAYLOAD.json'
    m.exclusive(payload, {'files_b64': {n: base64.b64encode(p.read_bytes()).decode() for n, p in files.items()}})
    code = (ROOT / 'operator/remote_submit.py').read_text()
    for key, value in {'__ROOT__': str(REMOTE), '__PACKAGE_SHA__': m.sha(ROOT / 'plan/PACKAGE_MANIFEST.json'),
                       '__PAYLOAD_SHA__': m.sha(payload), '__PYTHON__': m.PYTHON}.items():
        code = code.replace(key, repr(value))
    compile(code, '<independent_raw_submit>', 'exec')

    def setup(access):
        reserve = 'from pathlib import Path;r=Path(' + repr(str(REMOTE)) + ');r.mkdir(mode=0o700,exist_ok=False);(r/"evidence").mkdir();(r/"logs").mkdir();print("RAW_R1_RESERVED")'
        access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(reserve), timeout=30)
        access.scp(payload, str(REMOTE / 'evidence/PAYLOAD.json'))
    result, audit = run_remote('submit', code, timeout=600, setup=setup)
    assert len(result['jobs']) == len(result['held']) == len(result['releases']) == 5
    assert all(r['returncode'] == 0 for r in result['releases'].values())
    assert result['package_sha256'] == m.sha(ROOT / 'plan/PACKAGE_MANIFEST.json')
    m.exclusive(audit / 'LOCAL_ACCEPTANCE.json', dict(m.META, status='pass', jobs=result['jobs'],
                package_sha256=result['package_sha256'], utc=m.datetime.now(m.timezone.utc).isoformat(),
                five_held_audits=True, five_release_exit0=True, quality_pass=False))
    print(json.dumps(dict(status='submitted', jobs=result['jobs'], queue=result['queue_after_release']['stdout'], storage=result['storage'])))


if __name__ == '__main__':
    main()
