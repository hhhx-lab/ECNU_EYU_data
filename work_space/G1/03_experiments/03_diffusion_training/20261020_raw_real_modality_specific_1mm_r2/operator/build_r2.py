"""User-authorized independent recovery; keep R1 frozen and change only CPU test mock."""
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / '20261020_raw_real_modality_specific_1mm_r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def main():
    package = PARENT / 'plan/PACKAGE_MANIFEST.json'
    assert sha(package) == '4c250cfc4cc594f1c0135bacbbb37d2d0038e3642cb4369f690c1c57d86db8e4'
    manifest = json.loads(package.read_text())
    assert all(sha(PARENT / p) == h for p, h in manifest['files'].items())
    failure = PARENT / 'operator/submit_20261020T073522Z/RESULT.json'
    result = json.loads(failure.read_text())
    assert result['status'] == 'failed' and result['stage'] == 'static_preflight'
    assert result['jobs'] == result['held'] == result['releases'] == {}
    for name in manifest['files']:
        if name.startswith('operator/') or name in ('README.md', 'plan/SUBMISSION_RESERVATION.json', 'plan/LOCAL_TESTS.json'):
            continue
        target = ROOT / name
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PARENT / name, target)
    for name in ('common.py', 'submit.py', 'remote_submit.py'):
        source = PARENT / 'operator' / name
        target = ROOT / 'operator' / name
        assert not target.exists()
        text = source.read_text().replace('g1_diffusion_raw_real_modality_specific_1mm_20261020_r1',
                                          'g1_diffusion_raw_real_modality_specific_1mm_20261020_r2')
        if name == 'submit.py':
            text = text.replace("attempt='raw_real_modality_specific_1mm_r1'", "attempt='raw_real_modality_specific_1mm_r2'")
        target.write_text(text)
    test = ROOT / 'tests/test_training_boundary.py'
    original = test.read_text()
    anchor = "            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='A100 CPU test double'))"
    assert original.count(anchor) == 1
    amended = original.replace(anchor, anchor + "\n            stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))")
    test.write_text(amended)
    cfgpath = ROOT / 'config/RUN.json'
    cfg = json.loads(cfgpath.read_text())
    cfg['attempt'] = 'raw_real_modality_specific_1mm_r2'
    cfg['remote_root'] = cfg['remote_root'].replace('20261020_r1', '20261020_r2')
    cfgpath.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    for p in (ROOT / 'slurm').glob('*.slurm'):
        p.write_text(p.read_text().replace('20261020_r1', '20261020_r2').replace('g1raw1', 'g1raw2'))
    remote = ROOT / 'operator/remote_submit.py'
    remote.write_text(remote.read_text().replace('g1raw1', 'g1raw2'))
    write(ROOT / 'plan/PARENT_PREFLIGHT_FAILURE.json', result)
    write(ROOT / 'plan/R2_RECOVERY_AUTHORIZATION.json', dict(
        artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed',
        user_answer='修正测试后核验并提交 R2', parent_package_sha256=sha(package),
        parent_failed_receipt_sha256=sha(failure), parent_jobs={},
        production_train_sha256=sha(ROOT / 'runtime/train.py'),
        production_train_unchanged=sha(ROOT / 'runtime/train.py') == sha(PARENT / 'runtime/train.py'),
        changes='CPU test only: mock is_current_stream_capturing=False during synthetic CUDA availability; independent root/job names/attempt metadata',
        source_records_sha256=sha(ROOT / 'config/SOURCE_RECORDS.json'),
        no_automatic_retry=True, scope='one independent R2 CPU + four afterok A100 held-audit-release'))
    assert sha(ROOT / 'runtime/train.py') == sha(PARENT / 'runtime/train.py')
    assert sha(ROOT / 'runtime/prepare.py') == sha(PARENT / 'runtime/prepare.py')
    assert sha(ROOT / 'runtime/identity.py') == sha(PARENT / 'runtime/identity.py')
    assert sha(ROOT / 'config/SOURCE_RECORDS.json') == sha(PARENT / 'config/SOURCE_RECORDS.json')
    print(json.dumps({'status': 'pass', 'root': str(ROOT), 'production_train_unchanged': True}))


if __name__ == '__main__':
    main()
