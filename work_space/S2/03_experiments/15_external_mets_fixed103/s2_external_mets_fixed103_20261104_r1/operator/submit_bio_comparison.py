"""One audited CPU Bio MET job; immutable parent scores, no inference/install."""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess

from session import session, access_module
from submit_direct_full103 import encode, digest, expanded_queue
from submit_fixed103_comparison import canonical_qos_names
from s0_slurm_fields_r2 import job_fields
from s0_qos_table import parse_qos_table

ROOT = Path(__file__).resolve().parents[1]
OLD_AUDIT = ROOT / 'operator/submit_fixed103_comparison_r2_20261106T103950Z'
BIO_AUDIT = ROOT / 'operator/submit_bio_full103_r3_20261106T102000Z'
SNAPSHOT = ROOT / 'operator/snapshot_jobs_readonly_no_duplicate_download_20261107T072728Z/RESULT.json'
NAME = 's2extbiomet'


def entry(path, text):
    return dict(path=str(path), bytes=len(text.encode()), sha256=digest(text))


def prepare_payload(remote, revision):
    previous = json.loads((OLD_AUDIT / 'SUBMISSION_PAYLOAD.json').read_text())
    bio_payload = json.loads((BIO_AUDIT / 'SUBMISSION_PAYLOAD.json').read_text())
    assert access_module.sha(SNAPSHOT) == '156f1cd9a6f5989290fd115b06a53107b0e4f31e40fdbe7833478f9304626e2d'
    snapshot = json.loads(SNAPSHOT.read_text())
    for payload in (previous, bio_payload):
        for name, value in json.loads(payload['files']['contract/PACKAGE_SHA256.json'])['files'].items():
            assert digest(payload['files'][name]) == value, 'local parent frozen source drift'
        assert payload['contract']['remote_root'] == remote
    old = previous['contract']
    bio_root = Path(bio_payload['contract']['attempt'])
    bio_reports = bio_root / 'reports/EXT2026_BIO_INTEL_FULL'
    bio_prediction = bio_root / 'predictions/EXT2026_BIO_INTEL_FULL'
    manifest = snapshot['Bio_prediction_manifest']
    assert manifest['sha256'] == 'e63602a5e0d03ec1118a704a7bb247c2565ffef3baae1c2537986ce5f47120ad'
    old_complete = snapshot['comparison_reports']['COMPLETE.json']
    assert old_complete['data']['job_id'] == '4319117' and snapshot['Bio_output_identity'] == 'COMPLETED_0_0_COMPLETE_103_current_byte_SHA_pass'
    complete_bio = snapshot['Bio_reports']['COMPLETE.json']
    files_bio = manifest['data']['files']
    assert set(files_bio) == set(old['references']) and len(files_bio) == 103
    attempt = remote + '/attempts/compare_bio_four_models_r1'
    selected = {name: value for name, value in old_complete['data']['outputs'].items()
                if name in ('reports/HISTORICAL_REPLAY.json', 'reports/MEANS_216.csv', 'reports/PAIRED_ALL_216.csv')
                or name.endswith('/leaderboard_metrics.csv')}
    assert len(selected) == 6
    models = {model: dict(metrics=dict(path=old['attempt'] + '/evaluation/' + model + '/leaderboard_metrics.csv',
                    **selected['evaluation/' + model + '/leaderboard_metrics.csv']))
              for model in ('EXT2025', 'K1_175', 'original_E_200')}
    models['EXT2026_BIO_INTEL_FULL'] = dict(prediction_root=str(bio_prediction),
        predictions={name: dict(path=str(bio_prediction / name), **value) for name, value in files_bio.items()},
        complete=dict(path=str(bio_reports / 'COMPLETE.json'), **{k: complete_bio[k] for k in ('bytes', 'sha256')}),
        manifest=dict(path=str(bio_reports / 'PREDICTION_SHA256.json'), **{k: manifest[k] for k in ('bytes', 'sha256')}),
        contract_sha256=complete_bio['data']['contract_sha256'],
        contract=entry(bio_root / 'contract/EXECUTION_CONTRACT.json', bio_payload['files']['contract/EXECUTION_CONTRACT.json']),
        package=entry(bio_root / 'contract/PACKAGE_SHA256.json', bio_payload['files']['contract/PACKAGE_SHA256.json']),
        identity_markers={name: dict(path=str(bio_reports / name), **{k: snapshot['Bio_reports'][name][k] for k in ('bytes', 'sha256')})
                          for name in ('STARTED.json', 'RUN_CONTRACT.json')})
    contract = dict(access_module.META, attempt=attempt, remote_root=remote, cases=103,
        model_order=['EXT2025', 'K1_175', 'original_E_200', 'EXT2026_BIO_INTEL_FULL'], models=models,
        parent_comparison=dict(attempt=old['attempt'], job_id='4319117',
            complete=dict(path=old['attempt'] + '/reports/COMPLETE.json', **{k: old_complete[k] for k in ('bytes', 'sha256')}),
            contract_sha256=digest(previous['files']['contract/COMPARISON_CONTRACT.json']), outputs=selected,
            package=entry(Path(old['attempt']) / 'contract/PACKAGE_SHA256.json', previous['files']['contract/PACKAGE_SHA256.json'])),
        Bio_author_revision=bio_payload['contract']['author_revision'],
        git_revision=revision, bootstrap=old['bootstrap'],
        references=old['references'], patient_groups=old['patient_groups'], evaluator=old['evaluator'],
        evaluator_prefix=old['evaluator_prefix'], evaluator_bin_sha256=old['evaluator_bin_sha256'],
        source_evidence_sha256={str(path.relative_to(ROOT)): access_module.sha(path)
            for path in (SNAPSHOT, OLD_AUDIT / 'SUBMISSION_PAYLOAD.json', BIO_AUDIT / 'SUBMISSION_PAYLOAD.json')},
        user_authorization=json.loads((ROOT / 'BIO_MET_AUTHORIZATION_20261107.json').read_text()),
        locked_test_read=False, raw_MRI_read=False, automatic_retry=False, old_scores_and_CI_recomputed=False)
    assert contract['bootstrap'] == dict(iterations=20000, seed=20260831, frame='69_patient_groups_common_finite_per_metric')
    files = {'contract/COMPARISON_CONTRACT.json': encode(contract),
        'contract/USER_AUTHORIZATION.json': (ROOT / 'BIO_MET_AUTHORIZATION_20261107.json').read_text(),
        'contract/BIO_MET_PROTOCOL.md': (ROOT / 'BIO_MET_PROTOCOL_20261107.md').read_text(),
        'runtime/compare_bio_fixed103.py': (ROOT / 'runtime/compare_bio_fixed103.py').read_text(),
        'runtime/compare_fixed103.py': previous['files']['runtime/compare_fixed103.py'],
        'runtime/run_official_met_eval.sh': previous['files']['runtime/run_official_met_eval.sh']}
    expected = digest(files['contract/COMPARISON_CONTRACT.json'])
    files['slurm/compare.slurm'] = render_slurm(contract, expected)
    files['contract/PACKAGE_SHA256.json'] = encode(dict(access_module.META, files={name: digest(text) for name, text in files.items()}))
    return dict(contract=contract, files=files)


def render_slurm(contract, expected):
    attempt = contract['attempt']
    return '\n'.join(['#!/bin/bash', '#SBATCH --job-name=s2extbiomet', '#SBATCH --partition=cpu_192G',
        '#SBATCH --account=zqchen', '#SBATCH --qos=user_zqchen', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=8', '#SBATCH --mem=64G', '#SBATCH --time=24:00:00', '#SBATCH --no-requeue',
        '#SBATCH --export=NONE', '#SBATCH --output=' + attempt + '/logs/compare_%j.out',
        '#SBATCH --error=' + attempt + '/logs/compare_%j.err', 'set -e', 'set -o pipefail', 'umask 077',
        'source /etc/profile', 'set -u', 'unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY',
        'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LC_ALL=C OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8',
        'export COMPARE_EXPECTED_CONTRACT_SHA256=' + shlex.quote(expected), 'cd ' + shlex.quote(attempt),
        'exec ' + shlex.quote(contract['evaluator_prefix'] + '/bin/python') + ' -B ' + shlex.quote(attempt + '/runtime/compare_bio_fixed103.py') + ' --root ' + shlex.quote(attempt), ''])


def validate_held(raw, identity, attempt):
    fields = job_fields(raw)
    expected = dict(JobId=identity, JobName='s2extbiomet', JobState='PENDING', Reason='JobHeldUser', Priority='0',
        Partition='cpu_192G', Account='zqchen', QOS='user_zqchen', NumCPUs='8', NumTasks='1', Requeue='0',
        Command=attempt + '/slurm/compare.slurm', WorkDir=attempt, **{'CPUs/Task': '8'})
    for key, value in expected.items():
        assert fields.get(key) == value, 'Bio MET held drift: ' + key
    assert fields.get('UserId', '').split('(')[0] == 'zqchen' and fields.get('NumNodes') in ('1', '1-1')
    assert fields.get('TimeLimit') in ('1-00:00:00', '24:00:00') and fields.get('MinMemoryNode') in ('64G', '65536M', '65536')
    assert fields.get('Dependency') in ('(null)', 'None')
    tres = dict(item.split('=', 1) for item in fields.get('TRES', fields.get('ReqTRES', '')).split(',') if '=' in item)
    assert not any('gpu' in key and value != '0' for key, value in tres.items()), 'unexpected Bio scoring GPU'
    for key, extension in (('StdOut', 'out'), ('StdErr', 'err')):
        assert fields.get(key) in (attempt + '/logs/compare_' + identity + '.' + extension, attempt + '/logs/compare_%j.' + extension)
    return fields


def validate_parent_accounting(raw):
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    assert all(len(row) == 3 for row in rows), 'parent accounting schema drift'
    for identity in ('4319113', '4319113.batch', '4319117', '4319117.batch'):
        selected = [row for row in rows if row[0].strip() == identity]
        assert len(selected) == 1 and selected[0][1:] == ['COMPLETED', '0:0'], 'parent not completed: ' + identity


def remote_execute(action, payload_path, payload_sha, verified):
    import ast
    from datetime import datetime, timezone
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1048576), b''):
                h.update(block)
        return h.hexdigest()
    def write(path, value):
        with Path(path).open('x') as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
    assert sha(payload_path) == payload_sha, 'submission payload drift'
    payload = json.loads(Path(payload_path).read_text())
    contract = payload['contract']
    attempt = Path(contract['attempt'])
    identity = None
    owned_attempt = False
    stage = 'before_attempt'
    def command(argv, timeout=90, env=None):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        assert len(result.stdout) + len(result.stderr) < 262144, 'command output budget'
        if result.returncode:
            if attempt.exists():
                write(attempt / ('operator/COMMAND_FAILED_' + action + '_' + stage + '.json'),
                    dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            raise RuntimeError('Bio MET command failed: ' + result.stderr[-4000:])
        return result.stdout
    def static():
        for name, value in json.loads((attempt / 'contract/PACKAGE_SHA256.json').read_text())['files'].items():
            assert sha(attempt / name) == value, 'Bio MET package drift'
        accounting = command(['sacct', '-n', '-P', '-j', '4319113,4319117', '--format=JobID,State,ExitCode'])
        validate_parent_accounting(accounting)
        assert shutil.disk_usage(attempt).free > 2 * 1024**3, 'fresh Bio MET disk budget not met'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LC_ALL='C',
            COMPARE_EXPECTED_CONTRACT_SHA256=sha(attempt / 'contract/COMPARISON_CONTRACT.json'),
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        raw = command([contract['evaluator_prefix'] + '/bin/python', '-B', str(attempt / 'runtime/compare_bio_fixed103.py'),
            '--root', str(attempt), '--validate-only'], timeout=180, env=env)
        check = json.loads(raw.strip().splitlines()[-1])
        assert check['status'] == 'exact_environment_parent_inputs_SHA_pass_no_voxel_scoring'
        return dict(accounting=accounting, exact_environment_and_parent_inputs=check)
    try:
        if action == 'hold':
            assert attempt.name == 'compare_bio_four_models_r1' and attempt.parent == Path(contract['remote_root']) / 'attempts'
            assert not attempt.exists(), 'Bio MET attempt exists; inspect only, no duplicate submission'
            queue = command(['squeue', '-r', '-h', '-u', 'zqchen', '-o', '%i|%T|%P|%j|%r|%b|%C'])
            jobs = expanded_queue(queue)
            assert not any(row[3] == 's2extbiomet' for row in jobs), 'Bio MET already queued'
            accounted = command(['sacct', '-n', '-P', '-u', 'zqchen', '--name=s2extbiomet', '--starttime=2026-11-07T00:00:00', '--format=JobID,JobName,State,ExitCode'])
            assert not accounted.strip(), 'Bio MET submission history exists; inspect only'
            attempt.mkdir(mode=0o700)
            owned_attempt = True
            for folder in ('contract', 'runtime', 'slurm', 'operator', 'logs', 'reports', 'stage', 'evaluation'):
                (attempt / folder).mkdir()
            stage = 'freeze_and_static_preflight'
            for name, text in payload['files'].items():
                path = attempt / name
                assert path.resolve().is_relative_to(attempt.resolve())
                with path.open('x') as handle:
                    handle.write(text)
                path.chmod(0o550 if name.endswith(('.sh', '.slurm')) else 0o440)
                if name.endswith('.py'):
                    ast.parse(text, feature_version=(3, 10))
            command(['/bin/bash', '-n', str(attempt / 'slurm/compare.slurm')])
            command(['/bin/bash', '-n', str(attempt / 'runtime/run_official_met_eval.sh')])
            preflight = static()
            write(attempt / 'operator/STATIC_PREFLIGHT.json', preflight)
            stage = 'fresh_CPU_resources'
            partition = job_fields(command(['scontrol', 'show', 'partition', 'cpu_192G', '-o']))
            assert partition['State'] == 'UP' and (partition['AllowAccounts'] == 'ALL' or 'zqchen' in partition['AllowAccounts'].split(','))
            qos = parse_qos_table(command(['sacctmgr', '-P', 'show', 'qos']))
            selected = [qos[name] for name in canonical_qos_names(qos, ['user_zqchen', partition['QoS']])]
            usage = sum(int(row[6]) for row in jobs)
            for row in selected:
                if row.get('MaxTRESPU'):
                    assert re.fullmatch(r'cpu=\d+', row['MaxTRESPU']) and usage + 8 <= int(row['MaxTRESPU'].split('=')[1]), 'CPU QoS usage exceeded'
                for key in ('MaxJobsPU', 'MaxSubmitPU', 'MaxSubmitJobsPU'):
                    if row.get(key):
                        assert len(jobs) + 1 <= int(row[key]), 'CPU QoS job limit'
            resource = dict(partition=partition, qos=selected, expanded_own_queue=queue,
                prospective_CPU_upper_bound=usage + 8, free_bytes=shutil.disk_usage(attempt).free,
                quota_stdout=command(['quota', '-s']), minimum_new_output_budget_bytes=2 * 1024**3,
                remote_utc=datetime.now(timezone.utc).isoformat())
            write(attempt / 'operator/RESOURCE_SNAPSHOT.json', resource)
            stage = 'one_sbatch_held'
            write(attempt / 'operator/SUBMISSION_RESERVED.json', dict(one_submission=True))
            raw = command(['sbatch', '--hold', '--parsable', '--export=NONE', '--chdir=' + str(attempt), str(attempt / 'slurm/compare.slurm')])
            write(attempt / 'operator/SBATCH_RESULT.json', dict(stdout=raw, returncode=0))
            match = re.fullmatch(r'(\d+)(?:;[\w.-]+)?', raw.strip())
            assert match, 'ambiguous Bio MET JobID; inspect only'
            identity = match.group(1)
            write(attempt / 'operator/HELD_CREATED.json', dict(job_id=identity))
            raw_job = command(['scontrol', 'show', 'job', identity, '-o'])
            fields = validate_held(raw_job, identity, str(attempt))
            result = dict(status='Bio_MET_submitted_held', job_id=identity, raw_scontrol=raw_job, fields=fields,
                resource=resource, contract_sha256=sha(attempt / 'contract/COMPARISON_CONTRACT.json'),
                package_sha256=sha(attempt / 'contract/PACKAGE_SHA256.json'), remote_utc=datetime.now(timezone.utc).isoformat())
            write(attempt / 'operator/HELD_AUDIT.json', result)
        elif action == 'release':
            identity = json.loads((attempt / 'operator/HELD_CREATED.json').read_text())['job_id']
            assert verified['job_id'] == identity and verified['local_held_verified']
            assert verified['held_audit_sha256'] == sha(attempt / 'operator/HELD_AUDIT.json')
            owned_attempt = True
            assert not (attempt / 'operator/RELEASE_RESERVED.json').exists(), 'release already attempted'
            static()
            validate_held(command(['scontrol', 'show', 'job', identity, '-o']), identity, str(attempt))
            write(attempt / 'operator/LOCAL_HELD_VERIFICATION.json', verified)
            write(attempt / 'operator/RELEASE_RESERVED.json', dict(job_id=identity))
            stage = 'same_ID_first_release'
            raw = command(['scontrol', 'release', identity])
            result = dict(status='Bio_MET_released_not_scoring_complete', job_id=identity, release_returncode=0, stdout=raw,
                observed_job=job_fields(command(['scontrol', 'show', 'job', identity, '-o'])),
                contract_sha256=sha(attempt / 'contract/COMPARISON_CONTRACT.json'), package_sha256=sha(attempt / 'contract/PACKAGE_SHA256.json'),
                remote_utc=datetime.now(timezone.utc).isoformat(), free_bytes=shutil.disk_usage(attempt).free)
            write(attempt / 'operator/SUBMISSION_RECEIPT.json', result)
        else:
            raise RuntimeError('invalid Bio MET action')
        return result
    except BaseException as error:
        result = dict(status='Bio_MET_submission_stopped', stage=stage, job_id=identity, message=str(error),
            remote_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False)
        if owned_attempt and (attempt / 'operator').is_dir():
            write(attempt / ('operator/FAILED_' + action + '.json'), result)
        return result


def remote_source(action, path, expected, verified):
    source = 'import re,csv,io\n'
    for helper in (job_fields, expanded_queue, parse_qos_table, canonical_qos_names, validate_held, validate_parent_accounting, remote_execute):
        source += '\n' + inspect.getsource(helper)
    source += '\nimport json\nprint(json.dumps(remote_execute(' + repr(action) + ',' + repr(path) + ',' + repr(expected) + ',' + repr(verified) + ')))\n'
    ast.parse(source, feature_version=(3, 10))
    compile(source, '<Bio MET submission>', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    revision = subprocess.run(['/opt/homebrew/bin/git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    payload = prepare_payload(os.environ['EXTERNAL_MET_REMOTE_ROOT'], revision)
    for name, text in payload['files'].items():
        if name.endswith('.py'):
            ast.parse(text, feature_version=(3, 10))
        if name.endswith(('.sh', '.slurm')):
            checked = subprocess.run(['/bin/bash', '-n'], input=text, text=True, capture_output=True)
            assert checked.returncode == 0, checked.stderr
    if args.prepare_only:
        print(json.dumps(dict(status='local_Bio_MET_payload_syntax_pass_no_SSH',
            contract_sha256=digest(payload['files']['contract/COMPARISON_CONTRACT.json']), files=len(payload['files']))))
        return
    with session('submit_bio_comparison') as access:
        path = access.audit / 'SUBMISSION_PAYLOAD.json'
        access_module.exclusive(path, payload)
        audit = os.environ['EXTERNAL_MET_REMOTE_ROOT'] + '/operator/' + access.audit.name
        access.ssh('mkdir -m 700 ' + shlex.quote(audit))
        access.scp(path, audit + '/')
        def invoke(action, verified=None):
            source = remote_source(action, audit + '/SUBMISSION_PAYLOAD.json', access_module.sha(path), verified)
            access_module.exclusive(access.audit / (action.upper() + '_EXECUTED_SOURCE.json'), dict(source=source, sha256=digest(source)))
            result = json.loads(access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(source), timeout=300).strip().splitlines()[-1])
            access_module.exclusive(access.audit / (action.upper() + '_RESULT.json'), result)
            assert result['status'] != 'Bio_MET_submission_stopped', result
            return result
        held = invoke('hold')
        validate_held(held['raw_scontrol'], held['job_id'], payload['contract']['attempt'])
        assert held['contract_sha256'] == digest(payload['files']['contract/COMPARISON_CONTRACT.json'])
        assert held['package_sha256'] == digest(payload['files']['contract/PACKAGE_SHA256.json'])
        verified = dict(local_held_verified=True, job_id=held['job_id'], held_audit_sha256=digest(encode(held)))
        access_module.exclusive(access.audit / 'LOCAL_HELD_VERIFICATION.json', verified)
        released = invoke('release', verified)
        assert released['job_id'] == held['job_id'] and released['release_returncode'] == 0
        assert released['contract_sha256'] == held['contract_sha256'] and released['package_sha256'] == held['package_sha256']
        access_module.exclusive(access.audit / 'INDEPENDENT_LOCAL_ACCEPTANCE.json', dict(access_module.META,
            status='held_same_ID_release_and_frozen_identity_pass_not_scoring_complete', job_id=released['job_id'],
            contract_sha256=held['contract_sha256'], package_sha256=held['package_sha256'],
            local_accepted_utc=access_module.datetime.now(access_module.timezone.utc).isoformat()))
        print(json.dumps(dict(audit=str(access.audit), attempt=payload['contract']['attempt'], result=released)), flush=True)


if __name__ == '__main__':
    main()
