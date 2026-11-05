"""One authorized independent Bio submission, consuming the exact failed R1 runtime."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess

import submit_direct_full103 as submit

ROOT = submit.ROOT
MODEL = 'EXT2026_BIO_INTEL_FULL'
R1_PAYLOAD = ROOT / 'operator/submit_direct_full103_ext2026_bio_intel_full_20261105T171704Z/SUBMISSION_PAYLOAD.json'
R1_PAYLOAD_SHA = 'c1c909050edd74ce49859b0d0332b23339680fd168bb7643d9d3cba97f534dc6'
METADATA = ROOT / 'operator/accept_runtime_r5_20261105T171520Z/ACCEPTED_METADATA.json'
METADATA_SHA = '812e15d2620646414ac335f57ced1c73106f824c189d19697780aa527313e679'
PARENT = dict(attempt_name='full103_bio_r1',
    package_sha256='f42630f38735ba361f4425a8ba419673df72dec2456565a1294c2880362b401b',
    contract_sha256='62bce88b6516184481c3f5b5f757f0df5f8a83b4d6ce2257227d4dc707bbc94b',
    failure_sha256='3ced04769ce960d68544bed000e44b0f8e29d7a476f10936fd30b3b972ac9cad',
    protected_2025_attempt='full103_2025_r1',protected_2025_job_id='4310575',
    protected_2025_package_sha256='4536d7d16462ba784bdd3e8c9b518cc6656755ea53e8e23ef1e648b7f7ff1c44',
    protected_2025_contract_sha256='b6ede31460d7a67c156fb3693f79f6d1c747ffad6f51695e60de35105150c3ff')


def validate_submission_parent(parent, root, command, file_sha, check_scheduler):
    import json
    from pathlib import Path
    base = Path(root) / 'attempts' / parent['attempt_name']
    assert base.name == 'full103_bio_r1', 'unapproved parent attempt'
    for name, key in (('PACKAGE_SHA256.json', 'package_sha256'), ('EXECUTION_CONTRACT.json', 'contract_sha256')):
        assert file_sha(base / 'contract' / name) == parent[key], 'R1 frozen identity drift: ' + name
    members = json.loads((base / 'contract/PACKAGE_SHA256.json').read_text())['files']
    assert len(members) == 12, 'R1 package member count drift'
    for name, expected in members.items():
        path = base / name
        assert path.resolve().is_relative_to(base.resolve()), 'R1 package path escape'
        assert file_sha(path) == expected, 'R1 package member drift: ' + name
    failed_path = base / 'operator/SUBMISSION_FAILED_hold.json'
    assert file_sha(failed_path) == parent['failure_sha256'], 'R1 failure drift'
    failed = json.loads(failed_path.read_text())
    assert failed['stage'] == 'single_sbatch_held' and failed['job_id'] is None, 'R1 no-job failure drift'
    forbidden = ('SBATCH_RESULT.json', 'HELD_CREATED.json', 'HELD_AUDIT.json', 'HELD_JOB_MAP.json',
                 'RELEASE_RESERVED.json', 'RELEASE_RESULT.json', 'SUBMISSION_RECEIPT.json')
    assert not any((base / 'operator' / n).exists() for n in forbidden), 'R1 submission evidence appeared; inspect only'
    protected = Path(root) / 'attempts' / parent['protected_2025_attempt']
    assert protected.name == 'full103_2025_r1' and parent['protected_2025_job_id'] == '4310575'
    for name, key in (('PACKAGE_SHA256.json', 'protected_2025_package_sha256'), ('EXECUTION_CONTRACT.json', 'protected_2025_contract_sha256')):
        assert file_sha(protected / 'contract' / name) == parent[key], 'protected 2025 identity drift'
    mapping = json.loads((protected / 'operator/HELD_JOB_MAP.json').read_text())
    assert mapping['jobs']['EXT2025_MET_GLI']['job_id'] == '4310575', 'protected 2025 JobID drift'
    result = dict(parent=parent, parent_frozen_files_verified=True, protected_2025_not_modified=True)
    if check_scheduler:
        queue = command(['squeue', '-h', '-u', 'zqchen', '-o', '%i|%T|%P|%j|%r|%b'])
        rows = [line.split('|') for line in queue.splitlines() if line]
        assert all(len(row) == 6 for row in rows), 'queue column drift'
        assert not any(row[3] == 's2ext26full' for row in rows), 'Bio job already queued; inspect only'
        accounting = command(['sacct', '-n', '-P', '-u', 'zqchen', '--name=s2ext26full',
                              '--starttime=2026-11-05T00:00:00', '--format=JobID,JobName,State,ExitCode,Submit,Start'])
        assert not accounting.strip(), 'Bio accounting evidence appeared; inspect only'
        result.update(fresh_queue=queue, fresh_Bio_accounting=accounting)
    return result


def payload_for_r2():
    assert submit.access_module.sha(R1_PAYLOAD) == R1_PAYLOAD_SHA, 'local R1 payload drift'
    assert submit.access_module.sha(METADATA) == METADATA_SHA, 'accepted R5 metadata drift'
    original = json.loads(R1_PAYLOAD.read_text())
    assert original['model'] == MODEL
    assert submit.digest(original['files']['contract/PACKAGE_SHA256.json']) == PARENT['package_sha256']
    assert submit.digest(original['files']['contract/EXECUTION_CONTRACT.json']) == PARENT['contract_sha256']
    package = json.loads(original['files']['contract/PACKAGE_SHA256.json'])['files']
    assert set(package) == set(original['files']) - {'contract/PACKAGE_SHA256.json'}
    assert all(submit.digest(original['files'][n]) == d for n, d in package.items()), 'local R1 member drift'
    metadata = json.loads(METADATA.read_text())
    assert original['contract']['bio_runtime_acceptance'] == metadata['accepted_bio_runtime']
    assert original['contract']['bio_runtime_identity_files'] == metadata['independent_runtime_acceptance']['files']
    authorization = json.loads((ROOT / 'BIO_FULL103_R2_AUTHORIZATION_20261106.json').read_text())
    assert authorization['user_answer'] == '允许独立提交 R2' and authorization['selected_model'] == MODEL
    assert authorization['new_attempt'] == 'full103_bio_r2'
    assert authorization['one_held_submission_then_independent_audit_and_same_ID_first_release'] is True
    assert authorization['runtime_models_versions_data_resources_changed'] is False
    assert authorization['automatic_retry'] is False and authorization['reinstall_or_solver'] is False
    project = ROOT.parents[4]
    head = subprocess.check_output(['/opt/homebrew/bin/git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    assert head == authorization['source_git_commit'], 'authorized source commit drift'
    payload = copy.deepcopy(original)
    contract = payload['contract']
    assert contract['remote_root'] == os.environ['EXTERNAL_MET_REMOTE_ROOT'], 'remote root drift'
    old_attempt = contract['attempt']
    assert old_attempt == contract['remote_root'] + '/attempts/full103_bio_r1'
    contract.update(attempt=contract['remote_root'] + '/attempts/full103_bio_r2',
                    submission_parent=PARENT, source_git_commit=head)
    files = payload['files']
    files['contract/EXECUTION_CONTRACT.json'] = submit.encode(contract)
    prior_auth = json.loads(files['contract/USER_AUTHORIZATION.json'])
    prior_auth['independent_Bio_R2_submission'] = authorization
    files['contract/USER_AUTHORIZATION.json'] = submit.encode(prior_auth)
    contract_sha = submit.digest(files['contract/EXECUTION_CONTRACT.json'])
    script = submit.render_slurm(contract['attempt'], MODEL, contract_sha)
    expected_script = original['files']['slurm/full.slurm'].replace(old_attempt, contract['attempt']).replace(PARENT['contract_sha256'], contract_sha)
    assert script == expected_script, 'R2 Slurm changed beyond attempt and contract identity'
    result = subprocess.run(['/bin/bash', '-n'], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    files['slurm/full.slurm'] = script
    files['contract/PACKAGE_SHA256.json'] = submit.encode(dict(submit.META, files={n:submit.digest(t) for n,t in files.items() if n != 'contract/PACKAGE_SHA256.json'}))
    for n, text in files.items():
        if n.endswith('.py'): ast.parse(text, feature_version=(3,10))
    return payload


def main():
    payload = payload_for_r2()
    extra_source = inspect.getsource(validate_submission_parent)
    for action in ('hold', 'release'):
        submit.code_for(action, '/tmp/unexecuted_payload', 'a'*64, MODEL, extra_source=extra_source)
    submit.submit_payload(payload, 'submit_bio_full103_r2', extra_source)


if __name__ == '__main__':
    main()
