"""One user-authorized R3 submission; immutable R1/R2 runtime and failures preserved."""
import ast
import copy
import inspect
import json
import os
import subprocess

import submit_direct_full103 as submit
import submit_bio_full103_r2 as r2

ROOT = submit.ROOT
MODEL = r2.MODEL
R2_PAYLOAD = ROOT / 'operator/submit_bio_full103_r2_20261105T174640Z/SUBMISSION_PAYLOAD.json'
R2_PAYLOAD_SHA = 'a67e167e2f155cdc1f602449fbad46461808e57fec54709e9d33032699d37e69'
PARENT = dict(r1=r2.PARENT, attempt_name='full103_bio_r2',
    package_sha256='365120b27ddf2c03ca183119c453206fae53e8a9ddaa0b3acadb67aac13f1a73',
    contract_sha256='95df0f5461676ec99f8f967505dd10740e34bb0a1eb6d7bec98da274212c9361',
    failure_sha256='406626ad25fa72fb5d159e5364758dce1365f7fafb8797debd972e5d33f7a663',
    command_failure_sha256='9c1f30e3b76d0aa2aeeba53d8d90bd0cd038c74c546cdd2ce19c066fac19f3cc')


def validate_submission_parent(parent, root, command, file_sha, check_scheduler):
    import json
    from pathlib import Path
    prior = validate_r1_parent(parent['r1'], root, command, file_sha, False)
    base = Path(root) / 'attempts' / parent['attempt_name']
    assert base.name == 'full103_bio_r2', 'unapproved R2 parent'
    for name, key in (('PACKAGE_SHA256.json', 'package_sha256'), ('EXECUTION_CONTRACT.json', 'contract_sha256')):
        assert file_sha(base / 'contract' / name) == parent[key], 'R2 frozen identity drift: ' + name
    members = json.loads((base / 'contract/PACKAGE_SHA256.json').read_text())['files']
    assert len(members) == 12, 'R2 package member count drift'
    for name, expected in members.items():
        path = base / name
        assert path.resolve().is_relative_to(base.resolve()), 'R2 package escape'
        assert file_sha(path) == expected, 'R2 package member drift: ' + name
    failed_path = base / 'operator/SUBMISSION_FAILED_hold.json'
    assert file_sha(failed_path) == parent['failure_sha256'], 'R2 failure drift'
    failed = json.loads(failed_path.read_text())
    assert failed['stage'] == 'single_sbatch_held' and failed['job_id'] is None, 'R2 no-job failure drift'
    raw_path = base / 'operator/COMMAND_FAILED_hold_single_sbatch_held.json'
    assert file_sha(raw_path) == parent['command_failure_sha256'], 'R2 raw rejection drift'
    raw = json.loads(raw_path.read_text())
    assert raw['returncode'] != 0 and raw['argv'][0] == 'sbatch', 'R2 command failure drift'
    forbidden = ('SBATCH_RESULT.json', 'HELD_CREATED.json', 'HELD_AUDIT.json', 'HELD_JOB_MAP.json',
                 'RELEASE_RESERVED.json', 'RELEASE_RESULT.json', 'SUBMISSION_RECEIPT.json')
    assert not any((base / 'operator' / n).exists() for n in forbidden), 'R2 submission evidence appeared; inspect only'
    result = dict(r1=prior, r2=parent, both_failed_parents_verified=True)
    if check_scheduler:
        queue = command(['squeue', '-r', '-h', '-u', 'zqchen', '-o', '%i|%T|%P|%j|%r|%b|%C'])
        rows = expanded_queue(queue)
        assert not any(row[3] == 's2ext26full' for row in rows), 'Bio already queued; inspect only'
        accounting = command(['sacct', '-n', '-P', '-u', 'zqchen', '--name=s2ext26full',
            '--starttime=2026-11-05T00:00:00', '--format=JobID,JobName,State,ExitCode,Submit,Start'])
        assert not accounting.strip(), 'Bio accounting evidence appeared; inspect only'
        result.update(fresh_expanded_queue=queue, fresh_Bio_accounting=accounting)
    return result


def payload_for_r3():
    assert submit.access_module.sha(R2_PAYLOAD) == R2_PAYLOAD_SHA, 'local R2 payload drift'
    original = json.loads(R2_PAYLOAD.read_text())
    assert original['model'] == MODEL
    assert submit.digest(original['files']['contract/PACKAGE_SHA256.json']) == PARENT['package_sha256']
    assert submit.digest(original['files']['contract/EXECUTION_CONTRACT.json']) == PARENT['contract_sha256']
    package = json.loads(original['files']['contract/PACKAGE_SHA256.json'])['files']
    assert set(package) == set(original['files']) - {'contract/PACKAGE_SHA256.json'}
    assert all(submit.digest(original['files'][n]) == d for n, d in package.items()), 'R2 local member drift'
    assert submit.access_module.sha(r2.METADATA) == r2.METADATA_SHA
    metadata = json.loads(r2.METADATA.read_text())
    assert original['contract']['bio_runtime_acceptance'] == metadata['accepted_bio_runtime']
    assert original['contract']['bio_runtime_identity_files'] == metadata['independent_runtime_acceptance']['files']
    auth = json.loads((ROOT / 'BIO_FULL103_R3_AUTHORIZATION_20261106.json').read_text())
    assert auth['new_attempt'] == 'full103_bio_r3' and auth['selected_model'] == MODEL
    assert auth['one_held_submission_then_independent_audit_and_same_ID_first_release'] is True
    assert auth['automatic_retry'] is False and auth['reinstall_or_solver'] is False
    assert auth['runtime_models_versions_data_resources_changed'] is False
    head = subprocess.check_output(['/opt/homebrew/bin/git', 'rev-parse', 'HEAD'], cwd=ROOT.parents[4], text=True).strip()
    assert head == auth['source_git_commit'], 'authorized source commit drift'
    payload = copy.deepcopy(original)
    contract = payload['contract']
    assert contract['remote_root'] == os.environ['EXTERNAL_MET_REMOTE_ROOT'], 'remote root drift'
    old_attempt = contract['attempt']
    assert old_attempt == contract['remote_root'] + '/attempts/full103_bio_r2'
    contract.update(attempt=contract['remote_root'] + '/attempts/full103_bio_r3',
                    submission_parent=PARENT, source_git_commit=head)
    files = payload['files']
    files['contract/EXECUTION_CONTRACT.json'] = submit.encode(contract)
    prior_auth = json.loads(files['contract/USER_AUTHORIZATION.json'])
    prior_auth['independent_Bio_R3_submission'] = auth
    files['contract/USER_AUTHORIZATION.json'] = submit.encode(prior_auth)
    contract_sha = submit.digest(files['contract/EXECUTION_CONTRACT.json'])
    script = submit.render_slurm(contract['attempt'], MODEL, contract_sha)
    assert script == original['files']['slurm/full.slurm'].replace(old_attempt, contract['attempt']).replace(PARENT['contract_sha256'], contract_sha), 'Slurm changed beyond identity'
    check = subprocess.run(['/bin/bash', '-n'], input=script, text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
    files['slurm/full.slurm'] = script
    files['contract/PACKAGE_SHA256.json'] = submit.encode(dict(submit.META, files={n:submit.digest(t) for n,t in files.items() if n != 'contract/PACKAGE_SHA256.json'}))
    for name, text in files.items():
        if name.endswith('.py'): ast.parse(text, feature_version=(3,10))
    return payload


def main():
    payload = payload_for_r3()
    source = inspect.getsource(r2.validate_submission_parent).replace('def validate_submission_parent(', 'def validate_r1_parent(', 1)
    source += '\n' + inspect.getsource(validate_submission_parent)
    for action in ('hold', 'release'):
        compile(submit.code_for(action, '/tmp/unexecuted_payload', 'a'*64, MODEL, extra_source=source), '<R3>', 'exec')
    submit.submit_payload(payload, 'submit_bio_full103_r3', source)


if __name__ == '__main__': main()
