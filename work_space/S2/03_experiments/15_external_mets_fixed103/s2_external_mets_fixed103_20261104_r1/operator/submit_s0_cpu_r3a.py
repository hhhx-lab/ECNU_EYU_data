"""Authorized R3a path correction; keep the failed R3 inventory immutable."""
import hashlib
import json
import os
import shlex

import submit_s0_cpu_r3 as base
import s0_cpu_r3_sources as r3
from session import session, access_module
from s0_cpu_r2_sources import replace_one

ROOT = r3.ROOT
FAILED_GUARD = '''
def failed_r3_guard(root):
    parent=Path(root)/'attempts/s0_cpu_prepare_r3'
    expected={
      'contract/BOOTSTRAP_VERIFICATION.json':'7ce1cdac2c4bda31ea96e2fa41d14662818fe9e27003b416bced4ea91a08c146',
      'contract/BUILD_ENV_EXPLICIT.txt':'866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77',
      'contract/USER_AUTHORIZATION.json':'f31dae05deb4003a8f3d23cbcad6ddfed044a7c7e3f32a4cfc966e821474aa8c',
      'operator/SUBMISSION_FAILED.json':'48af2f4f423d66ff8c8a8d2158f52842028a05de1a0b6e08b8b9f3df50c77127'}
    actual={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()}
    require(actual==expected,'failed R3 partial inventory drift')
    failed=json.loads((parent/'operator/SUBMISSION_FAILED.json').read_text())
    require(failed['job_id'] is None and failed['stage']=='freeze_R3_package','failed R3 may have a job')
    return actual
'''


def runtime_files():
    old = r3.runtime_files()
    files = {name.replace('prepare_s0_cpu_r3.py', 'prepare_s0_cpu_r3a.py'):
             source.replace("ATTEMPT = 's0_cpu_prepare_r3'", "ATTEMPT = 's0_cpu_prepare_r3a'")
                   .replace("root/'attempts/s0_cpu_prepare_r3'", "root/'attempts/s0_cpu_prepare_r3a'")
             for name, source in old.items()}
    files['s0_build_contract.py'] += FAILED_GUARD
    runtime = files['prepare_s0_cpu_r3a.py']
    runtime = replace_one(runtime, 'parent_r2_guard, r3_evidence_guard\n',
                          'parent_r2_guard, r3_evidence_guard, failed_r3_guard\n')
    require_anchor = '        r3_evidence_guard(attempt, contract)'
    assert runtime.count(require_anchor) == 2
    runtime = runtime.replace(require_anchor, require_anchor+'\n'
             "        require(failed_r3_guard(root)==contract['failed_R3_attempt_files'], 'failed R3 inventory drift')")
    files['prepare_s0_cpu_r3a.py'] = runtime
    for name, source in files.items():
        compile(source, name, 'exec')
    return files


def common_source():
    source = r3.common_source().replace("ATTEMPT = 's0_cpu_prepare_r3'", "ATTEMPT = 's0_cpu_prepare_r3a'")
    source = source.replace("attempt.name == 's0_cpu_prepare_r3'", "attempt.name == 's0_cpu_prepare_r3a'")
    source = source.replace('s2exts0p3', 's2exts0p3a').replace('prepare_s0_cpu_r3.py', 'prepare_s0_cpu_r3a.py')
    return source+'\n'+FAILED_GUARD


def protocol_source():
    source = (ROOT/'operator/s0_cpu_r3_protocol.py').read_text()
    source = source.replace('s0_cpu_prepare_r3', 's0_cpu_prepare_r3a').replace('s2exts0p3', 's2exts0p3a')
    source = source.replace('prepare_s0_cpu_r3.py', 'prepare_s0_cpu_r3a.py')
    source = replace_one(source, '    r3_evidence_guard(attempt, contract)',
          "    r3_evidence_guard(attempt, contract)\n    require(failed_r3_guard(root)==contract['failed_R3_attempt_files'], 'failed R3 inventory changed')")
    source = replace_one(source, '    prior, first_parent, second_parent = prior_guard(root), parent_staging_guard(root), parent_r2_guard(root)',
          '    prior, first_parent, second_parent = prior_guard(root), parent_staging_guard(root), parent_r2_guard(root)\n'
          '    failed_before = failed_r3_guard(root)')
    source = replace_one(source, 'contract.update(attempt=str(attempt), parent_R2_attempt_files=second_parent,',
          'contract.update(attempt=str(attempt), parent_R2_attempt_files=second_parent, failed_R3_attempt_files=failed_before,')
    compile(source, 'R3a_protocol', 'exec')
    return source


def preparation_payload():
    payload = base.preparation_payload()
    authorization = json.loads((ROOT/'S0_CPU_R3A_AUTHORIZATION.json').read_text())
    assert authorization['independent_R3a_authorized'] and authorization['single_R3_CPU_held_submission_and_release_authorized']
    payload.update(authorization=authorization, files=runtime_files())
    return payload


def remote_code(action, payload):
    assert action in ('prepare_held', 'release_held')
    code = common_source()+'\n'+protocol_source()+'\nprint(json.dumps(execute('+repr(action)+','+repr(payload)+')))\n'
    compile(code, 'R3a_'+action, 'exec')
    assert len(code.encode()) < 120000
    return code


def snapshot_code(job_id):
    source = base.snapshot_code(job_id)
    source = source.replace("root/'attempts/s0_cpu_prepare_r3'", "root/'attempts/s0_cpu_prepare_r3a'")
    source = source.replace('read_only_CPU_R3_observation_not_inference_pass', 'read_only_CPU_R3a_observation_not_inference_pass')
    source = replace_one(source, "if parent_r2_guard(root)!=contract['parent_R2_attempt_files']:drift.append('parent_R2_attempt')",
          "if parent_r2_guard(root)!=contract['parent_R2_attempt_files']:drift.append('parent_R2_attempt')\n"
          "if failed_r3_guard(root)!=contract['failed_R3_attempt_files']:drift.append('failed_R3_inventory')")
    return FAILED_GUARD+'\n'+source


def main():
    payload = preparation_payload()
    with session('submit_s0_cpu_r3a') as access:
        held = base.invoke(access, 'prepare_held', payload, code_factory=remote_code, runtime_factory=runtime_files)
        if held['status'] != 'R3_CPU_submitted_held_not_released':
            raise RuntimeError('R3a stopped before release; preserve all attempts without retry')
        verification = base.verify_local_held(held, runtime_factory=runtime_files, helper_factory=common_source)
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json', verification)
        released = base.invoke(access, 'release_held', verification, code_factory=remote_code, runtime_factory=runtime_files)
        if released['status'] != 'CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('R3a held release stopped; no retry')
        source = snapshot_code(released['job_id'])
        compile(source, 'R3a_snapshot', 'exec')
        access_module.exclusive(access.audit/'SNAPSHOT_EXECUTED_SOURCE.json', dict(
            sha256=hashlib.sha256(source.encode()).hexdigest(), source=source))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        snapshot = json.loads(access.ssh(command, timeout=900).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'SNAPSHOT.json', snapshot)
        print(json.dumps(dict(audit=str(access.audit), status=released['status'], job_id=released['job_id'],
             observed_job=released['observed_job'], remote_utc=snapshot['remote_utc'], drift=snapshot['drift'])), flush=True)


if __name__ == '__main__':
    main()
