"""One R2 held submission, only after fresh same-Python verification acceptance."""
import hashlib
import json
import os
import shlex

import submit_s0_cpu_r3 as base
import submit_sif_recovery_r1 as previous
import s0_sif_recovery_sources as old_sources
from s0_sif_recovery_r2_sources import ROOT, runtime_files, common_source, protocol_source
from session import session, access_module
from sif_empty_layer_sources import candidate_source


def preparation_payload(verification=None):
    if verification is None:
        acceptance = json.loads((ROOT / 'S0_SIF_RECOVERY_R2_PYTHON310_ACCEPTANCE_20261105.json').read_text())
        assert acceptance['pass_same_remote_Python310_only_not_full_SIF_or_GPU_pass']
        directory = ROOT / acceptance['local_audit_directory']
        data = (directory / 'RESULT.json').read_bytes()
        assert hashlib.sha256(data).hexdigest() == acceptance['local_result_sha256']
        result = json.loads(data)
        assert result['all_eight_checks_pass'] and result['small_same_backend_native_probe_pass']
        assert result['candidate_source_sha256'] == hashlib.sha256(candidate_source().encode()).hexdigest()
        verification = dict(remote_complete_sha256=hashlib.sha256(base.encode_json(result)).hexdigest(),
                            candidate_source_sha256=result['candidate_source_sha256'])
        assert verification['remote_complete_sha256'] == acceptance['remote_complete_sha256']
    payload = previous.preparation_payload()
    authorization = json.loads((ROOT / 'S0_SIF_RECOVERY_R2_AUTHORIZATION_20261105.json').read_text())
    assert authorization['independent_R2_authorized'] and authorization['single_new_CPU_held_submission_after_verification_authorized']
    authorization.update(single_R3_CPU_held_submission_and_release_authorized=True)
    proof = payload['bootstrap_proof']['compatibility_verification']
    proof.update(original_normalizer_source_sha256=proof['normalizer_source_sha256'],
                 normalizer_source_sha256=hashlib.sha256(candidate_source().encode()).hexdigest(),
                 python310_verification=verification)
    payload.update(authorization=authorization, files=runtime_files())
    return payload


def remote_code(action, payload):
    assert action in ('prepare_held', 'release_held')
    source = common_source() + '\n' + protocol_source() + '\nprint(json.dumps(execute(' + repr(action) + ',' + repr(payload) + ')))\n'
    compile(source, 'SIF_recovery_R2_' + action, 'exec')
    assert len(source.encode()) < 120000
    return source


def snapshot_code(job_id):
    source = previous.snapshot_code(job_id)
    old = old_sources.runtime_files()['s0_build_contract.py']
    assert source.startswith(old + '\n')
    body = source[len(old):].replace('s0_sif_recovery_r1', 's0_sif_recovery_r2')
    source = runtime_files()['s0_build_contract.py'] + body
    compile(source, 'SIF_recovery_R2_snapshot', 'exec')
    return source


def main():
    payload = preparation_payload()
    with session('submit_sif_recovery_r2') as access:
        held = base.invoke(access, 'prepare_held', payload, code_factory=remote_code, runtime_factory=runtime_files)
        if held['status'] != 'R3_CPU_submitted_held_not_released':
            raise RuntimeError('R2 stopped before held verification; no automatic retry')
        verification = base.verify_local_held(held, runtime_factory=runtime_files, helper_factory=common_source)
        access_module.exclusive(access.audit / 'LOCAL_HELD_VERIFICATION.json', verification)
        released = base.invoke(access, 'release_held', verification, code_factory=remote_code, runtime_factory=runtime_files)
        if released['status'] != 'CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('R2 release stopped; preserve held identity without retry')
        source = snapshot_code(released['job_id'])
        access_module.exclusive(access.audit / 'SNAPSHOT_EXECUTED_SOURCE.json', dict(source=source, sha256=hashlib.sha256(source.encode()).hexdigest()))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(source)
        result = json.loads(access.ssh(command, timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'SNAPSHOT.json', result)
        print(json.dumps(dict(audit=str(access.audit), job_id=released['job_id'], remote_utc=result['remote_utc'],
                             drift=result['drift'], scheduler=result['scheduler'])), flush=True)


if __name__ == '__main__':
    main()
