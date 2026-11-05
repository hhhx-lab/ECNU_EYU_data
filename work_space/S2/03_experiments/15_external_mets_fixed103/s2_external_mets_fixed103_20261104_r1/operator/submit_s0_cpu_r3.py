"""One user-authorized R3 submission, with local held verification before release."""
import hashlib
import json
import os
from pathlib import Path
import shlex

from session import session, access_module
from s0_cpu_r3_sources import ROOT, runtime_files, common_source, source_identities


def encode_json(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def preparation_payload():
    directory = ROOT/'operator/review_s0_cpu_preparation_20261105T041618Z'
    proof = json.loads((directory/'BOOTSTRAP_VERIFICATION.json').read_text())
    assert proof['original_result_sha256'] == hashlib.sha256((directory/'RESULT.json').read_bytes()).hexdigest()
    assert proof['pass_bootstrap_only'] and not proof['full_preparation_pass'] and not proof['GPU_pass']
    authorization = json.loads((ROOT/'S0_CPU_R3_AUTHORIZATION.json').read_text())
    assert authorization['single_R3_CPU_held_submission_and_release_authorized']
    return dict(authorization=authorization, bootstrap_proof=proof, files=runtime_files())


def remote_code(action, payload):
    assert action in ('prepare_held', 'release_held')
    code = common_source()+'\n'+(ROOT/'operator/s0_cpu_r3_protocol.py').read_text()
    code += '\nprint(json.dumps(execute('+repr(action)+','+repr(payload)+')))\n'
    compile(code, 'R3_'+action, 'exec')
    assert len(code.encode()) < 120000, 'R3 remote source size budget'
    return code


def verify_local_held(result, runtime_factory=runtime_files, helper_factory=common_source):
    assert result['status'] == 'R3_CPU_submitted_held_not_released'
    for name, item in result['light_audit_files'].items():
        raw = encode_json(item['content'])
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'light audit transport drift: '+name
    contract = result['contract']
    assert hashlib.sha256(encode_json(contract)).hexdigest() == result['contract_sha256']
    assert hashlib.sha256(result['frozen_slurm'].encode()).hexdigest() == result['slurm_sha256']
    assert contract['package_files'] == {'runtime/'+n: h for n, h in source_identities(runtime_factory()).items()}
    helpers = {}
    exec(compile(helper_factory(), 'local_R3_identity_helpers', 'exec'), helpers)
    attempt = Path(contract['attempt'])
    validation = result['light_audit_files']['STATIC_PREFLIGHT.json']['content']['validation']
    observed = result['light_audit_files']['HELD_AUDIT.json']['content']
    fields = helpers['validate_held'](observed['raw_scontrol'], result['job_id'], attempt, [validation['account']])
    assert fields['QOS'].lower() in [q.lower() for q in validation['applicable_qos']]
    expected = helpers['render_slurm'](attempt, contract['build_env'], os.environ['EXTERNAL_MET_CONDA_BIN'], result['contract_sha256'])
    assert expected == result['frozen_slurm'], 'R3 rendered Slurm mismatch'
    return dict(access_module.META, local_held_verification_pass=True, job_id=result['job_id'],
                held_audit_sha256=result['light_audit_files']['HELD_AUDIT.json']['sha256'],
                contract_sha256=result['contract_sha256'], slurm_sha256=result['slurm_sha256'],
                local_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),
                resource=contract['resource'], models_loaded=False, images_read=False)


def invoke(access, action, payload, code_factory=remote_code, runtime_factory=runtime_files):
    code = code_factory(action, payload)
    access_module.exclusive(access.audit/(action.upper()+'_SOURCE_IDENTITY.json'), dict(
        remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(), runtime=source_identities(runtime_factory())))
    access_module.exclusive(access.audit/(action.upper()+'_EXECUTED_SOURCE.json'), dict(
        sha256=hashlib.sha256(code.encode()).hexdigest(), source=code))
    command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
    command += ' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
    command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
    result = json.loads(access.ssh(command, timeout=600).strip().splitlines()[-1])
    access_module.exclusive(access.audit/(action.upper()+'_RESULT.json'), result)
    return result


def snapshot_code(job_id):
    from snapshot_s0_cpu_r2 import CODE
    from s0_cpu_r2_sources import replace_one
    code = replace_one(CODE, "root/'attempts/s0_cpu_prepare_r2'", "root/'attempts/s0_cpu_prepare_r3'")
    code = replace_one(code, "job_id=receipt['job_id']", "job_id=receipt['job_id']\nrequire(job_id=="+repr(job_id)+",'R3 snapshot JobID drift')")
    code = replace_one(code, "if prior_guard(root)!=contract['prior_guard']:drift.append('prior_environment_or_failure')",
                       "if prior_guard(root)!=contract['prior_guard']:drift.append('prior_environment_or_failure')\n"
                       "if parent_r2_guard(root)!=contract['parent_R2_attempt_files']:drift.append('parent_R2_attempt')")
    code = code.replace('read_only_CPU_R2_observation_not_inference_pass', 'read_only_CPU_R3_observation_not_inference_pass')
    source = runtime_files()['s0_build_contract.py']+'\n'+code
    compile(source, 'single_R3_snapshot', 'exec')
    return source


def main():
    payload = preparation_payload()
    with session('submit_s0_cpu_r3') as access:
        held = invoke(access, 'prepare_held', payload)
        if held['status'] != 'R3_CPU_submitted_held_not_released':
            raise RuntimeError('R3 preparation/submission stopped; inspect evidence without retry')
        verification = verify_local_held(held)
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json', verification)
        released = invoke(access, 'release_held', verification)
        if released['status'] != 'CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('R3 release stopped; preserve held job without retry')
        source = snapshot_code(released['job_id'])
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        snapshot = json.loads(access.ssh(command, timeout=900).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'SNAPSHOT.json', snapshot)
        print(json.dumps(dict(audit=str(access.audit), status=released['status'], job_id=released['job_id'],
                              observed_job=released['observed_job'], remote_utc=snapshot['remote_utc'],
                              drift=snapshot['drift'])), flush=True)


if __name__ == '__main__':
    main()
