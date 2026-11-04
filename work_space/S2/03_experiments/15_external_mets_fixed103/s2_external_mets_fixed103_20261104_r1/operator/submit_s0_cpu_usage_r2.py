"""Unexecuted R2: preserve both failures and require new authorization before SSH."""
import hashlib
import inspect
import json
import os
import shlex

import submit_s0_cpu_usage_r1 as original
from session import session, access_module
from s0_cpu_r2_sources import ROOT, replace_function, replace_one
from s0_slurm_fields_r2 import job_fields

USAGE_R1_FAILED_SHA='9fa1bd3fd29656328d8d35b6fd9ae48f375bf79a197e665714b28ac5d11c5f75'
AUTHORIZATION_PATH=ROOT/'S0_CPU_USAGE_R2_AUTHORIZATION.json'


def remote_code():
    code=replace_function(original.remote_code(),'job_fields',inspect.getsource(job_fields))
    code=replace_one(code,"audit=old_audit/'submit_usage_r1'", """previous_usage=old_audit/'submit_usage_r1'
require(sha(previous_usage/'SUBMISSION_FAILED.json')=="""+repr(USAGE_R1_FAILED_SHA)+""",'first CPU usage failure changed')
for name in ['RESERVATION.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json','RELEASE_RESULT.json','SUBMISSION_RECEIPT.json']:
    require(not (previous_usage/name).exists(),'first CPU usage job may exist; no duplicate submission')
previous_usage_files={p.name:sha(p) for p in previous_usage.iterdir() if p.is_file()}
audit=old_audit/'submit_usage_r2'""")
    code=replace_one(code,"write(audit/'AUTHORIZATION.json',USAGE_AUTHORIZATION)",
        "write(audit/'AUTHORIZATION.json',USAGE_R2_AUTHORIZATION)")
    code=replace_one(code,"    stage='release_original_held_job'",
        "    require({p.name:sha(p) for p in previous_usage.iterdir() if p.is_file()}==previous_usage_files,'first CPU usage audit changed while held')\n"
        "    stage='release_original_held_job'")
    compile(code,'independent_CPU_usage_R2_pending_authorization','exec')
    return code


def main():
    authorization=json.loads(AUTHORIZATION_PATH.read_text())
    assert authorization['independent_R2_single_first_held_submission_authorized']
    assert authorization['parser_only_amendment'] and authorization['preserve_both_prior_failures']
    assert not authorization['solver_rerun_authorized'] and not authorization['environment_reinstallation_authorized']
    code='USAGE_R2_AUTHORIZATION='+repr(authorization)+'\n'+remote_code()
    with session('submit_s0_cpu_usage_r2') as access:
        access_module.exclusive(access.audit/'SOURCE_IDENTITY.json',dict(
            remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            parser_sha256=hashlib.sha256((ROOT/'operator/s0_slurm_fields_r2.py').read_bytes()).hexdigest(),
            frozen_slurm_sha256=original.SLURM_SHA,preparation_contract_sha256=original.CONTRACT_SHA,
            original_failed_sha256=original.FAILED_SHA,first_usage_failed_sha256=USAGE_R1_FAILED_SHA))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result.get('job_id'),remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('Independent CPU usage R2 stopped; preserve any held job and failures; no retry')


if __name__=='__main__':
    main()
