"""Submit only the independently accepted R2 CPU package, held then audited."""
import json
import os
import shlex
import submit_s0_cpu_r1 as r1
from session import session,access_module
from s0_cpu_r2_sources import ROOT,pinned,replace_one,runtime_files,held_helpers


def remote_code():
    pinned('operator/submit_s0_cpu_r1.py')
    code=runtime_files()['s0_build_contract.py']+'\n'+held_helpers()+'\n'
    qos=(ROOT/'operator/s0_qos_table.py').read_text().replace('from s0_cpu_submission_contract import memory_mib\n','')
    qos=qos.replace('    from s0_cpu_submission_contract import job_fields\n','')
    body=replace_one(r1.CODE,"root/'attempts/s0_cpu_prepare_r1'","root/'attempts/s0_cpu_prepare_r2'")
    body=replace_one(body,"    require(prior_guard(root)==contract['prior_guard'], 'old environment/failure changed')",
                     "    require(prior_guard(root)==contract['prior_guard'], 'old environment/failure changed')\n"
                     "    require(parent_staging_guard(root)==contract['parent_attempt_files'],'parent CPU staging files changed')")
    body=body.replace('|s2exts0p1|','|s2exts0p2|')
    code+=qos+'\n'+body
    compile(code,'single_independent_R2_CPU_submission','exec')
    return code


def main():
    assert json.loads((ROOT/'S0_STAGING_R2_AUTHORIZATION.json').read_text())['independent_CPU_preparation_R2_authorized']
    with session('submit_s0_cpu_r2') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=420).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result.get('job_id'),remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('R2 CPU submission needs review; no repeated submission or release')


if __name__=='__main__':
    main()
