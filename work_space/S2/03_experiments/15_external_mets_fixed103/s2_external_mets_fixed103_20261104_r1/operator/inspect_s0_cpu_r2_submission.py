"""Read-only diagnosis of the pre-sbatch aggregate QoS gate."""
import json
import os
import shlex

from session import session, access_module
from s0_cpu_r2_sources import runtime_files, held_helpers


CODE = r'''
import os,shutil,subprocess
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r2'
failed=attempt/'operator/SUBMISSION_FAILED.json'
failure=json.loads(failed.read_text())
require(failure['stage']=='CPU_package_and_fresh_scheduler_preflight' and failure['job_id'] is None,'unexpected CPU submission failure')
require(failure['message']=='aggregate QoS TRES requires current usage audit','unexpected CPU submission cause')
staging_path=attempt/'reports/STAGING_COMPLETE.json'
staging=json.loads(staging_path.read_text())
contract_path=attempt/'contract/PREPARATION_CONTRACT.json'
require(sha(contract_path)==staging['contract_sha256'],'CPU preparation contract drift')
contract=json.loads(contract_path.read_text())
script_path=attempt/'slurm/prepare.slurm'
script=script_path.read_text()
require(script==render_slurm(attempt,contract['build_env'],os.environ['EXTERNAL_MET_CONDA_BIN'],sha(contract_path)),
        'CPU frozen Slurm differs from accepted renderer')
queries={
    'qos':['sacctmgr','-P','show','qos'],
    'own_resources':['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%C|%m|%D|%q|%a|%b'],
    'association':['sacctmgr','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
    'partition':['scontrol','show','partition','cpu_192G','-o'],
    'quota':['quota','-s'],
}
scheduler={}
for name,argv in queries.items():
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    require(len(p.stdout.encode())+len(p.stderr.encode())<65536,'CPU diagnosis metadata budget')
    scheduler[name]=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
submission_files={name:dict(exists=(attempt/'operator'/name).exists(),
    sha256=sha(attempt/'operator'/name) if (attempt/'operator'/name).is_file() else None)
    for name in ['SUBMISSION_FAILED.json','RESERVATION.json','SBATCH_RESULT.json','HELD_CREATED.json',
                 'HELD_AUDIT.json','RELEASE_RESULT.json','SUBMISSION_RECEIPT.json']}
runtime_guard={name:sha(attempt/name)==digest for name,digest in contract['package_files'].items()}
build_guard={name:sha(Path(contract['build_env'])/name)==digest for name,digest in contract['build_env_files'].items()}
result=dict(META,status='read_only_pre_sbatch_CPU_R2_failure_diagnosis_not_retry',failure=failure,
    submission_files=submission_files,scheduler=scheduler,staging=staging,
    staging_sha256=sha(staging_path),contract_sha256=sha(contract_path),
    slurm=dict(path=str(script_path),sha256=sha(script_path),content=script),
    runtime_identity_unchanged=all(runtime_guard.values()),build_identity_unchanged=all(build_guard.values()),
    parent_CPU_staging_unchanged=parent_staging_guard(root)==contract['parent_attempt_files'],
    prior_environments_preserved=prior_guard(root)==contract['prior_guard'],
    free_bytes=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
    remote_writes=False,solver_rerun=False,models_loaded=False,images_read=False,jobs_created=False)
print(json.dumps(result))
'''


def remote_code():
    code = runtime_files()['s0_build_contract.py']+'\n'+held_helpers()+'\n'+CODE
    compile(code, 'read_only_CPU_R2_pre_submission_failure', 'exec')
    return code


def main():
    with session('inspect_s0_cpu_r2_submission') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result = json.loads(access.ssh(command,timeout=300).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__ == '__main__':
    main()
