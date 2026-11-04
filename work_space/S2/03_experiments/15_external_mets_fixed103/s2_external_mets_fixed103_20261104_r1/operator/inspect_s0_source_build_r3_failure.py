"""Read-only scheduler diagnosis and acceptance identities; never reruns solver."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import subprocess,shutil
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/'operator/s0_source_build_preflight_r3'
failure=json.loads((audit/'FAILED.json').read_text())
require(failure['stage']=='fresh_slurm_metadata' and failure['message']=='scheduler metadata unavailable: qos','unexpected R3 failure')
require(sha(audit/'SOLVER.json')==failure['solver_sha256'],'successful solver identity drift')
plan=json.loads((audit/'SOLVER.json').read_text())
selected=validate_build_plan(plan)
records=explicit_lock_records(plan)
require((audit/'BUILD_ENV_EXPLICIT.txt').read_text()==render_explicit_lock(records),'accepted solver lock differs')
commands={
    'qos_default':['sacctmgr','-n','-P','show','qos'],
    'qos_corrected_fields':['sacctmgr','-n','-P','show','qos','format=Name,MaxWall,MaxTRESPJ,MaxJobsPU,MaxSubmitJobsPU'],
    'partition':['scontrol','show','partition','cpu_192G','-o'],
    'nodes':['sinfo','-N','-h','-p','cpu_192G','-o','%N|%t|%c|%m|%G'],
    'association':['sacctmgr','-n','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
    'own_queue':['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b']}
scheduler={}
for name,command in commands.items():
    p=subprocess.run(command,capture_output=True,text=True,timeout=30)
    require(len(p.stdout)+len(p.stderr)<131072,'bounded scheduler metadata exceeded')
    scheduler[name]=dict(argv=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
started=json.loads((audit/'STARTED.json').read_text())
result=dict(META,status='read_only_R3_solver_and_scheduler_diagnosis_not_execution_acceptance',
    solver_sha256=sha(audit/'SOLVER.json'),explicit_lock_sha256=sha(audit/'BUILD_ENV_EXPLICIT.txt'),
    failed_marker_sha256=sha(audit/'FAILED.json'),selected_packages=selected,explicit_packages=records,
    effective_config=started['effective_config'],host_libc=started['host_libc'],scheduler=scheduler,
    free_bytes=shutil.disk_usage(root).free,prior_environments_preserved=prior_guard(root)==started['before_guard'],
    build_environment_exists=(root/'envs'/BUILD_ENV).exists(),runtime_R4_exists=(root/'envs/bio_intel_py310_r4').exists(),
    solver_rerun=False,jobs_created=False,remote_writes=False,models_loaded=False,
    remote_utc=datetime.now(timezone.utc).isoformat())
print(json.dumps(result))
'''


def remote_code():
    source=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    source+=(ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'+CODE
    compile(source,'s0_R3_read_only_diagnosis','exec')
    return source


def main():
    with session('inspect_s0_source_build_r3_failure') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=300).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':main()
