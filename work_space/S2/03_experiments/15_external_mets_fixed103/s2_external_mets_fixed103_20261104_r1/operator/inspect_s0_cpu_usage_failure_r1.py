"""One read-only diagnosis of the failed own-job partition comparison."""
import json
import os
import shlex

from session import session, access_module
from s0_cpu_r2_sources import runtime_files, held_helpers


CODE = r'''
import os, shutil, subprocess
from datetime import datetime, timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r2'
audit=attempt/'operator/submit_usage_r1'
failed=audit/'SUBMISSION_FAILED.json'
require(sha(failed)=='9fa1bd3fd29656328d8d35b6fd9ae48f375bf79a197e665714b28ac5d11c5f75','CPU usage failure identity changed')
failure=json.loads(failed.read_text())
require(failure['job_id'] is None and failure['stage']=='CPU_package_and_fresh_scheduler_preflight','unexpected CPU usage submission state')
require(failure['message']=='CPU usage partition changed','unexpected CPU usage failure cause')
contract_path=attempt/'contract/PREPARATION_CONTRACT.json'
contract=json.loads(contract_path.read_text())
require(sha(contract_path)=='8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980','CPU contract drift')

def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    require(len(p.stdout.encode())+len(p.stderr.encode())<131072,'CPU diagnosis output budget')
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

default_queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'])
wide_queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%128P|%128j|%r|%b'])
require(default_queue['returncode']==wide_queue['returncode']==0,'CPU diagnosis queue query failed')
records={}
for line in wide_queue['stdout'].splitlines():
    if not line:continue
    values=[x.strip() for x in line.split('|')]
    require(len(values)==6 and re.fullmatch(r'[0-9]+',values[0]),'CPU diagnosis own queue schema')
    require(values[0] not in records and len(records)<100,'CPU diagnosis own JobID budget')
    records[values[0]]=values
jobs={}
for identity,values in records.items():
    raw=command(['scontrol','show','job',identity,'-o'])
    fields=job_fields(raw['stdout']) if raw['returncode']==0 else {}
    if fields:
        require(fields.get('JobId')==identity and fields.get('UserId','').split('(')[0]=='zqchen','CPU diagnosis owner drift')
    jobs[identity]=dict(raw=raw,fields=fields,wide_queue_fields=values,
        partition_exact_match=fields.get('Partition')==values[2],
        state_exact_match=fields.get('JobState')==values[1])
after=command(['squeue','-h','-u','zqchen','-o','%i|%T|%128P|%128j|%r|%b'])
names=['RESERVATION.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json','RELEASE_RESULT.json','SUBMISSION_RECEIPT.json']
submission_files={str(directory.relative_to(attempt)): {name:(directory/name).exists() for name in names}
    for directory in [attempt/'operator',audit]}
result=dict(META,status='read_only_CPU_usage_partition_diagnosis_not_retry',
    failure=failure,failure_sha256=sha(failed),original_failure_sha256=sha(attempt/'operator/SUBMISSION_FAILED.json'),
    default_queue=default_queue,wide_queue=wide_queue,after_queue=after,
    wide_queue_unchanged=wide_queue['returncode']==after['returncode']==0 and wide_queue['stdout']==after['stdout'],
    own_jobs=jobs,submission_files=submission_files,
    qos=command(['sacctmgr','-P','show','qos']),quota=command(['quota','-s']),
    contract_sha256=sha(contract_path),slurm_sha256=sha(attempt/'slurm/prepare.slurm'),
    runtime_identity_unchanged=all(sha(attempt/name)==digest for name,digest in contract['package_files'].items()),
    build_identity_unchanged=all(sha(Path(contract['build_env'])/name)==digest for name,digest in contract['build_env_files'].items()),
    parent_CPU_staging_unchanged=parent_staging_guard(root)==contract['parent_attempt_files'],
    prior_environments_preserved=prior_guard(root)==contract['prior_guard'],
    free_bytes=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
    remote_writes=False,solver_rerun=False,jobs_created=False,models_loaded=False,images_read=False)
print(json.dumps(result))
'''


def remote_code():
    code=runtime_files()['s0_build_contract.py']+'\n'+held_helpers()+'\n'+CODE
    compile(code,'read_only_CPU_usage_partition_failure_diagnosis','exec')
    return code


def main():
    code=remote_code()
    with session('inspect_s0_cpu_usage_failure_r1') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=300).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':
    main()
