"""Read-only reconciliation after the sole Bio full103 sbatch returned nonzero."""
import json
import os
import shlex
from session import session, access_module


CODE = r'''
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT']);base=root/'attempts/full103_bio_r1'
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=45)
 assert len(p.stdout)+len(p.stderr)<131072,'bounded submission diagnosis'
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
reports={}
for name in ('SUBMISSION_FAILED_hold.json','RESOURCE_SNAPSHOT.json','SUBMISSION_RESERVED.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json','HELD_JOB_MAP.json','RELEASE_RESERVED.json','SUBMISSION_RECEIPT.json'):
 path=base/'operator'/name
 if path.is_file():
  raw=path.read_bytes();assert len(raw)<131072,'lightweight operator report budget'
  reports[name]=dict(data=json.loads(raw),sha256=hashlib.sha256(raw).hexdigest())
out=dict(reports=reports,
 queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b']),
 accounting=command(['sacct','-n','-P','-u','zqchen','--name=s2ext26full','--starttime=2026-11-05T00:00:00','--format=JobID,JobName,State,ExitCode,Submit,Start']),
 association=command(['sacctmgr','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs']),
 qos=command(['sacctmgr','-P','show','qos']),partition=command(['scontrol','show','partition','a100','-o']),
 remote_utc=datetime.now(timezone.utc).isoformat(),remote_writes=False,jobs_submitted=False,jobs_released=False,
 raw_original_sbatch_error_discarded_by_assertion=True,no_original_cause_inferred=True,automatic_retry=False)
print(json.dumps(out))
'''


def main():
    compile(CODE,'read_only_Bio_submission_reconciliation','exec')
    with session('inspect_bio_submission_failure') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=CODE))
        command='env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=330).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),result=result)),flush=True)


if __name__=='__main__':main()
