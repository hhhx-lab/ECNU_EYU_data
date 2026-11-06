"""One read-only diagnosis of the before-sbatch CPU comparison resource gate."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
root=Path(REMOTE_ROOT);attempt=root/'attempts/compare_2025_K1_E_r1'
failed=attempt/'operator/FAILED_hold.json'
files={str(p.relative_to(attempt)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (attempt/'operator').glob('*') if p.is_file()}
print(json.dumps(dict(remote_utc=datetime.now(timezone.utc).isoformat(),failed=json.loads(failed.read_text()),operator_files=files,
 partition=command(['scontrol','show','partition','cpu_192G','-o']),qos=command(['sacctmgr','-P','show','qos']),
 queue=command(['squeue','-r','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b|%C']),
 comparison_accounting=command(['sacct','-n','-P','-u','zqchen','--name=s2ext25k1e','--starttime=2026-11-06T00:00:00','--format=JobID,JobName,State,ExitCode']),
 Bio_queue=command(['squeue','-h','-j','4319113','-o','%i|%T|%M|%R']),remote_writes=False)))
'''


def main():
 code='REMOTE_ROOT='+repr(os.environ['EXTERNAL_MET_REMOTE_ROOT'])+'\n'+CODE
 with session('diagnose_comparison_resources') as access:
  access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=code))
  result=json.loads(access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code),timeout=180).strip().splitlines()[-1])
  access_module.exclusive(access.audit/'RESULT.json',dict(access_module.META,**result))
  print(json.dumps(dict(audit=str(access.audit),Bio_queue=result['Bio_queue'],comparison_accounting=result['comparison_accounting'])),flush=True)


if __name__=='__main__':main()
