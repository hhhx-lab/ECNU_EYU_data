"""One immediate bounded read-only snapshot, including capacity headers, never polling."""
import datetime
import json
import shlex
from common import ROOT, REMOTE, RECON, SOURCE_CODE, m, transport


def main():
    access = transport('snapshot')
    code = r'''
import csv,datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(REMOTE_ROOT);recon=pathlib.Path(RECON_ROOT);source=pathlib.Path(SOURCE_ROOT)
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=25)
 return {'exit_code':r.returncode,'stdout':r.stdout[-12000:],'stderr':r.stderr[-2000:]}
def small(path):
 if not path.exists():return {'exists':False}
 p=json.loads(path.read_text()); return {'exists':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**{k:p.get(k) for k in ('status','cases','images','labels','checkpoint_sha256','split_counts','rows','global_step')}}
def logs(path):
 if not path.is_file():return {'exists':False}
 with path.open('rb') as h:h.seek(max(0,path.stat().st_size-10000));s=h.read().decode(errors='replace')
 return {'exists':True,'bytes':path.stat().st_size,'tail':s[-2000:],'critical':re.findall(r'^.*(?:Traceback|CUDA error|OutOfMemory|ImportError|No space left|nonfinite|RuntimeError|DUE TO TIME LIMIT).*$',s,re.I|re.M)[-8:]}
jobs=['3941042','4058680']
if root.exists():
 for p in (root/'evidence').glob('*_CANONICAL_JOB.json'):jobs.append(json.loads(p.read_text())['job_id'])
data={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'public_disk':cmd(['df','-h','/public']),
 'account_disk':cmd(['df','-B1',str(recon.parent)]),'available_bytes':os.statvfs(recon.parent).f_bavail*os.statvfs(recon.parent).f_frsize,
 'new_root_exists':root.exists(),'squeue':cmd(['squeue','-h','-j',','.join(jobs),'-o','%i|%T|%j|%P|%r|%b|%M']),
 'sacct':cmd(['sacct','-n','-P','-j',','.join(jobs),'--format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES']),
 'user_squeue':cmd(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r|%b']),
 'partitions':{p:cmd(['scontrol','show','partition',p]) for p in ('cpu_96G','a100')},
 'resources':cmd(['sinfo','-N','-h','-p','a100','-o','%N|%t|%G|%m|%c']),
 'qos':cmd(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU']),
 'reconstruction_complete':small(recon/'dataset/COMPLETE.ok'),
 'source_csv_sha256':hashlib.sha256((source/'data/data_csv.csv').read_bytes()).hexdigest()}
if root.exists():
 data['package_sha256']=hashlib.sha256((root/'plan/PACKAGE_MANIFEST.json').read_bytes()).hexdigest() if (root/'plan/PACKAGE_MANIFEST.json').exists() else None
 data['evidence']={str(p.relative_to(root)):small(p) for p in [root/'dataset/STARTED.json',root/'dataset/FAILED.json',root/'dataset/COMPLETE.ok',*(root/'training'/m/n for m in ('t1c','t1n','t2w','t2f') for n in ('STARTED.json','RUN_CONTRACT.json','FAILED.json','SAFE_TERM.json','LATEST_CHECKPOINT.json','COMPLETE.ok'))]}
 data['logs']={p.name:logs(p) for p in (root/'logs').glob('*') if p.is_file()}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(data,sort_keys=True),flags=re.I))
'''
    for key, value in {'REMOTE_ROOT': str(REMOTE), 'RECON_ROOT': str(RECON), 'SOURCE_ROOT': str(SOURCE_CODE)}.items():
        code = code.replace(key, repr(value))
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=200).strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SNAPSHOT.json', result)
        print(json.dumps({'audit': str(access.audit), **result}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
