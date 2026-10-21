"""One progress snapshot, immutable old audit pull and bounded capacity probe."""
import base64
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, META, transport

CODE = r'''
import base64,datetime,hashlib,json,os,pathlib,subprocess
base=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated')
old=base/'s2_k1_small_lesion_p0_20261015_r3_diagnostics'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=45)
 return dict(return_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
def capacity(p):
 try:
  s=os.statvfs(p)
  return dict(available_bytes=s.f_bavail*s.f_frsize,total_bytes=s.f_blocks*s.f_frsize,device=os.stat(p).st_dev)
 except OSError as e:return dict(unavailable=str(e))
out=dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
 squeue=cmd(['squeue','-h','-j','4087034','-o','%i|%j|%T|%R']),
 sacct=cmd(['sacct','-n','-X','-j','4087034','-P','--format=JobID,JobName,State,ExitCode,Elapsed']),
 old_scontrol=cmd(['scontrol','show','job','4087034','-o']),
 cpu_partition=cmd(['scontrol','show','partition','cpu_192G','-o']),
 cpu_nodes=cmd(['sinfo','-h','-p','cpu_192G','-N','-o','%N|%m|%t']),
 capacity={p:capacity(p) for p in (str(base),'/public/home/zqchen','/tmp','/scratch','/dev/shm')},
 df=cmd(['df','-B1',str(base)]),old_files={})
assert sha(old/'CPU_EXECUTION_PLAN.json')=='471ce8b9ff87b4aaf1a83ca8eee86cc9f0c4525d83e7fd8c632ad42a6f6236bd'
for folder in ('evidence','logs'):
 for p in sorted((old/folder).glob('*')):
  if p.is_file() and p.stat().st_size<=2*1024*1024:
   b=p.read_bytes();out['old_files'][str(p.relative_to(old))]=dict(sha256=hashlib.sha256(b).hexdigest(),content_base64=base64.b64encode(b).decode())
out['completed_remote_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print('SNAPSHOT='+json.dumps(out))
'''


def main():
    t = transport(); access = t.Access('inspect_once')
    start = datetime.now(timezone.utc).isoformat(); status = 'failed'
    try:
        access.start_vpn()
        raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + t.PYTHON + ' -B -c ' + shlex.quote(CODE), timeout=180)
        data = json.loads(next(s.split('=', 1)[1] for s in raw.splitlines() if s.startswith('SNAPSHOT=')))
        t.exclusive(access.audit / 'SNAPSHOT.json', dict(**META, **data))
        for name, record in data['old_files'].items():
            path = access.audit / 'old_exact' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f: f.write(base64.b64decode(record['content_base64']))
            assert t.sha(path) == record['sha256']
        s = data['sacct']
        assert s['return_code'] == 0 and '4087034|' in s['stdout']
        assert 'CANCELLED' in s['stdout'], 'old canonical job cancellation not established'
        v = os.statvfs(ROOT)
        t.exclusive(access.audit / 'LOCAL_STORAGE.json', dict(**META, available_bytes=v.f_bavail*v.f_frsize,
            local_utc=datetime.now(timezone.utc).isoformat(), probability_storage_authorized=False))
        status = 'pass_old_cancelled_and_capacity_observed'
        print(json.dumps(dict(status=status,audit=str(access.audit),old_job=s['stdout'],
            remote_available_bytes=data['capacity'][str(Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated'))]['available_bytes'],
            old_lightweight_files=len(data['old_files']),remote_utc=data['remote_utc'])))
    finally:
        access.close_vpn()
        t.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(**META,status=status,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),
            vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,
            progress_snapshots=1,remote_writes=False))


if __name__ == '__main__': main()
