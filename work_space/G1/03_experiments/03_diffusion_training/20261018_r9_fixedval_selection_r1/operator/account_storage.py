"""Bounded read-only account capacity check; no recursive data scan."""
import datetime
import json
import shlex
from common import m, transport

CODE = r'''
import datetime,json,os,pathlib,shutil,subprocess
def command(argv):
 try:
  v=subprocess.run(argv,capture_output=True,text=True,timeout=25)
  return dict(argv=argv,returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
 except subprocess.TimeoutExpired:
  return dict(argv=argv,status='timeout',timeout_seconds=25)
paths=['/public','/public/home','/public/home/zqchen',str(pathlib.Path.home()),
 '/public/home/zqchen/projects/ECNU_EYU_data',
 '/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_r9_fixedval_selection_20261018_r1']
observations={}
for name in dict.fromkeys(paths):
 p=pathlib.Path(name)
 if not p.exists():observations[name]=dict(exists=False);continue
 s=p.stat();v=os.statvfs(p)
 observations[name]=dict(exists=True,resolved_path=str(p.resolve()),owner_uid=s.st_uid,device=s.st_dev,
  total_bytes=v.f_blocks*v.f_frsize,available_bytes=v.f_bavail*v.f_frsize,
  free_bytes=v.f_bfree*v.f_frsize,inodes=v.f_files,inodes_free=v.f_favail,
  df=command(['df','-B1',str(p)]),inode_df=command(['df','-i',str(p)]),
  mount=command(['findmnt','-T',str(p),'-n','-o','SOURCE,TARGET,FSTYPE,OPTIONS']) if shutil.which('findmnt') else None)
quota=command(['quota','-v','-u','zqchen']) if shutil.which('quota') else dict(status='quota_command_unavailable')
print(json.dumps(dict(status='read_only_account_storage',remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 hostname=os.uname().nodename,uid=os.getuid(),account=command(['id','-un']),paths=observations,quota=quota,
 user_report='account storage expansion requested',scope='df/statvfs/inodes/mount/quota only; no writes, du, dataset/model reads or jobs')))
'''


def main():
    access=transport('account_storage');started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=None;started_ok=False
    try:
        access.start_vpn();started_ok=True
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(CODE),timeout=240)
        result=json.loads(out.strip().splitlines()[-1])
        result['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'ACCOUNT_STORAGE.json',result)
        print(json.dumps(dict(audit=str(access.audit),**result)))
    finally:
        if started_ok:access.close_vpn()
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            local_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            startup_confirmed=started_ok,vpn_owned=access.owns_vpn if started_ok else None,
            preexisting_vpn_left_unchanged=started_ok and not access.owns_vpn,
            remote_snapshot_created=result is not None,read_only=True))


if __name__=='__main__':main()
