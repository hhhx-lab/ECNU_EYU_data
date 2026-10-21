"""Bounded targeted metadata accounting after an incomplete home walk."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_storage_targeted')
    code = r'''
import datetime,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen')
project=home/'projects/ECNU_EYU_data/work_space'
targets=[(project/'S2',50),(project/'G1',40),(home/'data',35),(project/'S1',10),(project/'G2',10)]
observations=[]
for path,budget in targets:
    if not path.exists():observations.append({'path':str(path),'exists':False});continue
    started=time.monotonic()
    try:
        r=subprocess.run(['du','-x','-B1','--max-depth=2',str(path)],capture_output=True,text=True,timeout=budget)
        stdout=r.stdout;stderr=r.stderr;exit_code=r.returncode
    except subprocess.TimeoutExpired as e:
        decode=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
        stdout=decode(e.stdout);stderr=decode(e.stderr);exit_code=124
    records=[]
    for line in stdout.splitlines():
        f=line.split('\t',1)
        if len(f)==2 and f[0].isdigit():records.append({'allocated_bytes':int(f[0]),'path':f[1]})
    total=next((r['allocated_bytes'] for r in records if r['path']==str(path)),None)
    observations.append({'path':str(path),'exists':True,'exit_code':exit_code,'complete':exit_code==0 and total is not None,
                         'allocated_bytes':total,'records':records,'stderr':stderr[-4000:],'elapsed_seconds':round(time.monotonic()-started,3)})
s=os.statvfs(home)
listing=[]
for directory in (home,home/'projects',project,project/'G1',project/'S2',home/'data'):
    if directory.is_dir():
        listing.append({'parent':str(directory),'entries':[{'name':p.name,'symlink':p.is_symlink()} for p in sorted(directory.iterdir())]})
payload={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
         'targeted_observations':observations,'directory_inventory':listing,
         'home_capacity':{'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize},
         'method':'targeted du -x -B1 --max-depth=2, bounded per target; no symlink dereference or remote writes'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(payload,sort_keys=True),flags=re.I))
'''
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=185).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'STORAGE_TARGETED.json',data)
        print(json.dumps({'audit':str(access.audit),'observations':data['targeted_observations'],'home_capacity':data['home_capacity']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
