"""Bounded sizes of experiment batches; no data or checkpoint contents read."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access=transport('readonly_storage_experiments')
    code=r'''
import datetime,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen')
workspace=home/'projects/ECNU_EYU_data/work_space'
targets=[(workspace/'S2/experimental_unvalidated',100),
         (workspace/'G1/results/experimental_unvalidated',35),
         (workspace/'G1/code/brats2025-latent-ensemble-synthesis-main-v3',30)]
result=[]
for path,timeout in targets:
    if not path.is_dir():result.append({'path':str(path),'exists':False});continue
    started=time.monotonic()
    try:
        r=subprocess.run(['du','-x','-B1','--max-depth=1',str(path)],capture_output=True,text=True,timeout=timeout)
        output=r.stdout;stderr=r.stderr;rc=r.returncode
    except subprocess.TimeoutExpired as e:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        output=decode(e.stdout);stderr=decode(e.stderr);rc=124
    records=[]
    for line in output.splitlines():
        f=line.split('\t',1)
        if len(f)==2 and f[0].isdigit():records.append({'path':f[1],'allocated_bytes':int(f[0])})
    total=next((r['allocated_bytes'] for r in records if r['path']==str(path)),None)
    result.append({'path':str(path),'complete':rc==0 and total is not None,'allocated_bytes':total,
                   'exit_code':rc,'elapsed_seconds':round(time.monotonic()-started,3),
                   'records':sorted(records,key=lambda r:r['allocated_bytes'],reverse=True),'stderr':stderr[-2500:]})
files=[]
for p in home.iterdir():
    if p.is_file() and not p.is_symlink():
        s=p.stat()
        if s.st_size>=1024**3:files.append({'path':str(p),'logical_bytes':s.st_size,'allocated_bytes':s.st_blocks*512})
payload={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
         'observations':result,'home_top_level_large_files':sorted(files,key=lambda r:r['allocated_bytes'],reverse=True),
         'method':'bounded du -x -B1 --max-depth=1, includes hidden contents, no symlink dereference; top-level file stat only'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(payload,sort_keys=True),flags=re.I))
'''
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=195).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'STORAGE_EXPERIMENTS.json',data)
        print(json.dumps({'audit':str(access.audit),'observations':[{k:o.get(k) for k in ('path','allocated_bytes','complete','elapsed_seconds')} for o in data['observations']], 'largest':[{ 'parent':o['path'],'entries':o.get('records',[])[:8]} for o in data['observations']]},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
