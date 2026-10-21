"""Account remaining top-level allocated sizes; read-only bounded metadata scan."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access=transport('readonly_storage_remaining')
    code=r'''
import datetime,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen')
workspace=home/'projects/ECNU_EYU_data/work_space'
targets=[home/n for n in ('zjd_projects','hzj_project','cpg_project','yy_mambacpg','yy_grapecpg','zdn_deepcpg','SegMamba-main','g1_diffusion_v3_production_20260715','g1_diffusion_v3_smoke_20260715','run','runs','Rlibs','dogs-vs-cats')]
targets += [home/'projects'/n for n in ('zjd_projects','financial_ts_20260903','financial_ts_overlap_20260819_ecnu')]
targets += [workspace/n for n in ('S3','S4')]
observations=[]
overall=time.monotonic()
for path in targets:
    if not path.is_dir():continue
    if time.monotonic()-overall>170:break
    started=time.monotonic()
    try:
        r=subprocess.run(['du','-x','-B1','--max-depth=0',str(path)],capture_output=True,text=True,timeout=12)
        stdout=r.stdout;stderr=r.stderr;rc=r.returncode
    except subprocess.TimeoutExpired as e:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        stdout=decode(e.stdout);stderr=decode(e.stderr);rc=124
    total=None
    for line in stdout.splitlines():
        f=line.split('\t',1)
        if len(f)==2 and f[0].isdigit() and f[1]==str(path):total=int(f[0])
    observations.append({'path':str(path),'allocated_bytes':total,'complete':rc==0 and total is not None,
                         'exit_code':rc,'stderr':stderr[-1000:],'elapsed_seconds':round(time.monotonic()-started,3)})
payload={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
         'observations':observations,'method':'metadata-only du -x -B1 --max-depth=0; per-target 12s; no symlink dereference or writes'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(payload,sort_keys=True),flags=re.I))
'''
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=215).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'STORAGE_REMAINING.json',data)
        print(json.dumps({'audit':str(access.audit),'observations':data['observations']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
