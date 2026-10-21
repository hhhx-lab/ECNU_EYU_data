"""Bounded account storage/quota inventory; no remote writes or file contents."""

import datetime
import json
import shlex

from common import REMOTE, m, transport


def main():
    access = transport('readonly_storage_inventory')
    code = r'''
import datetime,json,os,pathlib,re,shutil,subprocess,time
home=pathlib.Path('/public/home/zqchen')
project=home/'projects/ECNU_EYU_data'
experiment=pathlib.Path(EXPERIMENT)
def command(args,timeout=15):
    started=time.monotonic()
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-8000:],
                'elapsed_seconds':round(time.monotonic()-started,3),'timed_out':False}
    except subprocess.TimeoutExpired as e:
        decode=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
        return {'exit_code':124,'stdout':decode(e.stdout),'stderr':decode(e.stderr)[-8000:],
                'elapsed_seconds':round(time.monotonic()-started,3),'timed_out':True}
def capacity(path):
    if not path.exists(): return {'exists':False}
    s=os.statvfs(path)
    return {'exists':True,'resolved':str(path.resolve()),'device_id':path.stat().st_dev,
            'total_bytes':s.f_blocks*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize,
            'available_bytes':s.f_bavail*s.f_frsize,'total_inodes':s.f_files,
            'free_inodes':s.f_ffree,'available_inodes':s.f_favail}
paths=[pathlib.Path('/public'),home,home/'data',home/'projects',project,project/'work_space',experiment]
tool_names=('quota','findmnt','lfs','mmlsquota','xfs_quota')
available={name:shutil.which(name) for name in tool_names}
quota=command([available['quota'],'-vs','-u','zqchen'],25) if available['quota'] else {'available':False}
mounts=command([available['findmnt'],'-T',str(home),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],10) if available['findmnt'] else {'available':False}
before={str(path):capacity(path) for path in paths}
# Walk metadata once. Do not dereference source/data symlinks or cross filesystems.
du=command(['du','-x','-B1','--max-depth=4',str(home)],150)
records=[]
for line in du['stdout'].splitlines():
    fields=line.split('\t',1)
    if len(fields)==2 and fields[0].isdigit():
        records.append({'path':fields[1],'allocated_bytes':int(fields[0])})
by_path={r['path']:r for r in records}
top=[r for r in records if pathlib.Path(r['path']).parent==home]
modules=[r for r in records if pathlib.Path(r['path']).parent==project/'work_space']
largest=sorted(records,key=lambda r:r['allocated_bytes'],reverse=True)[:100]
storage={'remote_utc_started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'hostname':os.uname().nodename,'identity':command(['id'],5),
         'capacity_before':before,'df_bytes':command(['df','-B1']+[str(p) for p in paths if p.exists()],15),
         'df_inodes':command(['df','-i',str(home)],10),'quota':quota,'available_tools':available,
         'mount':mounts,'home_du':du,'home_total':by_path.get(str(home)),
         'top_level_directories':sorted(top,key=lambda r:r['allocated_bytes'],reverse=True),
         'project_modules':sorted(modules,key=lambda r:r['allocated_bytes'],reverse=True),
         'largest_reported_directories':largest,'reported_directory_count':len(records),
         'du_walk_complete':du['exit_code']==0 and str(home) in by_path,
         'capacity_after':capacity(home),'remote_utc_finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'method':'du -x -B1 --max-depth=4; allocated sizes, includes hidden files, no symlink dereference; nested sizes must not be added together'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(storage,sort_keys=True),flags=re.I))
'''.replace('EXPERIMENT', repr(str(REMOTE)))
    compile(code, '<readonly_storage_inventory>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code), timeout=235).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'STORAGE_INVENTORY.json',data)
        print(json.dumps({k:data[k] for k in ('local_utc','hostname','capacity_after','quota','home_total','du_walk_complete','top_level_directories','project_modules','reported_directory_count')},ensure_ascii=False))
        print('AUDIT='+str(access.audit))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
