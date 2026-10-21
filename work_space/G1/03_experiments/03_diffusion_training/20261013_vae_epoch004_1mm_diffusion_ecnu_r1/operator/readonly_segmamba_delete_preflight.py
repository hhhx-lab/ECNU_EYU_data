"""Read-only gate for the user-requested exact SegMamba directory deletion."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_segmamba_delete_preflight')
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen');target=home/'SegMamba-main'
project=home/'projects/ECNU_EYU_data/work_space'
def stat(p):
    s=p.lstat()
    return {'path':str(p),'symlink':p.is_symlink(),'uid':s.st_uid,'dev':s.st_dev,'inode':s.st_ino,
            'size_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink,
            'mtime_ns':s.st_mtime_ns,'directory':p.is_dir(),
            'link_target':os.readlink(p) if p.is_symlink() else None}
def run(cmd,timeout):
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-1500:]}
    except subprocess.TimeoutExpired:return {'exit_code':124,'error':'bounded_timeout'}
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'target':stat(target),'available_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize,
   'queue':run(['squeue','-h','-u','zqchen','-o','%i|%j|%T|%R'],10),
   'size':run(['du','-x','-B1','--max-depth=1',str(target)],20)}
files=[];crossdev=[];internal_links=[];errors=[];deadline=time.monotonic()+25
for directory,dirs,names in os.walk(target,followlinks=False,onerror=lambda e:errors.append(type(e).__name__)):
    if time.monotonic()>deadline:errors.append('target_inventory_timeout');break
    for name in dirs+names:
        p=pathlib.Path(directory,name)
        try:
            s=stat(p);files.append(s)
            if s['dev']!=v['target']['dev']:crossdev.append(str(p))
            if s['symlink']:internal_links.append(s)
        except OSError as e:errors.append(type(e).__name__)
v['target_manifest']=files;v['internal_links']=internal_links;v['cross_device_entries']=crossdev
refs=[];scan_errors=[];scanned=0;deadline=time.monotonic()+35
for base in (project,home/'.conda/envs'):
    for directory,dirs,names in os.walk(base,followlinks=False,onerror=lambda e:scan_errors.append(type(e).__name__)):
        if time.monotonic()>deadline:scan_errors.append('external_scan_timeout');break
        ds=pathlib.Path(directory)
        for name in dirs+names:
            p=ds/name
            try:
                if p.is_symlink() and (str(p.resolve())==str(target) or str(p.resolve()).startswith(str(target)+'/')):
                    refs.append({'path':str(p),'kind':'external_symlink','target':str(p.resolve())})
            except OSError as e:scan_errors.append(type(e).__name__)
        dirs[:]=[n for n in dirs if n not in ('.git','__pycache__','node_modules') and not (ds/n).is_symlink()]
        for name in names:
            p=ds/name
            if p.suffix.lower() not in ('.py','.sh','.slurm','.md','.json','.yaml','.yml','.pth','.csv'):continue
            if name.lower().startswith(('.env','credential','secret','token')):continue
            try:
                if p.is_symlink() or p.stat().st_size>500000:continue
                if base.name=='envs' and p.suffix!='.pth':continue
                scanned+=1
                if str(target) in p.read_text(errors='replace'):
                    refs.append({'path':str(p),'kind':'text_absolute_reference'})
            except OSError as e:scan_errors.append(type(e).__name__)
process_refs=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cwd=os.readlink(p/'cwd')
        cmd=(p/'cmdline').read_bytes().decode(errors='replace')
        # The inspection command itself necessarily contains the target literal.
        if p.name==str(os.getpid()) or 'readonly_segmamba_delete_preflight' in cmd or 'target_manifest' in cmd:continue
        mapped=(p/'maps').read_text(errors='replace')
        if cwd==str(target) or cwd.startswith(str(target)+'/') or str(target) in cmd or str(target) in mapped:
            process_refs.append({'pid':p.name,'cwd_in_target':cwd.startswith(str(target)),'cmd_reference':str(target) in cmd,'mapped_reference':str(target) in mapped})
    except OSError:continue
v.update({'external_references':refs,'external_scan_errors':scan_errors,'external_text_files_scanned':scanned,
          'target_inventory_errors':errors,'login_process_references':process_refs,
          'boundary':'read-only; project/environment references and login processes only; no deletion; no full data equality or backup claim'})
v['manifest_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''
    compile(code, '<readonly_segmamba_delete_preflight>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=115).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SEGMAMBA_DELETE_PREFLIGHT.json', data)
        print(json.dumps({'audit':str(access.audit),'target':data['target'],'size':data['size'],
                          'files':len(data['target_manifest']),'inventory_errors':data['target_inventory_errors'],
                          'external_references':data['external_references'],'scan_errors':data['external_scan_errors'],
                          'process_references':data['login_process_references'],'queue':data['queue']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
