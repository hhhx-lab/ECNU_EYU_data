"""Read-only SegMamba directory size and public project identity inventory."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_segmamba_inventory')
    code = r'''
import datetime,json,os,pathlib,re,subprocess,time
root=pathlib.Path('/public/home/zqchen/SegMamba-main')
started=time.monotonic()
try:
    r=subprocess.run(['du','-x','-B1','--max-depth=3',str(root)],capture_output=True,text=True,timeout=100)
    stdout=r.stdout;stderr=r.stderr;rc=r.returncode
except subprocess.TimeoutExpired as e:
    decode=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
    stdout=decode(e.stdout);stderr=decode(e.stderr);rc=124
records=[]
for line in stdout.splitlines():
    f=line.split('\t',1)
    if len(f)==2 and f[0].isdigit():records.append({'path':f[1],'allocated_bytes':int(f[0])})
entries=[]
for p in sorted(root.iterdir()):
    s=p.lstat()
    entries.append({'name':p.name,'directory':p.is_dir(),'symlink':p.is_symlink(),'logical_bytes':s.st_size,
                    'link_target':os.readlink(p) if p.is_symlink() else None})
readme={}
for name in ('README.md','README.txt'):
    p=root/name
    if p.is_file():readme[name]=p.read_text(errors='replace')[:8000]
files=[]
checkpoint_names=[]
# Metadata for selected bulky directories, bounded independently of du.
remaining=45
scan_start=time.monotonic()
for base in ('nnUNet_results','nnUNet_preprocessed','nnUNet_raw','results','data','checkpoint','checkpoints','outputs'):
    directory=root/base
    if not directory.is_dir():continue
    count=0
    for parent,dirs,names in os.walk(directory,followlinks=False):
        dirs[:]=[d for d in dirs if not pathlib.Path(parent,d).is_symlink()]
        if time.monotonic()-scan_start>remaining:break
        for name in names:
            p=pathlib.Path(parent,name)
            try:s=p.lstat()
            except OSError:continue
            count+=1
            if p.is_symlink():continue
            if s.st_size>=1024**3:files.append({'path':str(p),'logical_bytes':s.st_size,'allocated_bytes':s.st_blocks*512})
            if name in ('checkpoint_best.pth','checkpoint_final.pth','checkpoint_latest.pth') and len(checkpoint_names)<150:
                checkpoint_names.append({'path':str(p),'logical_bytes':s.st_size})
        if count>200000:break
payload={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
         'root':str(root),'directory_entries':entries,'readme_excerpt':readme,
         'du_exit_code':rc,'du_complete':rc==0 and any(r['path']==str(root) for r in records),
         'du_elapsed_seconds':round(time.monotonic()-started,3),'du_stderr':stderr[-4000:],
         'directory_sizes':sorted(records,key=lambda r:r['allocated_bytes'],reverse=True),
         'large_files':sorted(files,key=lambda r:r['allocated_bytes'],reverse=True)[:60],
         'checkpoint_paths':checkpoint_names,'selected_metadata_scan_budget_seconds':remaining,
         'method':'du -x allocated bytes, no symlink dereference; README excerpt and file metadata only; no checkpoint load or image content read'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(payload,sort_keys=True),flags=re.I))
'''
    compile(code, '<readonly_segmamba_inventory>', 'exec')
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=180).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'SEGMAMBA_INVENTORY.json',data)
        print(json.dumps({'audit':str(access.audit),'remote_utc':data['remote_utc'],'complete':data['du_complete'],
                          'sizes':data['directory_sizes'][:28], 'entries':[r['name'] for r in data['directory_entries']],
                          'readme':data['readme_excerpt'],'checkpoint_paths':data['checkpoint_paths'][:12]},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
