"""One bounded read-only storage inventory; no data transfer, mutation or job control."""
import datetime
import json
import shlex
from common import REMOTE, RECON, SOURCE_CODE, m, transport


def main():
    access = transport('storage_readonly')
    code = r'''
import collections,datetime,json,os,pathlib,re,subprocess,time
remote=pathlib.Path(__REMOTE__);recon=pathlib.Path(__RECON__);source=pathlib.Path(__SOURCE__)
g1=remote.parents[2];home=pathlib.Path('/public/home/zqchen')
def command(args,timeout=20):
 started=time.monotonic()
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
  return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-3000:],'seconds':time.monotonic()-started,'complete':r.returncode==0}
 except subprocess.TimeoutExpired as error:
  decode=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else v or ''
  return {'exit_code':124,'stdout':decode(error.stdout),'stderr':decode(error.stderr)[-3000:],'seconds':time.monotonic()-started,'complete':False}
def du(path,depth,timeout):
 r=command(['du','-x','-B1','--max-depth='+str(depth),str(path)],timeout)
 rows=[]
 for line in r.pop('stdout').splitlines():
  parts=line.split('\t',1)
  if len(parts)==2 and parts[0].isdigit():rows.append({'bytes':int(parts[0]),'path':parts[1]})
 return {**r,'rows':sorted(rows,key=lambda p:p['bytes'],reverse=True)}
def files_metadata(base):
 stats={'exists':base.exists(),'case_dirs':0,'nifti_files':0,'nifti_logical_bytes':0,'nifti_allocated_bytes':0,
        'checkpoint_files':0,'checkpoint_logical_bytes':0,'symlinks':0,'extensions':{},'errors':[]}
 if not base.exists():return stats
 extensions=collections.Counter();seen=set()
 for directory,dirs,files in os.walk(base,followlinks=False):
  for name in dirs:
   p=pathlib.Path(directory)/name
   if p.is_symlink():stats['symlinks']+=1
   elif name.startswith('BraTS-'):stats['case_dirs']+=1
  for name in files:
   p=pathlib.Path(directory)/name
   try:
    if p.is_symlink():stats['symlinks']+=1;continue
    s=p.stat();kind='nii.gz' if name.endswith('.nii.gz') else p.suffix or '(none)';extensions[kind]+=1
    if name.endswith(('.nii','.nii.gz')):
     stats['nifti_files']+=1;stats['nifti_logical_bytes']+=s.st_size
     key=(s.st_dev,s.st_ino)
     if key not in seen:stats['nifti_allocated_bytes']+=s.st_blocks*512;seen.add(key)
    if name.endswith(('.pth','.pt','.ckpt')):stats['checkpoint_files']+=1;stats['checkpoint_logical_bytes']+=s.st_size
   except OSError as error:stats['errors'].append(str(error))
 stats['extensions']=dict(extensions)
 return stats
def marker(path):
 if not path.exists():return {'exists':False}
 p=json.loads(path.read_text())
 return {'exists':True,**{k:p.get(k) for k in ('status','cases','split_counts','rows','global_step','message','utc','job_id')}}
def log(path):
 if not path.exists():return {'exists':False}
 with path.open('rb') as h:h.seek(max(0,path.stat().st_size-10000));text=h.read().decode(errors='replace')
 return {'exists':True,'bytes':path.stat().st_size,'tail':text[-2000:],
  'critical':re.findall(r'^.*(?:Traceback|CUDA error|OutOfMemory|ImportError|No space left|nonfinite|RuntimeError|FloatingPointError|AssertionError|DUE TO TIME LIMIT).*$',text,re.I|re.M)[-8:]}
jobs='4069915,4069916,4069917,4069918,4069919'
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
 'public_disk':command(['df','-h','/public']),'account_disk':command(['df','-B1',str(home)]),
 'available_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize,
 'squeue':command(['squeue','-h','-j',jobs,'-o','%i|%T|%j|%P|%r|%b|%M']),
 'sacct':command(['sacct','-n','-P','-j',jobs,'--format=JobID,State,ExitCode,Elapsed,AllocTRES']),
 'g1_du':du(g1,4,90),'home_du':du(home,1,90)}
roots=[p for p in sorted(remote.parent.iterdir()) if p.is_dir() and p.name.startswith(('g1_diffusion_1mm_vae_epoch004_','g1_vae_epoch004_full_reconstruction_'))]
result['new_attempts']={p.name:{'du':du(p,1,10),'data':files_metadata(p/'dataset'),'training':files_metadata(p/'training'),
 'markers':{n:marker(p/'dataset'/n) for n in ('STARTED.json','FAILED.json','COMPLETE.ok')}} for p in roots}
result['v3_data_children']=[{'name':p.name,'symlink':p.is_symlink(),'resolved':str(p.resolve())} for p in sorted((source/'data').iterdir())]
result['r5_logs']={p.name:log(p) for p in (remote/'logs').iterdir() if p.is_file()}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(result,sort_keys=True),flags=re.I))
'''
    for key,value in {'__REMOTE__':str(REMOTE),'__RECON__':str(RECON),'__SOURCE__':str(SOURCE_CODE)}.items():
        code = code.replace(key,repr(value))
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -u -c '+shlex.quote(code),timeout=420).strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'STORAGE_INVENTORY.json',result)
        print(json.dumps({'audit':str(access.audit),**result},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__=='__main__':
    main()
