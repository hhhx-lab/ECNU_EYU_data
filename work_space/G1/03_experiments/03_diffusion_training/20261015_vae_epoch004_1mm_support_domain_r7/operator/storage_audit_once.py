"""One bounded, read-only storage audit for the active workspace."""
import datetime
import json
import shlex
from pathlib import Path

from common import REMOTE, m, transport


def main():
    access = transport('storage_audit')
    code = r'''
import datetime,json,os,pathlib,subprocess,time

def cmd(argv, timeout=25):
    started=time.monotonic()
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return {'returncode':r.returncode,'stdout':r.stdout[-20000:],'stderr':r.stderr[-4000:],
                'seconds':round(time.monotonic()-started,3),'complete':r.returncode==0}
    except subprocess.TimeoutExpired as e:
        dec=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        return {'returncode':124,'stdout':dec(e.stdout)[-20000:],'stderr':dec(e.stderr)[-4000:],
                'seconds':round(time.monotonic()-started,3),'complete':False,'timeout':True}

def statvfs(path):
    p=pathlib.Path(path)
    try:
        s=os.statvfs(p)
        return {'path':str(p),'total_bytes':s.f_blocks*s.f_frsize,
                'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize,
                'used_bytes':(s.f_blocks-s.f_bfree)*s.f_frsize}
    except OSError as e:
        return {'path':str(p),'error':str(e)}

def du(path, depth=1, timeout=40):
    return cmd(['du','-x','-B1','--max-depth='+str(depth),path],timeout)

def tree_stats(base):
    root=pathlib.Path(base)
    out={'path':str(root),'exists':root.exists(),'files':0,'dirs':0,'logical_bytes':0,
         'allocated_bytes':0,'nifti_files':0,'nifti_logical_bytes':0,'nifti_allocated_bytes':0,
         'checkpoint_files':0,'checkpoint_logical_bytes':0,'completed_case_dirs':0,
         'preprocess_complete_markers':0,'errors':[]}
    if not root.exists(): return out
    try:
        for directory, dirs, files in os.walk(root, followlinks=False):
            out['dirs'] += len(dirs)
            if pathlib.Path(directory).name.startswith('BraTS-'):
                out['completed_case_dirs'] += 1
            for name in files:
                p=pathlib.Path(directory)/name
                try:
                    if p.is_symlink(): continue
                    s=p.stat(); out['files'] += 1; out['logical_bytes'] += s.st_size
                    out['allocated_bytes'] += s.st_blocks*512
                    if name.endswith('.nii.gz') or name.endswith('.nii'):
                        out['nifti_files'] += 1; out['nifti_logical_bytes'] += s.st_size; out['nifti_allocated_bytes'] += s.st_blocks*512
                    if name.endswith(('.pth','.pt','.ckpt')):
                        out['checkpoint_files'] += 1; out['checkpoint_logical_bytes'] += s.st_size
                    if name == 'preprocess_complete.json': out['preprocess_complete_markers'] += 1
                except OSError as e:
                    out['errors'].append(str(e))
    except OSError as e:
        out['errors'].append(str(e))
    return out

home='/public/home/zqchen'
projects=home+'/projects/ECNU_EYU_data'
ws=projects+'/work_space'
g1=projects+'/work_space/G1'
g2=projects+'/work_space/G2'
s1=projects+'/work_space/S1'
s2=projects+'/work_space/S2'
r7='/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261015_support_domain_r7'
selected={}
for p in [home,projects,ws,g1,g2,s1,s2,'/tmp','/dev/shm']:
    selected[p]={'statvfs':statvfs(p),'du_depth1':du(p,1,45)}
commands={}
for name,args in {
    'df_home':['df','-B1',home], 'df_public':['df','-B1','/public'],
    'quota':['quota','-s'], 'quota_v':['quota','-v'],
    'squeue_g1':['squeue','-h','-j','4096309,4096310,4096311,4096312,4096313','-o','%i|%T|%j|%P|%r|%b|%M'],
    'sacct_g1':['sacct','-n','-P','-j','4096309,4096310,4096311,4096312,4096313','--format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES'],
}.items(): commands[name]=cmd(args,35)
result={
 'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hostname':os.uname().nodename,
 'account_path':home, 'selected_capacity':selected, 'commands':commands,
 'r7_root_du':du(r7,1,60), 'r7_dataset_du':du(r7+'/dataset',1,60),
 'r7_training_du':du(r7+'/training',1,60),
 'r7_dataset_stats':tree_stats(r7+'/dataset'),
 'r7_training_stats':tree_stats(r7+'/training'),
 'r7_markers':{p:cmd(['find',r7+'/'+p,'-maxdepth','3','-type','f','-printf','%p\t%s\n'],30) for p in ['dataset','training']},
 'method':'bounded targeted du/statvfs/metadata walk; no content hashing and no image/probability transfer'
}
print(json.dumps(result,sort_keys=True))
'''.replace('__ROOT__', repr(str(REMOTE)))
    try:
        access.start_vpn()
        out = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=300)
        value = json.loads(out.strip().splitlines()[-1])
        value.update(m.META)
        value['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'STORAGE_AUDIT.json', value)
        print(json.dumps({'audit': str(access.audit), **value}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
