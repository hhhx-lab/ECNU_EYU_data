"""One bounded read-only inventory; no submission and no full-volume login I/O."""

import datetime
import json
import shlex

from common import RECON, REMOTE, ROOT, SOURCE_CODE, m, transport


def main():
    access = transport('readonly_inventory')
    code = r'''
import collections,csv,datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(RECON_ROOT)
code_root=pathlib.Path(SOURCE_ROOT)
def cmd(args,timeout=20):
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return {'exit_code':r.returncode,'stdout':r.stdout[-16000:],'stderr':r.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        return {'exit_code':124,'error':'bounded_timeout'}
def file(path):
    if not path.is_file(): return {'exists':False}
    p=json.loads(path.read_text())
    return {'exists':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':p.get('status'),'cases':p.get('cases'),'images':p.get('images'),'labels':p.get('labels'),'split_counts':p.get('split_counts'),'checkpoint_sha256':p.get('checkpoint_sha256'),'source_csv_sha256':p.get('source_csv_sha256'),'manifest_count':len(p.get('manifest_sha256',{}))}
def log(path):
    if not path.is_file(): return {'exists':False}
    with path.open('rb') as h:
        h.seek(max(0,path.stat().st_size-16000)); value=h.read().decode(errors='replace')
    return {'exists':True,'bytes':path.stat().st_size,'tail':value[-5000:],'critical':re.findall(r'^.*(?:Traceback|OutOfMemory|CUDA error|ImportError|nonfinite|No space left|FAILED|RuntimeError|unbound variable).*$',value,re.I|re.M)[-12:]}
with (code_root/'data/data_csv.csv').open(newline='') as h: rows=list(csv.DictReader(h))
selected=[r for r in rows if r['id'] in {'BraTS-MET-01094-003','BraTS-MET-01184-002'}]
source_paths={r['id']:{k:{'declared':r[k],'resolved':str((code_root/'data/input'/r['id']/pathlib.Path(r[k]).name).resolve()),'exists':(code_root/'data/input'/r['id']/pathlib.Path(r[k]).name).is_file()} for k in ('t1n','t1c','t2w','t2f','seg')} for r in selected}
imports=cmd([PYTHON,'-B','-c',"import torch,monai,nibabel,numpy,scipy,pandas;print(__import__('json').dumps({k:globals()[k].__version__ for k in ('torch','monai','nibabel','numpy','scipy','pandas')}))"],120)
p={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'public_disk':cmd(['df','-h','/public']),
   'experiment_parent_disk':cmd(['df','-B1',str(root.parent)]),
   'available_bytes':os.statvfs(root.parent).f_bavail*os.statvfs(root.parent).f_frsize,
   'new_root_exists':pathlib.Path(NEW_ROOT).exists(),'reconstruction_squeue':cmd(['squeue','-h','-j','3941042','-o','%i|%T|%j|%b']),
   'reconstruction_sacct':cmd(['sacct','-n','-P','-j','3941042','--format=JobID,State,ExitCode,Start,End,Elapsed,AllocTRES']),
   'user_squeue':cmd(['squeue','-h','-u',os.environ.get('USER','zqchen'),'-o','%i|%T|%j|%P|%r|%b']),
   'partitions':cmd(['scontrol','show','partition']), 'resources':cmd(['sinfo','-N','-h','-o','%N|%P|%t|%G|%m|%c']),
   'qos':cmd(['sacctmgr','-n','-P','show','qos','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU']),
   'associations':cmd(['sacctmgr','-n','-P','show','assoc','where','user=zqchen','format=Account,Partition,QOS,DefaultQOS']),
   'imports':imports,'source_csv_sha256':hashlib.sha256((code_root/'data/data_csv.csv').read_bytes()).hexdigest(),
   'source_csv_counts':dict(collections.Counter(r['split'] for r in rows)),'corrected_case_source_paths':source_paths,
   'reconstruction_files':{n:file(root/'dataset'/n) for n in ('RUN_CONTRACT.json','DATASET_VALIDATION.json','COMPLETE.ok','FAILED.json')},
   'logs':{n:log(root/'logs'/n) for n in ('g1vae4fullr2_3941042.out','g1vae4fullr2_3941042.err')},
   'reconstruction_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('RECONSTRUCTION_RECOVERY_PLAN.json','runtime/reconstruct_epoch004.py','slurm/run_full_reconstruction_r2.slurm')}}
print(re.sub('password:', '[AUTH_PROMPT]:', json.dumps(p,sort_keys=True),flags=re.I))
'''
    for key, value in {'RECON_ROOT':str(RECON),'SOURCE_ROOT':str(SOURCE_CODE),'NEW_ROOT':str(REMOTE),'PYTHON':m.PYTHON}.items():
        code = code.replace(key, repr(value))
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=220).strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'INVENTORY.json', result)
        print(json.dumps({'audit':str(access.audit),'remote_utc':result['remote_utc'],'new_root_exists':result['new_root_exists'],'available_gib':result['available_bytes']/1024**3,'queue':result['user_squeue'],'sacct':result['reconstruction_sacct'],'imports':result['imports'],'reconstruction_files':result['reconstruction_files'],'source_csv_counts':result['source_csv_counts'],'critical':{k:v['critical'] for k,v in result['logs'].items()}},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
