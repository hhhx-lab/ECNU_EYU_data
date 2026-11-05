"""One bounded read-only observation of direct inference and R5 preparation."""
import json
import os
import shlex
from session import session, access_module

CODE = r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
out={}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def tail(path,limit):
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-limit));return f.read(limit).decode(errors='replace')
for name,base in [('runtime_r5',root/'operator/environment_r5'),('full2025',root/'attempts/full103_2025_r1/operator'),('fullbio',root/'attempts/full103_bio_r1/operator')]:
 items={}
 for p in base.glob('*.json'):
  if p.name not in ('FAILED.json','COMPLETE.json','PLAN_ACCEPTANCE.json','BASE_PACKAGE_LOCK.json','HELD_CREATED.json','HELD_JOB_MAP.json','SUBMISSION_RECEIPT.json','SUBMISSION_FAILED_hold.json','SUBMISSION_FAILED_release.json','IMAGE_CPU_IMPORT_CHECK.json','RESOURCE_SNAPSHOT.json'):continue
  assert p.stat().st_size<131072,'lightweight marker budget'
  data=p.read_bytes();items[p.name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=json.loads(data))
 out[name]=items
out['runtime_log_tails']={p.name:tail(p,2500) for p in (root/'logs/environment_r5').glob('*.log')}
out['base_package_files_present']={p.name:p.stat().st_size for p in (root/'operator/environment_r5/base_packages').glob('*') if p.is_file()}
out['full_reports']={}
out['full_jobs']={}
for model,name in [('EXT2025_MET_GLI','full103_2025_r1'),('EXT2026_BIO_INTEL_FULL','full103_bio_r1')]:
 base=root/'attempts'/name
 out['full_reports'][model]={p.name:json.loads(p.read_text()) for p in (base/'reports'/model).glob('*.json') if p.name in ('STARTED.json','FAILED.json','COMPLETE.json','RUN_CONTRACT.json')}
 out[model+'_logtails']={p.name:tail(p,2000) for p in (base/'logs').glob('*') if p.is_file()}
 mapping=base/'operator/HELD_JOB_MAP.json'
 if mapping.is_file():
  data=json.loads(mapping.read_text());identity=data['jobs'][model]['job_id']
  package=base/'contract/PACKAGE_SHA256.json';contract=base/'contract/EXECUTION_CONTRACT.json'
  assert sha(package)==data['package_sha256'] and sha(contract)==data['contract_sha256'],'job package identity drift'
  for n,digest in json.loads(package.read_text())['files'].items():assert sha(base/n)==digest,'full frozen file drift'
  observed={}
  for label,argv in [('scontrol',['scontrol','show','job',identity,'-o']),('sacct',['sacct','-n','-P','-j',identity,'--format=JobID,State,ExitCode,Elapsed,NodeList,Reason'])]:
   p=subprocess.run(argv,capture_output=True,text=True,timeout=40)
   assert len(p.stdout)+len(p.stderr)<65536,'job observation budget'
   observed[label]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  out['full_jobs'][model]=dict(job_id=identity,package_sha256=sha(package),contract_sha256=sha(contract),observed=observed,
   prediction_files_present=len(list((base/'predictions'/model).glob('*.nii.gz'))))
p=subprocess.run(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'],capture_output=True,text=True,timeout=40)
assert p.returncode==0,'queue unavailable'
out['own_queue']=p.stdout;out['remote_utc']=datetime.now(timezone.utc).isoformat()
out['disk_free_bytes']=shutil.disk_usage(root).free
p=subprocess.run(['quota','-s'],capture_output=True,text=True,timeout=40)
out['quota']=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
out['remote_writes']=False;out['images_read']=False;out['models_loaded']=False
print(json.dumps(out))
'''


def main():
    compile(CODE,'snapshot_direct_preparation','exec')
    with session('snapshot_direct_preparation') as access:
        command='env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=180).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),remote_utc=result['remote_utc'],
            runtime_R5_markers={k:{n:v['data'].get(n) for n in ('status','stage','message')} for k,v in result['runtime_r5'].items()},
            base_package_files_present=len(result['base_package_files_present']),
            runtime_log_tails=result['runtime_log_tails'],full_reports=result['full_reports'],full_jobs=result['full_jobs'],
            disk_free_bytes=result['disk_free_bytes'],quota=result['quota'],
            full2025_logtails=result['EXT2025_MET_GLI_logtails'])),flush=True)


if __name__=='__main__':main()
