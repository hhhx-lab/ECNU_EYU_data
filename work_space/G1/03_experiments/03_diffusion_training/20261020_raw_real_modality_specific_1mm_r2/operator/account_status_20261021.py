"""One bounded read-only account and tracked-runtime snapshot, outside frozen package."""
from common import run_remote

CODE = r'''
import hashlib,json,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated')
MODS=('t1c','t1n','t2w','t2f')
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
BATCHES={
 'raw_r2':('g1_diffusion_raw_real_modality_specific_1mm_20261020_r2','plan/PACKAGE_MANIFEST.json','add5c89469053180559238ea7d9c385133679c72caeba0596eea966498930ba8',[4161285,4161286,4161287,4161288,4161289]),
 'r9_evaluation':('g1_r9_fixedval_selection_20261018_r1','PACKAGE_MANIFEST.json','3486d52cfa995aff8cd439932384d8393224b90630cb878793aba01f7b41d2d4',[4145277,4145278,4145279,4145280,4145281,4159189,4159190,4159191,4159192,4159193]),
 'r9_training':('g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9','plan/PACKAGE_MANIFEST.json','80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6',[4128262,4128266,4128267,4128268,4128269])}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(returncode=124,stdout='',stderr='bounded command timeout')

def tail(p,size=8192):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')

def compact(v):
    if isinstance(v,dict):return {k:compact(x) for k,x in v.items()}
    if isinstance(v,list):
        return dict(count=len(v),first=compact(v[:3])) if len(v)>15 else [compact(x) for x in v]
    return v

def marker(p):
    if not p.exists():return dict(exists=False)
    try:
        v=json.loads(p.read_text());return dict(exists=True,sha256=sha(p),bytes=p.stat().st_size,summary=compact(v))
    except Exception as e:return dict(exists=True,parse_error=str(e))

ids=','.join(str(j) for b in BATCHES.values() for j in b[3])
result=dict(META,status='pass',observation_kind='read_only_snapshot_not_job_acceptance',remote_started_utc=datetime.now(timezone.utc).isoformat(),
 hostname=os.uname().nodename,history_start='2026-10-01',
 account_squeue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r|%M|%l|%D|%b']),
 account_sacct=command(['sacct','-n','-P','-u','zqchen','-S','2026-10-01','-X','--format=JobID,JobName%64,Partition,State,ExitCode,Elapsed,Submit,Start,End,AllocTRES%100']),
 tracked_sacct=command(['sacct','-n','-P','-j',ids,'--format=JobID,State,ExitCode,Elapsed,Start,End']),
 disk=command(['df','-B1',str(BASE)]),batches={})
for name,(directory,manifest,expected,jobs) in BATCHES.items():
    root=BASE/directory;p=root/manifest
    batch=dict(root=str(root),jobs=jobs,package_expected_sha256=expected,markers={},logs={},candidates={})
    try:
        batch['package_sha256']=sha(p);package=json.loads(p.read_text())
        batch['package_file_count']=len(package['files'])
        batch['package_drift']=[n for n,h in package['files'].items() if not (root/n).is_file() or sha(root/n)!=h]
        batch['package_identity_match']=batch['package_sha256']==expected
    except Exception as e:batch['package_error']=str(e)
    if name=='r9_evaluation':
        dirs=['reference','selection',*['smoke/'+m for m in MODS],*['evaluation/'+m for m in MODS]]
        names=['STARTED.json','FAILED.json','COMPLETE.ok','REPORT.json','SELECTION.json']
        for mod in MODS:
            batch['candidates'][mod]=[]
            for step in range(20000,200001,20000):
                p=root/'evaluation'/mod/str(step)/'CANDIDATE_COMPLETE.json'
                if p.exists():
                    try:
                        v=json.loads(p.read_text());rows=v['cases'];positive=[r for r in rows if r.get('positive')]
                        batch['candidates'][mod].append(dict(step=step,sha256=sha(p),cases=len(rows),positive_cases=len(positive),
                            primary_ssim=sum(r['primary_ssim'] for r in positive)/len(positive),
                            secondary_ssim=sum(r['secondary_ssim'] for r in positive)/len(positive),
                            all_qc_pass=v.get('all_qc_pass'),seeds=v.get('seeds'),checkpoint_sha256=v.get('checkpoint_sha256')))
                    except Exception as e:batch['candidates'][mod].append(dict(step=step,error=str(e)))
    else:
        dirs=['dataset',*['training/'+m for m in MODS]]
        names=['STARTED.json','FAILED.json','PREPROCESS_REPORT.json','COMPLETE.ok','RUN_CONTRACT.json','SAFE_TERM.json','LATEST_CHECKPOINT.json','STORAGE_ESTIMATE.json']
    for d in dirs:
        batch['markers'][d]={n:marker(root/d/n) for n in names}
        p=root/d/'progress.jsonl'
        if p.exists():
            lines=tail(p,16384).splitlines();parsed=[]
            for line in lines[-4:]:
                try:parsed.append(json.loads(line))
                except ValueError:pass
            batch['markers'][d]['progress']=dict(bytes=p.stat().st_size,last_records=parsed,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat())
    for job in jobs:
        for p in sorted((root/'logs').glob('*_'+str(job)+'.*')):
            if p.suffix not in ('.out','.err'):continue
            value=tail(p)
            batch['logs'][p.name]=dict(bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),tail=value,
                critical_tail_lines=[s for s in value.splitlines() if re.search(r'Traceback|CUDA out of memory|AssertionError|RuntimeError|slurmstepd: error|TIME LIMIT',s,re.I)])
    result['batches'][name]=batch
result['remote_utc']=datetime.now(timezone.utc).isoformat()
print(re.sub(r'password:', '[AUTH_PROMPT]',json.dumps(result),flags=re.I))
'''

if __name__ == '__main__':
    run_remote('account_status', CODE, timeout=240)
