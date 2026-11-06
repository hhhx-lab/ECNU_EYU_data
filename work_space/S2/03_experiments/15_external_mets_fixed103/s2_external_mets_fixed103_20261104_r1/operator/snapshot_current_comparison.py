"""One immediate two-job snapshot; completed lightweight reports only are retrieved."""
import ast
import json
import os
from pathlib import Path
import shlex
from session import session, access_module

ROOT=Path(__file__).resolve().parents[1]

CODE=r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def read(path):
 path=Path(path)
 if not path.is_file():return None
 assert path.stat().st_size<2*1024**2,'marker budget'
 return dict(bytes=path.stat().st_size,sha256=sha(path),data=json.loads(path.read_text()))
def tail(path):
 with Path(path).open('rb') as f:f.seek(max(0,path.stat().st_size-6000));return f.read().decode(errors='replace')
root=Path(REMOTE_ROOT)
bio=root/'attempts/full103_bio_r3';comparison=Path(COMPARISON_ATTEMPT)
assert comparison.parent==root/'attempts'
result=dict(remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,
 quota=command(['quota','-s']),squeue=command(['squeue','-h','-j',','.join(JOBS),'-o','%i|%T|%M|%R']),
 sacct=command(['sacct','-n','-P','-j',','.join(JOBS),'--format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Start,End']),
 Bio_reports={n:read(bio/'reports/EXT2026_BIO_INTEL_FULL'/n) for n in ('STARTED.json','RUN_CONTRACT.json','FAILED.json','COMPLETE.json')},
 comparison_reports={n:read(comparison/'reports'/n) for n in ('STARTED.json','FAILED.json','COMPLETE.json','HISTORICAL_REPLAY.json','PROGRESS_EXT2025.json','PROGRESS_K1_175.json','PROGRESS_original_E_200.json')},
 Bio_prediction_file_count=sum(1 for p in (bio/'predictions/EXT2026_BIO_INTEL_FULL').glob('*.nii.gz') if p.is_file()),
 logtails={},drift=[],remote_writes=False,images_or_GT_read=False)
for attempt,expected in ((bio,BIO_PACKAGE_SHA),(comparison,COMPARISON_PACKAGE_SHA)):
 if sha(attempt/'contract/PACKAGE_SHA256.json')!=expected:result['drift'].append(str(attempt)+':package')
 for name,e in json.loads((attempt/'contract/PACKAGE_SHA256.json').read_text())['files'].items():
  if sha(attempt/name)!=e:result['drift'].append(str(attempt/name))
 for p in sorted((attempt/'logs').glob('*')):
  if p.is_file():result['logtails'][str(p.relative_to(root))]=tail(p)
result['lightweight_download_files']={}
result['comparison_acceptance']='not_complete'
complete=result['comparison_reports']['COMPLETE.json']
if complete:
 assert result['sacct']['returncode']==0 and not result['drift']
 records=[line.split('|') for line in result['sacct']['stdout'].splitlines() if line]
 accounted=all(any(row[0]==identity and row[2:4]==['COMPLETED','0:0'] for row in records) for identity in (JOBS[1],JOBS[1]+'.batch'))
 data=complete['data'];assert data['job_id']==JOBS[1] and data['cases']==103 and data['metric_columns']==72
 assert not result['comparison_reports']['FAILED.json'],'comparison FAILED and COMPLETE conflict'
 selected={k:v for k,v in data['outputs'].items() if k.startswith('reports/') or k.endswith('leaderboard_metrics.csv')}
 selected['reports/COMPLETE.json']=dict(bytes=complete['bytes'],sha256=complete['sha256'])
 assert sum(entry['bytes'] for entry in selected.values())<16*1024**2,'lightweight output budget'
 for name,e in selected.items():
  path=comparison/name;assert path.stat().st_size==e['bytes'] and sha(path)==e['sha256'],'completed output drift: '+name
 result['comparison_acceptance']='completed_accounting_and_output_SHA_pass' if accounted else 'complete_marker_waiting_accounting_acceptance'
 if accounted:result['lightweight_download_files']={name:dict(path=str(comparison/name),**entry) for name,entry in selected.items()}
result['finished_remote_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(result))
'''


def main():
    bio_path=sorted((ROOT/'operator').glob('submit_bio_full103_r3_*/RELEASE_RESULT.json'))[-1]
    comparison_path=sorted((ROOT/'operator').glob('submit_fixed103_comparison_*/RELEASE_RESULT.json'))[-1]
    bio=json.loads(bio_path.read_text());comparison=json.loads(comparison_path.read_text())
    payload=json.loads((comparison_path.parent/'SUBMISSION_PAYLOAD.json').read_text())
    import hashlib
    package_sha=hashlib.sha256(payload['files']['contract/PACKAGE_SHA256.json'].encode()).hexdigest()
    code='REMOTE_ROOT='+repr(os.environ['EXTERNAL_MET_REMOTE_ROOT'])+'\nJOBS='+repr([bio['job_id'],comparison['job_id']])
    code+='\nBIO_PACKAGE_SHA='+repr(bio['package_sha256'])+'\nCOMPARISON_PACKAGE_SHA='+repr(package_sha)+'\n'+CODE
    code='COMPARISON_ATTEMPT='+repr(payload['contract']['attempt'])+'\n'+code
    ast.parse(code,feature_version=(3,10))
    with session('snapshot_current_comparison') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=code))
        result=json.loads(access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code),timeout=180).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',dict(access_module.META,**result))
        files=result['lightweight_download_files']
        if files:
            destination=ROOT/'results'/('comparison_light_'+access.audit.name.rsplit('_',1)[1]);destination.mkdir(parents=True)
            downloaded={}
            for name,entry in files.items():
                local=destination/name;local.parent.mkdir(parents=True,exist_ok=True)
                access.call(['scp','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                    access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+entry['path'],str(local)],timeout=120)
                assert local.stat().st_size==entry['bytes'] and access_module.sha(local)==entry['sha256'],'comparison transfer drift'
                downloaded[name]=dict(bytes=entry['bytes'],sha256=entry['sha256'])
            access_module.exclusive(destination/'DOWNLOAD_ACCEPTANCE.json',dict(access_module.META,files=downloaded,
                status='completed_CPU_comparison_accounting_and_output_SHA_pass',local_received_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),
                remote_utc=result['finished_remote_utc'],job_id=comparison['job_id']))
        print(json.dumps(dict(audit=str(access.audit),squeue=result['squeue'],sacct=result['sacct'],drift=result['drift'],
            comparison_complete=bool(files),comparison_acceptance=result['comparison_acceptance'],
            Bio_prediction_file_count=result['Bio_prediction_file_count'],
            local_comparison=str(destination) if files else None,free_bytes=result['free_bytes'],remote_utc=result['finished_remote_utc'])),flush=True)


if __name__=='__main__':main()
