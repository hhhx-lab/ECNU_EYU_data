"""One bounded snapshot; download only completed, SHA-verified light scores."""
import ast
import json
import os
from pathlib import Path
import shlex

from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import hashlib,json,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,'snapshot command failed: '+p.stderr
 return p.stdout
def queue_snapshot(job):
 p=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%M|%R'],capture_output=True,text=True,timeout=60)
 retired=p.returncode==1 and not p.stdout.strip() and p.stderr.strip()=='slurm_load_jobs error: Invalid job id specified'
 assert p.returncode==0 or retired,'snapshot queue failed: '+p.stderr
 return dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode,retired_id=retired)
def read(path):
 if not path.is_file():return None
 assert path.stat().st_size<2*1024**2,'marker budget'
 return dict(bytes=path.stat().st_size,sha256=sha(path),data=json.loads(path.read_text()))
root=Path(ATTEMPT)
assert sha(root/'contract/COMPARISON_CONTRACT.json')==CONTRACT_SHA
assert sha(root/'contract/PACKAGE_SHA256.json')==PACKAGE_SHA
for name,value in json.loads((root/'contract/PACKAGE_SHA256.json').read_text())['files'].items():
 assert sha(root/name)==value,'Bio MET package drift'
queue=queue_snapshot(JOB)
result=dict(remote_started_utc=datetime.now(timezone.utc).isoformat(),job_id=JOB,
 squeue=queue['stdout'],squeue_receipt=queue,
 sacct=command(['sacct','-n','-P','-j',JOB,'--format=JobID,JobName,State,ExitCode,Elapsed,NodeList']),
 markers={name:read(root/'reports'/name) for name in ('STARTED.json','FAILED.json','PROGRESS_Bio.json','PARENT_REUSE_VERIFICATION.json','COMPLETE.json')},
 logtails={},free_bytes=shutil.disk_usage(root).free,quota_stdout=command(['quota','-s']),
 acceptance='pending_or_running_not_complete',download_files={},remote_writes=False,source_MRI_GT_voxels_read=False)
for p in sorted((root/'logs').glob('*')):
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-7000));result['logtails'][p.name]=f.read().decode(errors='replace')
records=[line.split('|') for line in result['sacct'].splitlines() if line.strip()]
assert all(len(row)==6 for row in records),'accounting columns drift'
bad=[row for row in records if row[2] not in ('PENDING','RUNNING','COMPLETED','CONFIGURING','COMPLETING') or row[3]!='0:0']
if result['markers']['FAILED.json'] or bad:
 result['acceptance']='failed_or_nonzero_do_not_retry'
elif result['markers']['COMPLETE.json']:
 complete=result['markers']['COMPLETE.json'];data=complete['data']
 assert data['job_id']==JOB and data['contract_sha256']==CONTRACT_SHA
 assert data['status']=='four_model_fixed103_MET_descriptive_comparison_complete'
 assert data['cases']==103 and data['metric_columns']==72 and data['mean_rows']==288 and data['paired_rows']==432
 assert data['models']==['EXT2025','K1_175','original_E_200','EXT2026_BIO_INTEL_FULL']
 assert data['old_mean_and_paired_rows_unchanged'] and not data['locked_test_read']
 selected={name:entry for name,entry in data['outputs'].items() if name.startswith('reports/') or name.startswith('evaluation/EXT2026_BIO_INTEL_FULL/')}
 selected['reports/COMPLETE.json']={k:complete[k] for k in ('bytes','sha256')}
 assert sum(entry['bytes'] for entry in selected.values())<16*1024**2,'light output budget'
 for name,entry in selected.items():
  path=root/name;assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'completed output drift: '+name
 done=all(any(row[0]==identity and row[2:4]==['COMPLETED','0:0'] for row in records) for identity in (JOB,JOB+'.batch'))
 result['acceptance']='completed_accounting_and_light_output_SHA_pass' if done else 'COMPLETE_waiting_accounting'
 if done:result['download_files']={name:dict(path=str(root/name),**entry) for name,entry in selected.items()}
elif not result['squeue'].strip() and not records:
 result['acceptance']='job_disappeared_without_success_evidence_do_not_retry'
elif not result['squeue'].strip() and any(row[0]==JOB and row[2]=='COMPLETED' for row in records):
 result['acceptance']='accounting_complete_without_COMPLETE_do_not_retry'
if result['markers']['STARTED.json']:
 started=result['markers']['STARTED.json']['data'];assert started['job_id']==JOB and started['contract_sha256']==CONTRACT_SHA
result['remote_finished_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(result))
'''


def main():
    paths = list((ROOT / 'operator').glob('submit_bio_comparison_*/RELEASE_RESULT.json'))
    assert len(paths) == 1, 'Bio MET release evidence missing or ambiguous; inspect only'
    audit = paths[0].parent
    release = json.loads(paths[0].read_text())
    payload = json.loads((audit / 'SUBMISSION_PAYLOAD.json').read_text())
    contract = payload['contract']
    assert release['release_returncode'] == 0
    prefix = '\n'.join(name + '=' + repr(value) for name, value in dict(ATTEMPT=contract['attempt'], JOB=release['job_id'],
        CONTRACT_SHA=release['contract_sha256'], PACKAGE_SHA=release['package_sha256']).items())
    source = prefix + '\n' + CODE
    ast.parse(source, feature_version=(3, 10))
    with session('snapshot_bio_comparison') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json', dict(source=source))
        raw = access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(source), timeout=180)
        result = json.loads(raw.strip().splitlines()[-1])
        result['local_received_utc'] = access_module.datetime.now(access_module.timezone.utc).isoformat()
        access_module.exclusive(access.audit / 'RESULT.json', dict(access_module.META, **result))
        destination = None
        if result['download_files']:
            existing = list((ROOT / 'results').glob('comparison_four_light_*/DOWNLOAD_ACCEPTANCE.json'))
            assert not existing, 'four-model download already exists; verify locally instead of downloading again'
            destination = ROOT / 'results' / ('comparison_four_light_' + access.audit.name.rsplit('_', 1)[1])
            destination.mkdir()
            for name, entry in result['download_files'].items():
                local = destination / name
                local.parent.mkdir(parents=True, exist_ok=True)
                access.call(['scp', '-P', access.env['HPC_LOGIN_PORT'], '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                    access.env['HPC_SSH_USER'] + '@' + access.env['HPC_LOGIN_HOST'] + ':' + entry['path'], str(local)], timeout=90)
                assert local.stat().st_size == entry['bytes'] and access_module.sha(local) == entry['sha256'], 'Bio MET light transfer drift'
            access_module.exclusive(destination / 'DOWNLOAD_ACCEPTANCE.json', dict(access_module.META,
                status=result['acceptance'], job_id=release['job_id'], files=result['download_files'],
                total_bytes=sum(value['bytes'] for value in result['download_files'].values()),
                remote_utc=result['remote_finished_utc'], local_received_utc=access_module.datetime.now(access_module.timezone.utc).isoformat()))
        print(json.dumps(dict(audit=str(access.audit), job_id=release['job_id'], acceptance=result['acceptance'],
            queue=result['squeue'], accounting=result['sacct'], destination=str(destination) if destination else None)), flush=True)


if __name__ == '__main__':
    main()
