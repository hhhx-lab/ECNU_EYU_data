"""Read-only R9 final-model pull, gated by one completed job snapshot."""
import argparse
import base64
import datetime
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path

from common import REMOTE, ROOT, m, transport

PACKAGE_SHA = '80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6'
MODALITIES = ('t1c', 't1n', 't2w', 't2f')
JOBS = {'prepare': '4128262', 't1c': '4128266', 't1n': '4128267', 't2w': '4128268', 't2f': '4128269'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gate(snapshot):
    assert snapshot['jobs'] == JOBS
    assert snapshot['package_sha256'] == PACKAGE_SHA and not snapshot['drift']
    assert snapshot['squeue']['returncode'] == snapshot['sacct']['returncode'] == 0
    assert not snapshot['squeue']['stdout'].strip()
    rows = {r.split('|')[0]: r.split('|')[1:3] for r in snapshot['sacct']['stdout'].splitlines()}
    for job in JOBS.values():
        assert rows[job] == rows[job + '.batch'] == ['COMPLETED', '0:0']
    e = snapshot['evidence']
    assert not e['dataset/FAILED.json']['exists']
    assert e['dataset/COMPLETE.ok']['cases'] == 926
    assert e['dataset/COMPLETE.ok']['split_counts'] == {'train': 823, 'val': 103}
    assert e['dataset/PREPROCESS_REPORT.json']['error_cases'] == 0
    for modality in MODALITIES:
        for name in ('FAILED.json', 'SAFE_TERM.json'):
            assert not e[f'training/{modality}/{name}']['exists']
        assert e[f'training/{modality}/COMPLETE.ok']['status'] == 'training_complete'
        assert e[f'training/{modality}/COMPLETE.ok']['global_step'] == 200000
        assert e[f'training/{modality}/LATEST_CHECKPOINT.json']['global_step'] == 200000
        assert not snapshot['training_progress'][modality]['critical']
    assert all(not value['critical'] for value in snapshot['logs'].values())


REMOTE_CODE = r'''
import base64,datetime,hashlib,json,math,os,pathlib,re,subprocess
root=pathlib.Path(__ROOT__)
expected=__EXPECTED__
mods=('t1c','t1n','t2w','t2f')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def checked(p):
 assert p.is_file() and not p.is_symlink() and p.resolve()==p and root in p.parents,str(p)
 return p
def load(p):return json.loads(checked(p).read_text())
def command(argv):
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
  return dict(returncode=r.returncode,stdout=r.stdout[-8000:],stderr=r.stderr[-2000:])
 except subprocess.TimeoutExpired:return dict(status='timeout_unknown_not_zero')
def disk():
 v=os.statvfs(root)
 return dict(utc=now(),available_bytes=v.f_bavail*v.f_frsize,total_bytes=v.f_blocks*v.f_frsize,
             account_df=command(['df','-B1',str(root)]),public_df=command(['df','-h','/public']))
manifest=root/'plan/PACKAGE_MANIFEST.json'
assert sha(checked(manifest))==__PACKAGE_SHA__
package=load(manifest)
assert len(package['files'])==60
assert not [n for n,h in package['files'].items() if sha(checked(root/n))!=h]
for n,h in expected.items():assert sha(checked(root/n))==h,n
assert not (root/'dataset/FAILED.json').exists()
complete=load(root/'dataset/COMPLETE.ok');report=load(root/'dataset/PREPROCESS_REPORT.json')
assert complete['cases']==report['cases_ok']==report['cases_processed']==926
assert complete['split_counts']=={'train':823,'val':103} and not report['errors']
assert len(report['warnings'])==len(complete['resolution_warnings'])==5
assert sha(root/'dataset/PREPROCESS_REPORT.json')==complete['report_sha256']
assert sha(checked(root/'dataset/lesions.csv'))==complete['csv_sha256']
cfg=load(root/'config/RUN.json')
assert complete['reference_parents']==load(root/'plan/REUSE_CANDIDATES.json')['parents']
for k in ('loss_policy','label_resampling_algorithm'):assert complete[k]==cfg[k]
assert not complete.get('local_exception_sha256') and not (root/'plan/LOCAL_LABEL_EXCEPTION.json').exists()
models={};summaries={};payloads={};total=0
def pack(n,data):
 global total
 assert len(data)<=16*1024*1024,n
 total+=len(data);assert total<=64*1024*1024
 payloads[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data_b64=base64.b64encode(data).decode())
for n in sorted(set(expected)|{'plan/PACKAGE_MANIFEST.json','dataset/lesions.csv','evidence/SUBMISSION_RECEIPT.json'}|
                {n for n in package['files'] if n.startswith(('plan/','config/')) and pathlib.Path(n).suffix in ('.json','.csv','.md')}):
 pack(n,checked(root/n).read_bytes())
for modality in mods:
 d=root/'training'/modality
 assert not (d/'FAILED.json').exists() and not (d/'SAFE_TERM.json').exists()
 c=load(d/'COMPLETE.ok');latest=load(d/'LATEST_CHECKPOINT.json');contract=load(d/'RUN_CONTRACT.json')
 assert c['status']=='training_complete' and c['global_step']==latest['global_step']==200000
 assert c['selection_status']=='not_run' and latest['safe_boundary'] is True
 assert contract['package_sha256']==__PACKAGE_SHA__
 assert contract['identity']['csv_sha256']==complete['csv_sha256']
 for k in ('batch_size','crop_size','num_steps','optim_lr','adam_beta1'):
  assert contract['config'][k]==cfg['training'][k],k
 p=checked(d/'weights/diffusion_200000.pt')
 assert c['checkpoint']==latest['path']==str(p)
 digest=sha(p);assert c['checkpoint_sha256']==latest['sha256']==digest
 assert 0<p.stat().st_size<1024*1024*1024
 models[modality]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                       complete_utc=c['utc'],global_step=200000)
 progress=checked(d/'progress.jsonl');h=hashlib.sha256();count=0;bins=[];loss_sum=0.;lo=math.inf;hi=-math.inf;first=None;last=None
 with progress.open('rb') as f:
  for line in f:
   h.update(line);row=json.loads(line);count+=1
   assert row['global_step']==count and math.isfinite(row['loss'])
   if first is None:first=row
   last=row;loss_sum+=row['loss'];lo=min(lo,row['loss']);hi=max(hi,row['loss'])
   if count%1000==0:
    bins.append(dict(step_start=count-999,step_end=count,mean_loss=loss_sum/1000,min_loss=lo,max_loss=hi))
    loss_sum=0.;lo=math.inf;hi=-math.inf
 assert count==200000 and not loss_sum
 summaries[modality]=dict(rows=count,contiguous_successful_updates=True,all_loss_finite=True,
  progress_sha256=h.hexdigest(),progress_bytes=progress.stat().st_size,first=first,last=last,bins_1000=bins)
 pack(f'training/{modality}/PROGRESS_SUMMARY.json',json.dumps(summaries[modality],indent=2).encode()+b'\n')
 with progress.open('rb') as f:f.seek(max(0,progress.stat().st_size-32768));tail=f.read()
 pack(f'training/{modality}/progress_tail.jsonl',tail[tail.find(b'\n')+1:])
critical=re.compile(r'Traceback|CUDA error|OutOfMemory|ImportError|No space left|nonfinite|RuntimeError|DUE TO TIME LIMIT|FloatingPointError|AssertionError|unbound variable',re.I)
log_inventory={}
for p in sorted((root/'logs').iterdir()):
 checked(p);h=sha(p)
 with p.open('r',errors='replace') as f:errors=[line.strip() for line in f if critical.search(line)]
 assert not errors,p.name
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-32768));tail=f.read()
 pack('logs/'+p.name+'.tail',tail)
 log_inventory[p.name]=dict(bytes=p.stat().st_size,sha256=h,whole_log_critical_errors=errors)
v=dict(remote_utc=now(),models=models,files=payloads,lightweight_bytes=total,progress_summaries=summaries,
       log_inventory=log_inventory,storage=disk(),targeted_usage={n:command(['du','-s','-B1',str(root/n)]) for n in ('dataset','training')},
       package_sha256=__PACKAGE_SHA__,independent_full_metadata_audit='not_performed',quality_evaluation='not_run')
print(json.dumps(v))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    snapshot_path = args.snapshot.resolve()
    assert snapshot_path.is_relative_to(ROOT / 'operator')
    snapshot = json.loads(snapshot_path.read_text())
    gate(snapshot)
    expected = {n: value['sha256'] for n, value in snapshot['evidence'].items() if value['exists']}
    access = transport('pull_final')
    model_dir = ROOT / 'final_models_200000'
    model_dir.mkdir(exist_ok=False)
    report = dict(m.META, status='started', local_start_utc=now(), snapshot=str(snapshot_path),
                  snapshot_sha256=m.sha(snapshot_path), model_directory=str(model_dir))
    try:
        access.start_vpn()
        code = (REMOTE_CODE.replace('__ROOT__', repr(str(REMOTE)))
                .replace('__EXPECTED__', repr(expected)).replace('__PACKAGE_SHA__', repr(PACKAGE_SHA)))
        compile(code, 'r9_readonly_final_inventory', 'exec')
        value = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=240).strip().splitlines()[-1])
        payloads = value.pop('files')
        m.exclusive(access.audit / 'REMOTE_FINAL_INVENTORY.json', {**m.META, **value})
        for name, entry in payloads.items():
            rel = Path(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            data = base64.b64decode(entry['data_b64'], validate=True)
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            dest = access.audit / 'lightweight' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(data)
            entry.pop('data_b64')
        required = sum(v['bytes'] for v in value['models'].values())
        free = shutil.disk_usage(model_dir).free
        assert free > required + 4 * 1024 ** 3, 'Insufficient local capacity plus 4GiB reserve'
        report.update(remote_inventory_utc=value['remote_utc'], storage_before=value['storage'],
                      local_available_before_models_bytes=free, model_total_bytes=required,
                      lightweight_bytes=value['lightweight_bytes'], lightweight_files=payloads, models={})
        for modality in MODALITIES:
            entry = value['models'][modality]
            dest = model_dir / modality / 'diffusion_200000.pt'
            dest.parent.mkdir()
            partial = dest.with_suffix('.pt.partial')
            with partial.open('xb'):
                pass
            access.call(['scp', '-P', access.env['HPC_LOGIN_PORT'], '-o', 'StrictHostKeyChecking=yes',
                         '-o', 'ConnectTimeout=15',
                         f"{access.env['HPC_SSH_USER']}@{access.env['HPC_LOGIN_HOST']}:{entry['path']}", str(partial)], timeout=600)
            assert partial.stat().st_size == entry['bytes'] and m.sha(partial) == entry['sha256'], 'Downloaded model identity mismatch'
            os.link(partial, dest)
            partial.unlink()
            report['models'][modality] = {**entry, 'local_path': str(dest), 'local_verified_utc': now()}
            m.exclusive(access.audit / f'{modality.upper()}_MODEL_VERIFIED.json', {**m.META, **report['models'][modality]})
            print(f'{modality.upper()}: final 200000 checkpoint downloaded, bytes/SHA256 verified', flush=True)
        final_code = r'''
import datetime,json,os,pathlib,subprocess
root=pathlib.Path(__ROOT__);models=__MODELS__;expected=__EXPECTED__
for modality,e in models.items():
 p=pathlib.Path(e['path']);s=p.stat();assert not p.is_symlink() and p.resolve()==p
 assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns']
 c=json.loads((root/'training'/modality/'COMPLETE.ok').read_text())
 assert c['global_step']==200000 and c['checkpoint_sha256']==e['sha256']
 assert not (root/'training'/modality/'FAILED.json').exists()
import hashlib
for n,h in expected.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
v=os.statvfs(root);r=subprocess.run(['df','-B1',str(root)],capture_output=True,text=True,timeout=25)
assert r.returncode==0
print(json.dumps(dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_bytes=v.f_bavail*v.f_frsize,account_df=r.stdout)))
'''.replace('__ROOT__', repr(str(REMOTE))).replace('__MODELS__', repr(value['models'])).replace('__EXPECTED__', repr(expected))
        report['storage_after'] = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(final_code), timeout=90).strip().splitlines()[-1])
        report.update(status='final_models_and_lightweight_verified', local_end_utc=now(),
                      local_available_after_bytes=shutil.disk_usage(model_dir).free,
                      independent_full_metadata_audit='not_performed', model_deserialization='not_performed',
                      quality_evaluation='not_run', selection_status='not_run', remote_writes=False)
    except Exception as error:
        m.exclusive(access.audit / 'PULL_FAILED.json', {**m.META, **report, 'status': 'failed', 'message': str(error), 'utc': now()})
        raise
    finally:
        access.close_vpn()
    report['owned_vpn_closed'] = access.owns_vpn
    report['local_acceptance_utc'] = now()
    m.exclusive(access.audit / 'PULL_ACCEPTANCE.json', report)
    print(json.dumps({'audit': str(access.audit), 'status': report['status'], 'model_total_bytes': required,
                      'lightweight_bytes': report['lightweight_bytes'], 'storage_after': report['storage_after'],
                      'local_available_after_bytes': report['local_available_after_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
