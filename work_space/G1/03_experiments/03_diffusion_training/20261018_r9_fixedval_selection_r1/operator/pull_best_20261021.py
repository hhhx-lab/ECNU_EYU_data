"""Download only four selected R9 checkpoints and lightweight identity evidence."""
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, REMOTE, m, transport

SELECTION_SHA = '541d0fd245ba66841f35574aed3ad227efa9363c580211a1c8733f528d808902'
TRAINING_SHA = '80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6'
EVALUATION_SHA = '3486d52cfa995aff8cd439932384d8393224b90630cb878793aba01f7b41d2d4'
MODS = ('t1c', 't1n', 't2w', 't2f')
STEPS = dict(t1c=120000, t1n=40000, t2w=200000, t2f=200000)
TRAINING = REMOTE.parent / 'g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9'
DESTINATION = ROOT / 'best_models_fixedval103'


def now():
    return datetime.now(timezone.utc).isoformat()


REMOTE_CODE = r'''
import base64,hashlib,json,math,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(__ROOT__);training=Path(__TRAINING__);steps=__STEPS__
def now():return datetime.now(timezone.utc).isoformat()
def checked(p):
 assert p.is_file() and not p.is_symlink() and p.resolve()==p,str(p)
 assert root in p.parents or training in p.parents,str(p)
 return p
def sha(p):
 h=hashlib.sha256()
 with checked(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(checked(p).read_text())
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 assert p.returncode==0,(args,p.stderr)
 return p.stdout
started=now()
jobs='4159189,4159190,4159191,4159192,4159193'
queue=command(['squeue','-h','-j',jobs,'-o','%i|%T']);assert not queue.strip()
accounting=command(['sacct','-n','-P','-j',jobs,'--format=JobID,State,ExitCode,Elapsed,Start,End'])
rows={r[0]:r[1:3] for line in accounting.splitlines() if (r:=line.split('|'))}
for j in jobs.split(','):assert rows[j]==rows[j+'.batch']==['COMPLETED','0:0']
for base,name,expected,count in [(root,'PACKAGE_MANIFEST.json',__EVALUATION_SHA__,24),
 (training,'plan/PACKAGE_MANIFEST.json',__TRAINING_SHA__,60)]:
 assert sha(base/name)==expected
 package=load(base/name);assert len(package['files'])==count
 for n,h in package['files'].items():assert sha(base/n)==h,n
selection_path=root/'selection/SELECTION.json';assert sha(selection_path)==__SELECTION_SHA__
selection=load(selection_path);complete=load(root/'selection/COMPLETE.ok')
assert selection['status']=='exploratory_selected' and complete['status']=='complete'
assert complete['selection_sha256']==__SELECTION_SHA__
for d in ('selection','reference'):assert not (root/d/'FAILED.json').exists()
ref=load(root/'reference/REPORT.json');refdone=load(root/'reference/COMPLETE.ok')
assert ref['errors']==[] and refdone['report_sha256']==sha(root/'reference/REPORT.json')
models={};files={};total=0
def pack(key,p):
 global total
 data=checked(p).read_bytes();assert len(data)<=16*1024**2
 total+=len(data);assert total<=64*1024**2
 files[key]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data_b64=base64.b64encode(data).decode())
for n in ('PACKAGE_MANIFEST.json','PROTOCOL.md','config/RUN.json','selection/STARTED.json',
 'selection/SELECTION.json','selection/COMPLETE.ok','reference/COMPLETE.ok','reference/REPORT.json'):
 pack('evaluation/'+n,root/n)
for n in ('plan/PACKAGE_MANIFEST.json','config/RUN.json','dataset/COMPLETE.ok','dataset/PREPROCESS_REPORT.json'):
 pack('training/'+n,training/n)
for mod,step in steps.items():
 d=root/'evaluation'/mod;assert not (d/'FAILED.json').exists()
 done=load(d/'COMPLETE.ok');assert done['status']=='candidate_evaluation_complete'
 assert done['steps']==list(range(20000,200001,20000)) and len(done['cases'])==103
 assert done['evaluation_package_sha256']==__EVALUATION_SHA__
 value=selection['modalities'][mod];assert value['selected_step']==step
 assert value['rule']=='max_case_equal_volume_weighted_lesion_ssim_mean4seed_tol1e-6_earliest'
 scores={}
 for candidate in done['steps']:
  p=d/str(candidate)/'CANDIDATE_COMPLETE.json';v=load(p);digest=sha(p)
  assert digest==done['candidate_summary_sha256'][str(candidate)]==value['candidates'][str(candidate)]['candidate_summary_sha256']
  assert v['modality']==mod and v['step']==candidate and v['all_qc_pass'] is True
  assert v['seeds']==[20261013,20261014,20261015,20261016]
  assert v['reference_report_sha256']==refdone['report_sha256'] and v['evaluation_package_sha256']==__EVALUATION_SHA__
  cases=v['cases'];assert len(cases)==len({r['case'] for r in cases})==103
  positive=[r for r in cases if r['positive']];assert len(positive)==94
  assert all(math.isfinite(r['primary_ssim']) and math.isfinite(r['secondary_ssim']) for r in positive)
  score=sum(r['primary_ssim'] for r in positive)/94;scores[candidate]=score
  assert abs(score-value['candidates'][str(candidate)]['score'])<1e-12
  assert v['checkpoint_sha256']==value['candidates'][str(candidate)]['checkpoint_sha256']
  pack('evaluation/evaluation/'+mod+'/'+str(candidate)+'/CANDIDATE_COMPLETE.json',p)
 assert step==min(s for s,v in scores.items() if max(scores.values())-v<=1e-6)
 pack('evaluation/evaluation/'+mod+'/COMPLETE.ok',d/'COMPLETE.ok')
 td=training/'training'/mod
 assert not (td/'FAILED.json').exists() and not (td/'SAFE_TERM.json').exists()
 train_complete=load(td/'COMPLETE.ok');assert train_complete['status']=='training_complete' and train_complete['global_step']==200000
 contract=load(td/'RUN_CONTRACT.json');assert contract['package_sha256']==__TRAINING_SHA__
 for n in ('COMPLETE.ok','RUN_CONTRACT.json'):pack('training/training/'+mod+'/'+n,td/n)
 p=checked(td/'weights'/('diffusion_'+str(step)+'.pt'))
 assert str(p)==value['selected_checkpoint_path']==ref['checkpoints'][mod][str(step)]['path']
 st=p.stat();digest=sha(p)
 assert 0<st.st_size<1024**3
 assert digest==value['selected_checkpoint_sha256']==ref['checkpoints'][mod][str(step)]['sha256']
 models[mod]=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest,selected_step=step,
  primary_ssim=scores[step],authentic_secondary_ssim=value['secondary_scores'][str(step)])
v=os.statvfs(root)
out=dict(status='pass',remote_started_utc=started,remote_utc=now(),models=models,files=files,lightweight_bytes=total,
 selection_sha256=__SELECTION_SHA__,training_package_sha256=__TRAINING_SHA__,evaluation_package_sha256=__EVALUATION_SHA__,
 squeue=queue,sacct=accounting,storage=dict(total_bytes=v.f_blocks*v.f_frsize,available_bytes=v.f_bavail*v.f_frsize,
 df=command(['df','-B1',str(root)])),remote_writes=False)
print(re.sub(r'password:','[AUTH_PROMPT]',json.dumps(out),flags=re.I))
'''


def remote_json(access, code, timeout=240):
    compile(code, 'read_only_best_inventory', 'exec')
    raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=timeout)
    value = json.loads(raw.strip().splitlines()[-1])
    value['local_received_utc'] = now()
    return value


def shutdown(access):
    if not access.owns_vpn:
        return dict(vpn_owned=False, preexisting_vpn_preserved=True, own_vpn_stop_completed=False)
    stop = subprocess.run([m.VPN, 'stop'], env=access.env, capture_output=True, timeout=40)
    confirmed = stop.returncode == 0
    if not confirmed:
        status = subprocess.run([m.VPN, 'status'], env=access.env, capture_output=True, text=True, timeout=40)
        states = [json.loads(line[line.index('{'):]) for line in status.stdout.splitlines() if '{' in line]
        confirmed = status.returncode == 0 and len(states) == 1 and states[0].get('status') == 'logout' and states[0].get('tunnel_status') == 'tunnelUnavailable'
        m.exclusive(access.audit/'INDEPENDENT_VPN_STATUS.json', dict(m.META, utc=now(), stop_returncode=stop.returncode, confirmed_logout=confirmed))
    assert confirmed, 'Own VPN shutdown not confirmed; preserve downloaded files'
    return dict(vpn_owned=True, preexisting_vpn_preserved=False, own_vpn_stop_completed=True)


def verify_local(audit):
    acceptance = json.loads((audit/'PULL_ACCEPTANCE.json').read_text())
    assert acceptance['status'] == 'selected_models_bytes_verified'
    for mod, entry in acceptance['models'].items():
        p = Path(entry['local_path'])
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == entry['bytes'] and m.sha(p) == entry['sha256']
        assert entry['selected_step'] == STEPS[mod]
    for n, entry in acceptance['lightweight_files'].items():
        p = audit/'lightweight'/n
        assert p.stat().st_size == entry['bytes'] and m.sha(p) == entry['sha256']
    selection = audit/'lightweight/evaluation/selection/SELECTION.json'
    assert m.sha(selection) == SELECTION_SHA
    manifest = json.loads((DESTINATION/'MODEL_MANIFEST.json').read_text())
    assert manifest['models'] == acceptance['models'] and manifest['selection_sha256'] == SELECTION_SHA
    value = dict(m.META, status='pass', local_utc=now(), model_files=4, model_bytes=sum(e['bytes'] for e in acceptance['models'].values()),
                 all_models_rehashed=True, all_lightweight_rehashed=True, acceptance_sha256=m.sha(audit/'PULL_ACCEPTANCE.json'),
                 model_deserialization='not_performed', manual_quality_review='not_performed', local_available_bytes=shutil.disk_usage(DESTINATION).free)
    m.exclusive(audit/'LOCAL_VERIFICATION.json', value)
    print(json.dumps(dict(audit=str(audit), **value)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-local':
        verify_local(Path(sys.argv[2]))
        return
    assert not DESTINATION.exists(), 'Preserve existing download; no automatic overwrite or retry'
    access = transport('pull_best')
    report = dict(m.META, status='started', local_started_utc=now(), destination=str(DESTINATION), remote_writes=False,
                  authorization='2026-10-21 user: download selected best models and explicitly document', models={})
    stage = 'vpn_start'
    try:
        start = subprocess.run([m.VPN, 'start'], env=access.env, capture_output=True, timeout=100)
        flags = re.findall(rb'VPN_SESSION_OWNED=([01])', start.stdout)
        assert len(flags) == 1, 'VPN ownership unconfirmed; no SSH'
        access.owns_vpn = flags[0] == b'1'
        assert start.returncode == 0, 'VPN start failed; private output suppressed'
        stage = 'vpn_check'
        check = subprocess.run([m.VPN, 'check'], env=access.env, capture_output=True, timeout=40)
        assert check.returncode == 0, 'VPN check failed; no SSH'
        stage = 'remote_identity'
        code = REMOTE_CODE
        for key, value in dict(ROOT=str(REMOTE), TRAINING=str(TRAINING), STEPS=STEPS, SELECTION_SHA=SELECTION_SHA,
                               TRAINING_SHA=TRAINING_SHA, EVALUATION_SHA=EVALUATION_SHA).items():
            code = code.replace('__'+key+'__', repr(value))
        inventory = remote_json(access, code)
        payloads = inventory.pop('files')
        assert inventory['status'] == 'pass'
        m.exclusive(access.audit/'REMOTE_BEST_INVENTORY.json', dict(m.META, **inventory))
        required = sum(v['bytes'] for v in inventory['models'].values()) + inventory['lightweight_bytes']
        free = shutil.disk_usage(ROOT).free
        assert free > required + 4*1024**3, 'Insufficient local capacity plus 4GiB reserve'
        DESTINATION.mkdir(exist_ok=False)
        stage = 'lightweight_download'
        entries = {}
        for name, entry in payloads.items():
            rel = Path(name);assert not rel.is_absolute() and '..' not in rel.parts
            data = base64.b64decode(entry['data_b64'], validate=True)
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            p = access.audit/'lightweight'/rel;p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:f.write(data)
            entries[name] = {k:v for k,v in entry.items() if k != 'data_b64'}
        report.update(remote_inventory_utc=inventory['remote_utc'], remote_inventory_local_received_utc=inventory['local_received_utc'],
                      selection_sha256=SELECTION_SHA, training_package_sha256=TRAINING_SHA, evaluation_package_sha256=EVALUATION_SHA,
                      storage_before=inventory['storage'], local_available_before_bytes=free, lightweight_bytes=inventory['lightweight_bytes'],
                      lightweight_files=entries)
        stage = 'model_download'
        for mod in MODS:
            entry = inventory['models'][mod]
            dest = DESTINATION/mod/('diffusion_'+str(entry['selected_step'])+'.pt');dest.parent.mkdir()
            partial = dest.with_suffix('.pt.partial')
            with partial.open('xb'):pass
            access.call(['scp','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                         access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+entry['path'],str(partial)], timeout=600)
            assert partial.stat().st_size == entry['bytes'] and m.sha(partial) == entry['sha256'], 'Downloaded model identity mismatch'
            os.link(partial,dest);partial.unlink()
            report['models'][mod] = dict(entry, local_path=str(dest), local_verified_utc=now())
            m.exclusive(access.audit/(mod.upper()+'_MODEL_VERIFIED.json'), dict(m.META, **report['models'][mod]))
            print(mod.upper()+': selected '+str(entry['selected_step'])+' checkpoint downloaded; bytes/SHA256 verified', flush=True)
        stage = 'remote_post_transfer'
        post = r'''
import datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path(__ROOT__);models=__MODELS__
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'selection/SELECTION.json')==__SELECTION_SHA__
assert json.loads((root/'selection/COMPLETE.ok').read_text())['selection_sha256']==__SELECTION_SHA__
assert not (root/'selection/FAILED.json').exists()
for e in models.values():
 p=pathlib.Path(e['path']);s=p.stat()
 assert not p.is_symlink() and p.resolve()==p and s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns']
 assert sha(p)==e['sha256']
v=os.statvfs(root);r=subprocess.run(['df','-B1',str(root)],capture_output=True,text=True,timeout=30);assert r.returncode==0
print(json.dumps(dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_bytes=v.f_bavail*v.f_frsize,
 total_bytes=v.f_blocks*v.f_frsize,df=r.stdout,selected_model_bytes_rehashed=True)))
'''.replace('__ROOT__',repr(str(REMOTE))).replace('__MODELS__',repr(inventory['models'])).replace('__SELECTION_SHA__',repr(SELECTION_SHA))
        report['storage_after'] = remote_json(access, post, timeout=120)
        report.update(status='selected_models_bytes_verified', local_end_utc=now(), local_available_after_bytes=shutil.disk_usage(DESTINATION).free,
                      model_total_bytes=sum(e['bytes'] for e in report['models'].values()), model_deserialization='not_performed',
                      manual_quality_review='not_performed', independent_per_seed_bootstrap_review='not_performed')
    except BaseException as error:
        m.exclusive(access.audit/'PULL_FAILED.json', dict(report, status='failed', stage=stage, message=str(error), utc=now(), no_automatic_retry=True))
        raise
    finally:
        session = shutdown(access)
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json', dict(m.META, stage=stage, completed_utc=now(), **session))
    report.update(session, local_acceptance_utc=now())
    m.exclusive(access.audit/'PULL_ACCEPTANCE.json', report)
    m.exclusive(DESTINATION/'MODEL_MANIFEST.json', dict(m.META, selection_sha256=SELECTION_SHA,
                 training_package_sha256=TRAINING_SHA,evaluation_package_sha256=EVALUATION_SHA,models=report['models'],
                 acceptance_path=str(access.audit/'PULL_ACCEPTANCE.json'), local_utc=now(), selection_scope='fixedval103_reconstructed_reference_rule'))
    verify_local(access.audit)


if __name__ == '__main__':
    main()
