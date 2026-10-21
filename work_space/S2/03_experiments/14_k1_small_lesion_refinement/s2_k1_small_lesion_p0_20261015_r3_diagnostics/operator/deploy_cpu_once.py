"""Unique CPU deployment, real subset preflight, held audited launch; no polling."""
import base64
import importlib.util
import json
import shlex
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT.parents[2]/'03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'


def main():
    spec=importlib.util.spec_from_file_location('transport',HELPER)
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.ROOT=ROOT
    access=t.Access('deploy_cpu_once')
    start=datetime.now(timezone.utc).isoformat();status='failed_no_automatic_retry'
    p=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text());digest=t.sha(ROOT/'CPU_EXECUTION_PLAN.json');remote=p['remote_root']
    try:
        assert digest=='471ce8b9ff87b4aaf1a83ca8eee86cc9f0c4525d83e7fd8c632ad42a6f6236bd'
        bundle=access.audit/'bundle';bundle.mkdir()
        for name,wanted in p['static_sha256'].items():
            assert t.sha(ROOT/name)==wanted,'local frozen drift'
            target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        shutil.copyfile(ROOT/'CPU_EXECUTION_PLAN.json',bundle/'CPU_EXECUTION_PLAN.json')
        access.start_vpn()
        reserve="import pathlib,json; r=pathlib.Path("+repr(remote)+");r.mkdir(mode=0o700);(r/'evidence').mkdir();(r/'logs').mkdir();f=(r/'evidence/DEPLOY_RESERVATION.json').open('x');json.dump(dict(plan_sha256="+repr(digest)+",status='reserved_unique_new_root',artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed'),f);f.close();print('UNIQUE_NEW_ROOT')"
        assert 'UNIQUE_NEW_ROOT' in access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(reserve),timeout=60)
        access.call(['scp','-r','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',str(bundle)+'/.',access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+remote+'/'],timeout=180)
        syntax="import hashlib,json,pathlib,py_compile,tempfile; r=pathlib.Path("+repr(remote)+");p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text());assert hashlib.sha256((r/'CPU_EXECUTION_PLAN.json').read_bytes()).hexdigest()=="+repr(digest)+";assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==d for n,d in p['static_sha256'].items());tmp=tempfile.TemporaryDirectory();[py_compile.compile(str(r/n),cfile=tmp.name+'/'+str(i)+'.pyc',doraise=True) for i,n in enumerate(p['static_sha256']) if n.endswith('.py')];tmp.cleanup();print('STATIC_SHA_COMPILE_PASS')"
        assert 'STATIC_SHA_COMPILE_PASS' in access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(syntax)+' && bash -n '+shlex.quote(remote+'/slurm/cpu_supplement.slurm'),timeout=60)
        output=access.ssh('env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" '+t.PYTHON+' -B '+shlex.quote(remote+'/runtime/supplement.py')+' preflight',timeout=600)
        t.exclusive(access.audit/'PREFLIGHT_LOG.json',dict(**t.META,status='returned_zero',output_tail=output[-5000:]))
        target=access.audit/'SUPPLEMENT_STATIC_PREFLIGHT.json'
        access.call(['scp','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+remote+'/evidence/SUPPLEMENT_STATIC_PREFLIGHT.json',str(target)],timeout=90)
        pre=json.loads(target.read_text());assert pre['status']=='pass' and pre['plan_sha256']==digest
        assert pre['actual_transport_subset']['reference_count']>0
        code=r'''
import datetime,hashlib,json,os,pathlib,subprocess
r=pathlib.Path(REMOTE);meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,data):
 with (r/'evidence'/name).open('x') as f:json.dump(dict(**meta,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data),f,indent=2)
def run(args):return subprocess.run(args,capture_output=True,text=True,timeout=30)
p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text());sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(r/'CPU_EXECUTION_PLAN.json')==DIGEST
assert all(sha(r/n)==d for n,d in p['static_sha256'].items())
pre=json.loads((r/'evidence/SUPPLEMENT_STATIC_PREFLIGHT.json').read_text());assert pre['status']=='pass' and pre['plan_sha256']==DIGEST
assert not list((r/'evidence').glob('*FAILED*'))
capacity=p['probability_capacity'];available=os.statvfs(r).f_bavail*os.statvfs(r).f_frsize
assert available-capacity['other_authorized_G1_full_reserve_bytes']>capacity['cpu_lightweight_reserve_bytes'],'CPU storage gate'
q=run(['squeue','-h','-n',p['job_name'],'-o','%i']);assert q.returncode==0 and not q.stdout.strip(),'same-name active job'
script=r/'slurm/cpu_supplement.slurm'
test=run(['sbatch','--test-only',str(script)])
emit('SCHEDULER_STATIC_PREFLIGHT.json',dict(status='pass' if test.returncode==0 else 'blocked',return_code=test.returncode,stdout=test.stdout,stderr=test.stderr))
if test.returncode:
 print('LAUNCH='+json.dumps(dict(status='blocked_scheduler_no_submit',error=test.stderr)));raise SystemExit(0)
emit('CPU_RESERVATION.json',dict(status='reserved',plan_sha256=DIGEST,job_name=p['job_name']))
emit('CPU_ENGINEERING_AUTHORIZATION.json',dict(status='authorized_cpu_only',source=p['authorization_source'],plan_sha256=DIGEST,training_authorized=False))
s=run(['sbatch','--hold','--parsable','--export=NONE',str(script)])
if s.returncode:
 emit('CPU_SUBMISSION_FAILED.json',dict(status='failed',return_code=s.returncode,stderr=s.stderr))
 print('LAUNCH='+json.dumps(dict(status='submission_failed',error=s.stderr)));raise SystemExit(0)
jid=s.stdout.strip().split(';')[0];assert jid.isdigit()
emit('CPU_LAUNCH.json',dict(status='submitted_held',canonical_job_id=jid,job_id=jid,plan_sha256=DIGEST))
audit=run(['scontrol','show','job',jid,'-o']);assert audit.returncode==0
fields=dict(x.split('=',1) for x in audit.stdout.split() if '=' in x)
expected=dict(JobName=p['job_name'],Partition='cpu_192G',NumCPUs='4',NumNodes='1',MinMemoryNode='64G',TimeLimit='1-00:00:00',Command=str(script))
assert all(fields.get(k)==v for k,v in expected.items()),'held allocation contract mismatch'
assert fields.get('JobState')=='PENDING' and fields.get('Reason')=='JobHeldUser','not user-held'
assert '/logs/' in fields['StdOut'] and fields['StdOut'].startswith(str(r)) and fields['StdErr'].startswith(str(r))
assert 'gres/gpu' not in fields.get('ReqTRES',''),'CPU requested GPU'
emit('CPU_HELD_AUDIT.json',dict(status='pass',canonical_job_id=jid,fields=fields))
released=run(['scontrol','release',jid])
if released.returncode:
 emit('CPU_RELEASE_FAILED.json',dict(status='failed',canonical_job_id=jid,error=released.stderr))
 print('LAUNCH='+json.dumps(dict(status='release_failed',canonical_job_id=jid,error=released.stderr)));raise SystemExit(0)
emit('CPU_RELEASED.json',dict(status='released',canonical_job_id=jid))
print('LAUNCH='+json.dumps(dict(status='released_not_progress_polled',canonical_job_id=jid,job_name=p['job_name'],remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''.replace('REMOTE',repr(remote)).replace('DIGEST',repr(digest))
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(code),timeout=180)
        result=json.loads(next(x.split('=',1)[1] for x in raw.splitlines() if x.startswith('LAUNCH=')))
        t.exclusive(access.audit/'LAUNCH_RECEIPT.json',dict(**t.META,**result,plan_sha256=digest))
        assert result['status'] in ('released_not_progress_polled','blocked_scheduler_no_submit'), 'unique launch failed; stop'
        status=result['status']
        print(json.dumps(dict(**t.META,**result,audit=str(access.audit),probability='hold_capacity',original_E='blocked_missing_masks')))
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**t.META,status=status,local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=0))


if __name__=='__main__':main()
