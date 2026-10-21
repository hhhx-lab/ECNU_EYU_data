"""Deploy once, execute real preflight, audit held allocation, then release CPU only."""
import json
import shlex
import shutil
from datetime import datetime,timezone
from common import ROOT,REMOTE,META,transport

EXPECTED='bfa5b07b2cdb2f8e0f6e75e477570c2305e2fc0bbca17b999a420dbe7a1e7dd4'
LAUNCH=r'''
import datetime,hashlib,json,pathlib,subprocess,sys,os
r=pathlib.Path(REMOTE);sys.path.insert(0,str(r/'runtime'))
from slurm_contract import validate
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,data):
 with (r/'evidence'/name).open('x') as f:json.dump(dict(**meta,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data),f,indent=2)
def run(args):return subprocess.run(args,capture_output=True,text=True,timeout=45)
p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text());jid=None
def guard():
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 assert sha(r/'CPU_EXECUTION_PLAN.json')==DIGEST,'plan drift'
 assert all(sha(r/n)==s for n,s in p['static_sha256'].items()),'static drift'
 assert not list((r/'evidence').glob('*FAILED*')),'failed guard'
 assert p['probability']['local_probability_storage_authorized'] is False,'latest server-only contract drift'
 script=(r/'slurm/cpu_supplement.slurm').read_text()
 assert '--exclusive' not in script and '--gres' not in script,'CPU request drift'
 available=os.statvfs(r).f_bavail*os.statvfs(r).f_frsize
 assert available>p['probability']['other_G1_full_reserve_bytes']+2**30,'CPU concurrent storage gate'
 return available
try:
 available=guard()
 pre=json.loads((r/'evidence/SUPPLEMENT_STATIC_PREFLIGHT.json').read_text())
 assert pre['status']=='pass' and pre['plan_sha256']==DIGEST
 assert pre['all_source_native_tests']['actual_small_large_merge']['source_degree']==2
 q=run(['squeue','-h','-n',p['job_name'],'-o','%i'])
 assert q.returncode==0 and not q.stdout.strip(),'same-name active job'
 nodes=run(['scontrol','show','nodes','node[201-228,230]','-o'])
 assert nodes.returncode==0
 node_rows=[dict(x.split('=',1) for x in line.split() if '=' in x) for line in nodes.stdout.splitlines() if line.strip()]
 fits=[x['NodeName'] for x in node_rows if int(x.get('RealMemory','0'))-int(x.get('MemSpecLimit','0'))>=65536]
 emit('CPU_NODE_PREFLIGHT.json',dict(status='pass' if fits else 'failed',nodes_raw=nodes.stdout,configured_fitting_nodes=fits,request_mib=65536))
 assert fits,'no configured CPU192 node supports64GiB after system reserve'
 script=r/'slurm/cpu_supplement.slurm';test=run(['sbatch','--test-only',str(script)])
 emit('SCHEDULER_STATIC_PREFLIGHT.json',dict(status='pass' if test.returncode==0 else 'blocked',return_code=test.returncode,stdout=test.stdout,stderr=test.stderr))
 if test.returncode:
  print('LAUNCH='+json.dumps(dict(status='blocked_scheduler_no_submit',error=test.stderr)));raise SystemExit(0)
 guard()
 emit('CPU_RESERVATION.json',dict(status='reserved_unique',plan_sha256=DIGEST,job_name=p['job_name']))
 emit('CPU_ENGINEERING_AUTHORIZATION.json',dict(status='authorized_cpu_only',source=p['authorization_source'],training_authorized=False,probability_gpu_authorized=False))
 submission=run(['sbatch','--hold','--parsable','--export=NONE',str(script)])
 if submission.returncode:raise RuntimeError('unique sbatch rejected: '+submission.stderr)
 jid=submission.stdout.strip().split(';')[0];assert jid.isdigit()
 emit('CPU_LAUNCH.json',dict(status='submitted_held',canonical_job_id=jid,plan_sha256=DIGEST))
 held=run(['scontrol','show','job',jid,'-o']);assert held.returncode==0
 report=validate(held.stdout,str(r),p['job_name'])
 emit('CPU_HELD_AUDIT.json',dict(**report,canonical_job_id=jid))
 assert report['status']=='pass','held allocation mismatch: '+str(report['checks'])
 guard()
 released=run(['scontrol','release',jid]);assert released.returncode==0,released.stderr
 emit('CPU_RELEASED.json',dict(status='released',canonical_job_id=jid))
 print('LAUNCH='+json.dumps(dict(status='released_no_progress_requery',canonical_job_id=jid,available_bytes=available)))
except Exception as e:
 result=dict(status='failed_no_automatic_retry',error=str(e),canonical_job_id=jid)
 if jid:
  cancel=run(['scancel',jid]);result.update(cancel_return_code=cancel.returncode,cancel_stderr=cancel.stderr)
 emit('CPU_LAUNCH_ABORTED_FAILED.json',result)
 print('LAUNCH='+json.dumps(result))
'''


def main():
    t=transport();access=t.Access('deploy_once')
    start=datetime.now(timezone.utc).isoformat();status='failed_no_automatic_retry'
    try:
        assert t.sha(ROOT/'CPU_EXECUTION_PLAN.json')==EXPECTED
        plan=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
        bundle=access.audit/'bundle';bundle.mkdir()
        for name,digest in plan['static_sha256'].items():
            assert t.sha(ROOT/name)==digest,'local frozen drift'
            target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        shutil.copyfile(ROOT/'CPU_EXECUTION_PLAN.json',bundle/'CPU_EXECUTION_PLAN.json')
        access.start_vpn()
        reserve='import pathlib,json; r=pathlib.Path('+repr(REMOTE)+');r.mkdir(mode=0o700);(r/"evidence").mkdir();(r/"logs").mkdir();f=(r/"evidence/DEPLOY_RESERVATION.json").open("x");json.dump('+repr(dict(**META,status='unique_new_root',plan_sha256=EXPECTED))+',f);f.close();print("RESERVED")'
        assert 'RESERVED' in access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(reserve),timeout=45)
        access.call(['scp','-r','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',str(bundle)+'/.',access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+REMOTE+'/'],timeout=180)
        compile_code='import hashlib,json,pathlib,py_compile,tempfile; r=pathlib.Path('+repr(REMOTE)+');p=json.loads((r/"CPU_EXECUTION_PLAN.json").read_text());assert hashlib.sha256((r/"CPU_EXECUTION_PLAN.json").read_bytes()).hexdigest()=='+repr(EXPECTED)+';assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in p["static_sha256"].items());tmp=tempfile.TemporaryDirectory();[py_compile.compile(str(r/n),cfile=tmp.name+"/"+str(i)+".pyc",doraise=True) for i,n in enumerate(p["static_sha256"]) if n.endswith(".py")];tmp.cleanup();print("STATIC_PASS")'
        assert 'STATIC_PASS' in access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(compile_code)+' && bash -n '+shlex.quote(REMOTE+'/slurm/cpu_supplement.slurm'),timeout=90)
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" '+t.PYTHON+' -B '+shlex.quote(REMOTE+'/runtime/supplement.py')+' preflight',timeout=600)
        t.exclusive(access.audit/'PREFLIGHT_LOG.json',dict(**META,status='returned_zero',output_tail=raw[-5000:]))
        target=access.audit/'SUPPLEMENT_STATIC_PREFLIGHT.json'
        access.call(['scp','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes',access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+REMOTE+'/evidence/SUPPLEMENT_STATIC_PREFLIGHT.json',str(target)],timeout=90)
        pre=json.loads(target.read_text());assert pre['status']=='pass' and pre['plan_sha256']==EXPECTED
        code=LAUNCH.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(EXPECTED))
        output=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(code),timeout=180)
        data=json.loads(next(s.split('=',1)[1] for s in output.splitlines() if s.startswith('LAUNCH=')))
        t.exclusive(access.audit/'LAUNCH_RECEIPT.json',dict(**META,**data,plan_sha256=EXPECTED))
        status=data['status']
        print(json.dumps(dict(status=status,canonical_job_id=data.get('canonical_job_id'),audit=str(access.audit),probability='server_only_capacity_hold')))
        assert status in ('released_no_progress_requery','blocked_scheduler_no_submit'),data
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**META,status=status,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=0,probability_gpu_submitted=False))


if __name__=='__main__':main()
