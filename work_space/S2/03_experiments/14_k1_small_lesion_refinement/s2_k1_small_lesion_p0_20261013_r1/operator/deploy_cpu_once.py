"""Unique new-root deploy, real preflight, held launch audit, and release."""
import importlib.util
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"


def main():
    spec = importlib.util.spec_from_file_location("transport",HELPER)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT=ROOT
    access = transport.Access("deploy_cpu_once")
    started=datetime.now(timezone.utc).isoformat()
    outcome="failed"
    try:
        plan=json.loads((ROOT/"CPU_EXECUTION_PLAN.json").read_text())
        remote=plan["remote_root"]
        assert remote.endswith("/"+ROOT.name)
        bundle=access.audit/"transfer_bundle"
        bundle.mkdir()
        for relative,digest in plan["static_sha256"].items():
            assert transport.sha(ROOT/relative)==digest,relative
            target=bundle/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        shutil.copyfile(ROOT/"CPU_EXECUTION_PLAN.json",bundle/"CPU_EXECUTION_PLAN.json")
        (bundle/"logs").mkdir()
        access.start_vpn()
        # One target/name probe is a deployment guard, not a second progress poll.
        probe=access.ssh("test ! -e "+shlex.quote(remote)+" && printf ROOT_ABSENT",timeout=60)
        assert "ROOT_ABSENT" in probe,"target root exists; no overwrite/redeploy"
        access.call(["scp","-r","-P",access.env["HPC_LOGIN_PORT"],"-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=15",str(bundle),f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}'],timeout=120)
        compile_code="import json,pathlib; r=pathlib.Path("+repr(remote)+"); p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text()); [compile((r/n).read_text(),n,'exec') for n in p['static_sha256'] if n.endswith('.py')]; print('CPU_COMPILE_PASS')"
        check=access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(compile_code)+" && bash -n "+shlex.quote(remote+"/slurm/cpu_diagnosis.slurm"),timeout=60)
        assert "CPU_COMPILE_PASS" in check
        command="env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "+transport.PYTHON+" -B "+shlex.quote(remote+"/runtime/cpu_roundtrip.py")+" preflight"
        checked=access.ssh(command,timeout=360)
        assert "P0_CPU_STATIC_PASS" in checked
        transport.exclusive(access.audit/"PREFLIGHT_RECEIPT.json",dict(**transport.META,status="pass",plan_sha256=transport.sha(ROOT/"CPU_EXECUTION_PLAN.json"),diagnostic_tail=checked[-1600:]))
        launch=r'''
import datetime,json,os,pathlib,subprocess,hashlib
root=pathlib.Path(REMOTE)
plan=json.loads((root/'CPU_EXECUTION_PLAN.json').read_text())
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,data):
 fd=os.open(root/'evidence'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
 with os.fdopen(fd,'w') as f:json.dump(dict(**meta,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data),f,indent=2)
pre=json.loads((root/'evidence/CPU_STATIC_PREFLIGHT.json').read_text())
digest=hashlib.sha256((root/'CPU_EXECUTION_PLAN.json').read_bytes()).hexdigest()
assert pre['status']=='pass' and pre['plan_sha256']==digest
assert not (root/'evidence/CPU_STATIC_PREFLIGHT_FAILED.json').exists()
q=subprocess.run(['squeue','-h','-n',plan['job_name'],'-o','%i'],capture_output=True,text=True,check=True)
assert not q.stdout.strip(),'same-name job already exists'
emit('CPU_RESERVATION.json',dict(status='reserved',plan_sha256=digest,job_name=plan['job_name']))
r=subprocess.run(['sbatch','--hold','--parsable','--job-name='+plan['job_name'],str(root/'slurm/cpu_diagnosis.slurm')],capture_output=True,text=True)
if r.returncode:
 emit('CPU_SUBMISSION_FAILED.json',dict(status='failed',return_code=r.returncode,stderr=r.stderr))
 print('CPU_LAUNCH='+json.dumps(dict(status='submission_failed',error=r.stderr)));raise SystemExit(0)
jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
emit('CPU_LAUNCH.json',dict(status='submitted_held',job_id=jid,canonical_job_id=jid,job_name=plan['job_name'],plan_sha256=digest))
r=subprocess.run(['scontrol','release',jid],capture_output=True,text=True)
if r.returncode:
 emit('CPU_RELEASE_FAILED.json',dict(status='failed',job_id=jid,error=r.stderr))
 print('CPU_LAUNCH='+json.dumps(dict(status='release_failed',job_id=jid,error=r.stderr)));raise SystemExit(0)
emit('CPU_RELEASED.json',dict(status='released',job_id=jid))
print('CPU_LAUNCH='+json.dumps(dict(status='released',job_id=jid,job_name=plan['job_name'],remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''.replace("REMOTE",repr(remote))
        raw=access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(launch),timeout=90)
        line=next(x for x in raw.splitlines() if x.startswith("CPU_LAUNCH="))
        result=json.loads(line.split("=",1)[1])
        transport.exclusive(access.audit/"LAUNCH_RECEIPT.json",dict(**transport.META,**result))
        assert result["status"]=="released","unique CPU launch failed; no retry"
        outcome="released"
        print(json.dumps(dict(status=outcome,audit=str(access.audit),**{k:v for k,v in result.items() if k!='status'})))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit/"LOCAL_OPERATION_RECEIPT.json",dict(**transport.META,status=outcome,
             local_start_utc=started,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
             preexisting_vpn_preserved=not access.owns_vpn,remote_operation="unique_deploy_preflight_launch_no_status_poll"))


if __name__ == "__main__":
    main()
