"""One old-job snapshot, new-root real preflight, and at most one held submit."""
import base64
import importlib.util
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"


def main():
    spec = importlib.util.spec_from_file_location("transport", HELPER)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT = ROOT
    access = transport.Access("deploy_checked_once")
    start = datetime.now(timezone.utc).isoformat()
    outcome = "failed_no_automatic_retry"
    try:
        plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
        remote = plan["remote_root"]
        digest = transport.sha(ROOT / "CPU_EXECUTION_PLAN.json")
        assert remote.endswith("/"+ROOT.name)
        bundle = access.audit / "transfer_bundle"
        bundle.mkdir()
        for name, expected in plan["static_sha256"].items():
            assert transport.sha(ROOT/name) == expected, "local static drift"
            target = bundle/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        shutil.copyfile(ROOT/"CPU_EXECUTION_PLAN.json", bundle/"CPU_EXECUTION_PLAN.json")
        access.start_vpn()
        snapshot = r'''
import base64,datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_k1_small_lesion_p0_20261013_r1')
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
files={}
for p in list((root/'evidence').glob('*'))+list((root/'logs').glob('*')):
 if p.is_file():
  raw=p.read_bytes(); assert len(raw)<5000000,'unexpected heavy evidence'
  files[str(p.relative_to(root))]=dict(sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
plan=json.loads((root/'CPU_EXECUTION_PLAN.json').read_text())
assert hashlib.sha256((root/'CPU_EXECUTION_PLAN.json').read_bytes()).hexdigest()=='d9d353513e5ce6e95e905c843f6174db17c3e651f8a060fff07876b2edbcf387'
for n,d in plan['static_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==d,'old runtime drift'
payload=dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
 disk=command(['df','-h','/public']),squeue=command(['squeue','-h','-j','4051333','-o','%i|%j|%T|%R']),
 sacct=command(['sacct','-n','-X','-j','4051333','--format=JobID,JobName,State,ExitCode','-P']),files=files,old_runtime_sha_pass=True)
print('OLD_SNAPSHOT='+json.dumps(payload))
'''
        raw = access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(snapshot), timeout=120)
        observation = json.loads(next(x.split("=",1)[1] for x in raw.splitlines() if x.startswith("OLD_SNAPSHOT=")))
        transport.exclusive(access.audit/"OLD_R1_SNAPSHOT.json", dict(**transport.META, **observation))
        for name, record in observation["files"].items():
            path = access.audit/"old_r1_readonly"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(record["base64"]))
            assert transport.sha(path) == record["sha256"]
        queue = observation["squeue"]
        assert not queue["stdout"].strip(), "r1 still active; stop new deployment"
        assert queue["rc"] == 0 or "Invalid job id" in queue["stderr"], "old queue read failed"
        accounting = observation["sacct"]
        assert accounting["rc"] == 0
        row = next(x.split("|") for x in accounting["stdout"].splitlines() if x.startswith("4051333|s2k1p0cpu1|"))
        assert row[2].split()[0] in {"CANCELLED", "FAILED", "COMPLETED", "TIMEOUT"}, "r1 terminal unconfirmed"
        # Atomic new-root reservation; never overwrite or reuse a deployed candidate.
        reserve = "import pathlib,json; r=pathlib.Path("+repr(remote)+"); r.mkdir(mode=0o700); (r/'evidence').mkdir(); (r/'logs').mkdir(); (r/'evidence/DEPLOY_RESERVATION.json').write_text(json.dumps(dict(plan_sha256="+repr(digest)+",status='reserved_new_root'))); print('NEW_ROOT_RESERVED')"
        assert "NEW_ROOT_RESERVED" in access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(reserve), timeout=60)
        access.call(["scp", "-r", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
                     *[str(p) for p in bundle.iterdir()], f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}/'], timeout=150)
        syntax = "import pathlib,json,hashlib,py_compile,tempfile; r=pathlib.Path("+repr(remote)+"); p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text()); assert hashlib.sha256((r/'CPU_EXECUTION_PLAN.json').read_bytes()).hexdigest()=="+repr(digest)+"; t=tempfile.TemporaryDirectory(); [py_compile.compile(str(r/n),cfile=t.name+'/'+str(i)+'.pyc',doraise=True) for i,n in enumerate(p['static_sha256']) if n.endswith('.py')]; t.cleanup(); print('PY_COMPILE_PASS')"
        checked = access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(syntax)+" && bash -n "+shlex.quote(remote+"/slurm/cpu_diagnosis.slurm"), timeout=60)
        assert "PY_COMPILE_PASS" in checked
        checked = access.ssh("env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "+transport.PYTHON+" -B "+shlex.quote(remote+"/runtime/cpu_roundtrip.py")+" preflight", timeout=420)
        assert "P0_CPU_STATIC_PASS" in checked
        for name in ("OFFICIAL_BOUNDARY_TESTS.json", "CPU_STATIC_PREFLIGHT.json"):
            target = access.audit/name
            access.call(["scp", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
                         f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}/evidence/{name}', str(target)], timeout=90)
            report = json.loads(target.read_text())
            assert report["status"] == "pass"
        pre = json.loads((access.audit/"CPU_STATIC_PREFLIGHT.json").read_text())
        boundary = json.loads((access.audit/"OFFICIAL_BOUNDARY_TESTS.json").read_text())
        assert pre["plan_sha256"] == digest and pre["official_boundary_report_sha256"] == transport.sha(access.audit/"OFFICIAL_BOUNDARY_TESTS.json")
        assert boundary["unittest_count"] == 11 and len(boundary["real_fixed103_checks"]) == 4
        transport.exclusive(access.audit/"PREFLIGHT_RECEIPT.json",dict(**transport.META,status="pass",plan_sha256=digest,
            actual_tests=11,real_subset_checks=4,old_terminal=row[2],stdout_tail=checked[-1500:]))
        launch = r'''
import datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path(REMOTE)
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,data):
 fd=os.open(root/'evidence'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
 with os.fdopen(fd,'w') as f:json.dump(dict(**meta,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data),f,indent=2)
plan=json.loads((root/'CPU_EXECUTION_PLAN.json').read_text())
digest=hashlib.sha256((root/'CPU_EXECUTION_PLAN.json').read_bytes()).hexdigest()
pre=json.loads((root/'evidence/CPU_STATIC_PREFLIGHT.json').read_text())
boundary=json.loads((root/'evidence/OFFICIAL_BOUNDARY_TESTS.json').read_text())
assert digest==DIGEST and pre['status']=='pass' and pre['plan_sha256']==digest
assert boundary['status']=='pass' and boundary['unittest_count']==11 and len(boundary['real_fixed103_checks'])==4
assert pre['official_boundary_report_sha256']==hashlib.sha256((root/'evidence/OFFICIAL_BOUNDARY_TESTS.json').read_bytes()).hexdigest()
assert not (root/'evidence/CPU_STATIC_PREFLIGHT_FAILED.json').exists()
for n,d in plan['static_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==d
q=subprocess.run(['squeue','-h','-n',plan['job_name'],'-o','%i'],capture_output=True,text=True,check=True,timeout=30)
assert not q.stdout.strip(),'same-name job exists'
emit('ENGINEERING_EXECUTION_AUTHORIZATION.json',dict(status='authorized_cpu_only',plan_sha256=digest,
 source=plan['authorization_source'],boundary_report_sha256=pre['official_boundary_report_sha256'],training_authorized=False))
emit('CPU_RESERVATION.json',dict(status='reserved',job_name=plan['job_name'],plan_sha256=digest))
r=subprocess.run(['sbatch','--hold','--parsable','--job-name='+plan['job_name'],str(root/'slurm/cpu_diagnosis.slurm')],capture_output=True,text=True,timeout=30)
if r.returncode:
 emit('CPU_SUBMISSION_FAILED.json',dict(status='failed',return_code=r.returncode,stderr=r.stderr))
 print('CPU_LAUNCH='+json.dumps(dict(status='submission_failed',error=r.stderr)));raise SystemExit(0)
jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
emit('CPU_LAUNCH.json',dict(status='submitted_held',job_id=jid,canonical_job_id=jid,job_name=plan['job_name'],plan_sha256=digest))
r=subprocess.run(['scontrol','release',jid],capture_output=True,text=True,timeout=30)
if r.returncode:
 emit('CPU_RELEASE_FAILED.json',dict(status='failed',job_id=jid,error=r.stderr))
 print('CPU_LAUNCH='+json.dumps(dict(status='release_failed',job_id=jid,error=r.stderr)));raise SystemExit(0)
emit('CPU_RELEASED.json',dict(status='released',job_id=jid))
print('CPU_LAUNCH='+json.dumps(dict(status='released',job_id=jid,job_name=plan['job_name'],plan_sha256=digest,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''.replace("REMOTE", repr(remote)).replace("DIGEST", repr(digest))
        raw = access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(launch), timeout=120)
        result = json.loads(next(x.split("=",1)[1] for x in raw.splitlines() if x.startswith("CPU_LAUNCH=")))
        transport.exclusive(access.audit/"LAUNCH_RECEIPT.json",dict(**transport.META,**result))
        assert result["status"] == "released", "unique submit/release failed; no retry"
        outcome = "released"
        print(json.dumps(dict(**transport.META,**result,audit=str(access.audit),old_terminal=row[2],real_tests=11)))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit/"LOCAL_OPERATION_RECEIPT.json",dict(**transport.META,status=outcome,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),
            vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,
            operation="one_old_progress_snapshot_unique_new_deploy_preflight_launch_no_poll"))


if __name__ == "__main__":
    main()
