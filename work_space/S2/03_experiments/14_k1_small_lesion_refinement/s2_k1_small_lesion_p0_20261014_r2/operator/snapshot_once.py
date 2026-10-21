"""One read-only remote observation; terminal lightweight evidence pull only."""
import base64
import hashlib
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"
PLAN_SHA = "e3169bf735a39d05c066ca34c5568285b48ba060c79007331a850c903c936beb"


def main():
    spec = importlib.util.spec_from_file_location("transport", HELPER)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT = ROOT
    access = transport.Access("snapshot_once")
    start = datetime.now(timezone.utc).isoformat()
    outcome = "failed_readonly_no_retry"
    try:
        assert transport.sha(ROOT / "CPU_EXECUTION_PLAN.json") == PLAN_SHA
        plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
        remote = plan["remote_root"]
        access.start_vpn()
        code = r'''
import base64,csv,datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(REMOTE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
plan=json.loads((root/'CPU_EXECUTION_PLAN.json').read_text())
launch=json.loads((root/'evidence/CPU_LAUNCH.json').read_text())
assert launch['canonical_job_id']=='4068477' and launch['job_name']=='s2k1p0cpu2'
drift=[]
if sha(root/'CPU_EXECUTION_PLAN.json')!=PLAN_SHA:drift.append('CPU_EXECUTION_PLAN.json')
for name,digest in plan['static_sha256'].items():
 if not (root/name).is_file() or sha(root/name)!=digest:drift.append(name)
queue=command(['squeue','-h','-j','4068477','-o','%i|%j|%T|%R'])
account=command(['sacct','-n','-X','-j','4068477','--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList','-P'])
logs={};errors=[];roundtrip_progress=[];identity_progress=[]
for p in sorted((root/'logs').glob('*')):
 if p.is_file():
  content=p.read_text(errors='replace')
  logs[str(p.relative_to(root))]=dict(size=p.stat().st_size,sha256=sha(p),tail=content[-10000:])
  for line in content.splitlines():
   if re.search(r'Traceback \(most recent call last\)|OutOfMemoryError|out.of.memory|CUDA error|CUDA_ERROR|ImportError:|ModuleNotFoundError:|RuntimeError:|ValueError:|AssertionError:|TimeoutExpired:|DUE TO TIME LIMIT|nonfinite.*(failed|error)|oom-kill',line,re.I):errors.append(line)
   if line.startswith('P0_ROUNDTRIP_CASE '):roundtrip_progress.append(line)
   if line.startswith('P0_IDENTITY_CASE '):identity_progress.append(line)
evidence={}
for p in sorted((root/'evidence').glob('*')):
 if p.is_file() and p.suffix in ('.json','.ok'):
  evidence[p.name]=dict(sha256=sha(p),content=p.read_text())
files={}
for parent in ('evidence','results','logs','manifests','slurm','sources'):
 for p in sorted((root/parent).rglob('*')):
  if p.is_file() and p.suffix in ('.json','.csv','.md','.ok','.out','.err','.log','.slurm'):
   files[str(p.relative_to(root))]=dict(size=p.stat().st_size,sha256=sha(p))
for relative in ('CPU_EXECUTION_PLAN.json','roundtrip/GEOMETRY.json'):
 p=root/relative
 if p.is_file():files[relative]=dict(size=p.stat().st_size,sha256=sha(p))
counts={}
for p in (root/'results').rglob('*.csv'):
 with p.open() as f:
  reader=csv.DictReader(f); rows=sum(1 for _ in reader)
 counts[str(p.relative_to(root))]=rows
pred_count=len(list((root/'roundtrip/prediction').glob('*.nii.gz')))
payload=dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
 disk=command(['df','-h','/public']),path_disk=command(['df','-h',str(root)]),squeue=queue,sacct=account,
 plan_sha256=sha(root/'CPU_EXECUTION_PLAN.json'),static_sha_drift=drift,logs=logs,evidence=evidence,
 critical_errors=errors,roundtrip_logged_count=len(roundtrip_progress),roundtrip_last=roundtrip_progress[-1:] or None,
 identity_logged_count=len(identity_progress),identity_last=identity_progress[-1:] or None,
 roundtrip_nifti_count=pred_count,csv_rows=counts,lightweight_files=files)
print('P0_SNAPSHOT='+json.dumps(payload))
'''.replace("REMOTE", repr(remote)).replace("PLAN_SHA", repr(PLAN_SHA))
        raw = access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(code), timeout=180)
        observed = json.loads(next(x.split("=", 1)[1] for x in raw.splitlines() if x.startswith("P0_SNAPSHOT=")))
        transport.exclusive(access.audit / "PROGRESS_SNAPSHOT.json", dict(**transport.META, local_start_utc=start, **observed))
        assert observed["sacct"]["rc"] == 0, "accounting read failed"
        row = next(x.split("|") for x in observed["sacct"]["stdout"].splitlines() if x.startswith("4068477|s2k1p0cpu2|"))
        state = row[2].split()[0]
        queue = observed["squeue"]
        assert queue["rc"] == 0 or (not queue["stdout"].strip() and "Invalid job id" in queue["stderr"]
                and state not in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}), "unexplained queue error"
        terminal = state not in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
        if terminal:
            pulled = access.audit / "remote_exact"
            pulled.mkdir()
            parents = {name.split('/')[0] for name in observed["lightweight_files"] if '/' in name}
            parents.discard("roundtrip")
            sources = [remote+'/'+p for p in sorted(parents)] + [remote+'/CPU_EXECUTION_PLAN.json']
            if "roundtrip/GEOMETRY.json" in observed["lightweight_files"]:
                (pulled / "roundtrip").mkdir()
                access.call(["scp", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                             "-o", "ConnectTimeout=15", f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}/roundtrip/GEOMETRY.json',
                             str(pulled / "roundtrip/GEOMETRY.json")], timeout=120)
            access.call(["scp", "-r", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                         "-o", "ConnectTimeout=15", *[f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{p}' for p in sources],
                         str(pulled)], timeout=240)
            mismatches, parsed = [], 0
            for name, expected in observed["lightweight_files"].items():
                p = pulled / name
                if not p.is_file() or transport.sha(p) != expected["sha256"]:
                    mismatches.append(name)
                if p.is_file() and p.suffix == ".json":
                    json.loads(p.read_text())
                    parsed += 1
            forbidden = [str(p.relative_to(pulled)) for p in pulled.rglob('*') if p.is_file() and
                         (p.name.endswith(('.nii.gz','.pth','.npz','.jsonl')))]
            transport.exclusive(access.audit / "LIGHTWEIGHT_PULL_VALIDATION.json", dict(**transport.META,
                status="pass" if not mismatches and not forbidden else "failed", file_count=len(observed["lightweight_files"]),
                json_parsed=parsed,sha_mismatches=mismatches,forbidden_files=forbidden))
            assert not mismatches and not forbidden, "lightweight transfer mismatch"
        failures = [name for name in observed["evidence"] if "FAILED" in name]
        healthy = state == "COMPLETED" and row[3] == "0:0" and not failures and not observed["critical_errors"] and not observed["static_sha_drift"]
        outcome = "completed_pending_local_acceptance" if healthy else ("active_no_retry" if not terminal and not failures and not observed["critical_errors"] and not observed["static_sha_drift"] else "failed_preserved_no_retry")
        print(json.dumps(dict(**transport.META,status=outcome,job_id="4068477",state=state,exit_code=row[3],
            elapsed=row[4],roundtrip_count=observed["roundtrip_nifti_count"],identity_logged_count=observed["identity_logged_count"],
            failed_markers=failures,critical_errors=observed["critical_errors"][-6:],sha_drift=observed["static_sha_drift"],audit=str(access.audit))))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json",dict(**transport.META,status=outcome,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn,readonly=True,progress_snapshots=1))


if __name__ == "__main__":
    main()
