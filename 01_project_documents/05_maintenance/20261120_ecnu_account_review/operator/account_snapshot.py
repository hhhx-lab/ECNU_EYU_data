"""One bounded read-only snapshot of shared-account scheduler metadata."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
TRANSPORT = PROJECT / "work_space/G1/03_experiments/03_diffusion_training/20261020_raw_real_modality_specific_1mm_r2/operator/common.py"
spec = importlib.util.spec_from_file_location("account_review_transport", TRANSPORT)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.m.ROOT = ROOT

CODE = r'''
import getpass,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(argv=argv,returncode=124,stdout='',stderr='bounded command timeout')

started=datetime.now(timezone.utc).isoformat()
user=getpass.getuser()
queue=command(['squeue','--array','-u',user,'-h','-o','%i|%j|%P|%T|%M|%S|%R|%Z'])
fields=['JobIDRaw','JobID','JobName%160','Partition','State','ExitCode','Elapsed','Submit','Start','End','WorkDir%800','AllocTRES%240']
history=command(['sacct','-X','-n','-P','-u',user,'-S','2026-11-06','--format='+','.join(fields)])
rows=[]
parse_errors=[]
keys=['job_id_raw','job_id','name','partition','state','exit_code','elapsed','submit','start','end','work_dir','allocated_tres']
for line in history['stdout'].splitlines():
    cells=line.split('|')
    if cells and cells[-1]=='':cells.pop()
    if len(cells)!=len(keys):parse_errors.append(line)
    else:rows.append(dict(zip(keys,[x.strip() for x in cells])))
queue_rows=[]
qkeys=['job_id','name','partition','state','elapsed','expected_start','reason_or_node','work_dir']
for line in queue['stdout'].splitlines():
    cells=line.split('|')
    if len(cells)!=len(qkeys):parse_errors.append(line)
    else:queue_rows.append(dict(zip(qkeys,[x.strip() for x in cells])))
home=Path.home()
disk=command(['df','-B1',str(home)])
quota=command(['quota','-v','-u',user])
fs=os.statvfs(home)
out=dict(status='pass' if queue['returncode']==history['returncode']==0 and not parse_errors else 'query_incomplete',
    observation_kind='scheduler_metadata_only_not_completion_acceptance',
    artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    user=user,hostname=os.uname().nodename,remote_started_utc=started,remote_utc=datetime.now(timezone.utc).isoformat(),
    history_start='2026-11-06',queue=queue,history=history,queue_rows=queue_rows,history_rows=rows,
    parse_errors=parse_errors,disk=disk,quota=quota,filesystem=dict(total_bytes=fs.f_blocks*fs.f_frsize,available_bytes=fs.f_bavail*fs.f_frsize),
    boundaries=dict(no_job_changes=True,no_project_file_reads=True,no_download=True,no_remote_writes=True))
print(json.dumps(out,sort_keys=True))
'''


def main():
    if os.environ.get("ECNU_SERVER_ENV_FILE") != transport.m.ENVFILE:
        raise RuntimeError("Exact authorized env path required")

    def record_source(access):
        transport.m.exclusive(access.audit / "EXECUTED_SOURCE.json", {
            **transport.m.META, "source": CODE,
            "local_operator_sha256": transport.m.sha(Path(__file__)),
            "purpose": "User clarified jobs submitted by another person within the last two weeks",
            "no_job_changes": True,
        })

    result, audit = transport.run_remote("account_metadata", CODE, timeout=140, setup=record_source)
    print(json.dumps({"audit": str(audit), "queue": result["queue_rows"],
        "history_count": len(result["history_rows"]),
        "recent_history": sorted(result["history_rows"], key=lambda r:r["submit"], reverse=True)[:100],
        "filesystem": result["filesystem"], "remote_utc": result["remote_utc"],
        "local_received_utc": result["local_received_utc"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
