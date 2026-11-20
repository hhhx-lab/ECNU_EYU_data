"""Bounded read-only Beijing timeline and original failure evidence snapshot."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("failure_transport", ROOT / "operator/account_snapshot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
transport = module.transport
m = transport.m

CODE = r'''
import hashlib,json,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
IDS=IDS_VALUE
FAILURES=FAILURES_VALUE
BASE=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated')
TRAIN=BASE/'s2_rc_train_batch1_20261112_r1'
AUDIT=BASE/'s2_rc_audit_20261111_r1'
started=datetime.now(timezone.utc).isoformat()
os.environ['TZ']='Asia/Shanghai'
time.tzset()
errors=[]
def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(returncode=124,stdout='',stderr='bounded command timeout')
def inspect(p,kind):
    if not p.exists():return dict(path=str(p),exists=False,kind=kind)
    if not p.is_file() or p.is_symlink():
        errors.append('Evidence is not a regular file: '+str(p));return dict(path=str(p),exists=False,kind=kind)
    size=p.stat().st_size
    if kind=='log':
        with p.open('rb') as f:
            f.seek(max(0,size-16384));b=f.read(16384)
        return dict(path=str(p),exists=True,kind=kind,bytes=size,excerpt_bytes=len(b),
            excerpt_sha256=hashlib.sha256(b).hexdigest(),is_tail_excerpt=True,text=b.decode(errors='replace'))
    if size>2097152:
        errors.append('Evidence exceeds lightweight budget: '+str(p));return dict(path=str(p),exists=True,kind=kind,bytes=size)
    b=p.read_bytes()
    out=dict(path=str(p),exists=True,kind=kind,bytes=size,sha256=hashlib.sha256(b).hexdigest())
    if p.suffix=='.json':
        try:out['value']=json.loads(b)
        except (ValueError,UnicodeError) as e:out['parse_error']=str(e);errors.append('Invalid JSON: '+str(p))
    else:out['text']=b.decode(errors='replace')
    return out
accounting=command(['sacct','-n','-P','-j',','.join(IDS),
    '--format=JobID,JobName%120,State,ExitCode,Submit,Start,End,Elapsed,NodeList%200,ReqMem,MaxRSS,MaxVMSize'])
if accounting['returncode']:errors.append('Accounting query failed')
keys=['job_id','name','state','exit_code','submit_beijing','start_beijing','end_beijing','elapsed','nodes','requested_memory','max_rss','max_vm_size']
rows=[]
for line in accounting['stdout'].splitlines():
    cells=line.split('|')
    if cells[-1]=='':cells.pop()
    if len(cells)!=len(keys):errors.append('Unparsed accounting row')
    else:rows.append(dict(zip(keys,[x.strip() for x in cells])))
records={}
for run,job in FAILURES.items():
    evidence=[]
    directory=TRAIN/'evidence'
    names=sorted(p for p in directory.glob('RUN_'+run+'_*') if p.suffix=='.json')
    if len(names)>30:errors.append('Unexpected evidence count: '+run);names=[]
    for p in names:evidence.append(inspect(p,'marker'))
    logs=[]
    for p in sorted((TRAIN/'slurm_logs').glob('*'+job+'*')):
        if p.suffix in {'.out','.err','.log'}:logs.append(inspect(p,'log'))
    records[run]=dict(job_id=job,evidence=evidence,logs=logs)
audit_logs=[inspect(p,'log') for p in sorted((AUDIT/'slurm_logs').glob('*4388852*')) if p.suffix in {'.out','.err','.log'}]
sources=[]
for root in (TRAIN,AUDIT):
    if not root.is_dir() or root.resolve()!=root:errors.append('Missing or redirected root: '+str(root));continue
    sources.append(inspect(root/'RUN.json','config'))
    for p in sorted((root/'slurm').glob('*.slurm')):sources.append(inspect(p,'source'))
    wanted=('runtime/p3.py','runtime/launch_run.py') if root==TRAIN else ('runtime/scan.py','runtime/rca.py')
    for name in wanted:sources.append(inspect(root/name,'source'))
output=dict(status='pass' if not errors else 'query_incomplete',
    artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    remote_started_utc=started,remote_utc=datetime.now(timezone.utc).isoformat(),
    query_timezone='Asia/Shanghai',query_tz_env=os.environ['TZ'],query_local_clock=datetime.now().astimezone().isoformat(),
    accounting=accounting,rows=rows,failed_training=records,rc_scan_failed_logs=audit_logs,sources=sources,errors=errors,
    boundaries=dict(no_job_changes=True,no_remote_writes=True,no_image_or_model_reads=True,no_unrelated_project_reads=True))
print(json.dumps(output,sort_keys=True))
'''


def main():
    with (ROOT / "NOV12_JOBS.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["job_id"] for r in rows]
    failures = {r["name"].removeprefix("rcb_"): r["job_id"] for r in rows
        if r["name"].startswith("rcb_B") and r["state"] == "FAILED"}
    assert len(ids) == len(set(ids)) == 72 and len(failures) == 17
    code = CODE.replace("IDS_VALUE", repr(ids)).replace("FAILURES_VALUE", repr(failures))

    def record_source(access):
        m.exclusive(access.audit / "EXECUTED_SOURCE.json", {**m.META, "source": code,
            "local_operator_sha256": m.sha(Path(__file__)), "prior_job_list_sha256": m.sha(ROOT / "NOV12_JOBS.csv"),
            "purpose": "User requested exact start/end times and original failure causes"})

    result, audit = transport.run_remote("failure_timeline", code, timeout=180, setup=record_source)
    print(json.dumps({"audit": str(audit), "rows": len(result["rows"]), "failures": len(result["failed_training"]),
        "timezone": result["query_timezone"], "remote_utc": result["remote_utc"], "errors": result["errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
