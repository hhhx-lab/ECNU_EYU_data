"""Observe only the canonical CPU supplement job, once, without resubmission."""
import importlib.util
import json
import shlex
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT.parents[2]/'03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'


def main():
    spec=importlib.util.spec_from_file_location('transport',HELPER)
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.ROOT=ROOT
    launches=list((ROOT/'operator').glob('deploy_cpu_once_*/LAUNCH_RECEIPT.json'))
    assert len(launches)==1,'unique launch receipt required'
    receipt=json.loads(launches[0].read_text())
    assert receipt['status']=='released_not_progress_polled','no released canonical job'
    jid=receipt['canonical_job_id'];assert jid.isdigit()
    plan=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
    remote=plan['remote_root'];digest=t.sha(ROOT/'CPU_EXECUTION_PLAN.json')
    access=t.Access('snapshot_once');start=datetime.now(timezone.utc).isoformat();status='failed'
    try:
        access.start_vpn()
        code=r'''
import csv,datetime,hashlib,json,os,pathlib,re,subprocess
r=pathlib.Path(REMOTE);jid=JID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(args):
 q=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(rc=q.returncode,stdout=q.stdout,stderr=q.stderr)
p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text())
assert sha(r/'CPU_EXECUTION_PLAN.json')==DIGEST
drift=[n for n,d in p['static_sha256'].items() if sha(r/n)!=d]
logs={str(f.relative_to(r)):dict(sha256=sha(f),tail=f.read_text(errors='replace')[-16000:]) for f in (r/'logs').glob('*'+jid+'*') if f.is_file()}
evidence={str(f.relative_to(r)):dict(sha256=sha(f),content=f.read_text()) for f in (r/'evidence').glob('*') if f.is_file()}
errors=[]
for n,v in logs.items():
 if re.search(r'Traceback|OutOfMemory|out of memory|CUDA error|ImportError|ModuleNotFoundError|DUE TO TIME LIMIT|\bnonfinite\b',v['tail'],re.I):errors.append(n)
counts={str(f.relative_to(r)):sum(1 for _ in csv.DictReader(f.open())) for f in (r/'results').rglob('*.csv')}
out=dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
 disk=cmd(['df','-B1',str(r)]),squeue=cmd(['squeue','-h','-j',jid,'-o','%i|%j|%T|%R']),
 sacct=cmd(['sacct','-n','-X','-j',jid,'--format=JobID,JobName,State,ExitCode,Elapsed','-P']),
 logs=logs,evidence=evidence,csv_rows=counts,static_sha_drift=drift,critical_errors=errors,canonical_job_id=jid)
print('SNAPSHOT='+json.dumps(out))
'''.replace('REMOTE',repr(remote)).replace('JID',repr(jid)).replace('DIGEST',repr(digest))
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(code),timeout=150)
        d=json.loads(next(x.split('=',1)[1] for x in raw.splitlines() if x.startswith('SNAPSHOT=')))
        t.exclusive(access.audit/'SNAPSHOT.json',dict(**t.META,**d))
        q=d['squeue'];a=d['sacct'];assert a['rc']==0
        row=next(x.split('|') for x in a['stdout'].splitlines() if x.startswith(jid+'|'+plan['job_name']+'|'))
        assert not d['critical_errors'] and not d['static_sha_drift'] and not any('FAILED' in n for n in d['evidence']),'new failure; stop branch'
        if row[2]=='COMPLETED':
            assert row[3]=='0:0' and 'evidence/SUPPLEMENT_COMPLETE.ok' in d['evidence'],'completion evidence required'
            assert q['rc']==0 or 'Invalid job id' in q['stderr'],'completed queue read unexpected error'
            status='completed_requires_independent_pull_acceptance'
        else:
            assert row[2] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'),'new terminal failure or vanished job'
            assert q['rc']==0 and q['stdout'].strip(),'active job missing from queue'
            status='active_no_new_failure'
        print(json.dumps(dict(status=status,canonical_job_id=jid,sacct=row,csv_rows=d['csv_rows'],audit=str(access.audit))))
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**t.META,status=status,local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=1,remote_writes=False))


if __name__=='__main__':main()
