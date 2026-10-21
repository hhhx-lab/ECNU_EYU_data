"""One bounded lightweight snapshot; never scan native images/checkpoints."""
import hashlib
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv):
    v=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    return dict(returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)


def main():
    package=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text());jobs={};receipts={}
    for stage in ('smoke','full'):
        p=ROOT/'evidence'/stage/'SUBMISSION_RECEIPT.json'
        if p.exists():
            v=json.loads(p.read_text());receipts[stage]=v; jobs.update(v['jobs'])
    result=dict(status='snapshot',remote_utc=datetime.now(timezone.utc).isoformat(),package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
        package_drift=[n for n,h in package['files'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h],
        jobs=jobs,receipts=receipts,disk=command(['df','-B1',str(ROOT)]),markers={},logs={})
    if jobs:
        result['squeue']=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%M'])
        result['sacct']=command(['sacct','-n','-P','-j',','.join(jobs.values()),'--format=JobID,State,ExitCode,Elapsed,Start,End'])
    directories=['reference','selection',*['smoke/'+m for m in ('t1c','t1n','t2w','t2f')],*['evaluation/'+m for m in ('t1c','t1n','t2w','t2f')]]
    for name in directories:
        result['markers'][name]={}
        for marker in ('STARTED.json','FAILED.json','COMPLETE.ok','REPORT.json','SELECTION.json'):
            p=ROOT/name/marker
            if p.exists():
                v=json.loads(p.read_text())
                result['markers'][name][marker]=dict(sha256=sha(p),bytes=p.stat().st_size,
                    summary={k:v[k] for k in ('status','utc','cases','positive_cases','negative_cases','errors','elapsed_seconds','sampled_cubes','estimated_full_seconds','message','modalities') if k in v and k!='modalities'})
        p=ROOT/name/'progress.jsonl'
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4096));tail=f.read().decode(errors='replace').splitlines()
            result['markers'][name]['progress_tail']=tail[-3:]
    for phase,job in jobs.items():
        for suffix in ('out','err'):
            p=ROOT/'logs'/(phase+'_'+job+'.'+suffix)
            if p.exists():
                with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));tail=f.read().decode(errors='replace')
                result['logs'][p.name]=dict(bytes=p.stat().st_size,tail=tail)
    result['stage_failures']={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in (ROOT/'evidence').glob('*/FAILED.json')}
    print(json.dumps(result))


if __name__=='__main__':main()
