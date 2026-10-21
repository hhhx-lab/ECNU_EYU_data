"""Read only immutable submission/preflight evidence, without querying job progress."""
import base64
import json
import shlex
from datetime import datetime,timezone
from common import ROOT,REMOTE,META,transport


def main():
    t=transport();access=t.Access('pull_submission_evidence_once')
    start=datetime.now(timezone.utc).isoformat();status='failed'
    names=['DEPLOY_RESERVATION.json','SUPPLEMENT_STATIC_PREFLIGHT.json','CPU_NODE_PREFLIGHT.json',
        'SCHEDULER_STATIC_PREFLIGHT.json','CPU_RESERVATION.json','CPU_ENGINEERING_AUTHORIZATION.json',
        'CPU_LAUNCH.json','CPU_HELD_AUDIT.json','CPU_RELEASED.json','OFFICIAL_BOUNDARY_TESTS.json']
    try:
        access.start_vpn()
        code='import base64,hashlib,json,pathlib,datetime; r=pathlib.Path('+repr(REMOTE)+'); names='+repr(names)+'; files={n:dict(content_base64=base64.b64encode((r/"evidence"/n).read_bytes()).decode(),sha256=hashlib.sha256((r/"evidence"/n).read_bytes()).hexdigest()) for n in names}; print("PULL="+json.dumps(dict(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files)))'
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(code),timeout=60)
        data=json.loads(next(x.split('=',1)[1] for x in raw.splitlines() if x.startswith('PULL=')))
        for name,record in data['files'].items():
            path=access.audit/'remote_exact/evidence'/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(base64.b64decode(record['content_base64']))
            assert t.sha(path)==record['sha256']
            json.loads(path.read_text())
        evidence=access.audit/'remote_exact/evidence'
        held=json.loads((evidence/'CPU_HELD_AUDIT.json').read_text())
        assert held['status']=='pass' and all(held['checks'].values())
        released=json.loads((evidence/'CPU_RELEASED.json').read_text())
        assert released['canonical_job_id']==held['canonical_job_id']=='4100381'
        status='pass_immutable_submission_evidence_only'
        t.exclusive(access.audit/'PULL_VALIDATION.json',dict(**META,status=status,remote_utc=data['remote_utc'],
            canonical_job_id='4100381',files={n:r['sha256'] for n,r in data['files'].items()},file_count=len(names),
            probability_files_pulled=0,checkpoint_files_pulled=0,job_progress_queries=0))
        print(json.dumps(dict(status=status,audit=str(access.audit),files=len(names),remote_utc=data['remote_utc'])))
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**META,status=status,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=0,remote_writes=False))


if __name__=='__main__':main()
