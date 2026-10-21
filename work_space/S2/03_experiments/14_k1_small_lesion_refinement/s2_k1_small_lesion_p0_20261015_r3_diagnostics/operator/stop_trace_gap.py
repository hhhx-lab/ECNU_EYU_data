"""Stop only this new branch after discovering incomplete merge provenance."""
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
    access=t.Access('stop_trace_gap');start=datetime.now(timezone.utc).isoformat();status='failed'
    plan=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
    try:
        access.start_vpn()
        code=r'''
import datetime,json,pathlib,subprocess
r=pathlib.Path(REMOTE)
assert (r/'CPU_EXECUTION_PLAN.json').is_file()
p=json.loads((r/'CPU_EXECUTION_PLAN.json').read_text())
assert p['job_name']=='s2k1p0supp3'
payload=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
 status='blocked_incomplete_all_WT_transport_provenance',remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 reason='small-only trace source degree misses possible mergers with non-small original WT components; do not accept survival categories',
 runtime_unchanged=True,no_automatic_retry=True)
with (r/'evidence/TRACE_CONTRACT_FAILED.json').open('x') as f:json.dump(payload,f,indent=2)
launch=r/'evidence/CPU_LAUNCH.json'
if launch.exists():
 d=json.loads(launch.read_text());jid=d['canonical_job_id'];assert jid.isdigit()
 result=subprocess.run(['scancel',jid],capture_output=True,text=True,timeout=30)
 payload.update(canonical_job_id=jid,cancel_return_code=result.returncode,cancel_stderr=result.stderr,status='cancel_requested_known_trace_gap')
 with (r/'evidence/TRACE_CANCEL_REQUEST.json').open('x') as f:json.dump(payload,f,indent=2)
else:payload['status']='blocked_before_launch'
print('STOP='+json.dumps(payload))
'''.replace('REMOTE',repr(plan['remote_root']))
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+t.PYTHON+' -B -c '+shlex.quote(code),timeout=60)
        d=json.loads(next(x.split('=',1)[1] for x in raw.splitlines() if x.startswith('STOP=')))
        t.exclusive(access.audit/'STOP_RECEIPT.json',d);status=d['status']
        print(json.dumps(dict(status=status,audit=str(access.audit),canonical_job_id=d.get('canonical_job_id'))))
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**t.META,status=status,local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=0,operation='target_only_new_branch_guard_no_runtime_overwrite'))


if __name__=='__main__':main()
