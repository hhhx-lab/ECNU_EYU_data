"""Stop only this new diagnostic job after detecting an uncovered empty-ref path."""
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"


def main():
    spec=importlib.util.spec_from_file_location("transport",HELPER)
    transport=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT=ROOT
    access=transport.Access("stop_known_unsafe_cpu")
    outcome="failed"
    try:
        access.start_vpn()
        code=r'''
import datetime,json,os,pathlib,subprocess
root=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_k1_small_lesion_p0_20261013_r1')
launch=json.loads((root/'evidence/CPU_LAUNCH.json').read_text())
assert launch['job_id']=='4051333' and launch['job_name']=='s2k1p0cpu1'
# Verify ownership/name with a bounded safety check before cancelling our job.
r=subprocess.run(['scontrol','show','job','4051333','-o'],capture_output=True,text=True,timeout=20)
if r.returncode==0:
 assert 'JobName=s2k1p0cpu1' in r.stdout and 'UserId=zqchen(' in r.stdout
 cancelled=subprocess.run(['scancel','4051333'],capture_output=True,text=True,timeout=20)
 assert cancelled.returncode==0,cancelled.stderr
 status='cancel_requested_known_runtime_edge_bug'
else:
 a=subprocess.run(['sacct','-n','-X','-j','4051333','--format=JobID,JobName,State,ExitCode','-P'],capture_output=True,text=True,timeout=20)
 assert a.returncode==0 and '4051333|s2k1p0cpu1|' in a.stdout
 status='already_terminal_no_cancel'
payload=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
 status=status,job_id='4051333',job_name='s2k1p0cpu1',remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 reason='Our adapter calls official map_instance_labels on empty reference; missing empty-region preflight. Runtime frozen, no in-place fix/retry.',
 failed_marker=(root/'evidence/P0_CPU_FAILED.json').is_file(),complete_marker=(root/'evidence/P0_CPU_COMPLETE.ok').is_file())
fd=os.open(root/'evidence/OPERATOR_STOP_KNOWN_EMPTY_REFERENCE_RISK.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
with os.fdopen(fd,'w') as f:json.dump(payload,f,indent=2)
print('CPU_STOP='+json.dumps(payload))
'''
        raw=access.ssh("env PYTHONDONTWRITEBYTECODE=1 "+transport.PYTHON+" -B -c "+shlex.quote(code),timeout=75)
        result=json.loads(next(r.split("=",1)[1] for r in raw.splitlines() if r.startswith("CPU_STOP=")))
        transport.exclusive(access.audit/"STOP_RECEIPT.json",result)
        outcome=result["status"]
        print(json.dumps(result))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit/"LOCAL_OPERATION_RECEIPT.json",dict(**transport.META,status=outcome,
            local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn,job_id="4051333"))


if __name__ == "__main__":
    main()
