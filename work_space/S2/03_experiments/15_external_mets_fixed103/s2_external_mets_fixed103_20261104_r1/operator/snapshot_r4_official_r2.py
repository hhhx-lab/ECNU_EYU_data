"""One read-only stage snapshot; borrow the verified manual VPN without logout."""
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone

from session import access_module

CODE = r'''
import hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/'operator/environment_r4_official_preflight_r2'
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='read_only_stage_snapshot_not_solver_acceptance',remote_utc=datetime.now(timezone.utc).isoformat(),
    audit_exists=audit.is_dir(),free_bytes=shutil.disk_usage(root).free,markers={},remote_writes=False,
    models_loaded=False,images_read=False,jobs_created=False)
for name in ['STARTED.json','FAILED.json','COMPLETE.json']:
    path=audit/name
    item=dict(exists=path.is_file())
    if path.is_file():
        size=path.stat().st_size
        assert size<2*1024**2, 'stage marker exceeds lightweight budget'
        raw=path.read_bytes()
        data=json.loads(raw)
        item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),fields={
            key:data.get(key) for key in ['status','stage','host_libc','channel','error_type',
                'remote_utc','environment_created','prior_environments_preserved'] if key in data})
    out['markers'][name]=item
log=audit/'SOLVER.json'
out['solver_log']=dict(exists=log.is_file())
if log.is_file():
    size=log.stat().st_size
    out['solver_log']['bytes_observed']=size
    if 0<size<2*1024**2:
        try:
            data=json.loads(log.read_text())
            out['solver_log']['parsed_fields']={key:data.get(key) for key in ['success','exception_name','packages'] if key in data}
        except json.JSONDecodeError:out['solver_log']['json_complete_at_observation']=False
out['remote_completed_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
'''


def remote_code():
    compile(CODE, 'r4_read_only_snapshot', 'exec')
    return CODE


def main():
    access = access_module.Access('snapshot_r4_official_r2')
    checked = subprocess.run([os.environ['ECNU_VPN_SCRIPT'], 'check'], env=access.env,
                             capture_output=True, timeout=40)
    assert checked.returncode == 0, 'existing manual VPN no longer verified; no SSH or automatic login'
    command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
    command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(remote_code())
    result = json.loads(access.ssh(command, timeout=90).strip().splitlines()[-1])
    result['local_received_utc'] = datetime.now(timezone.utc).isoformat()
    access_module.exclusive(access.audit/'RESULT.json', result)
    access_module.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json', dict(access_module.META,
        completed_utc=datetime.now(timezone.utc).isoformat(),borrowed_existing_manual_vpn=True,
        vpn_start_called=False,vpn_stop_called=False,remote_writes=False))
    print(json.dumps(dict(audit=str(access.audit),result=result)), flush=True)


if __name__ == '__main__':main()
