"""Accept the independently collected scheduler snapshot, leaving R3 FAILED intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from session import session, access_module
from inspect_s0_scheduler_r1 import ROOT, AUDIT, SOLVER_SHA, LOCK_SHA, FAILED_SHA
from s0_qos_table import validate_cpu_snapshot


def remote_code(snapshot_sha, validation):
    code=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    code+=(ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'
    code+='\n'.join(f'{k}={v!r}' for k,v in dict(AUDIT=AUDIT,SNAPSHOT_SHA=snapshot_sha,
                   SOLVER_SHA=SOLVER_SHA,LOCK_SHA=LOCK_SHA,FAILED_SHA=FAILED_SHA,VALIDATION=validation).items())
    code+=r'''
import os,shutil
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/AUDIT
require(not (audit/'COMPLETE.json').exists() and not (audit/'FAILED.json').exists(), 'scheduler acceptance already attempted')
stage='independent_scheduler_acceptance'
try:
    require(sha(audit/'SNAPSHOT.json')==SNAPSHOT_SHA, 'scheduler snapshot SHA drift')
    snapshot=json.loads((audit/'SNAPSHOT.json').read_text())
    original=Path(snapshot['source_preflight'])
    require(original==root/'operator/s0_source_build_preflight_r3', 'R3 source path drift')
    require({p.name:sha(p) for p in original.iterdir() if p.is_file()}==snapshot['source_preflight_files'], 'R3 audit changed')
    require(sha(original/'SOLVER.json')==SOLVER_SHA and sha(original/'BUILD_ENV_EXPLICIT.txt')==LOCK_SHA,
            'R3 plan or lock SHA drift')
    require(sha(original/'FAILED.json')==FAILED_SHA and not (original/'COMPLETE.json').exists(), 'R3 failure not preserved')
    plan=json.loads((original/'SOLVER.json').read_text())
    require(validate_build_plan(plan)==snapshot['selected_packages'], 'selected R3 packages drift')
    require(render_explicit_lock(explicit_lock_records(plan))==(original/'BUILD_ENV_EXPLICIT.txt').read_text(), 'lock content drift')
    require(prior_guard(root)==snapshot['prior_guard'], 'old environments or failures drift')
    require(not (root/'envs'/BUILD_ENV).exists() and not (root/'attempts'/ATTEMPT).exists(), 'staging already exists')
    require(shutil.disk_usage(root).free>150*1024**3, 'CPU preparation capacity below reservation')
    require(VALIDATION['status']=='CPU_scheduler_request_eligible_not_running_or_full_capacity_pass', 'CPU scheduler validation failed')
    outcome=dict(status='s0_scheduler_acceptance_pass_same_R3_plan_no_install_no_job',
        source_preflight=str(original),source_preflight_files=snapshot['source_preflight_files'],
        solver_sha256=SOLVER_SHA,explicit_lock_sha256=LOCK_SHA,failed_marker_sha256=FAILED_SHA,
        scheduler_snapshot_sha256=SNAPSHOT_SHA,validation=VALIDATION,prior_guard=snapshot['prior_guard'],
        free_bytes=shutil.disk_usage(root).free,solver_rerun=False,jobs_created=False,environment_created=False,
        models_loaded=False,images_read=False,remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'COMPLETE.json',outcome)
except BaseException as error:
    outcome=dict(status='independent_scheduler_acceptance_failed',stage=stage,message=str(error),
        error_type=type(error).__name__,no_automatic_retry=True,solver_rerun=False,jobs_created=False,
        remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'FAILED.json',outcome)
print(json.dumps(dict(META,**outcome)))
'''
    compile(code,'accept_independent_S0_scheduler','exec')
    return code


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    assert json.loads((ROOT/'S0_SCHEDULER_R1_AUTHORIZATION.json').read_text())['independent_scheduler_acceptance_authorized']
    snapshot=json.loads(args.snapshot.read_text())
    assert snapshot['solver_sha256']==SOLVER_SHA and snapshot['explicit_lock_sha256']==LOCK_SHA
    assert snapshot['failed_marker_sha256']==FAILED_SHA
    validation=validate_cpu_snapshot(snapshot)
    with session('accept_s0_scheduler_r1') as access:
        access_module.exclusive(access.audit/'LOCAL_SCHEDULER_VALIDATION.json',validation)
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code(snapshot['snapshot_sha256'],validation))
        result=json.loads(access.ssh(command,timeout=240).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='s0_scheduler_acceptance_pass_same_R3_plan_no_install_no_job':
            raise RuntimeError('Independent scheduler acceptance failed; preserve audit without retry')


if __name__=='__main__':
    main()
