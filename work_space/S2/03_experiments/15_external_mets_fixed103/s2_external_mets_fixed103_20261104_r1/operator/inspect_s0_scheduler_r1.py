"""Collect one independent headed scheduler snapshot without rerunning Conda."""
import json
import os
from pathlib import Path
import shlex
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
SOLVER_SHA = '68099499cdaee295940740b6144d1f9c0b73e9822338f310a05b8a44386d481c'
LOCK_SHA = '866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77'
FAILED_SHA = '7ab5ed495a16d03e697425e7c217eddbad2461f9b39c5716f92b925bad403894'
AUDIT = 'operator/s0_scheduler_acceptance_r1'
CODE = r'''
import shutil, subprocess
from datetime import datetime, timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
original=root/'operator/s0_source_build_preflight_r3'
require(sha(original/'SOLVER.json')==SOLVER_SHA, 'R3 solver SHA drift')
require(sha(original/'BUILD_ENV_EXPLICIT.txt')==LOCK_SHA, 'R3 explicit lock SHA drift')
require(sha(original/'FAILED.json')==FAILED_SHA, 'R3 failure SHA drift')
failure=json.loads((original/'FAILED.json').read_text())
require(failure['stage']=='fresh_slurm_metadata', 'unexpected R3 failure stage')
require(not (original/'COMPLETE.json').exists(), 'R3 unexpectedly modified to COMPLETE')
started=json.loads((original/'STARTED.json').read_text())
before=prior_guard(root)
require(before==started['before_guard'], 'old environment or failure drift')
require(not (root/'envs'/BUILD_ENV).exists() and not (root/'attempts'/ATTEMPT).exists(), 'staging already exists')
require(not (root/'envs/bio_intel_py310_r4').exists(), 'runtime R4 already exists')
plan=json.loads((original/'SOLVER.json').read_text())
selected=validate_build_plan(plan)
records=explicit_lock_records(plan)
require(len(selected)==len(records)==47, 'R3 package count drift')
require((original/'BUILD_ENV_EXPLICIT.txt').read_text()==render_explicit_lock(records), 'R3 lock content drift')
audit=root/AUDIT
require(not audit.exists(), 'independent scheduler snapshot already exists; no retry')
audit.mkdir(mode=0o700)
original_guard={p.name:sha(p) for p in original.iterdir() if p.is_file()}
write(audit/'STARTED.json', dict(status='independent_headed_scheduler_snapshot_started',
    remote_utc=datetime.now(timezone.utc).isoformat(), solver_sha256=SOLVER_SHA,
    explicit_lock_sha256=LOCK_SHA, failed_marker_sha256=FAILED_SHA))
stage='headed_scheduler_snapshot'
try:
    commands={
        'qos':['sacctmgr','-P','show','qos'],
        'partition':['scontrol','show','partition','cpu_192G','-o'],
        'nodes':['sinfo','-N','-h','-p','cpu_192G','-o','%N|%t|%c|%m|%G'],
        'association':['sacctmgr','-P','show','assoc','where','user=zqchen',
                       'format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
        'own_queue':['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'],
        'quota':['quota','-s'],
        'filesystem':['df','-B1',str(root)],
        'libc':['getconf','GNU_LIBC_VERSION']}
    scheduler={}
    for name,command in commands.items():
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=30)
            require(len(p.stdout)+len(p.stderr)<131072, 'scheduler output exceeds metadata budget')
            scheduler[name]=dict(argv=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except OSError as error:
            scheduler[name]=dict(argv=command,returncode=127,error_type=type(error).__name__)
    snapshot=dict(status='scheduler_snapshot_collected_acceptance_not_yet_run', scheduler=scheduler,
        solver_sha256=SOLVER_SHA,explicit_lock_sha256=LOCK_SHA,failed_marker_sha256=FAILED_SHA,
        source_preflight=str(original),source_preflight_files=original_guard,
        selected_packages=selected,explicit_packages=records,prior_guard=before,
        free_bytes=shutil.disk_usage(root).free,solver_rerun=False,jobs_created=False,
        build_environment_created=False,models_loaded=False,images_read=False,
        remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'SNAPSHOT.json',snapshot)
    for name,item in scheduler.items():
        if name!='quota':require(item['returncode']==0, 'scheduler command failed: '+name)
    require(scheduler['libc']['stdout'].strip()=='glibc 2.17', 'real host libc drift')
    require(prior_guard(root)==before, 'old environment or failure drift during snapshot')
    require({p.name:sha(p) for p in original.iterdir() if p.is_file()}==original_guard, 'original R3 audit drift')
    print(json.dumps(dict(META,**snapshot,snapshot_sha256=sha(audit/'SNAPSHOT.json'))))
except BaseException as error:
    outcome=dict(status='independent_scheduler_snapshot_failed',stage=stage,
        error_type=type(error).__name__,message=str(error),no_automatic_retry=True,
        remote_utc=datetime.now(timezone.utc).isoformat(),solver_rerun=False,jobs_created=False)
    write(audit/'FAILED.json',outcome)
    print(json.dumps(dict(META,**outcome)))
'''


def remote_code():
    constants = '\n'.join(f'{key}={value!r}' for key,value in
                          dict(SOLVER_SHA=SOLVER_SHA,LOCK_SHA=LOCK_SHA,FAILED_SHA=FAILED_SHA,AUDIT=AUDIT).items())
    code = (ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    code += (ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'+constants+'\n'+CODE
    compile(code, 'independent_S0_scheduler_snapshot', 'exec')
    return code


def main():
    authorization=json.loads((ROOT/'S0_SCHEDULER_R1_AUTHORIZATION.json').read_text())
    assert authorization['independent_scheduler_acceptance_authorized']
    assert not authorization['solver_rerun_authorized'] and not authorization['automatic_retry']
    with session('inspect_s0_scheduler_r1') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=400).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='scheduler_snapshot_collected_acceptance_not_yet_run':
            raise RuntimeError('Independent scheduler snapshot failed; no solver/install/submission retry')


if __name__=='__main__':
    main()
