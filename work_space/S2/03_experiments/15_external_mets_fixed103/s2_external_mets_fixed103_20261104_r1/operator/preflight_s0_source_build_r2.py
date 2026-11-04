"""One independently authorized compiler solver with a fresh official-only cache."""
import hashlib
import json
import os
from pathlib import Path
import shlex
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
R1_SHA = 'f8c84270f69ce83f4e4a0ae26eee36ad01d91d8b0a2b4ec6b30a2866d06e703f'
CODE = r'''
import shutil,subprocess
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
require(not (root/'attempts'/ATTEMPT).exists(),'CPU attempt exists; no retry')
require(not (root/'envs'/BUILD_ENV).exists(),'build env exists; no retry')
require(not (root/'envs/bio_intel_py310_r4').exists(),'runtime R4 exists')
old_audit=root/'operator/s0_source_build_preflight_r1'
require(json.loads((old_audit/'FAILED.json').read_text())['message']=='build package origin drift','R1 failure contract drift')
before=prior_guard(root)
old_guard={p.name:sha(p) for p in old_audit.iterdir() if p.is_file()}
audit=root/PREFLIGHT_R2
require(not audit.exists(),'R2 already executed; no retry')
audit.mkdir(mode=0o700)
stage='real_host_and_assets'
try:
    archive=root/'assets/container_conversion/fixed2025.docker-archive.tar'
    require(archive.stat().st_size==8409948160 and sha(archive)==ARCHIVE_SHA,'accepted archive drift')
    require(json.loads((root/'operator/ASSET_TRANSFER_COMPLETE.json').read_text())['status']=='all_server_asset_bytes_SHA_verified','transfer incomplete')
    libc=subprocess.check_output(['getconf','GNU_LIBC_VERSION'],text=True,timeout=20).strip()
    require(libc=='glibc 2.17','real host glibc drift')
    free=shutil.disk_usage(root).free
    require(free>150*1024**3,'fresh CPU preparation capacity below 150GiB')
    cache=audit/'pkgs'
    cache.mkdir(mode=0o700)
    require(not list(cache.iterdir()),'R2 cache must start empty')
    config=audit/'condarc.yaml'
    with config.open('x') as stream:stream.write(official_config())
    config.chmod(0o440)
    env=isolated_conda_env(audit,os.environ)
    conda=os.environ['EXTERNAL_MET_CONDA_BIN']
    stage='effective_official_config'
    observed=json.loads(subprocess.check_output([conda,'config','--show','channel_alias','custom_channels',
        'pkgs_dirs','ssl_verify','add_anaconda_token','--json'],env=env,text=True,timeout=60))
    effective=validate_effective_config(observed,audit)
    info=json.loads(subprocess.check_output([conda,'info','--json'],env=env,text=True,timeout=60))
    require([x for x in info['virtual_pkgs'] if x[0]=='__glibc']==[['__glibc','2.17','0']],'solver virtual glibc drift')
    specs=['python=3.10.22','numpy=2.2.6','gcc_linux-64=13','gxx_linux-64=13','binutils_linux-64',
           'sysroot_linux-64=2.17','setuptools>=77','wheel','cython','pip','tomli','packaging']
    write(audit/'STARTED.json',dict(status='R2_source_solver_started',host_libc=libc,
        virtual_packages=info['virtual_pkgs'],effective_config=effective,config_sha256=sha(config),
        cache_was_empty=True,requested_specs=specs,before_guard=before,prior_R1_audit_sha256=old_guard,
        free_bytes=free,remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='compatible_userland_compiler_solver'
    command=[conda,'create','--prefix',str(root/'envs'/BUILD_ENV),'--dry-run','--json','--override-channels',
             '-c',CHANNEL,'--repodata-fn','repodata.json',*specs]
    log=audit/'SOLVER.json'
    with log.open('x') as stream:p=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    require(log.stat().st_size<8*1024**2,'solver metadata budget')
    plan=json.loads(log.read_text())
    require(p.returncode==0,'build solver failed: '+str(plan.get('exception_name','unknown')))
    selected=validate_build_plan(plan)
    records=explicit_lock_records(plan)
    lock=audit/'BUILD_ENV_EXPLICIT.txt'
    with lock.open('x') as stream:stream.write(render_explicit_lock(records))
    lock.chmod(0o440)
    require(prior_guard(root)==before,'old environment/failure drift')
    require({p.name:sha(p) for p in old_audit.iterdir() if p.is_file()}==old_guard,'R1 audit drift')
    require(not (root/'envs'/BUILD_ENV).exists(),'dry-run created environment')
    stage='fresh_slurm_metadata'
    commands={
        'partition':['scontrol','show','partition','cpu_192G','-o'],
        'nodes':['sinfo','-N','-h','-p','cpu_192G','-o','%N|%t|%c|%m|%G'],
        'association':['sacctmgr','-n','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
        'qos':['sacctmgr','-n','-P','show','qos','format=Name,MaxWall,MaxTRESPJ,MaxJobsPU,MaxSubmitPU'],
        'own_queue':['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'],
        'quota':['quota','-s']}
    scheduler={}
    for name,command in commands.items():
        p=subprocess.run(command,capture_output=True,text=True,timeout=30)
        require(len(p.stdout)+len(p.stderr)<131072,'scheduler metadata budget')
        scheduler[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if name!='quota':require(p.returncode==0,'scheduler metadata unavailable: '+name)
    outcome=dict(status='s0_source_build_preflight_pass_no_install_no_job',selected_packages=selected,
        explicit_packages=records,solver_sha256=sha(log),explicit_lock_sha256=sha(lock),specs=specs,
        channel=CHANNEL,effective_config=effective,config_sha256=sha(config),host_libc=libc,
        free_bytes=shutil.disk_usage(root).free,archive_sha256=ARCHIVE_SHA,scheduler=scheduler,
        prior_environments_preserved=True,prior_R1_audit_sha256=old_guard,environment_created=False,
        jobs_created=False,models_loaded=False,remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'COMPLETE.json',outcome)
except BaseException as error:
    outcome=dict(status='s0_source_build_preflight_failed',stage=stage,error_type=type(error).__name__,message=str(error),
        no_automatic_retry=True,environment_created=(root/'envs'/BUILD_ENV).exists(),jobs_created=False,models_loaded=False,
        prior_environments_preserved=prior_guard(root)==before,remote_utc=datetime.now(timezone.utc).isoformat(),
        free_bytes=shutil.disk_usage(root).free)
    log=audit/'SOLVER.json'
    if log.is_file():outcome.update(solver_sha256=sha(log),solver_tail=log.read_text()[-5000:])
    write(audit/'FAILED.json',outcome)
print(json.dumps(dict(META,**outcome)))
'''


def remote_code():
    assert hashlib.sha256((ROOT/'operator/preflight_s0_source_build_r1.py').read_bytes()).hexdigest()==R1_SHA
    source = (ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    source += (ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'+CODE
    compile(source,'s0_source_build_preflight_r2','exec')
    return source


def main():
    authorization=json.loads((ROOT/'S0_SOURCE_BUILD_R2_AUTHORIZATION.json').read_text())
    assert authorization['isolated_empty_batch_conda_cache'] and authorization['partition_read_correction']=='cpu_192G'
    with session('preflight_s0_source_build_r2') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=1500).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='s0_source_build_preflight_pass_no_install_no_job':
            raise RuntimeError('R2 compiler preflight failed; preserve evidence; no install/job/retry')


if __name__=='__main__':main()
