"""Freeze, submit held, audit and release the single authorized CPU preparation."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import os,subprocess,shutil
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r1'
audit=attempt/'operator'
require(not (audit/'RESERVATION.json').exists(), 'CPU submission already reserved; no retry')
stage='CPU_package_and_fresh_scheduler_preflight'
job_id=None
try:
    contract_path=attempt/'contract/PREPARATION_CONTRACT.json'
    contract=json.loads(contract_path.read_text())
    staging=json.loads((attempt/'reports/STAGING_COMPLETE.json').read_text())
    require(staging['status']=='S0_offline_CPU_input_staging_pass_not_model_runtime_pass', 'CPU staging incomplete')
    require(sha(contract_path)==staging['contract_sha256'], 'staged CPU contract SHA drift')
    require(contract['attempt']==str(attempt) and contract['remote_root']==str(root), 'CPU attempt identity drift')
    require(contract['resource']==dict(partition='cpu_192G',cpus_per_task=8,memory_gib=64,time_limit='04:00:00',gpu=0), 'CPU resource drift')
    require(contract['single_conversion_deadline_seconds']==7200 and not contract['automatic_retry'], 'CPU bounded conversion contract drift')
    require(prior_guard(root)==contract['prior_guard'], 'old environment/failure changed')
    for name,digest in contract['package_files'].items():
        require(sha(attempt/name)==digest, 'CPU runtime package SHA drift: '+name)
        compile((attempt/name).read_text(),name,'exec')
    for name,digest in contract['build_env_files'].items():
        require(sha(Path(contract['build_env'])/name)==digest, 'CPU build environment identity drift: '+name)
    for name,item in SOURCES.items():
        path=attempt/'sources'/item['filename']
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'], 'CPU source SHA drift: '+name)
    require(sha(Path(contract['compiler_preflight'])/'COMPLETE.json')==contract['compiler_preflight_sha256'], 'independent acceptance drift')
    require(sha(Path(contract['compiler_solver_preflight'])/'FAILED.json')==contract['original_R3_failed_sha256'], 'R3 failure changed')
    require(sha(contract['archive'])==contract['archive_sha256']==ARCHIVE_SHA, 'accepted archive SHA drift')
    script=render_slurm(attempt,contract['build_env'],os.environ['EXTERNAL_MET_CONDA_BIN'],sha(contract_path))
    path=attempt/'slurm/prepare.slurm'
    with path.open('x') as stream:stream.write(script)
    path.chmod(0o440)
    subprocess.run(['bash','-n',str(path)],check=True,timeout=30)
    commands={
        'qos':['sacctmgr','-P','show','qos'],
        'partition':['scontrol','show','partition','cpu_192G','-o'],
        'nodes':['sinfo','-N','-h','-p','cpu_192G','-o','%N|%t|%c|%m|%G'],
        'association':['sacctmgr','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
        'own_queue':['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'],
        'quota':['quota','-s'],'libc':['getconf','GNU_LIBC_VERSION']}
    scheduler={}
    for name,argv in commands.items():
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        require(len(p.stdout)+len(p.stderr)<131072, 'CPU scheduler metadata budget exceeded')
        scheduler[name]=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    validation=validate_cpu_snapshot(dict(scheduler=scheduler,free_bytes=shutil.disk_usage(root).free))
    require(not [x for x in scheduler['own_queue']['stdout'].splitlines() if '|s2exts0p1|' in x], 'same CPU job already queued')
    write(audit/'STATIC_PREFLIGHT.json',dict(status='CPU_static_and_fresh_scheduler_pass',validation=validation,
        contract_sha256=sha(contract_path),slurm_sha256=sha(path),scheduler=scheduler,
        remote_utc=datetime.now(timezone.utc).isoformat()))
    write(audit/'RESERVATION.json',dict(status='CPU_single_submission_reserved',slurm_sha256=sha(path),
        contract_sha256=sha(contract_path),remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='sbatch_held_once'
    p=subprocess.run(['sbatch','--hold','--parsable','--chdir='+str(attempt),str(path)],
                     capture_output=True,text=True,timeout=60)
    write(audit/'SBATCH_RESULT.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        remote_utc=datetime.now(timezone.utc).isoformat()))
    require(p.returncode==0, 'CPU held submission returned nonzero; inspect receipt without retry')
    match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',p.stdout.strip())
    require(match is not None, 'CPU submission identity unclear; inspect queue without retry')
    job_id=match.group(1)
    write(audit/'HELD_CREATED.json',dict(status='CPU_held_created',job_id=job_id,slurm_sha256=sha(path),
        contract_sha256=sha(contract_path),remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='independent_held_identity_audit'
    raw=subprocess.check_output(['scontrol','show','job',job_id,'-o'],text=True,timeout=30)
    fields=validate_held(raw,job_id,attempt,[validation['account']])
    require(sha(path)==json.loads((audit/'HELD_CREATED.json').read_text())['slurm_sha256'], 'CPU script changed while held')
    write(audit/'HELD_AUDIT.json',dict(status='CPU_held_identity_pass',job_id=job_id,raw_scontrol=raw,
        fields=fields,remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='release_original_held_job'
    p=subprocess.run(['scontrol','release',job_id],capture_output=True,text=True,timeout=30)
    write(audit/'RELEASE_RESULT.json',dict(job_id=job_id,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        remote_utc=datetime.now(timezone.utc).isoformat()))
    require(p.returncode==0, 'CPU held release failed; no release retry')
    state=subprocess.check_output(['scontrol','show','job',job_id,'-o'],text=True,timeout=30)
    outcome=dict(status='CPU_preparation_job_released_not_artifact_or_model_runtime_pass',job_id=job_id,
        observed_job=job_fields(state),slurm_sha256=sha(path),contract_sha256=sha(contract_path),
        remote_utc=datetime.now(timezone.utc).isoformat(),original_R3_failed_preserved=True,
        models_loaded=False,images_read=False,original_inference_jobs_created=False)
    write(audit/'SUBMISSION_RECEIPT.json',outcome)
except BaseException as error:
    outcome=dict(status='CPU_submission_failed_or_held_for_review',stage=stage,job_id=job_id,
        error_type=type(error).__name__,message=str(error),remote_utc=datetime.now(timezone.utc).isoformat(),
        no_automatic_retry=True,original_inference_jobs_created=False)
    write(audit/'SUBMISSION_FAILED.json',outcome)
print(json.dumps(dict(META,**outcome)))
'''


def remote_code():
    code=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    # The helpers are concatenated for the existing remote audit interpreter.
    helpers=(ROOT/'operator/s0_cpu_submission_contract.py').read_text()+'\n'
    qos=(ROOT/'operator/s0_qos_table.py').read_text()
    qos=qos.replace('from s0_cpu_submission_contract import memory_mib\n','')
    qos=qos.replace('    from s0_cpu_submission_contract import job_fields\n','')
    code+=helpers+'\n'+qos+'\n'+CODE
    compile(code,'submit_single_S0_CPU_preparation','exec')
    return code


def main():
    assert json.loads((ROOT/'S0_BUILD_CONVERSION_AUTHORIZATION.json').read_text())['new_cpu_preparation_job_authorized']
    assert json.loads((ROOT/'S0_SCHEDULER_R1_AUTHORIZATION.json').read_text())['independent_scheduler_acceptance_authorized']
    with session('submit_s0_cpu_r1') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=420).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result.get('job_id'),
                             remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('CPU submission needs review; preserve any created held job; no retry')


if __name__=='__main__':
    main()
