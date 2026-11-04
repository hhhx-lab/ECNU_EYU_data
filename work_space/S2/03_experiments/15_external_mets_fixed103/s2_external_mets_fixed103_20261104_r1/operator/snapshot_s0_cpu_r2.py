"""One bounded read-only CPU observation; no installation or job mutation."""
import json
import os
from pathlib import Path
import shlex

from session import session, access_module
from s0_cpu_r2_sources import runtime_files


CODE = r'''
import os, re, shutil, subprocess
from datetime import datetime, timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r2'
receipt_path=attempt/'operator/SUBMISSION_RECEIPT.json'
receipt=json.loads(receipt_path.read_text())
require(receipt['status']=='CPU_preparation_job_released_not_artifact_or_model_runtime_pass','CPU submission not accepted')
job_id=receipt['job_id']
require(re.fullmatch(r'[0-9]+',job_id),'CPU JobID invalid')
queries={
    'queue':['squeue','-h','-j',job_id,'-o','%i|%T|%P|%j|%r'],
    'accounting':['sacct','-P','-j',job_id,'--format=JobID,JobName,State,ExitCode,Elapsed,Start,End'],
    'job':['scontrol','show','job',job_id,'-o'],
    'quota':['quota','-s'],
}
scheduler={}
for name,argv in queries.items():
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    require(len(p.stdout.encode())+len(p.stderr.encode())<65536,'CPU observation metadata budget')
    scheduler[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
contract=json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text())
drift=[]
identities={**contract['package_files'],
    'contract/PREPARATION_CONTRACT.json':receipt['contract_sha256'],
    'slurm/prepare.slurm':receipt['slurm_sha256']}
static={}
for name,digest in identities.items():
    path=attempt/name
    observed=sha(path) if path.is_file() else None
    static[name]=dict(expected_sha256=digest,observed_sha256=observed)
    if observed!=digest:drift.append(name)
if parent_staging_guard(root)!=contract['parent_attempt_files']:drift.append('parent_CPU_staging')
if prior_guard(root)!=contract['prior_guard']:drift.append('prior_environment_or_failure')
reports={}
for path in sorted((attempt/'reports').glob('*.json')):
    require(path.stat().st_size<65536,'CPU report too large for light observation')
    reports[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),content=json.loads(path.read_text()))
logs={}
for path in sorted((attempt/'logs').glob('*')):
    if not path.is_file():continue
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-16384))
        tail=stream.read(16384).decode(errors='replace')
    logs[path.name]=dict(bytes=path.stat().st_size,tail=tail,tail_only=True)
    require(len(logs)<=12,'CPU log count budget')
artifact_identity={}
if 'COMPLETE.json' in reports:
    complete=reports['COMPLETE.json']['content']
    require(complete['job_id']==job_id,'CPU completion JobID drift')
    require(complete['status']=='S0_CPU_artifacts_complete_runtime_environment_still_requires_consumption','CPU completion status drift')
    require('FAILED.json' not in reports,'CPU COMPLETE conflicts with FAILED')
    wheels_path=attempt/'reports/WHEELS_COMPLETE.json'
    require(sha(wheels_path)==complete['wheels_manifest_sha256'],'wheel manifest SHA drift')
    manifest=reports['WHEELS_COMPLETE.json']['content']
    require(manifest['status']=='three_same_version_wheels_CPU_probe_pass','CPU wheel probe incomplete')
    require({x['name'] for x in manifest['wheels']}==set(SOURCES) and len(manifest['wheels'])==3,'CPU wheel set drift')
    for item in [complete['sif'],*manifest['wheels']]:
        path=Path(item['path'])
        require(path.resolve().is_relative_to(attempt),'CPU artifact outside attempt')
        observed=sha(path) if path.is_file() else None
        valid=path.is_file() and path.stat().st_size==item['bytes'] and observed==item['sha256']
        artifact_identity[str(path.relative_to(attempt))]=dict(expected_bytes=item['bytes'],
            observed_bytes=path.stat().st_size if path.is_file() else None,
            expected_sha256=item['sha256'],observed_sha256=observed,valid=valid)
        if not valid:drift.append(str(path.relative_to(attempt)))
result=dict(META,status='read_only_CPU_R2_observation_not_inference_pass',job_id=job_id,
    scheduler=scheduler,reports=reports,logs=logs,static_identity=static,drift=drift,
    artifact_identity=artifact_identity,submission_receipt_sha256=sha(receipt_path),
    free_bytes=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
    remote_writes=False,solver_rerun=False,jobs_created=False,models_loaded=False,images_read=False)
print(json.dumps(result))
'''


def remote_code():
    code = runtime_files()['s0_build_contract.py']+'\n'+CODE
    compile(code, 'single_read_only_CPU_R2_snapshot', 'exec')
    return code


def main():
    with session('snapshot_s0_cpu_r2') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result = json.loads(access.ssh(command, timeout=900).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json', result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result['job_id'],
                             remote_utc=result['remote_utc'],drift=result['drift'])),flush=True)


if __name__ == '__main__':
    main()
