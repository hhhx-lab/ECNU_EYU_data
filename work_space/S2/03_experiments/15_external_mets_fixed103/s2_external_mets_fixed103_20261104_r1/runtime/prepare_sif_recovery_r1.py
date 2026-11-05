"""Allocated one-shot SIF recovery, reusing accepted wheels without rebuilding."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from s0_build_contract import require, sha, write, recovery_guard
from prepare_s0_cpu_r1 import run_owned
from normalize_oci_hardlinks import normalize_archive


def convert(attempt, contract, env):
    deadline=time.monotonic()+contract['single_conversion_deadline_seconds']
    conversion=attempt/'conversion'
    conversion.mkdir(mode=0o700,exist_ok=False)
    normalized=normalize_archive(Path(contract['archive']),conversion/'normalized',deadline,
                                 contract['maximum_normalized_layer_bytes'])
    require(normalized['original_archive_sha256']==contract['archive_sha256'],'original archive drift during normalization')
    write(attempt/'reports/NORMALIZATION_COMPLETE.json',normalized)
    write(attempt/'reports/NORMALIZATION_SUMMARY.json',dict(
        report_sha256=sha(attempt/'reports/NORMALIZATION_COMPLETE.json'),
        layers=len(normalized['layers']),hardlinks=sum(len(r['hardlinks']) for r in normalized['layers']),
        derived_archive_path=normalized['path'],derived_archive_bytes=normalized['bytes'],
        derived_archive_sha256=normalized['sha256'],original_archive_sha256=normalized['original_archive_sha256'],
        every_member_bytes_and_metadata_compared=normalized['every_member_bytes_and_metadata_compared'],
        inode_sharing_not_preserved=True,models_loaded=False))
    backend=contract['backend']
    require(subprocess.check_output([backend,'version'],text=True,env=env,timeout=20).strip()=='4.1.1','backend version drift')
    env={k:v for k,v in env.items() if not k.startswith(('SINGULARITY','APPTAINER'))}
    for name in ('tmp','cache'):(conversion/name).mkdir()
    env.update(SINGULARITY_TMPDIR=str(conversion/'tmp'),SINGULARITY_CACHEDIR=str(conversion/'cache'))
    sandbox,squash,partial=[conversion/name for name in ('native_rootfs','native_rootfs.squashfs','fixed2025.part.sif')]
    commands=[
        [backend,'build','--disable-cache','--sandbox','--no-cleanup',str(sandbox),'docker-archive://'+normalized['path']],
        ['/usr/sbin/mksquashfs',str(sandbox),str(squash),'-noappend','-all-root','-comp','gzip',
         '-processors','2','-read-queue','64','-write-queue','128','-fragment-queue','64','-no-progress'],
        [backend,'sif','new',str(partial)],
        [backend,'sif','add',str(partial),str(squash),'--datatype','4','--parttype','2','--partfs','1','--partarch','2']]
    write(attempt/'reports/NATIVE_BUILD_COMMANDS.json',dict(commands=commands,deadline_seconds=7200,
        normalization_included_in_deadline=True,compression_threads=2,compression_queue_mib=256,
        administrator_configuration_changed=False))
    with (attempt/'logs/SIF_BUILD.log').open('x') as log:
        for index,command in enumerate(commands):
            remaining=deadline-time.monotonic()
            require(remaining>0,'single conversion deadline exhausted')
            write(attempt/f'reports/SIF_STAGE_{index}_STARTED.json',dict(remote_utc=datetime.now(timezone.utc).isoformat(),command=command))
            run_owned(command,log,env,remaining,compression=index==1)
            if index==0:require((sandbox/'.singularity.d/runscript').is_file(),'native OCI runscript absent')
        listing=subprocess.check_output([backend,'sif','list',str(partial)],env=env,text=True,timeout=30)
    require(listing.count('FS (Squashfs/*System/amd64)')==1,'native primary SIF descriptor drift')
    final=conversion/'fixed2025.sif'
    require(not final.exists(),'SIF final destination already exists')
    partial.chmod(0o440)
    os.link(partial,final)
    return dict(path=str(final),bytes=final.stat().st_size,sha256=sha(final),descriptor=listing,
        source_archive_sha256=contract['archive_sha256'],derived_archive_sha256=normalized['sha256'],
        normalization_report_sha256=sha(attempt/'reports/NORMALIZATION_COMPLETE.json'),
        original_model_files_byte_equivalent=True,not_original_byte_identical_Docker_digest=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    attempt=parser.parse_args().attempt.resolve()
    path=attempt/'contract/PREPARATION_CONTRACT.json'
    contract=json.loads(path.read_text())
    require(attempt==Path(contract['remote_root'])/'attempts/s0_sif_recovery_r1','recovery attempt path drift')
    require(not any((attempt/'reports'/name).exists() for name in ('STARTED.json','FAILED.json','COMPLETE.json')),'recovery already attempted; no retry')
    stage='allocated_CPU_identity_and_reused_artifacts'
    try:
        require(sha(path)==os.environ.get('S0_EXPECTED_CONTRACT_SHA256'),'frozen contract SHA drift')
        require(os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])==8,'CPU allocation missing')
        require(not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES'),'GPU in CPU preparation')
        require(Path(sys.executable).resolve()==(Path(contract['build_env'])/'bin/python').resolve(),'build interpreter drift')
        recovery_guard(attempt,contract)
        require(sha(contract['archive'])==contract['archive_sha256'],'original archive SHA drift')
        require(shutil.disk_usage(attempt).free>contract['incremental_budget_bytes'],'fresh CPU capacity gate')
        env={k:v for k,v in os.environ.items() if not k.upper().endswith('_PROXY') and not k.startswith('CONDA_OVERRIDE_')}
        write(attempt/'reports/STARTED.json',dict(status='allocated_CPU_SIF_recovery_started',job_id=os.environ['SLURM_JOB_ID'],
            remote_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(path),wheels_recompiled=False,models_loaded=False))
        write(attempt/'reports/WHEELS_REUSED.json',dict(wheels=contract['reused_wheels'],
            original_wheels_manifest_sha256=contract['parent_wheels_manifest_sha256'],fresh_bytes_SHA_pass=True,wheels_recompiled=False))
        stage='single_normalized_offline_SIF_conversion'
        sif=convert(attempt,contract,env)
        recovery_guard(attempt,contract)
        require(sha(contract['archive'])==contract['archive_sha256'],'original archive changed after conversion')
        write(attempt/'reports/COMPLETE.json',dict(status='S0_recovery_artifacts_complete_runtime_and_GPU_still_require_validation',
            job_id=os.environ['SLURM_JOB_ID'],remote_utc=datetime.now(timezone.utc).isoformat(),sif=sif,
            reused_wheels_sha256=sha(attempt/'reports/WHEELS_REUSED.json'),original_failed_job_preserved='4304783',
            models_loaded=False,images_read=False,runtime_environment_installed=False,original_inference_jobs_created=False,
            free_bytes_after=shutil.disk_usage(attempt).free))
    except BaseException as error:
        write(attempt/'reports/FAILED.json',dict(status='S0_SIF_recovery_failed',stage=stage,error_type=type(error).__name__,
            message=str(error),job_id=os.environ.get('SLURM_JOB_ID'),remote_utc=datetime.now(timezone.utc).isoformat(),
            no_automatic_retry=True,models_loaded=False,images_read=False))
        raise


if __name__=='__main__':
    def interrupted(signum,frame):raise RuntimeError('CPU recovery interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    main()
