"""Immutable contract checks for the user-authorized direct full103 jobs."""
import os
from pathlib import Path

from contract import META, read, require, sha


def load_contract(attempt, model):
    attempt = Path(attempt).resolve()
    path = attempt / 'contract/EXECUTION_CONTRACT.json'
    contract = read(path)
    require(str(attempt) == contract['attempt'], 'full inference attempt drift')
    require(all(contract[k] == v for k, v in META.items()), 'scientific status drift')
    require(contract['stage'] == 'direct_full103_user_waived_smoke', 'direct execution authorization drift')
    require(contract['smoke_gate'] == 'waived_by_user_not_pass', 'unmeasured smoke claimed pass')
    require(contract['measured_runtime_peak_gate'] == 'waived_by_user_not_pass', 'unmeasured capacity claimed pass')
    require(model in ('EXT2025_MET_GLI', 'EXT2026_BIO_INTEL_FULL'), 'unapproved model')
    require(sha(path) == os.environ['FULL_EXPECTED_CONTRACT_SHA256'], 'full inference contract SHA drift')
    mapping = read(attempt / 'operator/HELD_JOB_MAP.json')
    require(mapping['contract_sha256'] == sha(path), 'job-map contract drift')
    require(mapping['package_sha256'] == sha(attempt / 'contract/PACKAGE_SHA256.json'), 'job-map package drift')
    job = mapping['jobs'][model]
    require(os.environ.get('SLURM_JOB_ID') == job['job_id'], 'allocated full JobID drift')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == '8', 'full CPU allocation drift')
    for name, digest in read(attempt / 'contract/PACKAGE_SHA256.json')['files'].items():
        require(sha(attempt / name) == digest, 'frozen full inference file drift: ' + name)
    require(contract['cases'] == 103 and contract['patient_groups'] == 69, 'fixed103 count drift')
    require(contract['bio_folds'] == {'A': [0, 1, 2, 3, 4], 'B': [1, 3]}, 'seven-fold ensemble drift')
    return contract


def verify_assets(contract, model):
    files = [contract['sif']]
    if model == 'EXT2026_BIO_INTEL_FULL':
        files.extend(contract['bio_files'])
        files.extend(contract['bio_source_files'])
        files.extend(contract['bio_runtime_identity_files'])
    for item in files:
        path = Path(item['path'])
        require(path.is_file() and not path.is_symlink(), 'required immutable asset absent/symlink')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'model/source asset identity drift')


def clean_process_env():
    allowed = ('PATH', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
               'CUDA_DEVICE_ORDER', 'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_GPUS')
    result = {k: v for k, v in os.environ.items() if k in allowed}
    result.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='8',
                  MKL_NUM_THREADS='8', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='8')
    return result
