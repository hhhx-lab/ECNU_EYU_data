"""Allocated full103 entry; no separate smoke, no training or automatic retry."""
import argparse
import errno
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

from contract import META, MODALITIES, fixed_rows, input_map, internal_map, label_array, native_geometry, read, require, sha, write
from full_contract import clean_process_env, load_contract, verify_assets
from gpu_isolation import device_identity


def prepare_inputs(attempt, contract, model):
    import nibabel as nib
    import numpy as np
    rows = fixed_rows(attempt / 'manifests/FIXED103_REAL_T2W_CASES.csv')
    mapped = input_map(rows, read(attempt / 'manifests/INPUT_REFERENCE_SHA256.json'))
    images = {Path(v['path']) for case in mapped.values() for k, v in case.items() if k != 'seg'}
    directories = {p.parent for p in images}
    require(len(directories) == 1, 'single fixed103 image-only directory required')
    source = next(iter(directories))
    require(set(source.glob('*.nii.gz')) == images, 'image bind contains unapproved cases or references')
    stage = attempt / 'scratch' / model
    stage.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ('input', 'raw_output', 'tmp', 'home'):
        (stage / name).mkdir(mode=0o700)
    internal = internal_map(rows)
    index = {}
    for row in rows:
        case = row['source_case_id']
        target = internal[case] if model == 'EXT2025_MET_GLI' else case
        folder = stage / 'input' / target
        folder.mkdir()
        reference = nib.load(mapped[case]['t1c']['path'])
        for modality in MODALITIES:
            item = mapped[case][modality]
            path = Path(item['path'])
            require(path.is_file() and sha(path) == item['sha256'], 'full input SHA drift: ' + case + ' ' + modality)
            image = nib.load(str(path))
            native_geometry(image, reference)
            data = np.asanyarray(image.dataobj)
            require(data.ndim == 3 and np.isfinite(data).all(), 'full image finite/geometry failure')
            del data
            destination = folder / (target + '-' + modality + '.nii.gz')
            try:
                os.link(path.resolve(), destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(path.resolve(), destination)
            require(not destination.is_symlink() and sha(destination) == item['sha256'], 'staged real input identity drift')
        index[case] = dict(internal_case_id=target, input_sha256={m: mapped[case][m]['sha256'] for m in MODALITIES})
    return stage, mapped, index


def container_command(attempt, contract, model, stage):
    runtime = attempt / 'runtime'
    binds = [(stage / 'input', '/input', 'ro'),
             (stage / 'raw_output', '/output', 'rw'), (stage / 'tmp', '/tmp', 'rw'),
             (runtime, '/runtime', 'ro')]
    env = clean_process_env()
    gpu = device_identity()
    env['SINGULARITYENV_CUDA_VISIBLE_DEVICES'] = gpu['visible_devices']
    env['SINGULARITYENV_CUDA_DEVICE_ORDER'] = gpu['device_order']
    env['SINGULARITYENV_FULL_EXPECTED_GPU_IDENTITY'] = json.dumps(gpu)
    for name in ('PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        env['SINGULARITYENV_' + name] = env[name]
    if model == 'EXT2025_MET_GLI':
        env['SINGULARITYENV_FULL_2025_MAIN_SHA256'] = contract['2025_main_sha256']
        program = ['python3', '-B', '/runtime/checked_2025.py']
    else:
        binds.extend((Path(item['host']), item['container'], 'ro') for item in contract['bio_mounts'])
        env['SINGULARITYENV_FULL_BIO_CONFIG'] = json.dumps(contract['bio_config'])
        for key, value in contract['bio_environment'].items():
            require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key), 'unsafe Bio environment key')
            env['SINGULARITYENV_' + key] = str(value)
        program = [str(Path(contract['bio_environment_prefix']) / 'bin/python'), '-B', '/runtime/checked_bio.py']
    argv = [contract['backend'], 'exec', '--cleanenv', '--containall', '--no-home',
            '--no-mount', 'hostfs,bind-paths', '--net', '--network', 'none',
            '--nv', '--writable-tmpfs', '--pwd', '/tmp']
    for host, destination, mode in binds:
        require(host.exists() and ':' not in str(host) and ',' not in str(host), 'unsafe container bind')
        argv.extend(['--bind', str(host) + ':' + destination + ':' + mode])
    argv.extend([contract['sif']['path'], *program])
    return argv, env, gpu


def run_owned(argv, env, log, cwd, timeout):
    process = subprocess.Popen(argv, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        result = process.wait(timeout=timeout)
        require(result == 0, 'author full inference exited nonzero: ' + str(result))
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--model', choices=('EXT2025_MET_GLI', 'EXT2026_BIO_INTEL_FULL'), required=True)
    args = parser.parse_args()
    attempt, model = args.attempt.resolve(), args.model
    reports = attempt / 'reports' / model
    reports.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage_name = 'allocated_full_job_contract_and_assets'
    try:
        contract = load_contract(attempt, model)
        require(Path(sys.executable).resolve() == (Path(contract['controller_environment_prefix']) / 'bin/python').resolve(), 'full controller interpreter drift')
        import importlib.metadata as metadata
        require({n: metadata.version(n) for n in contract['controller_versions']} == contract['controller_versions'], 'controller dependency version drift')
        verify_assets(contract, model)
        write(reports / 'STARTED.json', dict(META, model=model, cases=103, job_id=os.environ['SLURM_JOB_ID'],
              contract_sha256=sha(attempt / 'contract/EXECUTION_CONTRACT.json'), remote_utc=datetime.now(timezone.utc).isoformat(),
              smoke_gate='waived_by_user_not_pass', measured_runtime_peak_gate='waived_by_user_not_pass'))
        stage_name = 'all412_real_images_identity_and_geometry'
        stage, mapped, index = prepare_inputs(attempt, contract, model)
        write(reports / 'INPUT_ADAPTER.json', dict(META, cases=index, image_bind=str(stage / 'input'),
              input_files=412, labels_read=False, missing_modality_fallback=False))
        stage_name = 'isolated_author_complete_full103_inference'
        argv, env, gpu = container_command(attempt, contract, model, stage)
        write(reports / 'RUN_CONTRACT.json', dict(META, argv=argv, gpu=gpu, cases=103,
              author_revision=contract['author_revision'], fold_selection=contract['bio_folds'] if model != 'EXT2025_MET_GLI' else 'all',
              ground_truth_bound=False, credentials_bound=False, network='none', training=False, DDP=False,
              smoke_gate='waived_by_user_not_pass'))
        with (attempt / 'logs' / (model + '_INFERENCE.log')).open('x') as log:
            run_owned(argv, env, log, stage / 'home', 48 * 3600 - 600)
        stage_name = 'full103_native_output_validation'
        import nibabel as nib
        raw = stage / 'raw_output'
        expected = {v['internal_case_id'] + '.nii.gz' for v in index.values()}
        require({p.name for p in raw.glob('*.nii.gz')} == expected, '103 authored outputs incomplete/unexpected')
        require(not any(p.is_dir() for p in raw.iterdir()), 'author output not flat')
        prediction = attempt / 'predictions' / model
        prediction.mkdir(parents=True, exist_ok=False)
        files = {}
        for case, item in index.items():
            path = raw / (item['internal_case_id'] + '.nii.gz')
            require(path.is_file() and not path.is_symlink() and path.resolve().parent == raw.resolve(), 'unsafe authored output path')
            image = nib.load(str(path))
            native_geometry(image, nib.load(mapped[case]['t1c']['path']))
            label_array(image)
            destination = prediction / (case + '.nii.gz')
            os.link(path, destination)
            files[destination.name] = dict(bytes=destination.stat().st_size, sha256=sha(destination))
        verify_assets(contract, model)
        write(reports / 'PREDICTION_SHA256.json', dict(META, files=files, cases=103,
              input_adapter_sha256=sha(reports / 'INPUT_ADAPTER.json'), contract_sha256=sha(attempt / 'contract/EXECUTION_CONTRACT.json')))
        write(reports / 'COMPLETE.json', dict(META, status='full103_inference_outputs_validated_not_scientific_quality_pass',
              model=model, job_id=os.environ['SLURM_JOB_ID'], cases=103, elapsed_seconds=time.monotonic() - started,
              remote_utc=datetime.now(timezone.utc).isoformat(), prediction_manifest_sha256=sha(reports / 'PREDICTION_SHA256.json'),
              smoke_gate='waived_by_user_not_pass', measured_runtime_peak_gate='waived_by_user_not_pass',
              contract_sha256=sha(attempt / 'contract/EXECUTION_CONTRACT.json'), reference_labels_read=False))
    except BaseException as error:
        write(reports / 'FAILED.json', dict(META, status='full_inference_failed', stage=stage_name,
              model=model, job_id=os.environ.get('SLURM_JOB_ID'), error_type=type(error).__name__, message=str(error),
              remote_utc=datetime.now(timezone.utc).isoformat(), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Allocated full inference interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
