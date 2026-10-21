"""CPU-only acceptance of immutable existing MET outputs; no inference or training."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'CPU_RECOVERY_EXECUTION_PLAN_20260911_R2.json'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
METHODS = ('B', 'E_SNFH', 'F')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def audit(name, status, **data):
    value = dict(META, status=status, remote_utc=datetime.now(timezone.utc).isoformat(),
                 hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'), **data)
    write(ROOT / 'evidence' / name, value)
    return value


def copy_light(source, relative):
    source, dest = Path(source), ROOT / 'sources' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as a, dest.open('xb') as b:
        shutil.copyfileobj(a, b)
    require(sha(source) == sha(dest), 'copy integrity: ' + relative)


def label_values(value):
    if type(value) is int:
        return {value}
    require(isinstance(value, (list, tuple)), 'invalid dataset labels')
    result = set()
    for entry in value:
        result |= label_values(entry)
    return result


def ontology(dataset):
    result = set()
    for value in dataset['labels'].values():
        result |= label_values(value)
    require(result == {0, 1, 2, 3, 4}, 'frozen dataset ontology drift')
    return result


def validate_voxels(array, allowed):
    import numpy as np
    require(bool(np.isfinite(array).all()), 'nonfinite prediction')
    valid = np.zeros(array.shape, dtype=bool)
    observed = []
    for label in sorted(allowed):
        mask = array == label
        if bool(mask.any()):
            observed.append(label)
        valid |= mask
    require(bool(valid.all()), 'illegal prediction label')
    return observed


def contract(expected_sha=None):
    if expected_sha:
        require(sha(ROOT / PLAN) == expected_sha, 'recovery plan SHA drift')
    subprocess.run(['sha256sum', '-c', 'STATIC_SHA256SUMS.txt'], cwd=ROOT,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    p = read(ROOT / PLAN)
    source = Path(p['source_root'])
    require(sha(source / p['source_plan_name']) == p['source_plan_sha256'], 'source plan drift')
    subprocess.run(['sha256sum', '-c', 'STATIC_SHA256SUMS.txt'], cwd=source,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    require(read(source / 'evidence/REMOTE_STATIC_PREFLIGHT.json')['status'] == 'pass', 'source preflight')
    old = read(source / p['source_plan_name'])
    require(old['bootstrap'] == p['bootstrap'] == dict(replicates=20000, seed=20260831), 'bootstrap drift')
    require(sha(ROOT / 'runtime/compare_paired_met.py') == p['comparator_sha256'] == sha(source / 'runtime/compare_paired_met.py'), 'comparator drift')
    manifest = source / 'manifests/FIXED103_REAL_T2W_CASES.csv'
    require(sha(manifest) == old['case_contract']['manifest_sha256'], 'manifest drift')
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ids = {r['source_case_id'] for r in rows}
    require(len(rows) == len(ids) == 103, 'manifest coverage')
    for method in METHODS:
        route, frozen = old['routes'][method], p['routes'][method]
        failure = source / f'evidence/{method}_EVALUATION_FAILED.json'
        require(read(failure) == frozen['source_failure_json'], method + ' failure evidence drift')
        if (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json').exists():
            require(sha(failure) == read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')['source_failure_sha256'][method], method + ' failure bytes drift')
        failed = read(failure)
        require(failed['status'] == 'failed' and failed['error'].startswith('prediction labels:') and str(failed['job_id']) == frozen['failed_job_id'], method + ' unexpected failure')
        out = source / 'cohort/fixed103/models' / method
        inf = read(out / 'inference_audit.json')
        ev = read(out / 'evaluation/EVALUATION_VALIDATION.json')
        require(inf['status'] == ev['status'] == 'pass', method + ' source validations')
        require(inf['case_count'] == inf['prediction_count'] == ev['case_count'] == 103, method + ' source counts')
        require(inf['checkpoint_sha256'] == route['checkpoint_sha256'] == frozen['checkpoint_sha256'], method + ' checkpoint contract')
        require(inf['execution_mode'] == 'sequential_in_process' and inf['preprocess_workers'] == inf['export_workers'] == 0 and inf['multiprocessing_queue_used'] is False, method + ' inference contract')
        require((out / 'evaluation/EVALUATION_COMPLETE.ok').is_file(), method + ' evaluator not complete')
        require(not (source / f'evidence/{method}_EVALUATION_COMPLETE.ok').exists(), method + ' original failure overwritten')
        require({x.name.removesuffix('.nii.gz') for x in (out / 'prediction').glob('*.nii.gz')} == ids, method + ' predictions coverage')
        require(sha(out / 'evaluation/leaderboard_metrics.csv') == frozen['metrics175_sha256'] == ev['metrics_csv_sha256'], method + ' 175 CSV SHA')
        require(sha(out / 'evaluation/panoptica_evaluation_summary.json') == frozen['summary175_sha256'] == ev['summary_json_sha256'], method + ' summary SHA')
        require(sha(route['baseline_metrics_csv']) == route['baseline_metrics_sha256'], method + ' 125 CSV SHA')
        require(read(route['baseline_validation'])['status'] == 'pass' and Path(route['baseline_complete']).is_file(), method + ' baseline incomplete')
        dataset = read(source / f'model_snapshots/{method}/dataset.json')
        require(dataset == read(Path(route['baseline_model_root']) / 'dataset.json'), method + ' dataset mismatch')
        ontology(dataset)
    return p, old, ids


def preflight(expected_sha):
    require(not (ROOT / 'evidence/CPU_RESERVATION.json').exists(), 'existing reservation: read only')
    p, _, _ = contract(expected_sha)
    audit('REMOTE_STATIC_PREFLIGHT.json', 'pass', plan_sha256=sha(ROOT / PLAN),
          validation_scope='lightweight metadata, hashes and file counts; full voxel audit scheduled on CPU',
          source_failure_sha256={m: sha(Path(p['source_root']) / f'evidence/{m}_EVALUATION_FAILED.json') for m in METHODS},
          source_failed_jobs_preserved=p['failed_jobs'], gpu_jobs=0, cpu_jobs=1)


def launch(expected_sha):
    require(sha(ROOT / PLAN) == expected_sha, 'launch plan SHA')
    require(read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')['plan_sha256'] == expected_sha, 'launch preflight SHA')
    require(read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')['status'] == 'pass', 'launch preflight')
    audit('CPU_RESERVATION.json', 'reserved', plan_sha256=expected_sha)
    command = ['sbatch', '--hold', '--parsable', '--job-name=s2x175cr2', 'slurm/cpu_recovery.slurm']
    answer = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    job = answer.split(';')[0]
    require(job.isdigit(), 'unrecognized sbatch receipt; reservation must remain')
    audit('CPU_LAUNCH.json', 'submitted_held', canonical_job_id=job, plan_sha256=expected_sha, command=command)
    subprocess.run(['scontrol', 'release', job], check=True)
    audit('CPU_RELEASED.json', 'released', canonical_job_id=job)
    print(json.dumps(dict(status='submitted', job_id=job, gpu_jobs=0)))


def cpu():
    import nibabel as nib
    import numpy as np
    launch_record = read(ROOT / 'evidence/CPU_LAUNCH.json')
    require(os.environ.get('SLURM_JOB_ID') == launch_record['canonical_job_id'], 'CPU launch identity')
    require(not os.environ.get('SLURM_JOB_GPUS'), 'CPU-only job assigned GPUs')
    p, old, ids = contract(launch_record['plan_sha256'])
    audit('CPU_STARTED.json', 'started', plan_sha256=sha(ROOT / PLAN), gpu_jobs=0,
          operation='read-only original outputs, corrected 0-4 ontology, immutable CPU-only recovery')
    require(validate_voxels(np.array([0, 1, 2, 3, 4]), {0, 1, 2, 3, 4}) == [0, 1, 2, 3, 4], 'label4 selftest')
    for invalid in (np.array([5.0]), np.array([np.nan]), np.array([1.5])):
        try:
            validate_voxels(invalid, {0, 1, 2, 3, 4})
        except RuntimeError:
            pass
        else:
            raise RuntimeError('invalid label accepted by selftest')
    audit('LABEL_ONTOLOGY_TESTS.json', 'pass', allowed=[0, 1, 2, 3, 4], rejects=[5, 'NaN', 1.5])
    source, cohort = Path(p['source_root']), Path(old['case_contract']['source_root'])
    copy_light(source / p['source_plan_name'], p['source_plan_name'])
    copy_light(source / 'manifests/FIXED103_REAL_T2W_CASES.csv', 'FIXED103_REAL_T2W_CASES.csv')
    for name, expected in old['case_contract']['audit_sha256'].items():
        require(sha(cohort / name) == expected, 'input audit drift')
        copy_light(cohort / name, name)
    with (cohort / 'input_audit.csv').open(newline='') as f:
        input_rows = list(csv.DictReader(f))
    require(len(input_rows) == 103 and {r['source_case_id'] for r in input_rows} == ids, 'input audit cohort')
    recorded = read(source / 'manifests/INPUT_REFERENCE_SHA256.json')['files']
    checked = {}
    for row in input_rows:
        for index, channel in enumerate(('t1n', 't1c', 't2w', 't2f', 'seg')):
            path = cohort / ('reference' if channel == 'seg' else 'input') / (row['source_case_id'] + ('.nii.gz' if channel == 'seg' else f'_{index:04d}.nii.gz'))
            observed = sha(path)
            require(observed == row[channel + '_sha256'] == recorded[str(path)], 'input/reference SHA: ' + path.name)
            checked[str(path)] = observed
    require(len(checked) == len(recorded) == 515, 'input/reference file count')
    write(ROOT / 'manifests/INPUT_REFERENCE_VERIFIED.json', dict(META, files=checked))
    print('INPUT_REFERENCE_515_PASS', flush=True)
    reports = {}
    for method in METHODS:
        route = old['routes'][method]
        out = source / 'cohort/fixed103/models' / method
        checkpoint = source / f'model_snapshots/{method}/fold_0/checkpoint_final.pth'
        require(sha(checkpoint) == route['checkpoint_sha256'], method + ' checkpoint actual SHA')
        require(sha(Path(route['baseline_model_root']) / 'fold_0/checkpoint_final.pth') == route['baseline_checkpoint_sha256'], method + ' baseline checkpoint actual SHA')
        require(sha(route['training_validation']) == route['training_validation_sha256'] and read(route['training_validation'])['status'] == 'pass' and Path(route['training_complete']).is_file(), method + ' training provenance')
        allowed = ontology(read(checkpoint.parents[1] / 'dataset.json'))
        predictions = sorted((out / 'prediction').glob('*.nii.gz'))
        require(len(predictions) == 103, 'prediction count')
        voxel_records = []
        for path in predictions:
            img, ref = nib.load(path), nib.load(cohort / 'reference' / path.name)
            require(img.shape == ref.shape and np.allclose(img.affine, ref.affine, rtol=0, atol=1e-5), method + ' geometry: ' + path.name)
            labels = validate_voxels(np.asanyarray(img.dataobj), allowed)
            voxel_records.append(dict(case_id=path.name.removesuffix('.nii.gz'), labels=labels, sha256=sha(path), geometry_match=True))
        summary = read(out / 'evaluation/panoptica_evaluation_summary.json')
        require(summary['missings'] == [] and len(summary['metrics']) == 103 and not any('error' in r for r in summary['metrics']), method + ' official MET coverage')
        for epoch, csv_path in ((125, Path(route['baseline_metrics_csv'])), (175, out / 'evaluation/leaderboard_metrics.csv')):
            with csv_path.open(newline='') as f:
                reader = csv.DictReader(f)
                rows, fields = list(reader), reader.fieldnames
            require(len(rows) == 106 and len(fields) == 73, method + ' MET shape')
            require([r['subject_id'] for r in rows[-3:]] == ['mean', 'std', 'median'], 'MET summary rows')
            require({r['subject_id'].removesuffix('.nii.gz') for r in rows[:103]} == ids, 'MET case IDs')
            copy_light(csv_path, f'{method}/epoch{epoch}/leaderboard_metrics.csv')
        for relative in ('inference_audit.json', 'evaluation/EVALUATION_VALIDATION.json', 'evaluation/EVALUATION_COMPLETE.ok', 'evaluation/panoptica_evaluation_summary.json'):
            copy_light(out / relative, f'{method}/epoch175/{relative}')
        for path, name in ((source / f'evidence/{method}_EVALUATION_FAILED.json', 'ORIGINAL_FAILURE.json'),
                           (checkpoint.parents[1] / 'dataset.json', 'dataset.json'),
                           (Path(route['training_validation']), 'TRAINING_VALIDATION.json'),
                           (Path(route['baseline_validation']), 'BASELINE_VALIDATION.json')):
            copy_light(path, f'{method}/{name}')
        reports[method] = dict(prediction_count=103, selected_epoch=175, allowed_labels=sorted(allowed),
                               cases_with_label4=sum(4 in r['labels'] for r in voxel_records), geometry_mismatch_count=0,
                               nonfinite_count=0, invalid_label_count=0, checkpoint_sha256=route['checkpoint_sha256'],
                               baseline_checkpoint_sha256=route['baseline_checkpoint_sha256'], cases=voxel_records)
        audit(method + '_CPU_VALIDATION.json', 'pass', **reports[method])
        print(method + '_CPU_103_PASS', flush=True)
    for names, scope in (([m + '175' for m in METHODS], 'epoch175_cross_route'), *[([m + '175', m + '125'], m + '_175_vs125') for m in METHODS]):
        args = [sys.executable, str(ROOT / 'runtime/compare_paired_met.py'), '--output-root', str(ROOT / 'results' / scope),
                '--expected-count', '103', '--expected-methods', str(len(names)), '--bootstrap', '20000', '--seed', '20260831', '--scope', scope]
        for name in names:
            args += ['--input', name + '=' + str(ROOT / 'sources' / name[:-3] / ('epoch' + name[-3:]) / 'leaderboard_metrics.csv')]
        subprocess.run(args, check=True)
    subprocess.run([sys.executable, str(ROOT / 'runtime/report.py')], check=True)
    contract(launch_record['plan_sha256'])
    audit('CPU_RECOVERY_VALIDATION.json', 'pass', method_count=3, prediction_count=309, pair_count=6,
          metric_count=72, paired_metric_records=432, bootstrap=p['bootstrap'], original_jobs_remain_failed=True,
          original_failures_preserved=True, gpu_jobs=0, input_reference_count=515)
    write(ROOT / 'evidence/CPU_RECOVERY_COMPLETE.ok', dict(META, status='pass', original_jobs_remain_failed=True))
    inventory = {}
    for folder in ('sources', 'results', 'evidence', 'manifests', 'runtime', 'slurm'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                require(not path.is_symlink() and path.suffix not in ('.pth', '.gz', '.jsonl', '.pyc'), 'heavy/unexpected artifact')
                inventory[str(path.relative_to(ROOT))] = dict(sha256=sha(path), size=path.stat().st_size)
    for name in (PLAN, 'STATIC_SHA256SUMS.txt', 'README.md'):
        path = ROOT / name
        inventory[name] = dict(sha256=sha(path), size=path.stat().st_size)
    write(ROOT / 'LIGHTWEIGHT_SHA256.json', dict(META, remote_utc=datetime.now(timezone.utc).isoformat(), files=inventory,
                                                logs_policy='snapshot hashes after scheduler completion; logs excluded here while still open'))
    print('CPU_RECOVERY_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'launch', 'cpu'))
    parser.add_argument('--plan-sha')
    args = parser.parse_args()
    try:
        if args.action == 'cpu':
            cpu()
        else:
            require(args.plan_sha, 'explicit plan SHA required')
            {'preflight': preflight, 'launch': launch}[args.action](args.plan_sha)
    except Exception as exc:
        name = {'cpu': 'CPU_RECOVERY_FAILED.json', 'preflight': 'REMOTE_STATIC_PREFLIGHT_FAILED.json', 'launch': 'LAUNCH_FAILED.json'}[args.action]
        if not (ROOT / 'evidence' / name).exists():
            audit(name, 'failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
