"""Isolated epoch175 evaluation, immutable launch audits, and matched125 comparisons."""
import argparse
import csv
import gc
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
EXPECTED_ROOT = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1')
PLAN_NAME = 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'
PYTHON = '/public/home/zqchen/.conda/envs/segmamba/bin/python'
EVAL_ENV = '/public/home/zqchen/.conda/envs/brats_eval'
METHODS = ('B', 'E_SNFH', 'F')
TRAINER = 'nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def audit(name, status, **kwargs):
    payload = dict(META, status=status, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                   hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'))
    payload.update(kwargs)
    write(ROOT / 'evidence' / name, payload)
    return payload


def complete(name):
    path = ROOT / 'evidence' / name
    with path.open('x') as f:
        f.write('status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n')


def static_check(expected_sha=None):
    require(ROOT == EXPECTED_ROOT, 'unexpected experiment root')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'bytecode writes must be disabled')
    subprocess.run(['sha256sum', '-c', 'STATIC_SHA256SUMS.txt'], cwd=ROOT, check=True, stdout=subprocess.PIPE)
    p = read(ROOT / PLAN_NAME)
    require(all(p.get(k) == v for k, v in META.items()), 'artifact contract drift')
    require(list(p['routes']) == list(METHODS), 'route order drift')
    require(p['experiment_id'] == ROOT.name, 'experiment identity drift')
    require(p['bootstrap'] == dict(replicates=20000, seed=20260831), 'bootstrap drift')
    if expected_sha:
        require(sha(ROOT / PLAN_NAME) == expected_sha, 'frozen plan SHA drift')
    return p


def require_preflight():
    pre = read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')
    require(pre['status'] == 'pass' and not (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json').exists(), 'preflight failed')
    return static_check(pre['plan_sha256'])


def metric_csv(path, ids):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    require(len(rows) == 106 and [r['subject_id'] for r in rows[-3:]] == ['mean', 'std', 'median'], 'official CSV shape')
    require(len(fields) == 73, 'official MET metric count')
    observed = [r['subject_id'].removesuffix('.nii.gz') for r in rows[:103]]
    require(len(set(observed)) == 103 and set(observed) == ids, 'official CSV cohort')
    return rows, fields


def case_ids():
    with (ROOT / 'manifests/FIXED103_REAL_T2W_CASES.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 103 and len({r['source_case_id'] for r in rows}) == 103, 'manifest count')
    require(all(r['source_t2w_allowed'] == 'True' and r['t2w_status'] == 'authentic' for r in rows), 'real T2W contract')
    return {r['source_case_id'] for r in rows}


def preflight(expected_sha):
    import numpy as np
    import torch
    import nnunetv2.inference.predict_from_raw_data as predict_module
    from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import nnUNetTrainerBraTS2026RC
    torch.set_num_threads(2)
    p = static_check(expected_sha)
    require(not any((ROOT / 'evidence').glob('*RESERVATION*')), 'existing launch reservation')
    require(not (ROOT / 'model_snapshots').exists(), 'model snapshots already exist')
    require(not (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json').exists(), 'preflight already exists')
    require(not (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json').exists(), 'failed preflight already exists')
    for item in p['upstream_plans']:
        require(sha(item['path']) == item['sha256'], 'upstream plan SHA')
    ids = case_ids()
    require(sha(ROOT / 'manifests/FIXED103_REAL_T2W_CASES.csv') == p['case_contract']['manifest_sha256'], 'manifest SHA')
    cohort = Path(p['case_contract']['source_root'])
    for filename, expected in p['case_contract']['audit_sha256'].items():
        require(sha(cohort / filename) == expected, 'shared input audit SHA')
    inp = read(cohort / 'input_audit.json')
    require(inp['status'] == 'pass' and inp['case_count'] == 103 and inp['source_t2w_used'] is True, 'shared input audit')
    require(inp['geometry_mismatch_count'] == inp['geometry_repaired_count'] == 0, 'input geometry')
    require(len(list((cohort / 'input').glob('*.nii.gz'))) == 412, 'input file count')
    require({x.name.removesuffix('.nii.gz') for x in (cohort / 'reference').glob('*.nii.gz')} == ids, 'reference cohort')
    with (cohort / 'input_audit.csv').open(newline='') as f:
        source_rows = list(csv.DictReader(f))
    require({r['source_case_id'] for r in source_rows} == ids and len(source_rows) == 103, 'input audit cohort')
    hashes = {}
    for row in source_rows:
        for index, modality in enumerate(('t1n', 't1c', 't2w', 't2f', 'seg')):
            file = cohort / ('reference' if modality == 'seg' else 'input') / (row['source_case_id'] + ('.nii.gz' if modality == 'seg' else f'_{index:04d}.nii.gz'))
            observed = sha(file)
            require(observed == row[modality + '_sha256'], 'input/reference content drift: ' + file.name)
            hashes[str(file)] = observed
    write(ROOT / 'manifests/INPUT_REFERENCE_SHA256.json', dict(META, files=hashes))
    env_check = "import importlib.metadata as m,json; e={'BraTS-evaluation':'0.0.8','panoptica':'2.1.0','numpy':'1.26.4'}; a={k:m.version(k) for k in e}; assert a==e,a; print(json.dumps(a))"
    versions = json.loads(subprocess.check_output([EVAL_ENV + '/bin/python', '-c', env_check], text=True))
    for name in ('brats-evaluate', 'brats-parse-metrics'):
        require(os.access(EVAL_ENV + '/bin/' + name, os.X_OK), 'official evaluator executable')
    require(sha(ROOT / 'runtime/run_official_met_eval.sh') == p['reused_runtime_sha256']['run_official_met_eval.sh'], 'official MET runtime SHA')
    schedule = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(p['source_jobs']), '--format=JobIDRaw,State,ExitCode'], text=True)
    states = {r[0]: r[1:] for line in schedule.splitlines() if (r := line.split('|')) and r[0] in p['source_jobs']}
    require(set(states) == set(p['source_jobs']) and all(v[:2] == ['COMPLETED', '0:0'] for v in states.values()), 'source jobs not successfully complete')
    models = {}
    for method in METHODS:
        route = p['routes'][method]
        source = Path(route['checkpoint'])
        require(sha(source) == route['checkpoint_sha256'], method + ' source175 SHA')
        require(sha(route['training_validation']) == route['training_validation_sha256'], method + ' training audit SHA')
        val = read(route['training_validation'])
        require(val['status'] == 'pass' and val['completed_total_epochs'] == 175 and val['best_checkpoint']['sha256'] == route['checkpoint_sha256'], method + ' selected175 validation')
        require(Path(route['training_complete']).is_file(), method + ' training complete')
        require(not Path(route['training_validation'].replace('_VALIDATION.json', '_FAILED.json')).exists(), method + ' training failure marker')
        old = read(route['baseline_validation'])
        require(old['status'] == 'pass' and old['selected_epoch'] == 125 and old['prediction_count'] == 103, method + ' baseline125 validation')
        require(old['checkpoint_sha256'] == route['baseline_checkpoint_sha256'] and old['metrics_csv_sha256'] == route['baseline_metrics_sha256'], method + ' baseline provenance')
        require(Path(route['baseline_complete']).is_file(), method + ' baseline complete')
        require(sha(route['baseline_metrics_csv']) == route['baseline_metrics_sha256'], method + ' baseline CSV SHA')
        metric_csv(route['baseline_metrics_csv'], ids)
        ck = torch.load(source, map_location='cpu', weights_only=False)
        require(ck['current_epoch'] == 175 and ck['trainer_name'] == TRAINER, method + ' epoch/trainer')
        require(all(bool(torch.isfinite(t).all()) for t in ck['network_weights'].values()), method + ' network finite')
        require(float(ck['logging']['ema_fg_dice'][174]) == route['ema_fg_dice'], method + ' EMA selection')
        init = ck['init_args']
        baseline_model = Path(route['baseline_model_root'])
        require(init['plans'] == read(baseline_model / 'plans.json') and init['dataset_json'] == read(baseline_model / 'dataset.json'), method + ' architecture/dataset differs from125')
        model = ROOT / 'model_snapshots' / method
        (model / 'fold_0').mkdir(parents=True, exist_ok=False)
        target = model / 'fold_0/checkpoint_final.pth'
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        target.chmod(0o440)
        require(sha(target) == route['checkpoint_sha256'], method + ' snapshot SHA')
        write(model / 'plans.json', init['plans'])
        write(model / 'dataset.json', init['dataset_json'])
        def finder(name):
            require(name == TRAINER, 'unexpected trainer during CPU architecture smoke test')
            return nnUNetTrainerBraTS2026RC
        predict_module.recursive_find_trainer_class_by_name = finder
        predictor = predict_module.nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(model), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
        models[method] = dict(epoch=175, checkpoint_sha256=sha(target), trainer=TRAINER, cpu_network_strict_load=True, source_copy_not_hardlink=True)
        audit(method + '_MODEL_SNAPSHOT.json', 'pass', **models[method])
        del predictor, ck
        gc.collect()
    report = audit('REMOTE_STATIC_PREFLIGHT.json', 'pass', plan_sha256=sha(ROOT / PLAN_NAME),
        source_job_states=states, case_count=103, input_and_reference_sha_checked=len(hashes), models=models,
        evaluation_versions=versions, execution_mode='sequential_in_process', preprocess_workers=0, export_workers=0,
        multiprocessing_queue_used=False, single_gpu=True, exclusive_node=False, deep_validated=521, locked_test_used=False)
    print(json.dumps(report))


def run_logged(command, path):
    with Path(path).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate_route(method, p):
    import nibabel as nib
    import numpy as np
    route = p['routes'][method]
    out = ROOT / 'cohort/fixed103/models' / method
    inf, ev = read(out / 'inference_audit.json'), read(out / 'evaluation/EVALUATION_VALIDATION.json')
    require(inf['status'] == ev['status'] == 'pass' and inf['prediction_count'] == inf['case_count'] == ev['case_count'] == 103, 'evaluation validation')
    require(inf['checkpoint_sha256'] == route['checkpoint_sha256'] and inf['trainer_name_from_checkpoint'] == TRAINER, 'inference checkpoint')
    require(inf['execution_mode'] == 'sequential_in_process' and inf['preprocess_workers'] == inf['export_workers'] == 0 and inf['multiprocessing_queue_used'] is False, 'inference execution contract')
    require((out / 'evaluation/EVALUATION_COMPLETE.ok').exists(), 'official MET complete')
    ids = case_ids()
    predictions = list((out / 'prediction').glob('*.nii.gz'))
    require(len(predictions) == 103 and {x.name.removesuffix('.nii.gz') for x in predictions} == ids, 'prediction cohort')
    for path in predictions:
        image = nib.load(path)
        reference = nib.load(Path(p['case_contract']['source_root']) / 'reference' / path.name)
        require(image.shape == reference.shape and np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5), 'prediction geometry: ' + path.name)
        labels = np.asanyarray(image.dataobj)
        require(bool(np.isfinite(labels).all()) and set(np.unique(labels)).issubset({0, 1, 2, 3}), 'prediction labels: ' + path.name)
    csv_path = out / 'evaluation/leaderboard_metrics.csv'
    require(sha(csv_path) == ev['metrics_csv_sha256'], 'official metrics SHA')
    metric_csv(csv_path, ids)
    require(sha(out / 'evaluation/panoptica_evaluation_summary.json') == ev['summary_json_sha256'], 'official summary SHA')
    return dict(route=method, selected_epoch=175, checkpoint_sha256=route['checkpoint_sha256'], case_count=103,
        prediction_count=103, metrics_csv_sha256=sha(csv_path), geometry_mismatch_count=0, prediction_nonfinite_count=0,
        execution_mode='sequential_in_process', preprocess_workers=0, export_workers=0, multiprocessing_queue_used=False,
        locked_test_used=False)


def run_route(method):
    import torch
    p = require_preflight()
    torch.set_num_threads(1)
    require(method in METHODS, 'unsupported route')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'single A100 required')
    launch = read(ROOT / f'evidence/{method}_EVALUATION_LAUNCH.json')
    require(launch['job_id'] == os.environ.get('SLURM_JOB_ID'), 'job launch identity')
    out = ROOT / 'cohort/fixed103/models' / method
    for suffix in ('STARTED.json', 'FAILED.json', 'VALIDATION.json', 'COMPLETE.ok'):
        require(not (ROOT / f'evidence/{method}_EVALUATION_{suffix}').exists(), 'route already attempted')
    require(not out.exists(), 'route output already exists')
    ck = ROOT / 'model_snapshots' / method / 'fold_0/checkpoint_final.pth'
    require(sha(ck) == p['routes'][method]['checkpoint_sha256'], 'frozen175 checkpoint SHA')
    audit(method + '_EVALUATION_STARTED.json', 'started', route=method, selected_epoch=175,
          checkpoint_sha256=sha(ck), device_name=torch.cuda.get_device_name(0), single_gpu=True,
          execution_mode='sequential_in_process', preprocess_workers=0, export_workers=0, multiprocessing_queue_used=False,
          plan_sha256=sha(ROOT / PLAN_NAME), case_count=103, exclusive_node=False)
    out.mkdir(parents=True, exist_ok=False)
    cohort = Path(p['case_contract']['source_root'])
    job = os.environ['SLURM_JOB_ID']
    run_logged([PYTHON, str(ROOT / 'runtime/frozen_s2_inference.py'), '--input', str(cohort / 'input'), '--output', str(out / 'prediction'),
        '--model-root', str(ck.parents[1]), '--audit-json', str(out / 'inference_audit.json'), '--expected-count', '103',
        '--expected-checkpoint-sha256', p['routes'][method]['checkpoint_sha256'], '--model-name', method + '_175',
        '--execution-mode', 'sequential_in_process'], ROOT / f'logs/{method}_inference_{job}.log')
    run_logged(['bash', str(ROOT / 'runtime/run_official_met_eval.sh'), EVAL_ENV, str(out / 'prediction'), str(cohort / 'reference'), str(out / 'evaluation'), '103'], ROOT / f'logs/{method}_official_met_{job}.log')
    audit(method + '_EVALUATION_VALIDATION.json', 'pass', **validate_route(method, p))
    complete(method + '_EVALUATION_COMPLETE.ok')


def paired():
    p = require_preflight()
    launch = read(ROOT / 'evidence/PAIRED_COMPARISON_LAUNCH.json')
    require(launch['job_id'] == os.environ.get('SLURM_JOB_ID'), 'paired job identity')
    require(not (ROOT / 'cohort/fixed103/comparisons').exists(), 'comparison already attempted')
    for method in METHODS:
        val = read(ROOT / f'evidence/{method}_EVALUATION_VALIDATION.json')
        require(val['status'] == 'pass' and val['selected_epoch'] == 175 and val['checkpoint_sha256'] == p['routes'][method]['checkpoint_sha256'], 'route validation prerequisite')
        require((ROOT / f'evidence/{method}_EVALUATION_COMPLETE.ok').exists() and not (ROOT / f'evidence/{method}_EVALUATION_FAILED.json').exists(), 'route completion prerequisite')
        require(sha(p['routes'][method]['baseline_metrics_csv']) == p['routes'][method]['baseline_metrics_sha256'], 'baseline CSV changed')
    audit('PAIRED_COMPARISON_STARTED.json', 'started', plan_sha256=sha(ROOT / PLAN_NAME))
    csv175 = lambda m: str(ROOT / f'cohort/fixed103/models/{m}/evaluation/leaderboard_metrics.csv')
    groups = [('epoch175_three_routes', [(m, csv175(m)) for m in METHODS])]
    groups += [(m + '_175_vs_125', [(m + '_175', csv175(m)), (m + '_125', p['routes'][m]['baseline_metrics_csv'])]) for m in METHODS]
    reports = {}
    for name, inputs in groups:
        out = ROOT / 'cohort/fixed103/comparisons' / name
        command = [PYTHON, str(ROOT / 'runtime/compare_paired_met.py')]
        for method, path in inputs:
            command += ['--input', method + '=' + path]
        command += ['--expected-methods', str(len(inputs)), '--expected-count', '103', '--bootstrap', '20000', '--seed', '20260831', '--output-root', str(out), '--scope', name + '_exploratory_fixed103']
        run_logged(command, ROOT / f'logs/{name}_paired_{os.environ["SLURM_JOB_ID"]}.log')
        result = read(out / 'PAIRED_COMPARISON.json')
        require(result['status'] == 'pass' and result['case_count'] == 103 and set(result['methods']) == {x[0] for x in inputs}, 'paired cohort/methods')
        expected_pairs = 3 if len(inputs) == 3 else 1
        require(len(result['pairs']) == expected_pairs and result['bootstrap'] == p['bootstrap'], 'paired contract')
        require(result['metric_count'] == 72, 'paired metrics')
        for pair in result['pairs']:
            require(len(pair['metrics']) == 72, 'paired metric records')
            for row in pair['metrics']:
                n = row['paired_complete_case_count']
                require(n + row['excluded_nonfinite_pair_count'] == 103, 'paired exclusions')
                if row['status'] == 'available':
                    require(row['wins_a'] + row['ties'] + row['wins_b'] == n, 'paired WTL')
        reports[name] = dict(pair_count=expected_pairs, method_count=len(inputs), sha256=sha(out / 'PAIRED_COMPARISON.json'))
    audit('PAIRED_COMPARISON_VALIDATION.json', 'pass', comparisons=reports, total_pair_count=6,
          case_count=103, bootstrap=p['bootstrap'], locked_test_used=False)
    complete('PAIRED_COMPARISON_COMPLETE.ok')
    complete('EPOCH175_OFFICIAL_MET_PIPELINE_COMPLETE.ok')


def launch(expected_sha):
    p = require_preflight()
    require(sha(ROOT / PLAN_NAME) == expected_sha, 'launch plan SHA')
    require(not any((ROOT / 'evidence').glob('*RESERVATION*')), 'reservation already exists; never resubmit')
    audit('PIPELINE_SUBMISSION_RESERVATION.json', 'reserved', plan_sha256=expected_sha)
    launched = {}
    for method in (*METHODS, 'PAIRED'):
        name = p['job_names'][method]
        tag = method + '_EVALUATION' if method != 'PAIRED' else 'PAIRED_COMPARISON'
        dependency = None if method != 'PAIRED' else 'afterok:' + ':'.join(launched[m] for m in METHODS)
        audit(tag + '_SUBMISSION_RESERVATION.json', 'reserved', job_name=name, plan_sha256=expected_sha, dependency=dependency)
        command = ['sbatch', '--hold', '--parsable', '--job-name=' + name]
        if dependency:
            command += ['--dependency=' + dependency]
        command += [str(ROOT / 'slurm' / ('02_paired.slurm' if method == 'PAIRED' else '01_evaluate.slurm'))]
        if method != 'PAIRED':
            command += [method]
        response = subprocess.run(command, capture_output=True, text=True, check=True)
        job = response.stdout.strip().split(';')[0]
        require(job.isdigit(), 'unrecognized sbatch response; inspect reservation, do not retry')
        launched[method] = job
        audit(tag + '_LAUNCH.json', 'submitted', job_id=job, job_name=name, plan_sha256=expected_sha, dependency=dependency)
    result = audit('LAUNCH_SUMMARY.json', 'submitted', jobs=launched, job_names=p['job_names'], plan_sha256=expected_sha,
                   parallel_gpu_routes=True, single_gpu_per_route=True, paired_dependency=dependency)
    # Release only after all immutable launch records exist, including the CPU dependency.
    subprocess.run(['scontrol', 'release', ','.join(launched.values())], check=True)
    audit('LAUNCH_RELEASED.json', 'released', jobs=launched, plan_sha256=expected_sha)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'launch', 'route', 'paired'))
    parser.add_argument('--route', choices=METHODS)
    parser.add_argument('--plan-sha')
    args = parser.parse_args()
    failure_name = {'preflight': 'REMOTE_STATIC_PREFLIGHT_FAILED.json', 'launch': 'LAUNCH_FAILED.json',
                    'route': f'{args.route}_EVALUATION_FAILED.json', 'paired': 'PAIRED_COMPARISON_FAILED.json'}[args.action]
    try:
        if args.action == 'preflight':
            require(bool(args.plan_sha), 'expected plan SHA required')
            preflight(args.plan_sha)
        elif args.action == 'launch':
            launch(args.plan_sha)
        elif args.action == 'route':
            run_route(args.route)
        else:
            paired()
    except Exception as exc:
        if ROOT == EXPECTED_ROOT and not (ROOT / 'evidence' / failure_name).exists():
            audit(failure_name, 'failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
