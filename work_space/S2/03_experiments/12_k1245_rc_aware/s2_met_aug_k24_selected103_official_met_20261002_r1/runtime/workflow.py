"""Frozen K2/K4 checkpoint inference and official MET acceptance."""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/public/home/zqchen/.conda/envs/segmamba/bin/python'
EVAL_ENV = '/public/home/zqchen/.conda/envs/brats_eval'
TRAINER = 'nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def evidence(name, status, **values):
    payload = dict(META, status=status, remote_utc=datetime.now(timezone.utc).isoformat(),
                   hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'))
    payload.update(values)
    write(ROOT / 'evidence' / name, payload)
    return payload


def static_check():
    plan = read(ROOT / 'EXECUTION_PLAN.json')
    require(str(ROOT) == plan['remote_root'], 'root identity drift')
    require(all(plan[k] == v for k, v in META.items()), 'artifact status drift')
    require(plan['routes'].keys() == {'K2', 'K4'}, 'route drift')
    for relative, digest in read(ROOT / 'STATIC_SHA256.json')['files'].items():
        require(sha(ROOT / relative) == digest, 'static SHA drift: ' + relative)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'bytecode contract')
    return plan


def case_rows():
    with (ROOT / 'manifests/FIXED103_REAL_T2W_CASES.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 103 and len({r['source_case_id'] for r in rows}) == 103, 'fixed103 manifest')
    require(all(r['source_t2w_allowed'] == 'True' and r['t2w_status'] == 'authentic' for r in rows), 'real T2W contract')
    return rows


def preflight():
    import torch
    import nnunetv2.inference.predict_from_raw_data as module
    from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import nnUNetTrainerBraTS2026RC
    plan = static_check()
    require(not (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json').exists(), 'preflight already attempted')
    torch.set_num_threads(2)
    for path in (ROOT / 'runtime').rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('frozen_k24', ROOT / 'runtime/inference.py')
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    cohort = Path(plan['case_contract']['source_root'])
    rows = case_rows()
    ids = {r['source_case_id'] for r in rows}
    for name, digest in plan['case_contract']['audit_sha256'].items():
        require(sha(cohort / name) == digest, 'input audit SHA drift')
    input_audit = read(cohort / 'input_audit.json')
    require(input_audit['status'] == 'pass' and input_audit['case_count'] == 103, 'shared input audit')
    require(input_audit['source_t2w_used'] is True, 'authentic T2W input')
    with (cohort / 'input_audit.csv').open(newline='') as f:
        input_rows = list(csv.DictReader(f))
    require(len(input_rows) == 103 and {r['source_case_id'] for r in input_rows} == ids, 'input cohort drift')
    hashes = {}
    for row in input_rows:
        for index, modality in enumerate(('t1n', 't1c', 't2w', 't2f', 'seg')):
            file = cohort / ('reference' if modality == 'seg' else 'input') / (row['source_case_id'] + ('.nii.gz' if modality == 'seg' else f'_{index:04d}.nii.gz'))
            hashes[str(file)] = sha(file)
            require(hashes[str(file)] == row[modality + '_sha256'], 'input/reference SHA drift: ' + file.name)
    require(len(list((cohort / 'input').glob('*.nii.gz'))) == 412, 'input count')
    require({p.name.removesuffix('.nii.gz') for p in (cohort / 'reference').glob('*.nii.gz')} == ids, 'reference count')
    split_root = Path(plan['training_split_root'])
    def split_file(primary, fallback):
        p = split_root / primary
        return p if p.is_file() else split_root / fallback
    train_file = split_file('train_fixed.txt', 'train_full.txt')
    val_file = split_file('val_fixed.txt', 'val_full.txt')
    train = train_file.read_text().split()
    val = val_file.read_text().split()
    require(len(train) == len(set(train)) == 1035 and len(val) == len(set(val)) == 103, 'training split count')
    require(not set(train) & set(val), 'training split overlap')
    require(set(val) == {r['nnunet_case_id'] for r in rows}, 'selection/report split mismatch')
    write(ROOT / 'manifests/INPUT_REFERENCE_SHA256.json', dict(META, files=hashes,
          train_split_sha256=sha(train_file),val_split_sha256=sha(val_file)))
    env_code = "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ('BraTS-evaluation','panoptica','numpy')}))"
    versions = json.loads(subprocess.check_output([EVAL_ENV + '/bin/python', '-c', env_code], text=True))
    require(versions == {'BraTS-evaluation':'0.0.8','panoptica':'2.1.0','numpy':'1.26.4'}, 'official evaluator version drift')
    for name in ('brats-evaluate', 'brats-parse-metrics'):
        require(os.access(EVAL_ENV + '/bin/' + name, os.X_OK), 'missing official evaluator')
    models = {}
    for route, r in plan['routes'].items():
        root = Path(r['training_root'])
        schedule = subprocess.check_output(['sacct','-n','-P','-X','-j',r['training_job_id'],'-o','JobIDRaw,State,ExitCode'], text=True)
        require(any(x.split('|')[:3] == [r['training_job_id'],'COMPLETED','0:0'] for x in schedule.splitlines()), 'training scheduler drift')
        require(not (root / f'evidence/{route}_175_TRAINING_FAILED.json').exists(), 'training failure')
        require((root / f'evidence/{route}_175_TRAINING_COMPLETE.ok').is_file(), 'training complete absent')
        valpath = root / f'evidence/{route}_175_TRAINING_VALIDATION.json'
        require(sha(valpath) == r['training_validation_sha256'], 'training validation SHA drift')
        val = read(valpath)
        require(val['status'] == 'pass' and val['completed_epochs'] == 175, 'training validation')
        best = sorted(val['candidate_checkpoints'], key=lambda x:(-x['ema_fg_dice'],x['epoch']))[0]
        require(best['epoch'] == r['selected_epoch'] and best['sha256'] == r['checkpoint_sha256'], 'checkpoint selection drift')
        source = Path(r['checkpoint'])
        require(sha(source) == r['checkpoint_sha256'], 'checkpoint SHA drift')
        ck = torch.load(source, map_location='cpu', weights_only=False)
        require(ck['current_epoch'] == r['selected_epoch'] and ck['trainer_name'] == TRAINER, 'selected epoch/trainer')
        model = ROOT / 'model_snapshots' / route
        (model / 'fold_0').mkdir(parents=True, exist_ok=False)
        target = model / 'fold_0/checkpoint_final.pth'
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1048576)
        target.chmod(0o440)
        require(sha(target) == r['checkpoint_sha256'], 'model snapshot SHA drift')
        write(model / 'plans.json', ck['init_args']['plans'])
        write(model / 'dataset.json', ck['init_args']['dataset_json'])
        dataset = read(model / 'dataset.json')
        require(dataset['labels'] == {'background':0,'NETC':1,'SNFH':2,'ET':3,'RC':4}, 'dataset labels drift')
        def finder(name):
            require(name == TRAINER, 'unexpected checkpoint trainer')
            return nnUNetTrainerBraTS2026RC
        module.recursive_find_trainer_class_by_name = finder
        predictor = module.nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(model), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
        models[route] = dict(selected_epoch=r['selected_epoch'], checkpoint_sha256=sha(target), cpu_network_strict_load=True)
        evidence(route + '_MODEL_SNAPSHOT.json', 'pass', **models[route])
        del ck, predictor
        gc.collect()
    subprocess.run(['bash','-n',str(ROOT / 'slurm/evaluate.slurm')], check=True)
    print(json.dumps(evidence('REMOTE_STATIC_PREFLIGHT.json', 'pass', plan_sha256=sha(ROOT/'EXECUTION_PLAN.json'),
          input_reference_sha_checked=len(hashes),case_count=103,models=models,evaluation_versions=versions,
          fixed103_matches_training_split=True,preprocess_workers=0,export_workers=0,multiprocessing_queue_used=False)))


def run_route(route):
    import nibabel as nib
    import numpy as np
    import torch
    plan = static_check()
    pre = read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')
    require(pre['status'] == 'pass' and pre['plan_sha256'] == sha(ROOT/'EXECUTION_PLAN.json'), 'preflight prerequisite')
    require(not (ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json').exists(), 'failed preflight')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'one A100 required')
    job = os.environ['SLURM_JOB_ID']
    launch = read(ROOT / f'evidence/{route}_LAUNCH.json')
    require(launch['job_id'] == job, 'launch identity drift')
    r = plan['routes'][route]
    cohort = Path(plan['case_contract']['source_root'])
    model = ROOT / 'model_snapshots' / route
    require(sha(model / 'fold_0/checkpoint_final.pth') == r['checkpoint_sha256'], 'frozen checkpoint drift')
    out = ROOT / 'cohort/fixed103/models' / route
    require(not out.exists(), 'exclusive output already exists')
    out.mkdir(parents=True, exist_ok=False)
    evidence(route+'_STARTED.json','started',route=route,selected_epoch=r['selected_epoch'],checkpoint_sha256=r['checkpoint_sha256'],
             device_name=torch.cuda.get_device_name(0),single_gpu=True,exclusive_node=False,case_count=103,
             preprocess_workers=0,export_workers=0,multiprocessing_queue_used=False)
    argv = [PYTHON,str(ROOT/'runtime/inference.py'),'--input',str(cohort/'input'),'--output',str(out/'prediction'),
            '--model-root',str(model),'--audit-json',str(out/'inference_audit.json'),'--expected-count','103',
            '--expected-checkpoint-sha256',r['checkpoint_sha256'],'--model-name',route,'--execution-mode','sequential_in_process']
    with (ROOT/f'logs/{route}_inference_{job}.log').open('x') as f:
        subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True)
    inf=read(out/'inference_audit.json')
    require(inf['status']=='pass' and inf['prediction_count']==inf['case_count']==103, 'inference validation')
    require(inf['checkpoint_sha256']==r['checkpoint_sha256'] and inf['trainer_name_from_checkpoint']==TRAINER, 'inference checkpoint')
    require(inf['preprocess_workers']==inf['export_workers']==0 and inf['multiprocessing_queue_used'] is False, 'inference worker contract')
    ids={r['source_case_id'] for r in case_rows()}
    paths=list((out/'prediction').glob('*.nii.gz'))
    require(len(paths)==103 and {p.name.removesuffix('.nii.gz') for p in paths}==ids, 'prediction cases')
    prediction_hashes={}
    labels=set(read(model/'dataset.json')['labels'].values())
    for p in paths:
        image=nib.load(p); ref=nib.load(cohort/'reference'/p.name); data=np.asanyarray(image.dataobj)
        require(image.shape==ref.shape and np.allclose(image.affine,ref.affine,rtol=0,atol=1e-5), 'prediction geometry')
        require(bool(np.isfinite(data).all()) and set(np.unique(data)).issubset(labels), 'prediction finite/labels')
        prediction_hashes[p.name]=sha(p)
    write(out/'PREDICTION_SHA256.json',dict(META,files=prediction_hashes,allowed_labels=sorted(labels),status='pass'))
    with (ROOT/f'logs/{route}_official_met_{job}.log').open('x') as f:
        subprocess.run(['bash',str(ROOT/'runtime/run_official_met_eval.sh'),EVAL_ENV,str(out/'prediction'),str(cohort/'reference'),str(out/'evaluation'),'103'],stdout=f,stderr=subprocess.STDOUT,check=True)
    ev=read(out/'evaluation/EVALUATION_VALIDATION.json')
    require(ev['status']=='pass' and ev['case_count']==103 and ev['metric_column_count']==72, 'official MET validation')
    metrics=out/'evaluation/leaderboard_metrics.csv'
    require(sha(metrics)==ev['metrics_csv_sha256'], 'MET CSV SHA drift')
    with metrics.open(newline='') as f: rows=list(csv.DictReader(f))
    require(len(rows)==106 and {r['subject_id'].removesuffix('.nii.gz') for r in rows[:103]}==ids, 'official MET cohort')
    require([r['subject_id'] for r in rows[-3:]]==['mean','std','median'], 'official MET summary rows')
    evidence(route+'_VALIDATION.json','pass',route=route,case_count=103,prediction_count=103,selected_epoch=r['selected_epoch'],
             checkpoint_sha256=r['checkpoint_sha256'],metrics_csv_sha256=sha(metrics),metric_count=72,allowed_labels=sorted(labels))
    write(ROOT/f'evidence/{route}_COMPLETE.ok',dict(META,status='pass',job_id=job))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('preflight','route'))
    parser.add_argument('--route',choices=('K2','K4'))
    args=parser.parse_args()
    try:
        if args.action=='preflight': preflight()
        else: run_route(args.route)
    except Exception as exc:
        name='REMOTE_STATIC_PREFLIGHT_FAILED.json' if args.action=='preflight' else args.route+'_FAILED.json'
        if not (ROOT/'evidence'/name).exists(): evidence(name,'failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__': main()
