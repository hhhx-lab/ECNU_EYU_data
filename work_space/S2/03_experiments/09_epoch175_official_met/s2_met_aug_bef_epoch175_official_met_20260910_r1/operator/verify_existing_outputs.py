"""Read-only salvage verification. Never changes failed Slurm jobs to successful jobs."""
import base64
import csv
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1')
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def label_values(value):
    if isinstance(value, int):
        return {value}
    require(isinstance(value, (list, tuple)), 'invalid dataset label specification')
    result = set()
    for entry in value:
        result |= label_values(entry)
    return result


def main():
    plan_path = ROOT / 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'
    require(sha(plan_path) == '11e5dd46ceeaceb57594976a9e271a99e590a1cf692d8f369a47a969852b645e', 'plan drift')
    subprocess.run(['sha256sum', '-c', 'STATIC_SHA256SUMS.txt'], cwd=ROOT, stdout=subprocess.PIPE, check=True)
    plan = read(plan_path)
    require(read(ROOT / 'evidence/REMOTE_STATIC_PREFLIGHT.json')['status'] == 'pass', 'original preflight')
    with (ROOT / 'manifests/FIXED103_REAL_T2W_CASES.csv').open(newline='') as f:
        ids = {row['source_case_id'] for row in csv.DictReader(f)}
    require(len(ids) == 103, 'cohort size')
    cohort = Path(plan['case_contract']['source_root'])
    report = dict(META, status='verified_existing_outputs_only', recorded_remote_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), original_jobs_remain_failed=True, original_pipeline_remains_failed=True,
        new_jobs_submitted=False, paired_not_yet_run=True, routes={})
    for method in ('B', 'E_SNFH', 'F'):
        source = plan['routes'][method]
        failure = read(ROOT / f'evidence/{method}_EVALUATION_FAILED.json')
        require(failure['status'] == 'failed' and failure['error_type'] == 'RuntimeError' and failure['error'].startswith('prediction labels:'), method + ' different failure')
        ck = ROOT / f'model_snapshots/{method}/fold_0/checkpoint_final.pth'
        require(sha(ck) == source['checkpoint_sha256'], method + ' checkpoint SHA')
        dataset = read(ck.parents[1] / 'dataset.json')
        require(dataset == read(Path(source['baseline_model_root']) / 'dataset.json'), method + ' baseline dataset mismatch')
        allowed = set()
        for value in dataset['labels'].values():
            allowed |= label_values(value)
        require(allowed == {0, 1, 2, 3, 4}, method + ' unexpected label ontology')
        out = ROOT / f'cohort/fixed103/models/{method}'
        inf = read(out / 'inference_audit.json')
        ev = read(out / 'evaluation/EVALUATION_VALIDATION.json')
        require(inf['status'] == ev['status'] == 'pass' and inf['case_count'] == inf['prediction_count'] == ev['case_count'] == 103, method + ' inference/evaluation validation')
        require(inf['checkpoint_sha256'] == source['checkpoint_sha256'], method + ' inference checkpoint')
        require(inf['execution_mode'] == 'sequential_in_process' and inf['preprocess_workers'] == inf['export_workers'] == 0 and inf['multiprocessing_queue_used'] is False, method + ' inference contract')
        require((out / 'evaluation/EVALUATION_COMPLETE.ok').is_file(), method + ' official evaluation complete')
        predictions = list((out / 'prediction').glob('*.nii.gz'))
        require(len(predictions) == 103 and {p.name.removesuffix('.nii.gz') for p in predictions} == ids, method + ' prediction cohort')
        observed, with_label4 = set(), 0
        for path in predictions:
            image, reference = nib.load(path), nib.load(cohort / 'reference' / path.name)
            require(image.shape == reference.shape and np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5), method + ' geometry ' + path.name)
            values = np.asanyarray(image.dataobj)
            require(bool(np.isfinite(values).all()), method + ' nonfinite prediction')
            labels = set(np.unique(values).tolist())
            require(labels <= allowed, method + ' illegal prediction labels')
            observed |= labels
            with_label4 += int(4 in labels)
        csv_path = out / 'evaluation/leaderboard_metrics.csv'
        require(sha(csv_path) == ev['metrics_csv_sha256'], method + ' MET CSV hash')
        summary_path = out / 'evaluation/panoptica_evaluation_summary.json'
        require(sha(summary_path) == ev['summary_json_sha256'], method + ' MET summary hash')
        summary = read(summary_path)
        require(summary['missings'] == [] and len(summary['metrics']) == 103 and not any('error' in item for item in summary['metrics']), method + ' MET subject coverage')
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames
        require(len(rows) == 106 and len(fields) == 73 and [r['subject_id'] for r in rows[-3:]] == ['mean', 'std', 'median'], method + ' CSV shape')
        require({r['subject_id'].removesuffix('.nii.gz') for r in rows[:103]} == ids, method + ' CSV cohort')
        require(sha(source['baseline_metrics_csv']) == source['baseline_metrics_sha256'], method + ' original125 metrics drift')
        means = {}
        for key, value in rows[-3].items():
            if key == 'subject_id':
                continue
            number = float(value) if value else float('nan')
            means[key] = number if math.isfinite(number) else None
        report['routes'][method] = dict(original_failed_job_id=failure['job_id'], selected_epoch=175, prediction_count=103,
            allowed_labels=sorted(allowed), observed_labels=sorted(observed), cases_with_label4=with_label4,
            geometry_mismatch_count=0, nonfinite_prediction_count=0, official_met_status='pass', checkpoint_sha256=source['checkpoint_sha256'],
            metrics_csv_sha256=sha(csv_path), source_failure_sha256=sha(ROOT / f'evidence/{method}_EVALUATION_FAILED.json'), means=means)
    files, total = [], 0
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink() or any(part in rel.parts for part in ('operator', 'model_snapshots', 'prediction', 'input', 'reference', '__pycache__')):
            continue
        if path.suffix.lower() not in ('.json', '.csv', '.md', '.txt', '.log', '.err', '.out', '.ok'):
            continue
        data = path.read_bytes()
        total += len(data)
        require(len(data) < 16 * 1024**2 and total < 100 * 1024**2, 'large lightweight artifact')
        files.append(dict(relative_path=str(rel), sha256=sha(path), size=len(data), content_base64=base64.b64encode(data).decode('ascii')))
    report.update(remote_completed_utc=datetime.now(timezone.utc).isoformat(), light_file_count=len(files), light_total_bytes=total)
    return dict(report=report, files=files)


try:
    result = main()
except Exception as exc:
    result = dict(report=dict(META, status='verification_failed', error_type=type(exc).__name__, error=str(exc)), files=[])
print('OUTPUT_AUDIT_BEGIN')
print(json.dumps(result, allow_nan=False))
print('OUTPUT_AUDIT_END')
