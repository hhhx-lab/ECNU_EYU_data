"""Exact historical MET replay, three-model metrics and patient-group paired summaries."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')


def metrics(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f); rows = list(reader); fields = reader.fieldnames
    assert fields[0] == 'subject_id' and len(fields) == 73, 'MET schema drift'
    assert len(rows) == 106 and [r['subject_id'] for r in rows[-3:]] == ['mean', 'std', 'median']
    data = rows[:-3]
    assert len({r['subject_id'] for r in data}) == 103
    data = sorted(data, key=lambda r:r['subject_id'])
    values = np.array([[float(r[k]) if r[k] else np.nan for k in fields[1:]] for r in data])
    return fields[1:], [r['subject_id'] for r in data], values


def validate_replay(current, historical):
    fields, names, values = metrics(current)
    old_fields, old_names, old_values = metrics(historical)
    assert fields == old_fields and names == old_names, 'historical schema/case drift'
    assert np.array_equal(np.isnan(values), np.isnan(old_values)), 'historical NA drift'
    assert np.allclose(values, old_values, atol=1e-12, rtol=1e-9, equal_nan=True), 'historical MET replay mismatch'
    finite = np.isfinite(values) & np.isfinite(old_values)
    return dict(pass_per_case_72_metrics=True, atol=1e-12, rtol=1e-9,
                max_absolute_difference=float(np.max(np.abs(values[finite]-old_values[finite]))), cases=103)


def finite_mean(values):
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def direction(field):
    if '_std_' in field: return 0
    if any(token in field for token in ('_fp_', '_fn_', '_hd95_')): return -1
    return 1


def paired_bootstrap(delta, groups, rng, iterations=20000):
    labels = sorted(set(groups))
    sums = np.array([delta[np.array(groups) == g].sum() for g in labels])
    counts = np.array([sum(x == g for x in groups) for g in labels])
    draws = rng.multinomial(len(labels), np.repeat(1/len(labels), len(labels)), size=iterations)
    scan_means = (draws @ sums) / (draws @ counts)
    patient_means = (draws @ (sums/counts)) / len(labels)
    return dict(patient_groups=len(labels), ci_low=float(np.quantile(scan_means,.025)), ci_high=float(np.quantile(scan_means,.975)),
        patient_equal_delta=float((sums/counts).mean()), patient_equal_ci_low=float(np.quantile(patient_means,.025)),
        patient_equal_ci_high=float(np.quantile(patient_means,.975)))


def csv_write(path, rows):
    assert rows
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize(root, contract):
    loaded = {name:metrics(root / 'evaluation' / name / 'leaderboard_metrics.csv') for name in contract['models']}
    fields, names, _ = next(iter(loaded.values()))
    assert all(item[:2] == (fields, names) for item in loaded.values()), 'three-model cases/schema drift'
    assert set(names) == set(contract['patient_groups']) and len(set(contract['patient_groups'].values())) == 69
    means = []
    for model, (_, _, values) in loaded.items():
        for index, field in enumerate(fields):
            means.append(dict(model=model, metric=field, mean=finite_mean(values[:,index]), finite_cases=int(np.isfinite(values[:,index]).sum())))
    csv_write(root / 'reports/MEANS_216.csv', means)
    pairs = []; rng = np.random.default_rng(20260831)
    for first, second in itertools.combinations(loaded, 2):
        a, b = loaded[first][2], loaded[second][2]
        for index, field in enumerate(fields):
            valid = np.isfinite(a[:,index]) & np.isfinite(b[:,index]); d = a[valid,index] - b[valid,index]
            sign = direction(field); directional = d * sign if sign else d
            groups = [contract['patient_groups'][name] for name, use in zip(names, valid) if use]
            item = dict(first=first, second=second, metric=field, first_mean=finite_mean(a[:,index]), second_mean=finite_mean(b[:,index]),
                common_finite_cases=len(d), first_common_mean=finite_mean(a[valid,index]), second_common_mean=finite_mean(b[valid,index]),
                raw_delta=finite_mean(d), direction='higher' if sign == 1 else 'lower' if sign == -1 else 'descriptive_std',
                oriented_delta=finite_mean(directional), wins=int((directional>1e-12).sum()), ties=int((np.abs(directional)<=1e-12).sum()),
                losses=int((directional < -1e-12).sum()), bootstrap_iterations=20000, seed=20260831)
            if len(d): item.update(paired_bootstrap(directional, groups, rng))
            else: item.update(patient_groups=0,ci_low=None,ci_high=None,patient_equal_delta=None,patient_equal_ci_low=None,patient_equal_ci_high=None)
            pairs.append(item)
    csv_write(root / 'reports/PAIRED_ALL_216.csv', pairs)
    lookup = {(r['model'],r['metric']):r for r in means}
    lines = ['# Same-fixed103 descriptive comparison', '',
        '103 validation scans / 69 patient groups. Not locked test or independent generalization.',
        'BraTS-evaluation 0.0.8, panoptica 2.1.0, NumPy 1.26.4; mets / 27 mm^3 / parser DSC 0.2.',
        'TC and WT exclude RC. K1 epoch175 and original E epoch200 historical replay are mandatory.', '',
        'K1: development/selection cohort. EXT2025: public training overlap unknown.',
        'Bio-Intel is a separate submitted inference and is NOT in this completed comparison.', '',
        'All 72 metrics are retained. NA is not zero. n varies by metric and model.',
        'Paired CSV uses common-finite cases; 20,000 patient-group resamples, seed20260831.',
        'CI is descriptive, not a formal superiority claim; std columns have no better/worse direction.', '']
    models = list(loaded)
    for region in ('et','rc','tc','wt'):
        lines += ['## '+region.upper(), '', '| Metric | '+' | '.join(models)+' |', '|---|'+'---|'*len(models)]
        for field in fields:
            if not field.endswith('_'+region): continue
            cells = []
            for model in models:
                item = lookup[model,field]
                cells.append('NA (n=0)' if item['mean'] is None else f"{item['mean']:.6f} (n={item['finite_cases']})")
            lines.append('| '+field+' | '+' | '.join(cells)+' |')
        lines.append('')
    with (root / 'reports/COMPARISON_REPORT.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    return means


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    root = args.root; stage = 'contract'; started = datetime.now(timezone.utc).isoformat()
    try:
        contract_path = root / 'contract/COMPARISON_CONTRACT.json'
        assert sha(contract_path) == os.environ['COMPARE_EXPECTED_CONTRACT_SHA256'], 'comparison contract drift'
        contract = json.loads(contract_path.read_text())
        assert contract['attempt'] == str(root) and contract['cases'] == 103
        for name, expected in json.loads((root / 'contract/PACKAGE_SHA256.json').read_text())['files'].items():
            assert sha(root / name) == expected, 'package drift: '+name
        assert sys.version.split()[0] == '3.10.18'
        assert {n:metadata.version(n) for n in contract['evaluator']['versions']} == contract['evaluator']['versions']
        import brats_evaluation, panoptica
        source = {}
        for name, module in [('brats_evaluation',brats_evaluation),('panoptica',panoptica)]:
            base = Path(module.__file__).parent
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.py','.yaml','.yml'): source[name+'/'+str(path.relative_to(base))] = sha(path)
        assert source == contract['evaluator']['source_sha256'], 'evaluator source drift'
        write(root / 'reports/STARTED.json',dict(META,job_id=os.environ['SLURM_JOB_ID'],started_utc=started,contract_sha256=sha(contract_path)))
        stage = 'native_input_validation'
        import nibabel as nib
        ref_dir = root / 'stage/reference'; ref_dir.mkdir()
        identities = {}
        for name, entry in contract['references'].items():
            path = Path(entry['path']); assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            image = nib.load(str(path)); data = np.asanyarray(image.dataobj)
            assert data.ndim == 3 and np.isfinite(data).all() and set(np.unique(data)) <= {0,1,2,3,4}
            identities[name] = (image.shape,image.affine.copy()); (ref_dir / name).symlink_to(path)
        for model, files in contract['models'].items():
            assert set(files) == set(identities), 'model case identity drift'
            pred_dir = root / 'stage' / model; pred_dir.mkdir()
            for name, entry in files.items():
                path = Path(entry['path']); assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
                image = nib.load(str(path)); data = np.asanyarray(image.dataobj)
                shape, affine = identities[name]
                assert image.shape == shape and np.allclose(image.affine, affine, rtol=0, atol=1e-5), 'native geometry drift'
                assert np.isfinite(data).all() and set(np.unique(data)) <= {0,1,2,3,4}, 'illegal prediction labels'
                (pred_dir / name).symlink_to(path)
            stage = 'official_MET_'+model
            with (root / 'logs' / (model+'_evaluation.log')).open('x') as log:
                subprocess.run(['bash',str(root / 'runtime/run_official_met_eval.sh'),contract['evaluator_prefix'],str(pred_dir),str(ref_dir),str(root / 'evaluation' / model),'103'],stdout=log,stderr=subprocess.STDOUT,check=True)
            write(root / 'reports' / ('PROGRESS_'+model+'.json'),dict(META,model=model,cases=103,remote_utc=datetime.now(timezone.utc).isoformat()))
        stage = 'historical_replay_gate'
        replay = {}
        for model, entry in contract['historical'].items():
            path = Path(entry['path']); assert sha(path) == entry['sha256']
            replay[model] = validate_replay(root / 'evaluation' / model / 'leaderboard_metrics.csv',path)
        write(root / 'reports/HISTORICAL_REPLAY.json', dict(META,models=replay))
        stage = 'paired_summary'
        summarize(root,contract)
        output = {}
        for folder in ('reports','evaluation','logs'):
            for path in sorted((root / folder).rglob('*')):
                if path.is_file(): output[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,sha256=sha(path))
        write(root / 'reports/COMPLETE.json',dict(META,status='three_model_fixed103_MET_replay_and_descriptive_comparison_complete',
            job_id=os.environ['SLURM_JOB_ID'],cases=103,models=list(contract['models']),metric_columns=72,outputs=output,
            contract_sha256=sha(contract_path),started_utc=started,remote_utc=datetime.now(timezone.utc).isoformat(),historical_replay=replay,
            bootstrap_seed=20260831,bootstrap_iterations=20000,locked_test_read=False,scientific_superiority_pass=False))
    except BaseException as error:
        write(root / 'reports/FAILED.json',dict(META,stage=stage,error_type=type(error).__name__,message=str(error),remote_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
        raise


if __name__ == '__main__': main()
