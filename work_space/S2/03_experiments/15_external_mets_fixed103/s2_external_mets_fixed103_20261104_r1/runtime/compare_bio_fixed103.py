"""Score Bio once and append comparisons without recomputing completed results."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_fixed103 import META, sha, write, metrics, finite_mean, direction, paired_bootstrap, csv_write

BIO = 'EXT2026_BIO_INTEL_FULL'
OLD = ('EXT2025', 'K1_175', 'original_E_200')


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def check_file(entry):
    path = Path(entry['path'])
    assert path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'input SHA/bytes drift: ' + str(path)
    return path


def validate_contract(root, expected):
    contract_path = root / 'contract/COMPARISON_CONTRACT.json'
    assert sha(contract_path) == expected, 'Bio MET contract drift'
    contract = json.loads(contract_path.read_text())
    assert contract['attempt'] == str(root) and contract['cases'] == 103
    assert contract['model_order'] == [*OLD, BIO] and set(contract['models']) == set(contract['model_order'])
    assert set(contract['references']) == set(contract['patient_groups']) and len(contract['references']) == 103
    assert len(set(contract['patient_groups'].values())) == 69
    assert contract['bootstrap'] == dict(iterations=20000, seed=20260831, frame='69_patient_groups_common_finite_per_metric')
    for name, value in json.loads((root / 'contract/PACKAGE_SHA256.json').read_text())['files'].items():
        assert sha(root / name) == value, 'frozen Bio MET source drift: ' + name
    parent = contract['parent_comparison']
    parent_root = Path(parent['attempt'])
    complete = json.loads(check_file(parent['complete']).read_text())
    assert complete['job_id'] == '4319117' and complete['models'] == list(OLD)
    assert complete['cases'] == 103 and complete['metric_columns'] == 72 and not complete['locked_test_read']
    assert not (parent_root / 'reports/FAILED.json').exists(), 'parent FAILED/COMPLETE conflict'
    assert sha(parent_root / 'contract/COMPARISON_CONTRACT.json') == complete['contract_sha256'] == parent['contract_sha256']
    for name, value in json.loads(check_file(parent['package']).read_text())['files'].items():
        assert sha(parent_root / name) == value, 'parent comparison package drift'
    old_contract = json.loads((parent_root / 'contract/COMPARISON_CONTRACT.json').read_text())
    for key in ('references', 'patient_groups', 'evaluator', 'evaluator_prefix', 'evaluator_bin_sha256'):
        assert contract[key] == old_contract[key], 'parent scoring contract changed: ' + key
    for name, entry in parent['outputs'].items():
        assert entry == complete['outputs'][name]
        check_file(dict(path=str(parent_root / name), **entry))
    replay = json.loads((parent_root / 'reports/HISTORICAL_REPLAY.json').read_text())['models']
    assert set(replay) == {'K1_175', 'original_E_200'} and all(v['pass_per_case_72_metrics'] for v in replay.values())
    for model in OLD:
        entry = contract['models'][model]['metrics']
        assert entry == dict(path=str(parent_root / ('evaluation/' + model + '/leaderboard_metrics.csv')), **complete['outputs']['evaluation/' + model + '/leaderboard_metrics.csv'])
        fields, names, _ = metrics(check_file(entry))
        assert set(names) == set(contract['references'])
    for entry in contract['references'].values():
        check_file(entry)
    prediction = contract['models'][BIO]
    bio_complete = json.loads(check_file(prediction['complete']).read_text())
    manifest = json.loads(check_file(prediction['manifest']).read_text())
    assert bio_complete['job_id'] == '4319113'
    assert bio_complete['cases'] == manifest['cases'] == 103
    assert bio_complete['model'] == BIO
    assert bio_complete['prediction_manifest_sha256'] == prediction['manifest']['sha256']
    assert bio_complete['contract_sha256'] == manifest['contract_sha256'] == prediction['contract_sha256']
    assert sha(check_file(prediction['contract'])) == prediction['contract_sha256']
    package = json.loads(check_file(prediction['package']).read_text())
    bio_root = Path(prediction['contract']['path']).parents[1]
    for name, value in package['files'].items():
        assert sha(bio_root / name) == value, 'Bio inference package drift'
    assert not (Path(prediction['complete']['path']).parent / 'FAILED.json').exists()
    assert set(manifest['files']) == set(prediction['predictions']) == set(contract['references'])
    for name, entry in prediction['predictions'].items():
        assert {k: entry[k] for k in ('bytes', 'sha256')} == manifest['files'][name]
        assert Path(entry['path']).parent == Path(prediction['prediction_root']) and Path(entry['path']).name == name
        check_file(entry)
    assert {p.name for p in Path(prediction['prediction_root']).glob('*.nii.gz')} == set(manifest['files'])
    for name, entry in prediction['identity_markers'].items():
        marker = json.loads(check_file(entry).read_text())
        if name == 'STARTED.json':
            assert marker['job_id'] == '4319113' and marker['model'] == BIO, 'Bio STARTED identity drift'
            assert marker['contract_sha256'] == prediction['contract_sha256']
        else:
            assert name == 'RUN_CONTRACT.json' and marker['fold_selection'] == {'A': [0, 1, 2, 3, 4], 'B': [1, 3]}
            assert marker['cases'] == 103 and marker['author_revision'] == contract['Bio_author_revision']
    assert sys.version.split()[0] == '3.10.18'
    assert {name: metadata.version(name) for name in contract['evaluator']['versions']} == contract['evaluator']['versions']
    import brats_evaluation, panoptica
    sources = {}
    for name, module in [('brats_evaluation', brats_evaluation), ('panoptica', panoptica)]:
        base = Path(module.__file__).parent
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
                sources[name + '/' + str(path.relative_to(base))] = sha(path)
    assert sources == contract['evaluator']['source_sha256'], 'MET source drift'
    for name, value in contract['evaluator_bin_sha256'].items():
        assert sha(Path(contract['evaluator_prefix']) / 'bin' / name) == value, 'MET executable drift'
    return contract


def paired_rows(first, second, loaded, contract, rng):
    fields, names, a = loaded[first]
    b = loaded[second][2]
    result = []
    for index, field in enumerate(fields):
        valid = np.isfinite(a[:, index]) & np.isfinite(b[:, index])
        delta = a[valid, index] - b[valid, index]
        sign = direction(field)
        oriented = delta * sign if sign else delta
        groups = [contract['patient_groups'][name] for name, use in zip(names, valid) if use]
        row = dict(first=first, second=second, metric=field, first_mean=finite_mean(a[:, index]), second_mean=finite_mean(b[:, index]),
            common_finite_cases=len(delta), first_common_mean=finite_mean(a[valid, index]), second_common_mean=finite_mean(b[valid, index]),
            raw_delta=finite_mean(delta), direction='higher' if sign == 1 else 'lower' if sign == -1 else 'descriptive_std',
            oriented_delta=finite_mean(oriented), wins=int((oriented > 1e-12).sum()), ties=int((np.abs(oriented) <= 1e-12).sum()),
            losses=int((oriented < -1e-12).sum()), bootstrap_iterations=20000, seed=20260831)
        if len(delta):
            row.update(paired_bootstrap(oriented, groups, rng))
        else:
            row.update(patient_groups=0, ci_low=None, ci_high=None, patient_equal_delta=None, patient_equal_ci_low=None, patient_equal_ci_high=None)
        result.append(row)
    return result


def summarize_four(root, contract):
    loaded = {model: metrics(root / 'evaluation' / model / 'leaderboard_metrics.csv') for model in contract['model_order']}
    fields, names, _ = loaded[OLD[0]]
    assert all(item[:2] == (fields, names) for item in loaded.values()), 'four-model MET schema/case drift'
    assert set(names) == set(contract['patient_groups'])
    parent = Path(contract['parent_comparison']['attempt'])
    means = read_rows(parent / 'reports/MEANS_216.csv')
    pairs = read_rows(parent / 'reports/PAIRED_ALL_216.csv')
    assert len(means) == len(pairs) == 216
    assert len({(r['model'], r['metric']) for r in means}) == 216
    assert {(r['model'], r['metric']) for r in means} == {(m, f) for m in OLD for f in fields}
    expected_pairs = {(a, b, f) for a, b in ((OLD[0], OLD[1]), (OLD[0], OLD[2]), (OLD[1], OLD[2])) for f in fields}
    assert {(r['first'], r['second'], r['metric']) for r in pairs} == expected_pairs and len(expected_pairs) == 216
    for row in means:
        values = loaded[row['model']][2][:, fields.index(row['metric'])]
        mean = finite_mean(values)
        assert int(row['finite_cases']) == int(np.isfinite(values).sum())
        assert (row['mean'] == '' and mean is None) or (mean is not None and np.isclose(float(row['mean']), mean, rtol=1e-12, atol=1e-12)), 'old mean drift'
    for index, field in enumerate(fields):
        values = loaded[BIO][2][:, index]
        means.append(dict(model=BIO, metric=field, mean=finite_mean(values), finite_cases=int(np.isfinite(values).sum())))
    # Old confidence intervals remain untouched; the three new pairs have their own fixed RNG stream.
    rng = np.random.default_rng(20260831)
    for model in OLD:
        pairs.extend(paired_rows(model, BIO, loaded, contract, rng))
    assert len(means) == 288 and len(pairs) == 432
    csv_write(root / 'reports/MEANS_288.csv', means)
    csv_write(root / 'reports/PAIRED_ALL_432.csv', pairs)
    assert read_rows(root / 'reports/MEANS_288.csv')[:216] == read_rows(parent / 'reports/MEANS_216.csv')
    assert read_rows(root / 'reports/PAIRED_ALL_432.csv')[:216] == read_rows(parent / 'reports/PAIRED_ALL_216.csv')
    lookup = {(r['model'], r['metric']): r for r in means}
    lines = ['# Same-fixed103 four-model descriptive MET comparison', '',
        '103 validation scans / 69 patient groups. Not locked test104, external held-out validation, or official ranking.',
        'BraTS-evaluation0.0.8 / panoptica2.1.0 / NumPy1.26.4; mets, 27mm^3, parser DSC0.2. TC/WT exclude RC.',
        'Bio: complete A5+B2 author ensemble/postprocessing, inference4319113. Original scores: CPU4319117.',
        'Original 216 mean and 216 paired rows/CI are reused unchanged; only Bio is newly evaluated.',
        'NA is not zero. Paired comparisons use common-finite scans and 20000 patient-group resamples, seed20260831.',
        'K1 used this development cohort. EXT2025 public training overlap and Bio training exposure remain disclosed.',
        'Intervals are descriptive, not formal superiority. Standard-deviation columns have no quality direction.', '']
    for region in ('et', 'rc', 'tc', 'wt'):
        lines.extend(['## ' + region.upper(), '', '| Metric | ' + ' | '.join(contract['model_order']) + ' |', '|---|' + '---|' * 4])
        for field in fields:
            if not field.endswith('_' + region):
                continue
            cells = []
            for model in contract['model_order']:
                row = lookup[model, field]
                cells.append('NA (n=0)' if row['mean'] in ('', None) else f"{float(row['mean']):.6f} (n={row['finite_cases']})")
            lines.append('| ' + field + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    with (root / 'reports/COMPARISON_REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    return dict(mean_rows=288, paired_rows=432, old_mean_and_paired_rows_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = args.root
    stage = 'contract_and_parent_identity'
    started = datetime.now(timezone.utc).isoformat()
    try:
        contract = validate_contract(root, os.environ['COMPARE_EXPECTED_CONTRACT_SHA256'])
        if args.validate_only:
            print(json.dumps(dict(META, status='exact_environment_parent_inputs_SHA_pass_no_voxel_scoring', cases=103, remote_utc=datetime.now(timezone.utc).isoformat())))
            return
        write(root / 'reports/STARTED.json', dict(META, job_id=os.environ['SLURM_JOB_ID'], started_utc=started, contract_sha256=sha(root / 'contract/COMPARISON_CONTRACT.json')))
        stage = 'native_GT_Bio_validation'
        import nibabel as nib
        reference = root / 'stage/reference'
        prediction = root / 'stage' / BIO
        reference.mkdir()
        prediction.mkdir()
        for name, entry in contract['references'].items():
            ref_path = check_file(entry)
            pred_path = check_file(contract['models'][BIO]['predictions'][name])
            ref = nib.load(str(ref_path))
            pred = nib.load(str(pred_path))
            assert len(ref.shape) == 3 and pred.shape == ref.shape and np.allclose(pred.affine, ref.affine, rtol=0, atol=1e-5), 'native geometry drift: ' + name
            assert np.isfinite(ref.affine).all() and np.isfinite(pred.affine).all(), 'nonfinite affine'
            for image in (ref, pred):
                values = np.asanyarray(image.dataobj)
                assert np.isfinite(values).all() and set(np.unique(values)) <= {0, 1, 2, 3, 4}, 'illegal/finite label: ' + name
            (reference / name).symlink_to(ref_path)
            (prediction / name).symlink_to(pred_path)
        stage = 'official_MET_Bio_103'
        with (root / 'logs/Bio_evaluation.log').open('x') as log:
            subprocess.run(['/bin/bash', str(root / 'runtime/run_official_met_eval.sh'), contract['evaluator_prefix'], str(prediction), str(reference), str(root / 'evaluation' / BIO), '103'], stdout=log, stderr=subprocess.STDOUT, check=True)
        fields, names, _ = metrics(root / 'evaluation' / BIO / 'leaderboard_metrics.csv')
        assert set(names) == set(contract['references']) and len(fields) == 72
        write(root / 'reports/PROGRESS_Bio.json', dict(META, model=BIO, cases=103, metric_columns=72, remote_utc=datetime.now(timezone.utc).isoformat()))
        for model in OLD:
            folder = root / 'evaluation' / model
            folder.mkdir()
            (folder / 'leaderboard_metrics.csv').symlink_to(check_file(contract['models'][model]['metrics']))
        stage = 'four_model_summary'
        summary = summarize_four(root, contract)
        validate_contract(root, os.environ['COMPARE_EXPECTED_CONTRACT_SHA256'])
        write(root / 'reports/PARENT_REUSE_VERIFICATION.json', dict(META, parent_job_id='4319117', old_scores_recomputed=False, old_CI_recomputed=False, **summary))
        output = {}
        for folder in ('reports', 'evaluation', 'logs'):
            for path in sorted((root / folder).rglob('*')):
                if path.is_file():
                    output[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
        write(root / 'reports/COMPLETE.json', dict(META, status='four_model_fixed103_MET_descriptive_comparison_complete',
            job_id=os.environ['SLURM_JOB_ID'], cases=103, models=contract['model_order'], metric_columns=72, **summary,
            outputs=output, contract_sha256=sha(root / 'contract/COMPARISON_CONTRACT.json'), started_utc=started,
            remote_utc=datetime.now(timezone.utc).isoformat(), bootstrap_seed=20260831, bootstrap_iterations=20000,
            locked_test_read=False, raw_MRI_read=False, scientific_superiority_pass=False))
    except BaseException as error:
        if not args.validate_only:
            write(root / 'reports/FAILED.json', dict(META, stage=stage, error_type=type(error).__name__, message=str(error), remote_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
